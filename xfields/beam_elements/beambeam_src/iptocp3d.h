#ifndef XFIELDS_IPTOCP3D_H
#define XFIELDS_IPTOCP3D_H

#if !defined(mysign)
    #define mysign(a) (((a) >= 0) - ((a) < 0))
#endif

void MacropartToCP(double x,
                   double y,
                   double z,
                   double px,
                   double py,
                   const double x_c, // compute boosted centroids at IP of other slice in advance
                   const double y_c,
                   const double z_c,
                   const double px_c,
                   const double py_c,
                   double* Sx_i, double* Sy_i, double* Sz_i){

    // in order to move macroparts of slice 1 to their CP need to have centroid of slice 2 in advance
    // for this slice 2 is collapsed onto the Z of the centroid as a thin lens
    // each macropart of slice 1 will interact with this thin le§ns at a different CP
    (*Sz_i) = 0.5*(z - z_c); // CP for macropart
    (*Sx_i) = (x + px*(*Sz_i)) - (x_c - px_c*(*Sz_i)); // transversal coordinates w.r.t slice 2 centroid
    (*Sy_i) = (y + py*(*Sz_i)) - (y_c - py_c*(*Sz_i));

    // all 0 unless specified outside

    printf("x_c: %.10f",x_c);
    printf("y_c: %.10f",y_c);
    printf("z_c: %.10f",z_c);
    printf("px_c: %.10f",px_c);
    printf("py_c: %.10f",py_c);

}

/*gpufun*/
void IPToCP3D_track_local_particle(IPToCP3DData el, 
		 	   LocalParticle* part){
	
    // boosted centroid of other beam at IP
    /*gpuglmem*/ const double* x_centroid  = IPToCP3DData_getp_x_centroid(el);
    /*gpuglmem*/ const double* y_centroid  = IPToCP3DData_getp_y_centroid(el);
    /*gpuglmem*/ const double* z_centroid  = IPToCP3DData_getp_z_centroid(el);
    /*gpuglmem*/ const double* px_centroid = IPToCP3DData_getp_px_centroid(el);
    /*gpuglmem*/ const double* py_centroid = IPToCP3DData_getp_py_centroid(el);
    const int64_t slice_id                 = IPToCP3DData_get_slice_id(el);
    const int64_t is_sliced                = IPToCP3DData_get_is_sliced(el);
    
    int64_t const n_part = LocalParticle_get_num_particles(part); //only_for_context cpu_serial cpu_openmp
    
    // loop over macroparticles
    for (int ii=0; ii<n_part; ii++){ //only_for_context cpu_serial cpu_openmp
        part->ipart = ii;            //only_for_context cpu_serial cpu_openmp

        // these already boosted probably
    	double x = LocalParticle_get_x(part);
    	double px = LocalParticle_get_px(part);
    	double y = LocalParticle_get_y(part);
    	double py = LocalParticle_get_py(part);
    	double z = LocalParticle_get_zeta(part);
    	double delta = LocalParticle_get_delta(part);

        // macropart state: 0=dead, 1=alive, 100X: part of slice X
        int64_t state = LocalParticle_get_state(part);
        state -= 1000;

        // code is executed only if macropart is in correct slice or if there are no slices
        if(state == slice_id || is_sliced == 0){

            // move slice 1 macroparts to CP nusing slice 2 centroid
            double Sx_i, Sy_i, Sz_i; // slice 1 macropart coords at CP
            MacropartToCP(x, y, z, px, py,
                      *x_centroid, *y_centroid, *z_centroid, *px_centroid, *py_centroid,
                      &Sx_i, &Sy_i, &Sz_i);
 
            // variables at CP (only x,y,z changes; x,y are w.r.t to the centroid of the other slice)
    	    LocalParticle_set_x(part, Sx_i);
    	    LocalParticle_set_px(part, px);
    	    LocalParticle_set_y(part, Sy_i);
    	    LocalParticle_set_py(part, py);
    	    LocalParticle_set_zeta(part, Sz_i);
    	    LocalParticle_update_delta(part, delta);
	}
    } //only_for_context cpu_serial cpu_openmp
}


#endif
