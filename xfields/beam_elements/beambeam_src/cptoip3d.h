#ifndef XFIELDS_CPTOIP3D_H
#define XFIELDS_CPTOIP3D_H

#if !defined(mysign)
    #define mysign(a) (((a) >= 0) - ((a) < 0))
#endif

void MacropartToIP(double Sx_i,
                   double Sy_i,
                   double Sz_i,
                   double px,
                   double py,
                   const double x_c, // compute stats of other slice in advance
                   const double y_c,
                   const double z_c,
                   const double px_c,
                   const double py_c,
                   double* x, double* y, double* z){

    // in order to move macroparts of slice 1 to their CP need to have centroid of slice 2 in advance
    // for this slice 2 is collapsed onto the Z of the centroid as a thin lens
    // each macropart of slice 1 will interact with this thin le§ns at a different CP
    (*z) = 2*Sz_i + z_c; // CP for macropart
    
/*
    printf(" px: %.10f,", px); 
    printf(" py: %.10f,", py); 
    printf(" Sz_i*px: %.10f,", Sz_i*px); 
    printf(" Sz_i*py: %.10f\n", Sz_i*py); 
*/

    (*x) = (Sx_i - px*Sz_i) + (x_c - px_c*Sz_i);
    (*y) = (Sy_i - py*Sz_i) + (y_c - py_c*Sz_i);

}

/*gpufun*/
void CPToIP3D_track_local_particle(CPToIP3DData el, 
		 	   LocalParticle* part){
	
    // boosted centroid coords of beam 2 passed in with element.update()
    /*gpuglmem*/ const double* x_centroid  = CPToIP3DData_getp_x_centroid(el);
    /*gpuglmem*/ const double* y_centroid  = CPToIP3DData_getp_y_centroid(el);
    /*gpuglmem*/ const double* z_centroid  = CPToIP3DData_getp_z_centroid(el);
    /*gpuglmem*/ const double* px_centroid = CPToIP3DData_getp_px_centroid(el);
    /*gpuglmem*/ const double* py_centroid = CPToIP3DData_getp_py_centroid(el);
    const int64_t slice_id                 = CPToIP3DData_get_slice_id(el);
    const int64_t is_sliced                = CPToIP3DData_get_is_sliced(el);
    int64_t const n_part = LocalParticle_get_num_particles(part); //only_for_context cpu_serial cpu_openmp
    
    // loop over macroparticles
    for (int ii=0; ii<n_part; ii++){ //only_for_context cpu_serial cpu_openmp
        part->ipart = ii;            //only_for_context cpu_serial cpu_openmp

        // these already boosted probably
    	double Sx_i = LocalParticle_get_x(part);
    	double px = LocalParticle_get_px(part);
    	double Sy_i = LocalParticle_get_y(part);
    	double py = LocalParticle_get_py(part);
    	double Sz_i = LocalParticle_get_zeta(part);
    	double delta = LocalParticle_get_delta(part);

        // macropart state: 0=dead, 1=alive, 100X: part of slice X
        int64_t state = LocalParticle_get_state(part);
        state -= 1000;

        // code is executed only if macropart is in correct slice or if there are no slices
        if(state == slice_id || is_sliced == 0){

            // move slice 1 macroparts to CP nusing slice 2 centroid
            double x, y, z; // slice 1 macropart coords at CP
            MacropartToIP(Sx_i, Sy_i, Sz_i, px, py,
                  *x_centroid, *y_centroid, *z_centroid, *px_centroid, *py_centroid,
                  &x, &y, &z);
 
            // variables at CP (only x,y,z changes; x,y are w.r.t to the centroid of the other slice)
            LocalParticle_set_x(part, x);
      	    LocalParticle_set_px(part, px);
    	    LocalParticle_set_y(part, y);
      	    LocalParticle_set_py(part, py);
    	    LocalParticle_set_zeta(part, z);
    	    LocalParticle_update_delta(part, delta);
        }	
    } //only_for_context cpu_serial cpu_openmp
}


#endif
