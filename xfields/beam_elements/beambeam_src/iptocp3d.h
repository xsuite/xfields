#ifndef XFIELDS_IPTOCP3D_H
#define XFIELDS_IPTOCP3D_H

#if !defined(mysign)
    #define mysign(a) (((a) >= 0) - ((a) < 0))
#endif

void MacropartToCP(double x,
                   double y,
                   const double z,
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
    
    // x_w^cp - x_s^ip + x_s^cp = Sx_i w.r.t. strong slice centroid at CP
    (*Sx_i) = (x + px*(*Sz_i)) + px_c*(*Sz_i); //- (x_c - px_c*(*Sz_i)); 
    (*Sy_i) = (y + py*(*Sz_i)) + py_c*(*Sz_i); //- (y_c - py_c*(*Sz_i));

    // all 0 unless specified outside
//    printf("x_c: %.12f\n",   x_c);
//    printf("y_c: %.12f\n",   y_c);
//    printf("z_c: %.12f\n",   z_c);
//    printf("px_c: %.12f\n", px_c);
//    printf("py_c: %.12f\n", py_c);


}

/*gpufun*/
void IPToCP3D_track_local_particle(IPToCP3DData el, 
		 	   LocalParticle* part){
	
    // boosted centroid of other beam at IP
    /*gpuglmem*/ const double* x_bb_centroid  = IPToCP3DData_getp_x_bb_centroid(el);
    /*gpuglmem*/ const double* y_bb_centroid  = IPToCP3DData_getp_y_bb_centroid(el);
    /*gpuglmem*/ const double* z_bb_centroid  = IPToCP3DData_getp_z_bb_centroid(el);
    /*gpuglmem*/ const double* px_bb_centroid = IPToCP3DData_getp_px_bb_centroid(el);
    /*gpuglmem*/ const double* py_bb_centroid = IPToCP3DData_getp_py_bb_centroid(el);
    /*gpuglmem*/ const double* z_centroid     = IPToCP3DData_getp_z_centroid(el); 
    const int64_t slice_id                 = IPToCP3DData_get_slice_id(el);
    const int64_t is_sliced                = IPToCP3DData_get_is_sliced(el);
    
    int64_t const n_part = LocalParticle_get_num_particles(part); //only_for_context cpu_serial cpu_openmp
//    int64_t counter = 0;
 
//    printf("slice %d: x_c:  %.20f\n",  slice_id, *x_bb_centroid);
//    printf("slice %d: y_c:  %.20f\n",  slice_id, *y_bb_centroid);
//    printf("slice %d: z_c:  %.20f\n",  slice_id, *z_bb_centroid);
//    printf("slice %d: px_c: %.20f\n", slice_id, *px_bb_centroid);
//    printf("slice %d: py_c: %.20f\n", slice_id, *py_bb_centroid);

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
//        printf("part %d: sliceid, state: %d, %d\n", ii, slice_id, state);

        // code is executed only if macropart is in correct slice or if there are no slices
        if(state == slice_id || is_sliced == 0){

//           counter++;
           // move slice 1 macroparts to CP nusing slice 2 centroid
            double Sx_i, Sy_i, Sz_i; // slice 1 macropart coords at CP
            MacropartToCP(x, y, *z_centroid, px, py,
                      *x_bb_centroid, *y_bb_centroid, *z_bb_centroid, *px_bb_centroid, *py_bb_centroid,
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
//   printf("counter: %d\n", counter);
 
}


#endif
