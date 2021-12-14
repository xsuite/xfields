#ifndef XFIELDS_IPTOCP3D_H
#define XFIELDS_IPTOCP3D_H

#if !defined(mysign)
    #define mysign(a) (((a) >= 0) - ((a) < 0))
#endif

void MacropartToCP(const unsigned int use_strongstrong,
// first beam macropart
                   double x1,
                   double y1,
                   const double z1,  // this is z of the slice centroid not the single macropart
                   double px1,
                   double py1,
// second beam slice
                   const double x2_c, // boosted centroid of other slice at IP (= vector pointing from my full beam/slice centoid to other beam slice centroid)
                   const double y2_c,
                   const double z2_c,
                   const double px2_c,
                   const double py2_c,
                   const double x2_bc, // boosted centroid of other full beam from before slicing (or slice, when ref frames are transformed slice by slice, so same as x_c) at IP (= vector pointing from my full beam/slice centroid to other full beam/slice centroid), w.r.t other full beam centroid
                   const double y2_bc,
                   double* Sx_i, double* Sy_i, double* Sz_i){

    // in order to move macroparts of slice 1 to their CP need to have centroid of slice 2 in advance
    // for this slice 2 is collapsed onto the Z of the centroid as a thin lens
    // each macropart of slice 1 will interact with this thin le§ns at a different CP
    (*Sz_i) = 0.5*(z1 - z2_c); // CP for macropart

    // weakstrong model: center of ref. is full beam 2 centroid: x^cp_weak_macropart + xc_strong_fullbeam - xc^cp_strong_slice = Sx_i w.r.t. strong slice centroid at CP  
    if (use_strongstrong == 0){
        (*Sx_i) = x1 + px1*(*Sz_i) + x2_bc - (x2_c - px2_c*(*Sz_i)); //- (x_c - px_c*(*Sz_i)); 
        (*Sy_i) = y1 + py1*(*Sz_i) + y2_bc - (y2_c - py2_c*(*Sz_i)); //- (y_c - py_c*(*Sz_i));

    // strongstorng model: center of ref. is the barycentric frame with common origin: x^cp_beam1_macropart - xc^cp_beam2_slice = Sx_i w.r.t. strong slice centroid at CP  
    }else if (use_strongstrong == 1){
        (*Sx_i) = x1 + px1*(*Sz_i) - (x2_c - px2_c*(*Sz_i)); 
        (*Sy_i) = y1 + py1*(*Sz_i) - (y2_c - py2_c*(*Sz_i));
    }
}


/*gpufun*/
void IPToCP3D_track_local_particle(IPToCP3DData el, 
		 	   LocalParticle* part){
//    clock_t tt;
//    tt = clock(); 
	
    // boosted centroid of other beam slice at IP
    /*gpuglmem*/ const double* x_bb_centroid  = IPToCP3DData_getp_x_bb_centroid(el);
    /*gpuglmem*/ const double* y_bb_centroid  = IPToCP3DData_getp_y_bb_centroid(el);
    /*gpuglmem*/ const double* z_bb_centroid  = IPToCP3DData_getp_z_bb_centroid(el);
    /*gpuglmem*/ const double* px_bb_centroid = IPToCP3DData_getp_px_bb_centroid(el);
    /*gpuglmem*/ const double* py_bb_centroid = IPToCP3DData_getp_py_bb_centroid(el);

    // boosted centroid of other full beam at IP
    /*gpuglmem*/ const double* x_full_bb_centroid  = IPToCP3DData_getp_x_full_bb_centroid(el);
    /*gpuglmem*/ const double* y_full_bb_centroid  = IPToCP3DData_getp_y_full_bb_centroid(el);

    // boosted centroid of this beam slice at IP
    /*gpuglmem*/ const double* z_centroid     = IPToCP3DData_getp_z_centroid(el);
 
    // get additional params of the beambeam element
    const int64_t slice_id                 = IPToCP3DData_get_slice_id(el);
    const int64_t is_sliced                = IPToCP3DData_get_is_sliced(el);
    const unsigned int use_strongstrong    = IPToCP3DData_get_use_strongstrong(el);
    
    int64_t const n_part = LocalParticle_get_num_particles(part); //only_for_context cpu_serial cpu_openmp

    // loop over macroparticles
    for (int ii=0; ii<n_part; ii++){ //only_for_context cpu_serial cpu_openmp
        part->ipart = ii;            //only_for_context cpu_serial cpu_openmp

        // macropart state: 0=dead, 1=alive, 100X: part of slice X
        int64_t state = LocalParticle_get_state(part);

        // code is executed only if macropart is in correct slice or if there are no slices
        int cond;
        cond = (state == slice_id || is_sliced == 0);

//        clock_t t2;
//        t2 = clock();
 
        if(cond){
	
            // these already boosted
       	    double x = LocalParticle_get_x(part);
    	    double px = LocalParticle_get_px(part);
    	    double y = LocalParticle_get_y(part);
    	    double py = LocalParticle_get_py(part);
    	    double z = LocalParticle_get_zeta(part);
    	    double delta = LocalParticle_get_delta(part);

            // move slice 1 macroparts to CP nusing slice 2 centroid
            double Sx_i, Sy_i, Sz_i; // slice 1 macropart coords at CP
            MacropartToCP(use_strongstrong,
                      x, y, *z_centroid, px, py,
                      *x_bb_centroid, *y_bb_centroid, *z_bb_centroid, *px_bb_centroid, *py_bb_centroid,
                      *x_full_bb_centroid, *y_full_bb_centroid,
                      &Sx_i, &Sy_i, &Sz_i);
 
            // variables at CP (only x,y,z changes; x,y are w.r.t to the centroid of the other slice)
    	    LocalParticle_set_x(part, Sx_i);
    	    LocalParticle_set_px(part, px);
    	    LocalParticle_set_y(part, Sy_i);
    	    LocalParticle_set_py(part, py);
    	    LocalParticle_set_zeta(part, Sz_i);
    	    LocalParticle_update_delta(part, delta);
        }
 
//        t2 = clock() - t2;
//        double time_taken_2 = ((double)t2)/CLOCKS_PER_SEC;
//        printf("[iptocp.h] IPtoCP one if: el_sliceid: %d, part_state: %d, entered_if: %d, took %.8f seconds to execute\n", slice_id, state, cond, time_taken_2);


   } //only_for_context cpu_serial cpu_openmp
//   tt = clock() - tt;
//   double ttime_taken = ((double)tt)/CLOCKS_PER_SEC;
//   printf("[iptocp.h] IPtoCP full took %.8f seconds to execute\n", ttime_taken);


}


#endif
