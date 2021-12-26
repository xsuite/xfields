#ifndef XFIELDS_IPTOCP3D_H
#define XFIELDS_IPTOCP3D_H

#if !defined(mysign)
    #define mysign(a) (((a) >= 0) - ((a) < 0))
#endif

void MacropartToCP(const unsigned int use_strongstrong,
                   double x1, double y1, const double z1, double px1, double py1,
                   const double x2_c, const double y2_c, const double z2_c, const double px2_c, const double py2_c,// boosted centroid of other slice at IP (= vector pointing from my full beam/slice centoid to other beam slice centroid)
                   const double x2_bc, const double y2_bc, // boosted centroid of other full beam (or slice, when ref frames are transformed slice by slice, so same as x_c) at IP (= vector pointing from my full beam/slice centroid to other full beam/slice centroid), w.r.t other full beam centroid
                   double* Sx_i, double* Sy_i, double* Sz_i){

    // in order to move macroparts of slice 1 to their CP need to have centroid of slice 2 in advance
    // for this slice 2 is collapsed onto the Z of the centroid as a thin lens
    // each macropart of slice 1 will interact with this thin lens at CP
    (*Sz_i) = 0.5*(z1 - z2_c); // CP for macropart

    // weakstrong model: center of ref. is full beam 2 centroid: x^cp_weak_macropart + xc_strong_fullbeam - xc^cp_strong_slice = Sx_i w.r.t. strong slice centroid at CP  
    if (use_strongstrong == 0){

        // if beams have 1 slice, then x2_bc = x2_c
        (*Sx_i) = (x1 + px1*(*Sz_i)) + x2_bc - (x2_c - px2_c*(*Sz_i)); 
        (*Sy_i) = (y1 + py1*(*Sz_i)) + y2_bc - (y2_c - py2_c*(*Sz_i));

    // strongstorng model: center of ref. is the barycentric frame with common origin: x^cp_beam1_macropart - xc^cp_beam2_slice = Sx_i w.r.t. strong slice centroid at CP  
    }else if (use_strongstrong == 1){
        (*Sx_i) = (x1 + px1*(*Sz_i)) - (x2_c - px2_c*(*Sz_i)); 
        (*Sy_i) = (y1 + py1*(*Sz_i)) - (y2_c - py2_c*(*Sz_i));
    }
}


/*gpufun*/
void IPToCP3D_track_local_particle(IPToCP3DData el, 
		 	   LocalParticle* part0){
//    clock_t tt;
//    tt = clock(); 
//    tt = clock() - tt;

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
    /*gpuglmem*/ const double* z_centroid  = IPToCP3DData_getp_z_centroid(el);
 
    // get additional params of the beambeam element
    const int64_t slice_id                 = IPToCP3DData_get_slice_id(el);
    const int64_t is_sliced                = IPToCP3DData_get_is_sliced(el);
    const unsigned int use_strongstrong    = IPToCP3DData_get_use_strongstrong(el);

    int counter = 0;   

    //start_per_particle_block (part0->part)
        
        // macropart state: 0=dead, 1=alive, 100X: part of slice X
        int64_t state = LocalParticle_get_state(part);

        // code is executed only if macropart is in correct slice or if there are no slices 
        if(state == slice_id || is_sliced == 0){
	    counter+=1;
 
            // these already boosted
            double x = LocalParticle_get_x(part);
      	    double px = LocalParticle_get_px(part);
            double y = LocalParticle_get_y(part);
    	    double py = LocalParticle_get_py(part);
    	    double zeta = LocalParticle_get_zeta(part);
    	    double delta = LocalParticle_get_delta(part);       
   
            // slice 1 macropart coords at CP
            double Sx_i, Sy_i, Sz_i;

            MacropartToCP(use_strongstrong,
                      x, y, *z_centroid, px, py,
                      *x_bb_centroid, *y_bb_centroid, *z_bb_centroid, *px_bb_centroid, *py_bb_centroid,
                      *x_full_bb_centroid, *y_full_bb_centroid,
                      &Sx_i, &Sy_i, &Sz_i);

            // variables at CP (only x,y,z changes; x,y are w.r.t to the centroid of the other slice at CP)
       	    LocalParticle_set_x(part, Sx_i);
    	    LocalParticle_set_px(part, px);
    	    LocalParticle_set_y(part, Sy_i);
    	    LocalParticle_set_py(part, py);
    	    LocalParticle_set_zeta(part, Sz_i);
    	    LocalParticle_update_delta(part, delta);
        }   
   //end_per_particle_block
//    double ttime_taken = ((double)tt)/CLOCKS_PER_SEC;
//    printf("[iptocp.h] iptocp full took %.8f seconds to execute\nCounter: %d", ttime_taken, counter);

}

#endif
