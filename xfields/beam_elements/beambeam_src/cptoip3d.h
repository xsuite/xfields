#ifndef XFIELDS_CPTOIP3D_H
#define XFIELDS_CPTOIP3D_H

#if !defined(mysign)
    #define mysign(a) (((a) >= 0) - ((a) < 0))
#endif

void MacropartToIP(const unsigned int use_strongstrong,
// first beam macropart
                   double Sx_i,
                   double Sy_i,
                   double Sz_i,
                   double px1,
                   double py1,
// second beam slice
                   const double x2_c, // compute stats of other slice in advance
                   const double y2_c,
                   const double z2_c,
                   const double px2_c,
                   const double py2_c,
                   const double x2_bc, // centroid of other full beam at IP 
                   const double y2_bc,
                   double* x1, double* y1, double* z1){

    // in order to move macroparts of slice 1 to their CP need to have centroid of slice 2 in advance
    // for this slice 2 is collapsed onto the Z of the centroid as a thin lens
    // each macropart of slice 1 will interact with this thin le§ns at a different CP
    (*z1) = 2*Sz_i + z2_c; // CP for macropart, here we get back the centroid of beam 1 if combi

    // weakstrong model: center of ref. is full beam 2 centroid: x^cp_weak_macropart + xc_strong_fullbeam - xc^cp_strong_slice = Sx_i w.r.t. strong slice centroid at CP   
    if (use_strongstrong == 0){        

        // if beams have 1 slice, then x2_bc = x2_c
        (*x1) = (Sx_i - px1*Sz_i) - x2_bc + (x2_c - px2_c*Sz_i);//+ px_c*Sz_i; // 
        (*y1) = (Sy_i - py1*Sz_i) - y2_bc + (y2_c - py2_c*Sz_i);//+ py_c*Sz_i; //

    // strongstorng model: center of ref. is the barycentric frame with common origin: x^cp_beam1_macropart - xc^cp_beam2_slice = Sx_i w.r.t. strong slice centroid at CP  
    }else if (use_strongstrong == 1){

        // if beams have 1 slice, then x2_bc = x2_c
        (*x1) = (Sx_i - px1*Sz_i) + (x2_c - px2_c*Sz_i); 
        (*y1) = (Sy_i - py1*Sz_i) + (y2_c - py2_c*Sz_i);
    }
}

/*gpufun*/
void CPToIP3D_track_local_particle(CPToIP3DData el, 
		 	   LocalParticle* part){
	
    // boosted centroid coords of beam 2 passed in with element.update()
    /*gpuglmem*/ const double* x_bb_centroid  = CPToIP3DData_getp_x_bb_centroid(el);
    /*gpuglmem*/ const double* y_bb_centroid  = CPToIP3DData_getp_y_bb_centroid(el);
    /*gpuglmem*/ const double* z_bb_centroid  = CPToIP3DData_getp_z_bb_centroid(el);
    /*gpuglmem*/ const double* px_bb_centroid = CPToIP3DData_getp_px_bb_centroid(el);
    /*gpuglmem*/ const double* py_bb_centroid = CPToIP3DData_getp_py_bb_centroid(el);

    // boosted centroid of other full beam at IP
    /*gpuglmem*/ const double* x_full_bb_centroid  = CPToIP3DData_getp_x_full_bb_centroid(el);
    /*gpuglmem*/ const double* y_full_bb_centroid  = CPToIP3DData_getp_y_full_bb_centroid(el);

    // boosted centroid of my beam slice at IP
    /*gpuglmem*/ const double* z_centroid  = CPToIP3DData_getp_z_centroid(el);

    // get additional params of the beambeam element
    const int64_t slice_id                 = CPToIP3DData_get_slice_id(el);
    const int64_t is_sliced                = CPToIP3DData_get_is_sliced(el);
    const unsigned int use_strongstrong    = CPToIP3DData_get_use_strongstrong(el);
  
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
        // state -= 1000;

        // code is executed only if macropart is in correct slice or if there are no slices
        if(state == slice_id || is_sliced == 0){

            // move slice 1 macroparts to CP nusing slice 2 centroid
            double x, y, z; // slice 1 macropart coords at CP
            MacropartToIP(use_strongstrong,
                  Sx_i, Sy_i, Sz_i, px, py,
                  *x_bb_centroid, *y_bb_centroid, *z_bb_centroid, *px_bb_centroid, *py_bb_centroid, 
                  *x_full_bb_centroid, *y_full_bb_centroid,
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
