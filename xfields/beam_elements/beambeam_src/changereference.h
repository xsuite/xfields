#ifndef XFIELDS_CHANGEREFERENCE_H
#define XFIELDS_CHANGEREFERENCE_H

#if !defined(mysign)
    #define mysign(a) (((a) >= 0) - ((a) < 0))
#endif

#include <stdio.h>

/*gpufun*/
void ChangeReference_track_local_particle(ChangeReferenceData el, LocalParticle* part){

    // part= single macropart
    // el= beambeam element (other beam)	
    // Get data from memory
    const double delta_x            = ChangeReferenceData_get_delta_x(el);
    const double delta_y            = ChangeReferenceData_get_delta_y(el);
    const double delta_px           = ChangeReferenceData_get_delta_px(el);
    const double delta_py           = ChangeReferenceData_get_delta_py(el);
    const double x_CO               = ChangeReferenceData_get_x_CO(el);     
    const double px_CO              = ChangeReferenceData_get_px_CO(el);
    const double y_CO               = ChangeReferenceData_get_y_CO(el);
    const double py_CO              = ChangeReferenceData_get_py_CO(el);
    const double z_CO               = ChangeReferenceData_get_z_CO(el);
    const double delta_CO           = ChangeReferenceData_get_delta_CO(el);

    const int64_t slice_id                 = ChangeReferenceData_get_slice_id(el);
    const int64_t is_sliced                = ChangeReferenceData_get_is_sliced(el);
    int64_t const n_part = LocalParticle_get_num_particles(part); //only_for_context cpu_serial cpu_openmp

    for (int ii=0; ii<n_part; ii++){ //only_for_context cpu_serial cpu_openmp
        part->ipart = ii;            //only_for_context cpu_serial cpu_openmp
      
        // get macropart properties, one number each, using generated C code of the particle element
    	double x     = LocalParticle_get_x(part);
    	double px    = LocalParticle_get_px(part);
    	double y     = LocalParticle_get_y(part);
    	double py    = LocalParticle_get_py(part);
    	double z     = LocalParticle_get_zeta(part);
    	double delta = LocalParticle_get_delta(part);

        // macropart state: 0=dead, 1=alive, 100X: part of slice X
        int64_t state = LocalParticle_get_state(part); 
        state -= 1000;

        // code is executed only if macropart is in correct slice or if there are no slices
        if(state == slice_id || is_sliced == 0){

            // x_co: distance of other beam closed orbit from my closed orbit
            // x_bb_co: distance of other beam centroid from its own closed orbit
            // for inverse transformation put in negatives
            double x_star, px_star, y_star, py_star, z_star, delta_star;
            x_star     = x     - x_CO    - delta_x;
            px_star    = px    - px_CO   - delta_px;
            y_star     = y     - y_CO    - delta_y;
            py_star    = py    - py_CO   - delta_py;
            z_star     = z     - z_CO;
            delta_star = delta - delta_CO;
     	
            LocalParticle_set_x(part, x_star);
            LocalParticle_set_px(part, px_star);
    	    LocalParticle_set_y(part, y_star);
    	    LocalParticle_set_py(part, py_star);
    	    LocalParticle_set_zeta(part, z_star);
    	    LocalParticle_update_delta(part, delta_star);
          }
      } //only_for_context cpu_serial cpu_openmp
}


#endif
