// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_BEAMBEAMPIC_METHODS_H
#define XFIELDS_BEAMBEAMPIC_METHODS_H


/*gpufun*/
void BeamBeamPIC3D_change_ref_frame_local_particle(
        BeamBeamPIC3DData el, LocalParticle* part0){

    // Get data from memory
    double const sin_phi = BeamBeamPIC3DData_get__sin_phi(el);
    double const cos_phi = BeamBeamPIC3DData_get__cos_phi(el);
    double const tan_phi = BeamBeamPIC3DData_get__tan_phi(el);
    double const sin_alpha = BeamBeamPIC3DData_get__sin_alpha(el);
    double const cos_alpha = BeamBeamPIC3DData_get__cos_alpha(el);

    const double shift_x = BeamBeamPIC3DData_get_ref_shift_x(el)
                           + BeamBeamPIC3DData_get_other_beam_shift_x(el);
    const double shift_px = BeamBeamPIC3DData_get_ref_shift_px(el)
                            + BeamBeamPIC3DData_get_other_beam_shift_px(el);
    const double shift_y = BeamBeamPIC3DData_get_ref_shift_y(el)
                            + BeamBeamPIC3DData_get_other_beam_shift_y(el);
    const double shift_py = BeamBeamPIC3DData_get_ref_shift_py(el)
                            + BeamBeamPIC3DData_get_other_beam_shift_py(el);
    const double shift_zeta = BeamBeamPIC3DData_get_ref_shift_zeta(el)
                            + BeamBeamPIC3DData_get_other_beam_shift_zeta(el);
    const double shift_pzeta = BeamBeamPIC3DData_get_ref_shift_pzeta(el)
                            + BeamBeamPIC3DData_get_other_beam_shift_pzeta(el);


    //start_per_particle_block (part0->part)
        double x = LocalParticle_get_x(part);
        double px = LocalParticle_get_px(part);
        double y = LocalParticle_get_y(part);
        double py = LocalParticle_get_py(part);
        double zeta = LocalParticle_get_zeta(part);
        double pzeta = LocalParticle_get_pzeta(part);

        // Change reference frame
        change_ref_frame_coordinates(
            &x, &px, &y, &py, &zeta, &pzeta,
            shift_x, shift_px, shift_y, shift_py, shift_zeta, shift_pzeta,
            sin_phi, cos_phi, tan_phi, sin_alpha, cos_alpha);

        // Store
        LocalParticle_set_x(part, x);
        LocalParticle_set_px(part, px);
        LocalParticle_set_y(part, y);
        LocalParticle_set_py(part, py);
        LocalParticle_set_zeta(part, zeta);
        LocalParticle_update_pzeta(part, pzeta);

    //end_per_particle_block
}

/*gpufun*/
void BeamBeamPIC3D_change_back_ref_frame_and_subtract_dipolar_local_particle(
        BeamBeamPIC3DData el, LocalParticle* part0){

    // Get data from memory
    double const sin_phi = BeamBeamPIC3DData_get__sin_phi(el);
    double const cos_phi = BeamBeamPIC3DData_get__cos_phi(el);
    double const tan_phi = BeamBeamPIC3DData_get__tan_phi(el);
    double const sin_alpha = BeamBeamPIC3DData_get__sin_alpha(el);
    double const cos_alpha = BeamBeamPIC3DData_get__cos_alpha(el);

    const double shift_x = BeamBeamPIC3DData_get_ref_shift_x(el)
                           + BeamBeamPIC3DData_get_other_beam_shift_x(el);
    const double shift_px = BeamBeamPIC3DData_get_ref_shift_px(el)
                            + BeamBeamPIC3DData_get_other_beam_shift_px(el);
    const double shift_y = BeamBeamPIC3DData_get_ref_shift_y(el)
                            + BeamBeamPIC3DData_get_other_beam_shift_y(el);
    const double shift_py = BeamBeamPIC3DData_get_ref_shift_py(el)
                            + BeamBeamPIC3DData_get_other_beam_shift_py(el);
    const double shift_zeta = BeamBeamPIC3DData_get_ref_shift_zeta(el)
                            + BeamBeamPIC3DData_get_other_beam_shift_zeta(el);
    const double shift_pzeta = BeamBeamPIC3DData_get_ref_shift_pzeta(el)
                            + BeamBeamPIC3DData_get_other_beam_shift_pzeta(el);

    const double post_subtract_x = BeamBeamPIC3DData_get_post_subtract_x(el);
    const double post_subtract_px = BeamBeamPIC3DData_get_post_subtract_px(el);
    const double post_subtract_y = BeamBeamPIC3DData_get_post_subtract_y(el);
    const double post_subtract_py = BeamBeamPIC3DData_get_post_subtract_py(el);
    const double post_subtract_zeta = BeamBeamPIC3DData_get_post_subtract_zeta(el);
    const double post_subtract_pzeta = BeamBeamPIC3DData_get_post_subtract_pzeta(el);

    //start_per_particle_block (part0->part)
        double x = LocalParticle_get_x(part);
        double px = LocalParticle_get_px(part);
        double y = LocalParticle_get_y(part);
        double py = LocalParticle_get_py(part);
        double zeta = LocalParticle_get_zeta(part);
        double pzeta = LocalParticle_get_pzeta(part);


        // Go back to original reference frame and remove dipolar effect
        change_back_ref_frame_and_subtract_dipolar_coordinates(
            &x, &px, &y, &py, &zeta, &pzeta,
            shift_x, shift_px, shift_y, shift_py, shift_zeta, shift_pzeta,
            post_subtract_x, post_subtract_px,
            post_subtract_y, post_subtract_py,
            post_subtract_zeta, post_subtract_pzeta,
            sin_phi, cos_phi, tan_phi, sin_alpha, cos_alpha);

        // Store
        LocalParticle_set_x(part, x);
        LocalParticle_set_px(part, px);
        LocalParticle_set_y(part, y);
        LocalParticle_set_py(part, py);
        LocalParticle_set_zeta(part, zeta);
        LocalParticle_update_pzeta(part, pzeta);

    //end_per_particle_block

}

/*gpufun*/
void BeamBeamPIC3D_track_local_particle(BeamBeamPIC3DData el, LocalParticle* part0){


}



#endif
