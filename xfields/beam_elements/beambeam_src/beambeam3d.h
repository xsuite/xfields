// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_BEAMBEAM3D_H
#define XFIELDS_BEAMBEAM3D_H


/*gpufun*/
void synchrobeam_kick(
        BeamBeamBiGaussian3DData el, const int i_slice,
        double const q0, double const p0c,
        double* x_star,
        double* px_star,
        double* y_star,
        double* py_star,
        double* zeta_star,
        double* pzeta_star){

    // Get data from memory
    double const scale_strength = BeamBeamBiGaussian3DData_get_scale_strength(el);
    const double q0_bb  = scale_strength*BeamBeamBiGaussian3DData_get_other_beam_q0(el);
    const double min_sigma_diff = BeamBeamBiGaussian3DData_get_min_sigma_diff(el);
    const double threshold_singular = BeamBeamBiGaussian3DData_get_threshold_singular(el);
    

    double const Sig_11_0 = BeamBeamBiGaussian3DData_get_slices_other_beam_Sigma_11_star(el, i_slice);
    double const Sig_12_0 = BeamBeamBiGaussian3DData_get_slices_other_beam_Sigma_12_star(el, i_slice);
    double const Sig_13_0 = BeamBeamBiGaussian3DData_get_slices_other_beam_Sigma_13_star(el, i_slice);
    double const Sig_14_0 = BeamBeamBiGaussian3DData_get_slices_other_beam_Sigma_14_star(el, i_slice);
    double const Sig_22_0 = BeamBeamBiGaussian3DData_get_slices_other_beam_Sigma_22_star(el, i_slice);
    double const Sig_23_0 = BeamBeamBiGaussian3DData_get_slices_other_beam_Sigma_23_star(el, i_slice);
    double const Sig_24_0 = BeamBeamBiGaussian3DData_get_slices_other_beam_Sigma_24_star(el, i_slice);
    double const Sig_33_0 = BeamBeamBiGaussian3DData_get_slices_other_beam_Sigma_33_star(el, i_slice);
    double const Sig_34_0 = BeamBeamBiGaussian3DData_get_slices_other_beam_Sigma_34_star(el, i_slice);
    double const Sig_44_0 = BeamBeamBiGaussian3DData_get_slices_other_beam_Sigma_44_star(el, i_slice);

    double const num_part_slice = BeamBeamBiGaussian3DData_get_slices_other_beam_num_particles(el, i_slice);

    const double x_slice_star = BeamBeamBiGaussian3DData_get_slices_other_beam_x_center_star(el, i_slice);
    const double y_slice_star = BeamBeamBiGaussian3DData_get_slices_other_beam_y_center_star(el, i_slice);
    double const zeta_slice_star = BeamBeamBiGaussian3DData_get_slices_other_beam_zeta_center_star(el, i_slice);

    const double P0 = p0c/C_LIGHT*QELEM;

    //Compute force scaling factor
    const double Ksl = num_part_slice*QELEM*q0_bb*QELEM*q0/(P0 * C_LIGHT);

    //Identify the Collision Point (CP)
    const double S = 0.5*(*zeta_star - zeta_slice_star);

    // Propagate sigma matrix
    double Sig_11_hat_star, Sig_33_hat_star, costheta, sintheta;
    double dS_Sig_11_hat_star, dS_Sig_33_hat_star, dS_costheta, dS_sintheta;

    // Get strong beam shape at the CP
    Sigmas_propagate(
            Sig_11_0,
            Sig_12_0,
            Sig_13_0,
            Sig_14_0,
            Sig_22_0,
            Sig_23_0,
            Sig_24_0,
            Sig_33_0,
            Sig_34_0,
            Sig_44_0,
            S, threshold_singular, 1,
            &Sig_11_hat_star, &Sig_33_hat_star,
            &costheta, &sintheta,
            &dS_Sig_11_hat_star, &dS_Sig_33_hat_star,
            &dS_costheta, &dS_sintheta);

    // Evaluate transverse coordinates of the weak baem w.r.t. the strong beam centroid
    const double x_bar_star = *x_star + *px_star * S - x_slice_star;
    const double y_bar_star = *y_star + *py_star * S - y_slice_star;

    // Move to the uncoupled reference frame
    const double x_bar_hat_star = x_bar_star*costheta +y_bar_star*sintheta;
    const double y_bar_hat_star = -x_bar_star*sintheta +y_bar_star*costheta;

    // Compute derivatives of the transformation
    const double dS_x_bar_hat_star = x_bar_star*dS_costheta +y_bar_star*dS_sintheta;
    const double dS_y_bar_hat_star = -x_bar_star*dS_sintheta +y_bar_star*dS_costheta;

    // Get transverse fields
    double Ex, Ey;
    get_Ex_Ey_gauss(x_bar_hat_star, y_bar_hat_star,
        sqrt(Sig_11_hat_star), sqrt(Sig_33_hat_star),
        min_sigma_diff,
        &Ex, &Ey);

    //compute Gs
    double Gx, Gy;
    compute_Gx_Gy(x_bar_hat_star, y_bar_hat_star,
          sqrt(Sig_11_hat_star), sqrt(Sig_33_hat_star),
                      min_sigma_diff, Ex, Ey, &Gx, &Gy);

    // Compute kicks
    double Fx_hat_star = Ksl*Ex;
    double Fy_hat_star = Ksl*Ey;
    double Gx_hat_star = Ksl*Gx;
    double Gy_hat_star = Ksl*Gy;

    // Move kisks to coupled reference frame
    double Fx_star = Fx_hat_star*costheta - Fy_hat_star*sintheta;
    double Fy_star = Fx_hat_star*sintheta + Fy_hat_star*costheta;

    // Compute longitudinal kick
    double Fz_star = 0.5*(Fx_hat_star*dS_x_bar_hat_star  + Fy_hat_star*dS_y_bar_hat_star+
                   Gx_hat_star*dS_Sig_11_hat_star + Gy_hat_star*dS_Sig_33_hat_star);

    // Apply the kicks (Hirata's synchro-beam)
    *pzeta_star = *pzeta_star + Fz_star+0.5*(
                Fx_star*(*px_star+0.5*Fx_star)+
                Fy_star*(*py_star+0.5*Fy_star));
    *x_star = *x_star - S*Fx_star;
    *px_star = *px_star + Fx_star;
    *y_star = *y_star - S*Fy_star;
    *py_star = *py_star + Fy_star;

    }



/*gpufun*/
void BeamBeamBiGaussian3D_track_local_particle(BeamBeamBiGaussian3DData el,
                LocalParticle* part0){

    // Get data from memory
    double const sin_phi = BeamBeamBiGaussian3DData_get__sin_phi(el);
    double const cos_phi = BeamBeamBiGaussian3DData_get__cos_phi(el);
    double const tan_phi = BeamBeamBiGaussian3DData_get__tan_phi(el);
    double const sin_alpha = BeamBeamBiGaussian3DData_get__sin_alpha(el);
    double const cos_alpha = BeamBeamBiGaussian3DData_get__cos_alpha(el);

    const int N_slices = BeamBeamBiGaussian3DData_get_num_slices_other_beam(el);

    const double shift_x = BeamBeamBiGaussian3DData_get_ref_shift_x(el)
                           + BeamBeamBiGaussian3DData_get_other_beam_shift_x(el);
    const double shift_px = BeamBeamBiGaussian3DData_get_ref_shift_px(el)
                            + BeamBeamBiGaussian3DData_get_other_beam_shift_px(el);
    const double shift_y = BeamBeamBiGaussian3DData_get_ref_shift_y(el)
                            + BeamBeamBiGaussian3DData_get_other_beam_shift_y(el);
    const double shift_py = BeamBeamBiGaussian3DData_get_ref_shift_py(el)
                            + BeamBeamBiGaussian3DData_get_other_beam_shift_py(el);
    const double shift_zeta = BeamBeamBiGaussian3DData_get_ref_shift_zeta(el)
                            + BeamBeamBiGaussian3DData_get_other_beam_shift_zeta(el);
    const double shift_pzeta = BeamBeamBiGaussian3DData_get_ref_shift_pzeta(el)
                            + BeamBeamBiGaussian3DData_get_other_beam_shift_pzeta(el);

    double const scale_strength = BeamBeamBiGaussian3DData_get_scale_strength(el);
    const double post_subtract_x = scale_strength*BeamBeamBiGaussian3DData_get_post_subtract_x(el);
    const double post_subtract_px = scale_strength*BeamBeamBiGaussian3DData_get_post_subtract_px(el);
    const double post_subtract_y = scale_strength*BeamBeamBiGaussian3DData_get_post_subtract_y(el);
    const double post_subtract_py = scale_strength*BeamBeamBiGaussian3DData_get_post_subtract_py(el);
    const double post_subtract_zeta = scale_strength*BeamBeamBiGaussian3DData_get_post_subtract_zeta(el);
    const double post_subtract_pzeta = scale_strength*BeamBeamBiGaussian3DData_get_post_subtract_pzeta(el);

    //start_per_particle_block (part0->part)
        double x = LocalParticle_get_x(part);
        double px = LocalParticle_get_px(part);
        double y = LocalParticle_get_y(part);
        double py = LocalParticle_get_py(part);
        double zeta = LocalParticle_get_zeta(part);
        double pzeta = LocalParticle_get_pzeta(part);

        const double q0 = LocalParticle_get_q0(part);
        const double p0c = LocalParticle_get_p0c(part); // eV


        // Change reference frame
        change_ref_frame_coordinates(
            &x, &px, &y, &py, &zeta, &pzeta,
            shift_x, shift_px, shift_y, shift_py, shift_zeta, shift_pzeta,
            sin_phi, cos_phi, tan_phi, sin_alpha, cos_alpha);

        // Synchro beam
        for (int i_slice=0; i_slice<N_slices; i_slice++)
        {
            synchrobeam_kick(el, i_slice, q0, p0c,
                             &x,
                             &px,
                             &y,
                             &py,
                             &zeta,
                             &pzeta);

        }

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



#endif
