// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_BEAMBEAMPIC_METHODS_H
#define XFIELDS_BEAMBEAMPIC_METHODS_H

/*gpufun*/
void do_beamstrahlung_pic(BeamBeamPIC3DData el, LocalParticle *part,
                      double Fx_star, double Fy_star,
                      double* pzeta_star, double const dz,
                      const int64_t flag_beamstrahlung){

        // init record table
        BeamBeamPIC3DRecordData beamstrahlung_record = NULL;
        BeamstrahlungTableData beamstrahlung_table   = NULL;
        RecordIndex beamstrahlung_table_index        = NULL;
        beamstrahlung_record = BeamBeamPIC3DData_getp_internal_record(el, part);
        if (beamstrahlung_record){
            beamstrahlung_table       = BeamBeamPIC3DRecordData_getp_beamstrahlungtable(beamstrahlung_record);
            beamstrahlung_table_index =               BeamstrahlungTableData_getp__index(beamstrahlung_table);
        }

        LocalParticle_update_pzeta(part, *pzeta_star);  // update energy vars with boost and/or last kick

	if(flag_beamstrahlung==1){
            // no average beamstrahlung implemented
	} else if (flag_beamstrahlung==2){
            double const Fr = hypot(Fx_star, Fy_star) * LocalParticle_get_rpp(part); // radial kick [1]
            beamstrahlung(part, beamstrahlung_record, beamstrahlung_table_index, beamstrahlung_table, Fr, dz);
        }

        *pzeta_star = LocalParticle_get_pzeta(part);  // BS rescales energy vars, so load again before kick
}


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
void BeamBeamPIC3D_propagate_transverse_coords_at_step(
        BeamBeamPIC3DData el, LocalParticle* part0, double z_step_other)
{
    //start_per_particle_block (part0->part)
        if (LocalParticle_get_state(part) <= 0) return;

        double x = LocalParticle_get_x(part);
        double y = LocalParticle_get_y(part);
        double px = LocalParticle_get_px(part);
        double py = LocalParticle_get_py(part);
        double ptau = LocalParticle_get_ptau(part);
        double zeta = LocalParticle_get_zeta(part);
        double beta0 = LocalParticle_get_beta0(part);

        double gamma_gamma0 = ptau * beta0 + 1;
        x += px / gamma_gamma0 * (zeta - z_step_other);
        y += py / gamma_gamma0 * (zeta - z_step_other);

        LocalParticle_set_x(part, x);
        LocalParticle_set_y(part, y);
    //end_per_particle_block
}


/*gpufun*/
void BeamBeamPIC3D_kick_and_propagate_transverse_coords_back(
        BeamBeamPIC3DData el, LocalParticle* part0,
        const double* dphi_dx, const double* dphi_dy, const double* dphi_dz,
        const double z_step_other)
{
    const double q0 = LocalParticle_get_q0(part0);
    const double mass0 = LocalParticle_get_mass0(part0);
    const double dz = BeamBeamPIC3DData_get_fieldmap_self_dz(el);

    //start_per_particle_block (part0->part)
        if (LocalParticle_get_state(part) <= 0) return;

        double x = LocalParticle_get_x(part);
        double y = LocalParticle_get_y(part);
        double px = LocalParticle_get_px(part);
        double py = LocalParticle_get_py(part);
        double pzeta = LocalParticle_get_pzeta(part);
        double zeta = LocalParticle_get_zeta(part);
        double beta0 = LocalParticle_get_beta0(part);
        double gamma0 = LocalParticle_get_gamma0(part);
        double chi = LocalParticle_get_chi(part);

        // Compute factor for the kick, assume Assume ultrarelativistic for now
        double pp_beta0 = 1.;
        double beta0_other = 1.;
        // double charge_mass_ratio = chi * QELEM * q0 / (mass0 * QELEM / (C_LIGHT * C_LIGHT));
        // double factor = -(charge_mass_ratio
        //         / (gamma0 * pp_beta0 * pp_beta0 * C_LIGHT * C_LIGHT)
        //         * (1 + beta0_other * pp_beta0));
        // Simplified:
        double factor = - (chi * q0 * (1 + beta0_other * pp_beta0)) / (gamma0 * pp_beta0 * pp_beta0 * mass0);

        // Compute kick
        double dpx = factor * dphi_dx[part->ipart] * dz;
        double dpy = factor * dphi_dy[part->ipart] * dz;

        // Effect of the particle angle as in Hirata
        double dpz = 0.5 * (dpx * (px + 0.5 * dpx) + dpy * (py + 0.5 * dpy));

        // emit beamstrahlung photons from single macropart
        #ifndef XFIELDS_BB3D_NO_BEAMSTR
        const int64_t flag_beamstrahlung = BeamBeamPIC3DData_get_flag_beamstrahlung(el);
        if(flag_beamstrahlung!=0){
            do_beamstrahlung_pic(el, part, dpx, dpy, &pzeta, dz, flag_beamstrahlung); // no avg only quantum
        }
        #endif 

        // Apply kick
        px += dpx;
        py += dpy;
        LocalParticle_set_px(part, px);
        LocalParticle_set_py(part, py);
        LocalParticle_update_pzeta(part, pzeta + dpz);

        // Propagate transverse coordinates back to IP
        double ptau = LocalParticle_get_ptau(part);
        double gamma_gamma0 = ptau * beta0 + 1;
        x -= px / gamma_gamma0 * (zeta - z_step_other);
        y -= py / gamma_gamma0 * (zeta - z_step_other);

        LocalParticle_set_x(part, x);
        LocalParticle_set_y(part, y);
    //end_per_particle_block
}


/*gpufun*/
void BeamBeamPIC3D_track_local_particle(BeamBeamPIC3DData el, LocalParticle* part0){

// Dummy, to avoid error    on compilation

}

#endif
