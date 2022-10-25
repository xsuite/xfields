// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_BEAMBEAM3D_H
#define XFIELDS_BEAMBEAM3D_H

//#if !defined(dump_path)
//    #define dump_path "/Users/pkicsiny/phd/cern/xsuite/outputs/n74c" 
//#endif

/*gpufun*/
void synchrobeam_kick(
        BeamBeamBiGaussian3DData el, LocalParticle *part, 
        const int64_t flag_beamstrahlung, const int64_t flag_luminosity,
        BeamBeamBiGaussian3DRecordData beamstrahlung_record, RecordIndex beamstrahlung_table_index, BeamstrahlungTableData beamstrahlung_table,
        BeamBeamBiGaussian3DRecordData luminosity_record, RecordIndex luminosity_table_index, LuminosityTableData luminosity_table,
        const int i_slice,
        double const q0, double const p0c,
        double* x_star,
        double* px_star,
        double* y_star,
        double* py_star,
        double* zeta_star,
        double* pzeta_star, double const tan_phi){

    // Get data from memory
    const double q0_bb  = BeamBeamBiGaussian3DData_get_other_beam_q0(el);
    const double min_sigma_diff = BeamBeamBiGaussian3DData_get_min_sigma_diff(el);
    const double threshold_singular = BeamBeamBiGaussian3DData_get_threshold_singular(el);

    // debugging
    const int64_t turn    = BeamBeamBiGaussian3DData_get_turn(el);
//    const double flat_y_i = BeamBeamBiGaussian3DData_get_flat_y_i(el); 

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

    //printf("[turn: %d] Sig_11_hat_star: %.6e, Sig_33_hat_star: %.6e, dS_Sig_11_hat_star: %6e, dS_Sig_33_hat_star: %6e\n", turn, Sig_11_hat_star, Sig_33_hat_star, dS_Sig_11_hat_star, dS_Sig_33_hat_star);
    // apply kick only if Sig_11_hat_star > 0 and Sig_33_hat_star > 0 (corresponds to num_macroparts_in_slice > 2)
    if (Sig_11_hat_star<=0 || Sig_33_hat_star<=0){
        return;    
    } 
/*
    if (part->ipart==6 && turn==0){
        char dump_file[1024];
        sprintf(dump_file, "%s/strong_moments_cp_%.2e_phi_%.2e_flaty.txt", dump_path, atan(tan_phi), flat_y_i);
        FILE *f = fopen(dump_file, "a");
        fprintf(f, "%d %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e\n", i_slice, x_slice_star, y_slice_star, S, Sig_11_0, Sig_22_0, Sig_33_0, Sig_44_0, Sig_11_hat_star, Sig_33_hat_star);
        fclose(f);
    }
*/

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

    //printf("[turn: %d] Ex: %6e, Ey: %6e\n", turn, Ex, Ey);

    //compute Gs
    double Gx, Gy;
    compute_Gx_Gy(x_bar_hat_star, y_bar_hat_star,
          sqrt(Sig_11_hat_star), sqrt(Sig_33_hat_star),
                      min_sigma_diff, Ex, Ey, &Gx, &Gy);
    //printf("[turn: %d] Gx: %6e, Gy: %6e\n", turn, Gx, Gy);
/*
    if (part->ipart==6 && turn==0){
        char dump_file[1024];
        sprintf(dump_file, "%s/forces_%.2e_phi_%.2e_flaty.txt", dump_path, atan(tan_phi), flat_y_i);
        FILE *f = fopen(dump_file, "a");
        fprintf(f, "%d %.8e %.8e %.8e %.8e %.8e\n", i_slice, Ksl, Ex, Ey, Gx, Gy);
        fclose(f);
    }
*/

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
    //printf("[before bs] pzeta: %.30e\n", *pzeta_star);
    // emit beamstrahlung photons from single macropart
    if (flag_beamstrahlung==1){
        LocalParticle_update_pzeta(part, *pzeta_star);  // update energy vars with boost and/or last kick
        double const Fr = hypot(Fx_star, Fy_star) * LocalParticle_get_rpp(part); // total kick [1]
        /*gpuglmem*/ double const dz = .5*BeamBeamBiGaussian3DData_get_slices_other_beam_zeta_bin_width_star(el, i_slice);  // bending radius [m]
        beamstrahlung(part, beamstrahlung_record, beamstrahlung_table_index, beamstrahlung_table, Fr, dz);
        *pzeta_star = LocalParticle_get_pzeta(part);  // BS rescales energy vars, so load again before kick 
    }
    else if(flag_beamstrahlung==2){
        double var_z_bb = 0.0121;
        LocalParticle_update_pzeta(part, *pzeta_star);  // update energy vars with boost and/or last kick
        beamstrahlung_avg(part, num_part_slice, sqrt(Sig_11_hat_star), sqrt(Sig_33_hat_star), var_z_bb);  // slice intensity and RMS slice sizes
        *pzeta_star = LocalParticle_get_pzeta(part);  
    }


    // Bhabha scattering
    double charge_density;
    get_rho_charge(x_bar_hat_star, y_bar_hat_star, sqrt(Sig_11_hat_star), sqrt(Sig_33_hat_star), Sig_13_0, &charge_density);

    //printf("[before kick] pzeta: %.30e\n", *pzeta_star);

    // Apply the kicks (Hirata's synchro-beam)
    *pzeta_star = *pzeta_star + Fz_star+0.5*(
                Fx_star*(*px_star+0.5*Fx_star)+
                Fy_star*(*py_star+0.5*Fy_star));
    *x_star = *x_star - S*Fx_star;
    *px_star = *px_star + Fx_star;
    *y_star = *y_star - S*Fy_star;
    *py_star = *py_star + Fy_star;

    if (flag_luminosity){
        double lumi = num_part_slice / (sqrt(Sig_11_hat_star)*sqrt(Sig_33_hat_star));
        if (luminosity_record){
            // Get a slot in the record (this is thread safe)
            int64_t i_slot = RecordIndex_get_slot(luminosity_table_index);
            // The returned slot id is negative if record is NULL or if record is full
            if (i_slot>=0){
                LuminosityTableData_set_particle_id(      luminosity_table, i_slot, LocalParticle_get_particle_id(part));
                LuminosityTableData_set_opposite_slice_id(luminosity_table, i_slot, i_slice);
                LuminosityTableData_set_charge_density(   luminosity_table, i_slot, charge_density);
                LuminosityTableData_set_luminosity(       luminosity_table, i_slot, lumi);
            }
        }
    }
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

    const double post_subtract_x = BeamBeamBiGaussian3DData_get_post_subtract_x(el);
    const double post_subtract_px = BeamBeamBiGaussian3DData_get_post_subtract_px(el);
    const double post_subtract_y = BeamBeamBiGaussian3DData_get_post_subtract_y(el);
    const double post_subtract_py = BeamBeamBiGaussian3DData_get_post_subtract_py(el);
    const double post_subtract_zeta = BeamBeamBiGaussian3DData_get_post_subtract_zeta(el);
    const double post_subtract_pzeta = BeamBeamBiGaussian3DData_get_post_subtract_pzeta(el);

    // Extract the record and record_index
    const int64_t flag_beamstrahlung = BeamBeamBiGaussian3DData_get_flag_beamstrahlung(el);
    BeamBeamBiGaussian3DRecordData beamstrahlung_record = NULL; 
    BeamstrahlungTableData beamstrahlung_table          = NULL;
    RecordIndex beamstrahlung_table_index               = NULL;
    if (flag_beamstrahlung > 0) {
      beamstrahlung_record = BeamBeamBiGaussian3DData_getp_internal_record(el, part0);
      if (beamstrahlung_record){
        beamstrahlung_table       = BeamBeamBiGaussian3DRecordData_getp_beamstrahlungtable(beamstrahlung_record);
        beamstrahlung_table_index =                      BeamstrahlungTableData_getp__index(beamstrahlung_table);
      }
    }

    const int64_t flag_luminosity = BeamBeamBiGaussian3DData_get_flag_luminosity(el);
    BeamBeamBiGaussian3DRecordData luminosity_record = NULL;
    LuminosityTableData luminosity_table             = NULL;
    RecordIndex luminosity_table_index               = NULL;
    if (flag_luminosity > 0) {
      luminosity_record = BeamBeamBiGaussian3DData_getp_internal_record(el, part0);
      if (luminosity_record){
        luminosity_table       = BeamBeamBiGaussian3DRecordData_getp_luminositytable(luminosity_record);
        luminosity_table_index =                      LuminosityTableData_getp__index(luminosity_table);
      }
    }

    const int64_t turn    = BeamBeamBiGaussian3DData_get_turn(el);
    //start_per_particle_block (part0->part)
        double x = LocalParticle_get_x(part);
        double px = LocalParticle_get_px(part);
        double y = LocalParticle_get_y(part);
        double py = LocalParticle_get_py(part);
        double zeta = LocalParticle_get_zeta(part);
        double pzeta = LocalParticle_get_pzeta(part);

        const double q0 = LocalParticle_get_q0(part);
        const double p0c = LocalParticle_get_p0c(part); // eV

        //printf("[before boost] pzeta: %.30e\n", pzeta);

        // Change reference frame
        change_ref_frame_coordinates(
            &x, &px, &y, &py, &zeta, &pzeta,
            shift_x, shift_px, shift_y, shift_py, shift_zeta, shift_pzeta,
            sin_phi, cos_phi, tan_phi, sin_alpha, cos_alpha);

        // here pzeta is not updated but has to be! changes delta
        //LocalParticle_update_pzeta(part, pzeta);

        //printf("[after boost] pzeta: %.30e\n", pzeta);

        // Synchro beam
        for (int i_slice=0; i_slice<N_slices; i_slice++)
        {
            //printf("[slice %d, before synchrobeam and get] pzeta: %.30e\n", i_slice, pzeta);

            // new: reload boosted pzeta after each slice kick to compare with sbc6d; these are boosted
       	    //pzeta = LocalParticle_get_pzeta(part);

            //printf("[slice %d, before synchrobeam] pzeta: %.30e\n", i_slice, pzeta);

                synchrobeam_kick(
                             el, part, 
                             flag_beamstrahlung, flag_luminosity,
                             beamstrahlung_record, beamstrahlung_table_index, beamstrahlung_table,
                             luminosity_record, luminosity_table_index, luminosity_table,
                             i_slice, q0, p0c,
                             &x,
                             &px,
                             &y,
                             &py,
                             &zeta,
                             &pzeta, tan_phi);
//                printf("[turn: %d] returned, x: %6e, y: %6e, z: %6e, px: %.6e, py: %.6e, pzeta: %.6e\n", turn, x, y, zeta, px, py, pzeta);

                //printf("[slice %d, after synchrobeam] pzeta: %.30e\n", i_slice, pzeta);

                // here pzeta is not updated but has to be! changes delta
                //LocalParticle_update_pzeta(part, pzeta);
           // }
        }

        //printf("[before inverse boost] pzeta: %.30e\n", pzeta);

        // Go back to original reference frame and remove dipolar effect
        change_back_ref_frame_and_subtract_dipolar_coordinates(
            &x, &px, &y, &py, &zeta, &pzeta,
            shift_x, shift_px, shift_y, shift_py, shift_zeta, shift_pzeta,
            post_subtract_x, post_subtract_px,
            post_subtract_y, post_subtract_py,
            post_subtract_zeta, post_subtract_pzeta,
            sin_phi, cos_phi, tan_phi, sin_alpha, cos_alpha);

//        printf("[turn: %d] after boost inv, x: %6e, y: %6e, z: %6e, px: %.6e, py: %.6e, pzeta: %.6e\n", turn, x, y, zeta, px, py, pzeta);
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
