// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_BEAMBEAM3D_H
#define XFIELDS_BEAMBEAM3D_H

#ifndef min
#define min(a,b) ((a) <= (b) ? (a) : (b))
#endif

/*gpufun*/
void do_luminosity(BeamBeamBiGaussian3DData el, LocalParticle *part,
               double* rho, double* wgt,
               double const x_bar_hat_star, double const y_bar_hat_star,
               double const Sig_11_hat_star, double const Sig_33_hat_star,
               double const num_part_slice){

    // gaussian charge density: at x, y density given by the 2D gaussian, local lumi depending on x y, total lumi sum of all
    get_charge_density(x_bar_hat_star, y_bar_hat_star, sqrt(Sig_11_hat_star), sqrt(Sig_33_hat_star), rho);
    *wgt =  LocalParticle_get_weight(part) * num_part_slice * (*rho);  // [m^-2] integrated lumi of a single electron colliding with the opposing slice

    // init record table
    BeamBeamBiGaussian3DRecordData lumi_record = NULL;
    LumiTableData lumi_table                   = NULL;
    RecordIndex lumi_table_index               = NULL;
    lumi_record = BeamBeamBiGaussian3DData_getp_internal_record(el, part);
    if (lumi_record){
        lumi_table       = BeamBeamBiGaussian3DRecordData_getp_lumitable(lumi_record);
        lumi_table_index =                      LumiTableData_getp__index(lumi_table);

    const int at_turn = LocalParticle_get_at_turn(part);
    /*gpuglmem*/ double* lumi_address = LumiTableData_getp1_luminosity(lumi_table, at_turn);  // double pointer
    atomicAdd(lumi_address, *wgt);
    }
}

/*gpufun*/
void do_bhabha(BeamBeamBiGaussian3DData el, LocalParticle *part,
               double* rho, double* wgt,
               const int64_t flag_luminosity, double const q0,
               double const x_bar_hat_star, double const y_bar_hat_star,
               double const Sig_11_hat_star, double const Sig_33_hat_star,
               double const num_part_slice, double const S,
               double* px_star, double* py_star, double* pzeta_star,
               double px_slice_star, double py_slice_star, double pzeta_slice_star){

    // init record table
    BeamBeamBiGaussian3DRecordData bhabha_record = NULL;
    BhabhaTableData bhabha_table                 = NULL;
    RecordIndex bhabha_table_index               = NULL;
    bhabha_record = BeamBeamBiGaussian3DData_getp_internal_record(el, part);
    if (bhabha_record){
        bhabha_table       = BeamBeamBiGaussian3DRecordData_getp_bhabhatable(bhabha_record);
        bhabha_table_index =                      BhabhaTableData_getp__index(bhabha_table);
    }

    // switch for beam size effect
    const int64_t flag_beamsize_effect = BeamBeamBiGaussian3DData_get_flag_beamsize_effect(el);

    // gaussian charge density, we are centered at the strong slice centroid
    if (flag_luminosity != 1 && flag_beamsize_effect == 0){
      get_charge_density(x_bar_hat_star, y_bar_hat_star, sqrt(Sig_11_hat_star), sqrt(Sig_33_hat_star), rho);
      *wgt =  LocalParticle_get_weight(part) * num_part_slice * (*rho);  // [m^-2] integrated lumi of a single electron colliding with the opposing slice
    }

    LocalParticle_update_pzeta(part, *pzeta_star);  // update energy vars with boost and/or last kick

    const double other_beam_slice_energy =  LocalParticle_get_energy0(part)*(1 + pzeta_slice_star) * 1e-9;  // [GeV] for now betastar is 1; later change to other beam E0

    const double compt_x_min = BeamBeamBiGaussian3DData_get_compt_x_min(el);
    int n_photons = requiv(part, other_beam_slice_energy, compt_x_min);  // generate virtual photons of the opposite slice using the average energy of the opposite slice

    // generate virtual photons of the opposite slice
    double xmin, e_photon, q2, one_m_x, x_photon, y_photon, px_photon, py_photon, pzeta_photon, radius, theta;
    for (int i_phot=0; i_phot<n_photons; i_phot++){

      mequiv(part, other_beam_slice_energy, compt_x_min, &xmin, &e_photon, &q2, &one_m_x);  // here again use opposite slice energy average

      // apply beam size effect here (affects x and y only)
      switch(flag_beamsize_effect){
      case 0:  // this is w.r.t of the strong slice centroid
          radius = 0.0;
          x_photon = x_bar_hat_star;
          y_photon = y_bar_hat_star;
          break;
      case 1:  // photons distributed on a disc around centroid
          radius = HBAR_GEVS*C_LIGHT / sqrt(q2*one_m_x);  // [m]
          //printf("radius: %.6e\n", radius);
          radius = min(radius, 1e5);
          x_photon = x_bar_hat_star + rndm_sincos(part, &theta) * radius;
          y_photon = y_bar_hat_star + theta * radius;

          // resample charge density at randomized photon location
          get_charge_density(x_photon, y_photon, sqrt(Sig_11_hat_star), sqrt(Sig_33_hat_star), rho);
          *wgt =  LocalParticle_get_weight(part) * num_part_slice * (*rho);  // [m^-2] integrated lumi of a single electron colliding with the opposing slice
          break;
      }

      // virtual photons are located at the opposite slice centroid

      px_photon = px_slice_star;
      py_photon = py_slice_star;
      pzeta_photon = pzeta_slice_star;

      //if (radius < sqrt(Sig_33_hat_star)){
        // for each virtual photon get compton scatterings; updates pzeta and energy vars inside
        compt_do(part, bhabha_record, bhabha_table_index, bhabha_table,
                 e_photon, compt_x_min, q2,
                 x_photon, y_photon, S, px_photon, py_photon, pzeta_photon,
                 *wgt, px_star, py_star, pzeta_star, q0);

        // reload pzeta since they changed from compton; px and py are changed only locally
        *pzeta_star = LocalParticle_get_pzeta(part);  // bhabha rescales energy vars, so load again before kick
     // }
    }
}


/*gpufun*/
void do_beamstrahlung(BeamBeamBiGaussian3DData el, LocalParticle *part,
                      double Fx_star, double Fy_star,
                      double* pzeta_star, const int i_slice, double const num_part_slice,
                      const int64_t flag_beamstrahlung){

        // init record table
        BeamBeamBiGaussian3DRecordData beamstrahlung_record = NULL;
        BeamstrahlungTableData beamstrahlung_table          = NULL;
        RecordIndex beamstrahlung_table_index               = NULL;
        beamstrahlung_record = BeamBeamBiGaussian3DData_getp_internal_record(el, part);
        if (beamstrahlung_record){
            beamstrahlung_table       = BeamBeamBiGaussian3DRecordData_getp_beamstrahlungtable(beamstrahlung_record);
            beamstrahlung_table_index =                      BeamstrahlungTableData_getp__index(beamstrahlung_table);
        }

        LocalParticle_update_pzeta(part, *pzeta_star);  // update energy vars with boost and/or last kick
        if(flag_beamstrahlung==1){

            // get unboosted strong slice RMS [m]
            double sqrtSigma_11 = BeamBeamBiGaussian3DData_get_slices_other_beam_sqrtSigma_11_beamstrahlung(el, i_slice);
            double sqrtSigma_33 = BeamBeamBiGaussian3DData_get_slices_other_beam_sqrtSigma_33_beamstrahlung(el, i_slice);
            double sqrtSigma_55 = BeamBeamBiGaussian3DData_get_slices_other_beam_sqrtSigma_55_beamstrahlung(el, i_slice);
            beamstrahlung_avg(part, beamstrahlung_record, beamstrahlung_table_index, beamstrahlung_table,
                num_part_slice, sqrtSigma_11, sqrtSigma_33, sqrtSigma_55);
        } else if (flag_beamstrahlung==2){
            double const Fr = hypot(Fx_star, Fy_star) * LocalParticle_get_rpp(part); // radial kick [1]
            double const dz = .5*BeamBeamBiGaussian3DData_get_slices_other_beam_zeta_bin_width_star_beamstrahlung(el, i_slice);  // half slice width [m]
            beamstrahlung(part, beamstrahlung_record, beamstrahlung_table_index, beamstrahlung_table, Fr, dz);
        }
        *pzeta_star = LocalParticle_get_pzeta(part);  // BS rescales energy vars, so load again before kick
    }



/*gpufun*/
void synchrobeam_kick(
        BeamBeamBiGaussian3DData el, LocalParticle *part,
        const int i_slice,
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

    // no kick if not sufficient macroparticles; should be taken care of when slicing
    if (num_part_slice == 0){
        return;
    }

    const double x_slice_star = BeamBeamBiGaussian3DData_get_slices_other_beam_x_center_star(el, i_slice);
    const double y_slice_star = BeamBeamBiGaussian3DData_get_slices_other_beam_y_center_star(el, i_slice);
    const double px_slice_star = BeamBeamBiGaussian3DData_get_slices_other_beam_px_center_star(el, i_slice);
    const double py_slice_star = BeamBeamBiGaussian3DData_get_slices_other_beam_py_center_star(el, i_slice);
    const double zeta_slice_star = BeamBeamBiGaussian3DData_get_slices_other_beam_zeta_center_star(el, i_slice);
    const double pzeta_slice_star = BeamBeamBiGaussian3DData_get_slices_other_beam_pzeta_center_star(el, i_slice);

    const double P0 = p0c/C_LIGHT*QELEM;

    //Compute force scaling factor
    const double Ksl = num_part_slice*QELEM*q0_bb*QELEM*q0/(P0 * C_LIGHT);

    //Identify the Collision Point (CP)
    #ifdef XFIELDS_BEAMBEAM3D_FORCE_CP0
    const double S = 0.0;
    #else
    const double S = 0.5*(*zeta_star - zeta_slice_star);
    #endif
    //fflush(stdout);

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
    const double x_bar_star = *x_star + *px_star * S - x_slice_star + px_slice_star * S;
    const double y_bar_star = *y_star + *py_star * S - y_slice_star + py_slice_star * S;

    // Move to the uncoupled reference frame
    const double x_bar_hat_star = x_bar_star*costheta + y_bar_star * sintheta;
    const double y_bar_hat_star = -x_bar_star*sintheta + y_bar_star * costheta;

    // Compute derivatives of the transformation
    const double dS_x_bar_hat_star = x_bar_star * dS_costheta + y_bar_star * dS_sintheta;
    const double dS_y_bar_hat_star = -x_bar_star * dS_sintheta + y_bar_star * dS_costheta;

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

    // Move kicks to coupled reference frame
    double Fx_star = Fx_hat_star*costheta - Fy_hat_star*sintheta;
    double Fy_star = Fx_hat_star*sintheta + Fy_hat_star*costheta;

    // Compute longitudinal kick
    double Fz_star = 0.5*(Fx_hat_star*dS_x_bar_hat_star  + Fy_hat_star*dS_y_bar_hat_star+
                   Gx_hat_star*dS_Sig_11_hat_star + Gy_hat_star*dS_Sig_33_hat_star);

    double rho, wgt;

    // calculate luminosity
    const int64_t flag_luminosity = BeamBeamBiGaussian3DData_get_flag_luminosity(el);
    if (flag_luminosity == 1){
        do_luminosity(el, part, &rho, &wgt,
            x_bar_hat_star, y_bar_hat_star, Sig_11_hat_star, Sig_33_hat_star,
            num_part_slice);
    }

    // emit bhabha photons from single macropart
    #ifndef XFIELDS_BB3D_NO_BHABHA
    const int64_t flag_bhabha = BeamBeamBiGaussian3DData_get_flag_bhabha(el);
    if (flag_bhabha == 1) {
        do_bhabha(el, part, &rho, &wgt, flag_luminosity, q0,
            x_bar_hat_star, y_bar_hat_star, Sig_11_hat_star, Sig_33_hat_star,
            num_part_slice, S, px_star, py_star, pzeta_star,
            px_slice_star, py_slice_star, pzeta_slice_star);
    }
    #endif

    // emit beamstrahlung photons from single macropart
    #ifndef XFIELDS_BB3D_NO_BEAMSTR
    const int64_t flag_beamstrahlung = BeamBeamBiGaussian3DData_get_flag_beamstrahlung(el);
    if(flag_beamstrahlung!=0){
        do_beamstrahlung(el, part, Fx_star, Fy_star, pzeta_star,
           i_slice, num_part_slice, flag_beamstrahlung);
    }
    #endif

    // Apply the kicks (Hirata's synchro-beam)
    *pzeta_star = *pzeta_star + Fz_star + 0.5*(
                Fx_star*(*px_star+0.5*Fx_star + px_slice_star)+
                Fy_star*(*py_star+0.5*Fy_star + py_slice_star));
    *x_star = *x_star - S*Fx_star;
    *px_star = *px_star + Fx_star;
    *y_star = *y_star - S*Fy_star;
    *py_star = *py_star + Fy_star;
}


/*gpufun*/
void BeamBeamBiGaussian3D_track_local_particle(BeamBeamBiGaussian3DData el, LocalParticle* part0){

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
                synchrobeam_kick(
                             el, part,
                             i_slice, q0, p0c,
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
