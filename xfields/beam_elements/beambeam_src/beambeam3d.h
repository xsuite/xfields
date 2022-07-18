// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_BEAMBEAM3D_H
#define XFIELDS_BEAMBEAM3D_H


/*gpufun*/
void BoostParameters_boost_coordinates(
        double const sphi,
        double const cphi,
        double const tphi,
        double const salpha,
        double const calpha,
        double* x_star,
        double* px_star,
        double* y_star,
        double* py_star,
        double* sigma_star,
        double* delta_star){


    double const x = *x_star;
    double const px = *px_star;
    double const y = *y_star;
    double const py = *py_star ;
    double const sigma = *sigma_star;
    double const delta = *delta_star ;

    double const h = delta + 1. - sqrt((1.+delta)*(1.+delta)-px*px-py*py);


    double const px_st = px/cphi-h*calpha*tphi/cphi;
    double const py_st = py/cphi-h*salpha*tphi/cphi;
    double const delta_st = delta -px*calpha*tphi-py*salpha*tphi+h*tphi*tphi;

    double const pz_st =
        sqrt((1.+delta_st)*(1.+delta_st)-px_st*px_st-py_st*py_st);

    double const hx_st = px_st/pz_st;
    double const hy_st = py_st/pz_st;
    double const hsigma_st = 1.-(delta_st+1)/pz_st;

    double const L11 = 1.+hx_st*calpha*sphi;
    double const L12 = hx_st*salpha*sphi;
    double const L13 = calpha*tphi;

    double const L21 = hy_st*calpha*sphi;
    double const L22 = 1.+hy_st*salpha*sphi;
    double const L23 = salpha*tphi;

    double const L31 = hsigma_st*calpha*sphi;
    double const L32 = hsigma_st*salpha*sphi;
    double const L33 = 1./cphi;

    double const x_st = L11*x + L12*y + L13*sigma;
    double const y_st = L21*x + L22*y + L23*sigma;
    double const sigma_st = L31*x + L32*y + L33*sigma;

    *x_star = x_st;
    *px_star = px_st;
    *y_star = y_st;
    *py_star = py_st;
    *sigma_star = sigma_st;
    *delta_star = delta_st;

}

/*gpufun*/
void BoostParameters_boost_coordinates_inv(
        double const sphi,
        double const cphi,
        double const tphi,
        double const salpha,
        double const calpha,
        double* x,
        double* px,
        double* y,
        double* py,
        double* sigma,
        double* delta){

    double const x_st = *x;
    double const px_st = *px;
    double const y_st = *y;
    double const py_st = *py ;
    double const sigma_st = *sigma;
    double const delta_st = *delta ;

    double const pz_st = sqrt((1.+delta_st)*(1.+delta_st)-px_st*px_st-py_st*py_st);
    double const hx_st = px_st/pz_st;
    double const hy_st = py_st/pz_st;
    double const hsigma_st = 1.-(delta_st+1)/pz_st;

    double const Det_L =
        1./cphi + (hx_st*calpha + hy_st*salpha-hsigma_st*sphi)*tphi;

    double const Linv_11 =
        (1./cphi + salpha*tphi*(hy_st-hsigma_st*salpha*sphi))/Det_L;

    double const Linv_12 =
        (salpha*tphi*(hsigma_st*calpha*sphi-hx_st))/Det_L;

    double const Linv_13 =
        -tphi*(calpha - hx_st*salpha*salpha*sphi + hy_st*calpha*salpha*sphi)/Det_L;

    double const Linv_21 =
        (calpha*tphi*(-hy_st + hsigma_st*salpha*sphi))/Det_L;

    double const Linv_22 =
        (1./cphi + calpha*tphi*(hx_st-hsigma_st*calpha*sphi))/Det_L;

    double const Linv_23 =
        -tphi*(salpha - hy_st*calpha*calpha*sphi + hx_st*calpha*salpha*sphi)/Det_L;

    double const Linv_31 = -hsigma_st*calpha*sphi/Det_L;
    double const Linv_32 = -hsigma_st*salpha*sphi/Det_L;
    double const Linv_33 = (1. + hx_st*calpha*sphi + hy_st*salpha*sphi)/Det_L;

    double const x_i = Linv_11*x_st + Linv_12*y_st + Linv_13*sigma_st;
    double const y_i = Linv_21*x_st + Linv_22*y_st + Linv_23*sigma_st;
    double const sigma_i = Linv_31*x_st + Linv_32*y_st + Linv_33*sigma_st;

    double const h = (delta_st+1.-pz_st)*cphi*cphi;

    double const px_i = px_st*cphi+h*calpha*tphi;
    double const py_i = py_st*cphi+h*salpha*tphi;

    double const delta_i = delta_st + px_i*calpha*tphi + py_i*salpha*tphi - h*tphi*tphi;


    *x = x_i;
    *px = px_i;
    *y = y_i;
    *py = py_i;
    *sigma = sigma_i;
    *delta = delta_i;

}

/*gpufun*/ void compute_Gx_Gy(
        const double  x,
        const double  y,
        const double  sigma_x,
        const double  sigma_y,
        const double  min_sigma_diff,
        const double  Ex,
        const double  Ey,
        double* Gx_ptr,
        double* Gy_ptr){

    double Gx, Gy;

    if (fabs(sigma_x-sigma_y) < min_sigma_diff){

        const double sigma = 0.5*(sigma_x+sigma_y);
        Gx = 1/(2.*(x*x+y*y))*(y*Ey-x*Ex+1./(2*PI*EPSILON_0*sigma*sigma)
                            *x*x*exp(-(x*x+y*y)/(2.*sigma*sigma)));
        Gy = 1./(2*(x*x+y*y))*(x*Ex-y*Ey+1./(2*PI*EPSILON_0*sigma*sigma)
                            *y*y*exp(-(x*x+y*y)/(2.*sigma*sigma)));
    }
    else{

        const double Sig_11 = sigma_x*sigma_x;
        const double Sig_33 = sigma_y*sigma_y;

        Gx =-1./(2*(Sig_11-Sig_33))*(x*Ex+y*Ey+1./(2*PI*EPSILON_0)
                   *(sigma_y/sigma_x*exp(-x*x/(2*Sig_11)-y*y/(2*Sig_33))-1.));
        Gy =1./(2*(Sig_11-Sig_33))*(x*Ex+y*Ey+1./(2*PI*EPSILON_0)*
                      (sigma_x/sigma_y*exp(-x*x/(2*Sig_11)-y*y/(2*Sig_33))-1.));

    }

    *Gx_ptr = Gx;
    *Gy_ptr = Gy;
}

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
    const double q0_bb  = BeamBeamBiGaussian3DData_get_q0_other_beam(el);
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

    // Get transverse fieds
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


void change_ref_frame(
        double* x, double* px, double* y, double* py, double* zeta, double* pzeta,
        double const shift_x, double const shift_px,
        double const shift_y, double const shift_py,
        double const shift_zeta, double const shift_pzeta,
        double const sin_phi, double const cos_phi, double const tan_phi,
        double const sin_alpha, double const cos_alpha){

    // Change reference frame
    double x_star =     *x     - shift_x;
    double px_star =    *px    - shift_px;
    double y_star =     *y     - shift_y;
    double py_star =    *py    - shift_py;
    double zeta_star =  *zeta  - shift_zeta;
    double pzeta_star = *pzeta - shift_pzeta;

    // Boost coordinates of the weak beam
    BoostParameters_boost_coordinates(
        sin_phi, cos_phi, tan_phi, sin_alpha, cos_alpha,
        &x_star, &px_star, &y_star, &py_star,
        &zeta_star, &pzeta_star);

    *x = x_star;
    *px = px_star;
    *y = y_star;
    *py = py_star;
    *zeta = zeta_star;
    *pzeta = pzeta_star;
    }

void change_back_ref_frame_and_subtract_dipolar(
        double* x, double* px,
        double* y, double* py,
        double* zeta, double* pzeta,
        double const shift_x, double const shift_px,
        double const shift_y, double const shift_py,
        double const shift_zeta, double const shift_pzeta,
        double const post_subtract_x, double const post_subtract_px,
        double const post_subtract_y, double const post_subtract_py,
        double const post_subtract_zeta, double const post_subtract_pzeta,
        double const sin_phi, double const cos_phi, double const tan_phi,
        double const sin_alpha, double const cos_alpha){

    // Inverse boost on the coordinates of the weak beam
    BoostParameters_boost_coordinates_inv(
        sin_phi, cos_phi, tan_phi, sin_alpha, cos_alpha,
        x, px, y, py, zeta, pzeta);

    // Go back to original reference frame and remove dipolar effect
    *x =     *x     + shift_x     - post_subtract_x;
    *px =    *px    + shift_px    - post_subtract_px;
    *y =     *y     + shift_y     - post_subtract_y;
    *py =    *py    + shift_py    - post_subtract_py;
    *zeta =  *zeta  + shift_zeta  - post_subtract_zeta;
    *pzeta = *pzeta + shift_pzeta - post_subtract_pzeta;

    }

/*gpufun*/
void BeamBeamBiGaussian3D_track_local_particle(BeamBeamBiGaussian3DData el,
                LocalParticle* part0){

    // Get data from memory
    double const sin_phi = BeamBeamBiGaussian3DData_get_sin_phi(el);
    double const cos_phi = BeamBeamBiGaussian3DData_get_cos_phi(el);
    double const tan_phi = BeamBeamBiGaussian3DData_get_tan_phi(el);
    double const sin_alpha = BeamBeamBiGaussian3DData_get_sin_alpha(el);
    double const cos_alpha = BeamBeamBiGaussian3DData_get_cos_alpha(el);

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
        change_ref_frame(
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
        change_back_ref_frame_and_subtract_dipolar(
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
void boost_local_particle(BeamBeamBiGaussian3DData el,
                LocalParticle* part0){

    // Get data from memory
    double const sin_phi = BeamBeamBiGaussian3DData_get_sin_phi(el);
    double const cos_phi = BeamBeamBiGaussian3DData_get_cos_phi(el);
    double const tan_phi = BeamBeamBiGaussian3DData_get_tan_phi(el);
    double const sin_alpha = BeamBeamBiGaussian3DData_get_sin_alpha(el);
    double const cos_alpha = BeamBeamBiGaussian3DData_get_cos_alpha(el);

    //start_per_particle_block (part0->part)

        double x = LocalParticle_get_x(part);
        double px = LocalParticle_get_px(part);
        double y = LocalParticle_get_y(part);
        double py = LocalParticle_get_py(part);
        double zeta = LocalParticle_get_zeta(part);
        double pzeta = LocalParticle_get_pzeta(part);

        // Change reference frame
        double x_star =     x;
        double px_star =    px;
        double y_star =     y;
        double py_star =    py;
        double sigma_star = zeta;
        double pzeta_star = pzeta; // TODO: could be fixed, in any case we assume beta=beta0=1
                                          //       in the synchrobeam

        // Boost coordinates of the weak beam
        BoostParameters_boost_coordinates(
            sin_phi, cos_phi, tan_phi, sin_alpha, cos_alpha,
            &x_star, &px_star, &y_star, &py_star,
            &sigma_star, &pzeta_star);

        LocalParticle_set_x(part, x);
        LocalParticle_set_px(part, px);
        LocalParticle_set_y(part, y);
        LocalParticle_set_py(part, py);
        LocalParticle_set_zeta(part, zeta);
        LocalParticle_update_pzeta(part, pzeta);

    //end_per_particle_block

}

/*gpufun*/
void BeamBeam3D_selective_apply_synchrobeam_kick(BeamBeamBiGaussian3DData el,
                LocalParticle* part0,
                const int64_t i_step,
                /*gpuglmem*/ int64_t* i_slice_for_particles){


    //start_per_particle_block (part0->part)

        const int64_t N_slices = BeamBeamBiGaussian3DData_get_num_slices_other_beam(el);
        const int64_t i_slice = i_step - i_slice_for_particles[part->ipart];

        if (i_slice >= 0 && i_slice < N_slices){

            double x_star = LocalParticle_get_x(part);
            double px_star = LocalParticle_get_px(part);
            double y_star = LocalParticle_get_y(part);
            double py_star = LocalParticle_get_py(part);
            double zeta_star = LocalParticle_get_zeta(part);
            double pzeta_star = LocalParticle_get_pzeta(part);

            const double q0 = LocalParticle_get_q0(part);
            const double p0c = LocalParticle_get_p0c(part); // eV

            synchrobeam_kick(
                el, i_slice,
                q0, p0c,
                &x_star,
                &px_star,
                &y_star,
                &py_star,
                &zeta_star,
                &pzeta_star);

            LocalParticle_set_x(part, x_star);
            LocalParticle_set_px(part, px_star);
            LocalParticle_set_y(part, y_star);
            LocalParticle_set_py(part, py_star);
            LocalParticle_set_zeta(part, zeta_star);
            LocalParticle_update_pzeta(part, pzeta_star);

        }

    //end_per_particle_block

}


#endif
