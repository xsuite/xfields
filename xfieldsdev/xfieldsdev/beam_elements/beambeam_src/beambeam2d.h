// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_BEAMBEAM_H
#define XFIELDS_BEAMBEAM_H

#if !defined(mysign)
    #define mysign(a) (((a) >= 0) - ((a) < 0))
#endif

/*gpufun*/
void BeamBeamBiGaussian2D_track_local_particle(
        BeamBeamBiGaussian2DData el, LocalParticle* part0){

    double const ref_shift_x = BeamBeamBiGaussian2DData_get_ref_shift_x(el);
    double const ref_shift_y = BeamBeamBiGaussian2DData_get_ref_shift_y(el);

    double const other_beam_shift_x = BeamBeamBiGaussian2DData_get_other_beam_shift_x(el);
    double const other_beam_shift_y = BeamBeamBiGaussian2DData_get_other_beam_shift_y(el);

    double const scale_strength = BeamBeamBiGaussian2DData_get_scale_strength(el);
    double const post_subtract_px = scale_strength*BeamBeamBiGaussian2DData_get_post_subtract_px(el);
    double const post_subtract_py = scale_strength*BeamBeamBiGaussian2DData_get_post_subtract_py(el);

    double const other_beam_q0 = scale_strength*BeamBeamBiGaussian2DData_get_other_beam_q0(el);
    double const other_beam_beta0 = BeamBeamBiGaussian2DData_get_other_beam_beta0(el);

    double const other_beam_num_particles = BeamBeamBiGaussian2DData_get_other_beam_num_particles(el);

    double const other_beam_Sigma_11 = BeamBeamBiGaussian2DData_get_other_beam_Sigma_11(el);
    double const other_beam_Sigma_13 = BeamBeamBiGaussian2DData_get_other_beam_Sigma_13(el);
    double const other_beam_Sigma_33 = BeamBeamBiGaussian2DData_get_other_beam_Sigma_33(el);

    double const min_sigma_diff = BeamBeamBiGaussian2DData_get_min_sigma_diff(el);

    //start_per_particle_block (part0->part)

        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);
        double const part_q0 = LocalParticle_get_q0(part);
        double const part_mass0 = LocalParticle_get_mass0(part);
        double const part_chi = LocalParticle_get_chi(part);
        double const part_beta0 = LocalParticle_get_beta0(part);
        double const part_gamma0 = LocalParticle_get_gamma0(part);

        double const x_bar = x - ref_shift_x - other_beam_shift_x;
        double const y_bar = y - ref_shift_y - other_beam_shift_y;

        // Move to rotated frame to account for transverse coupling (if needed)
        double x_hat, y_hat, costheta, sintheta, Sig_11_hat, Sig_33_hat;
        if (fabs(other_beam_Sigma_13) > 1e-13) {
            double const R = other_beam_Sigma_11 - other_beam_Sigma_33;
            double const W = other_beam_Sigma_11 + other_beam_Sigma_33;
            double const T = R * R + 4 * other_beam_Sigma_13 * other_beam_Sigma_13;
            double const sqrtT = sqrt(T);
            double const signR = mysign(R);
            double const cos2theta = signR*R/sqrtT;
            costheta = sqrt(0.5*(1.+cos2theta));
            sintheta = signR*mysign(other_beam_Sigma_13)*sqrt(0.5*(1.-cos2theta));
            x_hat = x_bar*costheta +y_bar*sintheta;
            y_hat = -x_bar*sintheta +y_bar*costheta;
            Sig_11_hat = 0.5*(W+signR*sqrtT);
            Sig_33_hat = 0.5*(W-signR*sqrtT);
        }
        else{
            sintheta = 0;
            costheta = 1;
            x_hat = x_bar;
            y_hat = y_bar;
            Sig_11_hat = other_beam_Sigma_11;
            Sig_33_hat = other_beam_Sigma_33;
        }

        // Get transverse fields
        double Ex, Ey; // Ex = -dphi/dx, Ey = -dphi/dy
        get_Ex_Ey_gauss(x_hat, y_hat,
            sqrt(Sig_11_hat), sqrt(Sig_33_hat),
            min_sigma_diff,
            &Ex, &Ey);

        const double charge_mass_ratio = part_chi*QELEM*part_q0
                    /(part_mass0*QELEM/(C_LIGHT*C_LIGHT));
        const double factor = (charge_mass_ratio
                    * other_beam_num_particles * other_beam_q0 * QELEM
                    / (part_gamma0*part_beta0*C_LIGHT*C_LIGHT)
                    * (1+other_beam_beta0 * part_beta0)
                    / (other_beam_beta0 + part_beta0));

        double const dpx_hat = factor * Ex;
        double const dpy_hat = factor * Ey;

        double const dpx = dpx_hat*costheta - dpy_hat*sintheta;
        double const dpy = dpx_hat*sintheta + dpy_hat*costheta;

        LocalParticle_add_to_px(part, dpx - post_subtract_px);
        LocalParticle_add_to_py(part, dpy - post_subtract_py);

    //end_per_particle_block

}

#endif
