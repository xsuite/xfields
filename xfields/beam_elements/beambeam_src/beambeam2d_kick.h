// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_BEAMBEAM2D_KICK_H
#define XFIELDS_BEAMBEAM2D_KICK_H

#include "xtrack/headers/track.h"
#include "xfields/fieldmaps/bigaussian_src/bigaussian.h"


#if !defined(mysign)
    #define mysign(a) (((a) >= 0) - ((a) < 0))
#endif


GPUFUN
void BeamBeamBiGaussian2D_apply_kick(
        LocalParticle* part,
        double const x_bar,
        double const y_bar,
        double const other_beam_num_particles,
        double const other_beam_q0,
        double const other_beam_beta0,
        double const other_beam_Sigma_11,
        double const other_beam_Sigma_13,
        double const other_beam_Sigma_33,
        double const min_sigma_diff,
        double const post_subtract_px,
        double const post_subtract_py){

    double const part_q0 = LocalParticle_get_q0(part);
    double const part_mass0 = LocalParticle_get_mass0(part);
    double const part_chi = LocalParticle_get_chi(part);
    double const part_beta0 = LocalParticle_get_beta0(part);
    double const part_gamma0 = LocalParticle_get_gamma0(part);

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
}

#endif
