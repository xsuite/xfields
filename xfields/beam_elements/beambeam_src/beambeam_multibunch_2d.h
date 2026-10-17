// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_BEAMBEAM_MULTIBUNCH_2D_H
#define XFIELDS_BEAMBEAM_MULTIBUNCH_2D_H

#include "xtrack/headers/track.h"
#include "xfields/fieldmaps/bigaussian_src/bigaussian.h"


// Find, in a zeta-SORTED bunch array `zeta_arr` of length `n`, the bunch
// closest to `target` within `tol`, or -1 if none. If `period` > 0 the
// bunch-label axis is periodic (circular machine): the distance is evaluated
// modulo the period, so encounter offsets that wrap around the ring still find
// their partner. The nearest (mod period) bunch is either a linear neighbour of
// the folded target (found by binary search) or, across the wrap, one of the
// two ends.
GPUFUN
int64_t BeamBeamBiGaussianMultibunch2D_match_bunch(
        GPUGLMEM double const* zeta_arr, int64_t const n,
        double const target, double const tol, double const period){
    if (n <= 0){
        return -1;
    }
    double tt = target;
    if (period > 0.){
        double const z_first = zeta_arr[0];
        double const z_last = zeta_arr[n - 1];
        double const z_mid = 0.5 * (z_first + z_last);
        tt -= period * round((tt - z_mid) / period);
    }
    int64_t lo = 0;                          // lower bound: first z >= tt
    int64_t hi = n;
    while (lo < hi){
        int64_t const mid = (lo + hi) / 2;
        if (zeta_arr[mid] < tt){
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    int64_t const cand[4] = {lo - 1, lo, 0, n - 1};
    int64_t i_match = -1;
    double best_dist = tol;
    for (int cc = 0; cc < 4; cc++){
        int64_t const jj = cand[cc];
        if (jj < 0 || jj >= n) continue;
        double dist = zeta_arr[jj] - target;
        if (period > 0.){
            dist -= period * round(dist / period);
        }
        dist = fabs(dist);
        if (dist <= best_dist){
            best_dist = dist;
            i_match = jj;
        }
    }
    return i_match;
}


GPUFUN
void BeamBeamBiGaussianMultibunch2D_track_local_particle(
        BeamBeamBiGaussianMultibunch2DData el, LocalParticle* part0){

    double const scale_strength = BeamBeamBiGaussianMultibunch2DData_get_scale_strength(el);

    double const zeta_offset = BeamBeamBiGaussianMultibunch2DData_get_zeta_offset(el);
    double const zeta_match_tol = BeamBeamBiGaussianMultibunch2DData_get_zeta_match_tol(el);
    double const zeta_period = BeamBeamBiGaussianMultibunch2DData_get_zeta_period(el);

    double const other_beam_q0 = scale_strength*BeamBeamBiGaussianMultibunch2DData_get_other_beam_q0(el);
    double const other_beam_beta0 = BeamBeamBiGaussianMultibunch2DData_get_other_beam_beta0(el);

    int64_t const coherent = BeamBeamBiGaussianMultibunch2DData_get_coherent(el);

    double const min_sigma_diff = BeamBeamBiGaussianMultibunch2DData_get_min_sigma_diff(el);

    int64_t const num_other_bunches = BeamBeamBiGaussianMultibunch2DData_get_num_other_bunches(el);
    int64_t const num_own_bunches = BeamBeamBiGaussianMultibunch2DData_get_num_own_bunches(el);

    // Sorted zeta grids of both beams (for the binary-search bunch matching)
    // and this beam's own per-bunch sizes.
    GPUGLMEM double const* other_beam_zeta =
        BeamBeamBiGaussianMultibunch2DData_getp1_other_beam_zeta(el, 0);
    GPUGLMEM double const* own_beam_zeta =
        BeamBeamBiGaussianMultibunch2DData_getp1_own_beam_zeta(el, 0);
    GPUGLMEM double const* own_sigma_x_arr =
        BeamBeamBiGaussianMultibunch2DData_getp1_sigma_x(el, 0);
    GPUGLMEM double const* own_sigma_y_arr =
        BeamBeamBiGaussianMultibunch2DData_getp1_sigma_y(el, 0);

    START_PER_PARTICLE_BLOCK(part0, part);
        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);
        double const zeta = LocalParticle_get_zeta(part);
        double const part_q0 = LocalParticle_get_q0(part);
        double const part_mass0 = LocalParticle_get_mass0(part);
        double const part_chi = LocalParticle_get_chi(part);
        double const part_beta0 = LocalParticle_get_beta0(part);
        double const part_gamma0 = LocalParticle_get_gamma0(part);

        // This particle (bunch) at `zeta` encounters the opposing bunch located
        // at `zeta + zeta_offset` (indexing of the OTHER beam), found by the
        // binary-search match on the sorted opposing-beam zeta grid.
        int64_t const i_match = BeamBeamBiGaussianMultibunch2D_match_bunch(
            other_beam_zeta, num_other_bunches, zeta + zeta_offset,
            zeta_match_tol, zeta_period);

        if (i_match < 0){
            // No opposing bunch at the encounter position -> no kick
            continue;
        }

        double const other_beam_shift_x = BeamBeamBiGaussianMultibunch2DData_get_other_beam_x(el, i_match);
        double const other_beam_shift_y = BeamBeamBiGaussianMultibunch2DData_get_other_beam_y(el, i_match);
        double const other_beam_num_particles =
            BeamBeamBiGaussianMultibunch2DData_get_other_beam_num_particles(el, i_match);

        // Transverse size of the matched opposing bunch (indexed by the OTHER
        // beam). In the coherent (rigid-bunch) mode the effective Gaussian size
        // is the convolution with this beam's OWN size: the own size is indexed
        // by THIS beam -- the particle is matched to its own bunch on the
        // own-beam zeta grid (a single own bunch -> uniform size, index 0).
        double sigma_x = BeamBeamBiGaussianMultibunch2DData_get_other_beam_sigma_x(el, i_match);
        double sigma_y = BeamBeamBiGaussianMultibunch2DData_get_other_beam_sigma_y(el, i_match);
        if (coherent){
            int64_t i_own = 0;
            if (num_own_bunches > 1){
                i_own = BeamBeamBiGaussianMultibunch2D_match_bunch(
                    own_beam_zeta, num_own_bunches, zeta,
                    zeta_match_tol, zeta_period);
                if (i_own < 0) i_own = 0;   // fall back to the first own bunch
            }
            double const own_sigma_x = own_sigma_x_arr[i_own];
            double const own_sigma_y = own_sigma_y_arr[i_own];
            sigma_x = sqrt(sigma_x*sigma_x + own_sigma_x*own_sigma_x);
            sigma_y = sqrt(sigma_y*sigma_y + own_sigma_y*own_sigma_y);
        }

        double const x_bar = x - other_beam_shift_x;
        double const y_bar = y - other_beam_shift_y;

        // Get transverse fields
        double Ex, Ey; // Ex = -dphi/dx, Ey = -dphi/dy
        get_Ex_Ey_gauss(x_bar, y_bar,
            sigma_x, sigma_y,
            min_sigma_diff,
            &Ex, &Ey);

        const double charge_mass_ratio = part_chi*QELEM*part_q0
                    /(part_mass0*QELEM/(C_LIGHT*C_LIGHT));
        const double factor = (charge_mass_ratio
                    * other_beam_num_particles * other_beam_q0 * QELEM
                    / (part_gamma0*part_beta0*C_LIGHT*C_LIGHT)
                    * (1+other_beam_beta0 * part_beta0)
                    / (other_beam_beta0 + part_beta0));

        double const dpx = factor * Ex;
        double const dpy = factor * Ey;

        LocalParticle_add_to_px(part, dpx);
        LocalParticle_add_to_py(part, dpy);
    END_PER_PARTICLE_BLOCK;
}

#endif
