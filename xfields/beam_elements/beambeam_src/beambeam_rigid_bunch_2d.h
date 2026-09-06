// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_BEAMBEAM_RIGID_BUNCH_2D_H
#define XFIELDS_BEAMBEAM_RIGID_BUNCH_2D_H

#include "xfields/beam_elements/beambeam_src/beambeam2d_kick.h"


// Find, in a zeta-SORTED bunch array `zeta_arr` of length `n`, the bunch
// closest to `target` within `tol`, or -1 if none. If `period` > 0 the
// bunch-label axis is periodic (circular machine): the distance is evaluated
// modulo the period, so encounter offsets that wrap around the ring still find
// their partner. The nearest (mod period) bunch is either a linear neighbour of
// the folded target (found by binary search) or, across the wrap, one of the
// two ends.
GPUFUN
int64_t BeamBeamBiGaussianRigidBunch2D_match_bunch(
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
void BeamBeamBiGaussianRigidBunch2D_track_local_particle(
        BeamBeamBiGaussianRigidBunch2DData el, LocalParticle* part0){

    double const scale_strength = BeamBeamBiGaussianRigidBunch2DData_get_scale_strength(el);

    double const zeta_offset = BeamBeamBiGaussianRigidBunch2DData_get_zeta_offset(el);
    double const zeta_match_tol = BeamBeamBiGaussianRigidBunch2DData_get_zeta_match_tol(el);
    double const zeta_period = BeamBeamBiGaussianRigidBunch2DData_get_zeta_period(el);

    double const other_beam_q0 = scale_strength*BeamBeamBiGaussianRigidBunch2DData_get_other_beam_q0(el);
    double const other_beam_beta0 = BeamBeamBiGaussianRigidBunch2DData_get_other_beam_beta0(el);

    int64_t const coherent = BeamBeamBiGaussianRigidBunch2DData_get_coherent(el);

    double const min_sigma_diff = BeamBeamBiGaussianRigidBunch2DData_get_min_sigma_diff(el);

    int64_t const num_other_bunches = BeamBeamBiGaussianRigidBunch2DData_get_num_other_bunches(el);
    int64_t const num_own_bunches = BeamBeamBiGaussianRigidBunch2DData_get_num_own_bunches(el);

    // Sorted zeta grids of both beams (for the binary-search bunch matching)
    // and this beam's own per-bunch covariance.
    GPUGLMEM double const* other_beam_zeta =
        BeamBeamBiGaussianRigidBunch2DData_getp1_other_beam_zeta(el, 0);
    GPUGLMEM double const* own_beam_zeta =
        BeamBeamBiGaussianRigidBunch2DData_getp1_own_beam_zeta(el, 0);
    GPUGLMEM double const* own_beam_Sigma_11 =
        BeamBeamBiGaussianRigidBunch2DData_getp1_own_beam_Sigma_11(el, 0);
    GPUGLMEM double const* own_beam_Sigma_33 =
        BeamBeamBiGaussianRigidBunch2DData_getp1_own_beam_Sigma_33(el, 0);

    START_PER_PARTICLE_BLOCK(part0, part);
        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);
        double const zeta = LocalParticle_get_zeta(part);

        // This particle (bunch) at `zeta` encounters the opposing bunch located
        // at `zeta + zeta_offset` (indexing of the OTHER beam), found by the
        // binary-search match on the sorted opposing-beam zeta grid.
        int64_t const i_match = BeamBeamBiGaussianRigidBunch2D_match_bunch(
            other_beam_zeta, num_other_bunches, zeta + zeta_offset,
            zeta_match_tol, zeta_period);

        if (i_match < 0){
            // No opposing bunch at the encounter position -> no kick
            continue;
        }

        double const other_beam_shift_x = BeamBeamBiGaussianRigidBunch2DData_get_other_beam_shift_x(el, i_match);
        double const other_beam_shift_y = BeamBeamBiGaussianRigidBunch2DData_get_other_beam_shift_y(el, i_match);
        double const other_beam_num_particles =
            BeamBeamBiGaussianRigidBunch2DData_get_other_beam_num_particles(el, i_match);

        // Diagonal transverse covariance of the matched opposing bunch
        // (indexed by the OTHER beam). In coherent mode the effective diagonal
        // covariance is the sum with this beam's OWN covariance. Sigma_13 is
        // intentionally ignored until coupled rigid-bunch operation is
        // validated.
        double Sigma_11 = BeamBeamBiGaussianRigidBunch2DData_get_other_beam_Sigma_11(el, i_match);
        double Sigma_33 = BeamBeamBiGaussianRigidBunch2DData_get_other_beam_Sigma_33(el, i_match);
        if (coherent){
            int64_t i_own = 0;
            if (num_own_bunches > 1){
                i_own = BeamBeamBiGaussianRigidBunch2D_match_bunch(
                    own_beam_zeta, num_own_bunches, zeta,
                    zeta_match_tol, zeta_period);
                if (i_own < 0) i_own = 0;   // fall back to the first own bunch
            }
            Sigma_11 += own_beam_Sigma_11[i_own];
            Sigma_33 += own_beam_Sigma_33[i_own];
        }

        double const x_bar = x - other_beam_shift_x;
        double const y_bar = y - other_beam_shift_y;

        BeamBeamBiGaussian2D_apply_kick(
            part,
            x_bar,
            y_bar,
            other_beam_num_particles,
            other_beam_q0,
            other_beam_beta0,
            Sigma_11,
            0., // Transverse coupling is stored but not yet used by this model.
            Sigma_33,
            min_sigma_diff,
            0.,
            0.);
    END_PER_PARTICLE_BLOCK;
}

#endif
