// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef XFIELDS_BEAMBEAM_H
#define XFIELDS_BEAMBEAM_H

#include "xfields/beam_elements/beambeam_src/beambeam2d_kick.h"


GPUFUN
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

    START_PER_PARTICLE_BLOCK(part0, part);
        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);

        double const x_bar = x - ref_shift_x - other_beam_shift_x;
        double const y_bar = y - ref_shift_y - other_beam_shift_y;

        BeamBeamBiGaussian2D_apply_kick(
            part,
            x_bar,
            y_bar,
            other_beam_num_particles,
            other_beam_q0,
            other_beam_beta0,
            other_beam_Sigma_11,
            other_beam_Sigma_13,
            other_beam_Sigma_33,
            min_sigma_diff,
            post_subtract_px,
            post_subtract_py);
    END_PER_PARTICLE_BLOCK;
}

#endif
