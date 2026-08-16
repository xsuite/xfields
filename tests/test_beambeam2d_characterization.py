# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import numpy as np

import xfields as xf
import xobjects as xo
import xpart as xp
import xtrack as xt

from xobjects.test_helpers import for_all_test_contexts


P0C = 450e9
BETA0 = 0.999999
INTENSITY = 1.8e11


def _make_particles(test_context):
    return xp.Particles(
        _context=test_context,
        p0c=P0C,
        q0=1,
        mass0=xp.PROTON_MASS_EV,
        x=np.array([-1.7e-3, -0.4e-3, 0.8e-3, 2.1e-3]),
        y=np.array([0.9e-3, -1.2e-3, 0.3e-3, 1.5e-3]),
        px=np.array([1.0e-6, -2.0e-6, 0.5e-6, 3.0e-6]),
        py=np.array([-0.7e-6, 1.3e-6, -2.2e-6, 0.4e-6]),
    )


def _track_increment(element, particles):
    px0 = particles._context.nparray_from_context_array(particles.px).copy()
    py0 = particles._context.nparray_from_context_array(particles.py).copy()
    element.track(particles)
    ctx2np = particles._context.nparray_from_context_array
    return ctx2np(particles.px) - px0, ctx2np(particles.py) - py0


@for_all_test_contexts
def test_bb2d_modern_and_legacy_interfaces_are_equivalent(test_context):
    sigma_x = 1.3e-3
    sigma_y = 2.2e-3
    common = dict(
        scale_strength=0.63,
        ref_shift_x=0.17e-3,
        ref_shift_y=-0.23e-3,
        min_sigma_diff=1e-12,
        _context=test_context,
    )
    modern = xf.BeamBeamBiGaussian2D(
        other_beam_q0=1,
        other_beam_beta0=BETA0,
        other_beam_num_particles=INTENSITY,
        other_beam_Sigma_11=sigma_x**2,
        other_beam_Sigma_33=sigma_y**2,
        other_beam_shift_x=-0.31e-3,
        other_beam_shift_y=0.29e-3,
        post_subtract_px=1.7e-7,
        post_subtract_py=-2.4e-7,
        **common,
    )
    legacy = xf.BeamBeamBiGaussian2D(
        n_particles=INTENSITY,
        q0=1,
        beta0=BETA0,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        mean_x=-0.31e-3,
        mean_y=0.29e-3,
        d_px=1.7e-7,
        d_py=-2.4e-7,
        **common,
    )

    fields = (
        'scale_strength', 'ref_shift_x', 'ref_shift_y',
        'other_beam_shift_x', 'other_beam_shift_y',
        'post_subtract_px', 'post_subtract_py', 'other_beam_q0',
        'other_beam_beta0', 'other_beam_num_particles',
        'other_beam_Sigma_11', 'other_beam_Sigma_13',
        'other_beam_Sigma_33', 'min_sigma_diff',
    )
    for field in fields:
        assert getattr(modern, field) == getattr(legacy, field)

    p_modern = _make_particles(test_context)
    p_legacy = p_modern.copy()
    kick_modern = _track_increment(modern, p_modern)
    kick_legacy = _track_increment(legacy, p_legacy)
    xo.assert_allclose(kick_modern, kick_legacy, rtol=0, atol=0)


@for_all_test_contexts
def test_bb2d_shifts_subtraction_strength_copy_and_dict(test_context):
    bb = xf.BeamBeamBiGaussian2D(
        other_beam_q0=1,
        other_beam_beta0=BETA0,
        other_beam_num_particles=INTENSITY,
        other_beam_Sigma_11=(1.1e-3)**2,
        other_beam_Sigma_33=(1.9e-3)**2,
        ref_shift_x=0.23e-3,
        ref_shift_y=-0.19e-3,
        other_beam_shift_x=-0.37e-3,
        other_beam_shift_y=0.41e-3,
        post_subtract_px=1.3e-7,
        post_subtract_py=-0.8e-7,
        min_sigma_diff=1e-12,
        _context=test_context,
    )

    particles = _make_particles(test_context)
    kick_full = _track_increment(bb, particles.copy())

    for scale in (0.0, 0.37, 1.0):
        bb_scaled = bb.copy(_context=test_context)
        bb_scaled.scale_strength = scale
        kick_scaled = _track_increment(bb_scaled, particles.copy())
        xo.assert_allclose(
            kick_scaled, np.asarray(kick_full) * scale,
            rtol=2e-14, atol=1e-30)

    bb_unshifted = bb.copy(_context=test_context)
    bb_unshifted.ref_shift_x = 0
    bb_unshifted.ref_shift_y = 0
    bb_unshifted.other_beam_shift_x = 0
    bb_unshifted.other_beam_shift_y = 0
    shifted_coordinates = particles.copy()
    shifted_coordinates.x -= bb.ref_shift_x + bb.other_beam_shift_x
    shifted_coordinates.y -= bb.ref_shift_y + bb.other_beam_shift_y
    kick_unshifted = _track_increment(bb_unshifted, shifted_coordinates)
    xo.assert_allclose(kick_unshifted, kick_full, rtol=2e-14, atol=1e-30)

    bb_no_subtraction = bb.copy(_context=test_context)
    bb_no_subtraction.post_subtract_px = 0
    bb_no_subtraction.post_subtract_py = 0
    kick_no_subtraction = _track_increment(
        bb_no_subtraction, particles.copy())
    xo.assert_allclose(
        kick_full[0] - kick_no_subtraction[0],
        -bb.post_subtract_px, rtol=0, atol=2e-22)
    xo.assert_allclose(
        kick_full[1] - kick_no_subtraction[1],
        -bb.post_subtract_py, rtol=0, atol=2e-22)

    copies = (
        bb.copy(_context=test_context),
        xf.BeamBeamBiGaussian2D.from_dict(
            bb.to_dict(), _context=test_context),
    )
    for restored in copies:
        kick_restored = _track_increment(restored, particles.copy())
        xo.assert_allclose(kick_restored, kick_full, rtol=0, atol=0)


@for_all_test_contexts
def test_bb2d_scale_strength_through_line_variable(test_context):
    bb = xf.BeamBeamBiGaussian2D(
        other_beam_q0=1,
        other_beam_beta0=BETA0,
        other_beam_num_particles=INTENSITY,
        other_beam_Sigma_11=(1.2e-3)**2,
        other_beam_Sigma_33=(1.8e-3)**2,
        post_subtract_px=0.7e-7,
        post_subtract_py=-1.1e-7,
        _context=test_context,
    )
    line = xt.Line(elements=[bb], element_names=['bb'])
    line['beambeam_scale'] = 1.0
    line['bb'].scale_strength = 'beambeam_scale'
    line.build_tracker(_context=test_context)

    particles = _make_particles(test_context)
    ctx2np = test_context.nparray_from_context_array
    px_initial = ctx2np(particles.px).copy()
    py_initial = ctx2np(particles.py).copy()
    line.track(particles)
    kick_full = (
        ctx2np(particles.px) - px_initial,
        ctx2np(particles.py) - py_initial,
    )

    for scale in (0.0, 0.41, 1.0):
        line['beambeam_scale'] = scale
        scaled_particles = _make_particles(test_context)
        px0 = ctx2np(scaled_particles.px).copy()
        py0 = ctx2np(scaled_particles.py).copy()
        line.track(scaled_particles)
        kick_scaled = (
            ctx2np(scaled_particles.px) - px0,
            ctx2np(scaled_particles.py) - py0,
        )
        xo.assert_allclose(
            kick_scaled, np.asarray(kick_full) * scale,
            rtol=2e-14, atol=1e-30)
