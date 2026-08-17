# copyright ############################### #
# This file is part of the Xfields Package. #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
import pytest

import xfields as xf
import xobjects as xo
import xtrack as xt


ZETA_BUNCHES = np.array([0.0, -0.2, -0.4])


@pytest.fixture
def beam_beam_rigid_bunch_study():
    def make_line():
        opposing = xt.Particles(
            p0c=7e12,
            x=[0.2e-3, -0.35e-3, 0.5e-3],
            y=[-0.1e-3, 0.25e-3, 0.15e-3],
            zeta=ZETA_BUNCHES,
            weight=[0.8e11, 1.1e11, 1.4e11],
        )
        bb = xf.BeamBeamBiGaussianRigidBunch2D(
            other_particles=opposing,
            zeta_match_tol=0.02,
            zeta_period=0.6,
            other_beam_q0=1,
            other_beam_beta0=float(opposing.beta0[0]),
            other_beam_sigma_x=[1.0e-3, 1.2e-3, 0.9e-3],
            other_beam_sigma_y=[1.3e-3, 0.8e-3, 1.1e-3],
        )
        arc = xt.LineSegmentMap(
            length=0.6,
            qx=0.31,
            qy=0.32,
            betx=[12, 12],
            bety=[15, 15],
        )
        line = xt.Line(
            elements=[bb, arc],
            element_names=['bb', 'arc'],
            particle_ref=xt.Particles(p0c=7e12),
        )
        line.build_tracker()
        return line

    study = xf.BeamBeamRigidBunchStudy(
        make_line(), make_line(), ips=['ip1'],
        num_long_range_encounters_per_side=0,
        harmonic_number=3, bunch_spacing_buckets=1)
    study.set_filling(
        filling_scheme_cw=[1, 1, 1],
        filling_scheme_acw=[1, 1, 1],
        bunch_intensity_particles_cw=1.0,
        bunch_intensity_particles_acw=1.0)
    return study


def test_rigid_bunch_twiss_fast_modes_against_full(beam_beam_rigid_bunch_study):
    study = beam_beam_rigid_bunch_study
    names = ['slot_0', 'slot_1', 'slot_2']
    full, full_acw = study.twiss(
        mode='full',
        show_progress=False,
    )
    fast, fast_acw = study.twiss(
        mode='fast',
        show_progress=False,
        co_tol=1e-12,
    )
    fast_orbit, fast_orbit_acw = study.twiss(
        mode='fast_orbit',
        show_progress=False,
        co_tol=1e-12,
    )

    assert len(full) == len(fast) == len(fast_orbit) == len(ZETA_BUNCHES)
    assert fast.bunch_names == names
    assert fast.bunch('slot_1') is fast[1]
    xo.assert_allclose(fast.zeta_bunches, ZETA_BUNCHES, rtol=0, atol=1e-15)
    xo.assert_allclose(fast_acw['x'], fast['x'], rtol=0, atol=0)
    xo.assert_allclose(full_acw['x'], full['x'], rtol=0, atol=0)
    xo.assert_allclose(fast_orbit_acw['x'], fast_orbit['x'], rtol=0, atol=0)

    for column in ('x', 'px', 'y', 'py'):
        xo.assert_allclose(
            fast[column], full[column], rtol=0, atol=5e-13)
        xo.assert_allclose(
            fast_orbit[column], full[column], rtol=0, atol=5e-13)
    for column in (
            'betx', 'alfx', 'bety', 'alfy', 'mux', 'muy',
            'dx', 'dpx', 'dy', 'dpy'):
        xo.assert_allclose(
            fast[column], full[column], rtol=2e-10, atol=2e-10)

    xo.assert_allclose(fast.qx, full.qx, rtol=0, atol=2e-10)
    xo.assert_allclose(fast.qy, full.qy, rtol=0, atol=2e-10)
    # fast_orbit has no accumulated phase advance, so qx/qy are fractional.
    # This toy ring has tunes below one and can be compared directly to full.
    xo.assert_allclose(fast_orbit.qx, full.qx, rtol=0, atol=2e-10)
    xo.assert_allclose(fast_orbit.qy, full.qy, rtol=0, atol=2e-10)
    xo.assert_allclose(
        fast['x', 'bb'], full['x', 'bb'], rtol=0, atol=5e-13)


def test_rigid_bunch_twiss_rejects_invalid_inputs(beam_beam_rigid_bunch_study):
    study = beam_beam_rigid_bunch_study

    with pytest.raises(ValueError, match='Unknown mode'):
        study.twiss(mode='unknown')
    with pytest.raises(ValueError, match="requires method='4d'"):
        study.twiss(mode='fast', method='6d')
    with pytest.raises(ValueError, match='not supported'):
        study.twiss(mode='fast', freeze_longitudinal=True)
    with pytest.raises(ValueError, match='cannot be provided'):
        study.twiss(zeta0=0.0)
