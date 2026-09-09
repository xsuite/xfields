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
            other_beam_Sigma_11=np.array([1.0e-3, 1.2e-3, 0.9e-3])**2,
            other_beam_Sigma_13=0,
            other_beam_Sigma_33=np.array([1.3e-3, 0.8e-3, 1.1e-3])**2,
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
        harmonic_number=3, bunch_spacing_buckets=1,
        num_particles=1.0)
    study.apply_filling_pattern(
        filling_pattern_cw=[1, 1, 1],
        filling_pattern_acw=[1, 1, 1])
    return study


def test_rigid_bunch_twiss_fast_modes_against_full(beam_beam_rigid_bunch_study):
    study = beam_beam_rigid_bunch_study
    names = ['slot_0', 'slot_1', 'slot_2']
    full_result = study.twiss(
        mode='full',
        show_progress=False,
    )
    fast_result = study.twiss(
        mode='fast',
        show_progress=False,
        co_tol=1e-12,
    )
    fast_orbit_result = study.twiss(
        mode='fast_orbit',
        show_progress=False,
        co_tol=1e-12,
    )

    assert isinstance(fast_result, xf.BeamBeamRigidBunchTwiss)
    assert isinstance(fast_result.cw, xf.MultiBunchTwiss)
    assert fast_result['cw'] is fast_result.cw
    assert fast_result['acw'] is fast_result.acw
    with pytest.raises(KeyError):
        fast_result['b1']
    with pytest.raises(AttributeError):
        fast_result.b1
    full, full_acw = full_result.cw, full_result.acw
    fast, fast_acw = fast_result.cw, fast_result.acw
    fast_orbit = fast_orbit_result.cw
    fast_orbit_acw = fast_orbit_result.acw

    assert len(full) == len(fast) == len(fast_orbit) == len(ZETA_BUNCHES)
    assert fast.bunch_names == tuple(names)
    assert np.array_equal(fast.filled_slots, [0, 1, 2])
    assert fast.bunch('slot_1') is fast.bunch(index=1)
    assert fast.bunch(slot=1) is fast.bunch(index=1)
    assert fast['qx', 'slot_1'] == fast.qx[1]
    assert np.array_equal(fast.rows['slot_1'].slot, [1])
    with pytest.raises(ValueError, match='exactly one'):
        fast.bunch()
    with pytest.raises(ValueError, match='exactly one'):
        fast.bunch('slot_1', slot=1)
    with pytest.raises(ValueError, match='read-only'):
        fast.filled_slots[0] = 2
    with pytest.raises(ValueError, match='read-only'):
        fast.slot[0] = 2
    xo.assert_allclose(fast.zeta_bunches, ZETA_BUNCHES, rtol=0, atol=1e-15)

    def stack(mbtw, column):
        return np.asarray([tw[column] for tw in mbtw.twiss_tables])

    xo.assert_allclose(stack(fast_acw, 'x'), stack(fast, 'x'), rtol=0, atol=0)
    xo.assert_allclose(stack(full_acw, 'x'), stack(full, 'x'), rtol=0, atol=0)
    xo.assert_allclose(
        stack(fast_orbit_acw, 'x'), stack(fast_orbit, 'x'), rtol=0, atol=0)

    for column in ('x', 'px', 'y', 'py'):
        xo.assert_allclose(
            stack(fast, column), stack(full, column), rtol=0, atol=5e-13)
        xo.assert_allclose(
            stack(fast_orbit, column), stack(full, column),
            rtol=0, atol=5e-13)
    for column in (
            'betx', 'alfx', 'bety', 'alfy', 'mux', 'muy',
            'dx', 'dpx', 'dy', 'dpy'):
        xo.assert_allclose(
            stack(fast, column), stack(full, column),
            rtol=2e-10, atol=2e-10)

    xo.assert_allclose(fast.qx, full.qx, rtol=0, atol=2e-10)
    xo.assert_allclose(fast.qy, full.qy, rtol=0, atol=2e-10)
    # fast_orbit has no accumulated phase advance, so qx/qy are fractional.
    # This toy ring has tunes below one and can be compared directly to full.
    xo.assert_allclose(fast_orbit.qx, full.qx, rtol=0, atol=2e-10)
    xo.assert_allclose(fast_orbit.qy, full.qy, rtol=0, atol=2e-10)
    xo.assert_allclose(
        fast.at_element('bb').x, full.at_element('bb').x,
        rtol=0, atol=5e-13)


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
