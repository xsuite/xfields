# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
import pytest

import xobjects as xo
import xpart as xp
import xfields as xf

from xobjects.test_helpers import for_all_test_contexts

P0C = 450e9
GAMMA0 = P0C / xp.PROTON_MASS_EV
BETA0 = np.sqrt(1 - 1 / GAMMA0**2)
NEMITT = 1.5e-6
INTENSITY = 1.8e11
N_SLOTS = 3564
DZ = 1e-3   # zeta label spacing per slot
SIGMA = np.sqrt(11. * NEMITT / GAMMA0)   # reference transverse size


def _make_element(test_context, opp, zeta_offset_slots, zeta_period,
                  sigma_x=SIGMA, sigma_y=SIGMA):
    return xf.BeamBeamBiGaussianRigidBunch2D(
        other_particles=opp,
        zeta_offset=zeta_offset_slots * DZ,
        zeta_match_tol=0.4 * DZ,
        zeta_period=zeta_period,
        other_beam_q0=1.0, other_beam_beta0=BETA0,
        other_beam_sigma_x=sigma_x, other_beam_sigma_y=sigma_y,
        _context=test_context)


def _kick(test_context, bb, slot, x=1e-4, y=5e-5):
    p = xp.Particles(_context=test_context, p0c=P0C, q0=1,
                     mass0=xp.PROTON_MASS_EV, x=x, y=y, zeta=slot * DZ)
    bb.track(p)
    p.move(_context=None)  # to CPU
    return float(p.px[0]), float(p.py[0])


@for_all_test_contexts
def test_multibunch_matches_bb2d(test_context):
    # A single head-on encounter must reproduce BeamBeamBiGaussian2D exactly
    opp = xp.Particles(_context=test_context, p0c=P0C, q0=1,
                       mass0=xp.PROTON_MASS_EV,
                       x=[2e-4], y=[-1e-4], zeta=[0.0], weight=INTENSITY)
    bb = _make_element(test_context, opp, 0, N_SLOTS * DZ)

    x_probe = np.linspace(-3e-3, 3e-3, 7)
    p = xp.Particles(_context=test_context, p0c=P0C, q0=1,
                     mass0=xp.PROTON_MASS_EV,
                     x=x_probe, y=4e-4, zeta=0.0)
    p_ref = p.copy()
    bb.track(p)

    bb_ref = xf.BeamBeamBiGaussian2D(
        other_beam_q0=1.0, other_beam_beta0=BETA0,
        other_beam_num_particles=INTENSITY,
        other_beam_Sigma_11=SIGMA**2, other_beam_Sigma_33=SIGMA**2,
        other_beam_shift_x=2e-4, other_beam_shift_y=-1e-4,
        _context=test_context)
    bb_ref.track(p_ref)

    p.move(_context=None)
    p_ref.move(_context=None)
    assert np.allclose(p.px, p_ref.px, rtol=1e-13, atol=1e-30)
    assert np.allclose(p.py, p_ref.py, rtol=1e-13, atol=1e-30)


@for_all_test_contexts
def test_multibunch_coherent(test_context):
    # coherent=True must convolve the own and opposing sizes:
    # equivalent to BeamBeamBiGaussian2D with Sigma = sigma_own^2 + sigma_other^2
    opp = xp.Particles(_context=test_context, p0c=P0C, q0=1,
                       mass0=xp.PROTON_MASS_EV,
                       x=[2e-4], y=[-1e-4], zeta=[0.0], weight=INTENSITY)
    sig_own_x, sig_own_y = 0.8 * SIGMA, 1.3 * SIGMA
    bb = xf.BeamBeamBiGaussianRigidBunch2D(
        other_particles=opp, zeta_offset=0.0,
        zeta_match_tol=0.4 * DZ, zeta_period=N_SLOTS * DZ,
        other_beam_q0=1.0, other_beam_beta0=BETA0,
        coherent=True,
        own_beam_sigma_x=sig_own_x, own_beam_sigma_y=sig_own_y,
        other_beam_sigma_x=SIGMA, other_beam_sigma_y=SIGMA,
        _context=test_context)

    x_probe = np.linspace(-3e-3, 3e-3, 7)
    p = xp.Particles(_context=test_context, p0c=P0C, q0=1,
                     mass0=xp.PROTON_MASS_EV,
                     x=x_probe, y=4e-4, zeta=0.0)
    p_ref = p.copy()
    bb.track(p)

    bb_ref = xf.BeamBeamBiGaussian2D(
        other_beam_q0=1.0, other_beam_beta0=BETA0,
        other_beam_num_particles=INTENSITY,
        other_beam_Sigma_11=SIGMA**2 + sig_own_x**2,
        other_beam_Sigma_33=SIGMA**2 + sig_own_y**2,
        other_beam_shift_x=2e-4, other_beam_shift_y=-1e-4,
        _context=test_context)
    bb_ref.track(p_ref)

    p.move(_context=None)
    p_ref.move(_context=None)
    assert np.allclose(p.px, p_ref.px, rtol=1e-13, atol=1e-30)
    assert np.allclose(p.py, p_ref.py, rtol=1e-13, atol=1e-30)

    # own sizes are required in the coherent mode
    try:
        xf.BeamBeamBiGaussianRigidBunch2D(
            other_particles=opp,
            other_beam_q0=1.0, other_beam_beta0=BETA0,
            coherent=True, _context=test_context)
        raise AssertionError(
            'coherent=True without own-beam covariance must raise')
    except ValueError:
        pass


@for_all_test_contexts
def test_multibunch_coherent_per_bunch_own_size(test_context):
    # coherent=True with PER-BUNCH covariance: own_beam_Sigma_* is indexed by
    # THIS beam (own_beam_zeta), other_beam_Sigma_* by the opposing beam; the
    # kernel matches the tracked particle to its own bunch AND to its opposing
    # partner independently. The own- and opposing-bunch INDEXING differs: 2 own
    # bunches at slots [0, 20] with offset +10 pair with opposing bunches at
    # slots 10 (index 0) and 30 (index 2) among opposing slots [10, 20, 30, 40].
    off = 10
    opp_slots = np.array([10, 20, 30, 40])
    opp = xp.Particles(_context=test_context, p0c=P0C, q0=1,
                       mass0=xp.PROTON_MASS_EV,
                       x=[2e-4, 1e-4, -1e-4, 3e-4], y=[-1e-4, 0.5e-4, 2e-4, 1e-4],
                       zeta=opp_slots * DZ, weight=INTENSITY)
    oth_sx = np.array([1.0, 1.2, 0.7, 0.9]) * SIGMA
    oth_sy = np.array([0.6, 1.1, 1.5, 0.8]) * SIGMA

    own_slots = np.array([0, 20])
    own_sx = np.array([0.8, 1.4]) * SIGMA
    own_sy = np.array([1.3, 1.2]) * SIGMA
    bb = xf.BeamBeamBiGaussianRigidBunch2D(
        other_particles=opp, zeta_offset=off * DZ,
        zeta_match_tol=0.4 * DZ, zeta_period=N_SLOTS * DZ,
        other_beam_q0=1.0, other_beam_beta0=BETA0, coherent=True,
        own_beam_zeta=own_slots * DZ,
        own_beam_sigma_x=own_sx, own_beam_sigma_y=own_sy,
        other_beam_sigma_x=oth_sx, other_beam_sigma_y=oth_sy,
        _context=test_context)
    assert bb.num_own_bunches == 2
    assert np.allclose(bb.own_beam_Sigma_11, own_sx**2, rtol=1e-15)
    assert np.allclose(bb.own_beam_Sigma_13, 0, rtol=0, atol=0)
    assert np.allclose(bb.own_beam_Sigma_33, own_sy**2, rtol=1e-15)

    # own bunch k (slot own_slots[k]) pairs with opposing bunch i_opp
    for k, slot in enumerate(own_slots):
        i_opp = int(np.where(opp_slots == slot + off)[0][0])
        p = xp.Particles(_context=test_context, p0c=P0C, q0=1,
                         mass0=xp.PROTON_MASS_EV, x=1e-3, y=4e-4, zeta=slot * DZ)
        p_ref = p.copy()
        bb.track(p)
        bb_ref = xf.BeamBeamBiGaussian2D(
            other_beam_q0=1.0, other_beam_beta0=BETA0,
            other_beam_num_particles=INTENSITY,
            other_beam_Sigma_11=own_sx[k]**2 + oth_sx[i_opp]**2,
            other_beam_Sigma_33=own_sy[k]**2 + oth_sy[i_opp]**2,
            other_beam_shift_x=float(opp.x[i_opp]),
            other_beam_shift_y=float(opp.y[i_opp]),
            _context=test_context)
        bb_ref.track(p_ref)
        p.move(_context=None)
        p_ref.move(_context=None)
        assert np.allclose(p.px, p_ref.px, rtol=1e-13, atol=1e-30)
        assert np.allclose(p.py, p_ref.py, rtol=1e-13, atol=1e-30)


@for_all_test_contexts
def test_multibunch_zeta_period(test_context):
    # Opposing bunches at slots 200..204 with distinct offsets so the matched
    # partner can be identified through the kick it produces.
    slots_opp = np.arange(200, 205)
    opp = xp.Particles(_context=test_context, p0c=P0C, q0=1,
                       mass0=xp.PROTON_MASS_EV,
                       x=(slots_opp - 199) * 1e-4, y=np.zeros(5),
                       zeta=slots_opp * DZ, weight=INTENSITY)
    period = N_SLOTS * DZ

    def kicked(bb, slot):
        return max(abs(v) for v in _kick(test_context, bb, slot)) > 0

    # 1) plain pairing, no wrap: probe at slot 200 with offset +2 -> partner 202
    bb = _make_element(test_context, opp, 2, period)
    assert kicked(bb, 200)
    # probe whose partner slot (102) is empty -> no kick
    assert not kicked(bb, 100)

    # 2) left-LR-style offset stored mod N_SLOTS (i.e. -2 stored as N-2):
    #    probe at 202 pairs with 200 only through the periodic wrap
    bb = _make_element(test_context, opp, N_SLOTS - 2, period)
    assert kicked(bb, 202)
    assert not kicked(bb, 200)  # 200-2=198 not populated

    # 3) same but with the periodicity disabled -> no match (legacy behaviour)
    bb = _make_element(test_context, opp, N_SLOTS - 2, 0.0)
    assert not kicked(bb, 202)

    # 4) large IP2-style offset wrapping around the ring: probe at slot 2875
    #    with offset 891 -> 3766 = 202 (mod 3564). The kick must be identical
    #    to the direct pairing with bunch 202.
    bb_wrap = _make_element(test_context, opp, 891, period)
    kick_wrap = _kick(test_context, bb_wrap, 2875)
    bb_direct = _make_element(test_context, opp, 2, period)
    kick_direct = _kick(test_context, bb_direct, 200)
    assert max(abs(v) for v in kick_wrap) > 0
    assert np.allclose(kick_wrap, kick_direct, rtol=1e-14, atol=0)

    # 5) unsorted opposing bunches WITH per-bunch covariance:
    #    update_from_other_beam must sort bunches AND covariance in zeta
    #    (the kernel partner search is a binary search); pairing and covariance
    #    must be unchanged.
    sig_arr = SIGMA * (1.0 + 0.1 * (slots_opp - 200))
    bb_sig = _make_element(test_context, opp, 2, period,
                           sigma_x=sig_arr, sigma_y=sig_arr[::-1])
    kick_sig = _kick(test_context, bb_sig, 200)
    shuffle = np.array([3, 0, 4, 1, 2])
    opp_shuffled = xp.Particles(_context=test_context, p0c=P0C, q0=1,
                                mass0=xp.PROTON_MASS_EV,
                                x=((slots_opp - 199) * 1e-4)[shuffle],
                                y=np.zeros(5),
                                zeta=(slots_opp * DZ)[shuffle],
                                weight=INTENSITY)
    bb_shuffled = _make_element(test_context, opp_shuffled, 2, period,
                                sigma_x=sig_arr[shuffle],
                                sigma_y=sig_arr[::-1][shuffle])
    kick_shuffled = _kick(test_context, bb_shuffled, 200)
    assert max(abs(v) for v in kick_sig) > 0
    assert np.allclose(kick_shuffled, kick_sig, rtol=1e-14, atol=0)
    # stored covariance follows the zeta ordering
    assert np.allclose(
        bb_shuffled.other_beam_Sigma_11, sig_arr**2, rtol=1e-15)
    assert np.allclose(bb_shuffled.other_beam_Sigma_13, 0, rtol=0, atol=0)
    assert np.allclose(
        bb_shuffled.other_beam_Sigma_33, sig_arr[::-1]**2, rtol=1e-15)


@for_all_test_contexts
def test_multibunch_heterogeneous_bunches_match_bb2d(test_context):
    # Check that a multibunch element selects the centroid, population and
    # elliptical sizes of each matched opposing bunch and gives the same kick
    # as the corresponding scalar BB2D. Also check that a tracked bunch paired
    # with an empty opposing slot receives no kick, and cover partial strength
    # scaling.
    slots = np.array([3, 8, 14])
    centroids_x = np.array([0.2e-3, -0.5e-3, 0.9e-3])
    centroids_y = np.array([-0.7e-3, 0.4e-3, 0.1e-3])
    populations = np.array([0.8e11, 1.7e11, 2.6e11])
    sigma_x = np.array([0.7, 1.2, 1.8]) * SIGMA
    sigma_y = np.array([1.6, 0.9, 1.3]) * SIGMA
    opposing = xp.Particles(
        _context=test_context, p0c=P0C, q0=1,
        mass0=xp.PROTON_MASS_EV,
        x=centroids_x, y=centroids_y, zeta=slots * DZ,
        weight=populations)
    bb = xf.BeamBeamBiGaussianRigidBunch2D(
        other_particles=opposing,
        zeta_match_tol=0.4 * DZ,
        zeta_period=N_SLOTS * DZ,
        other_beam_q0=1,
        other_beam_beta0=BETA0,
        other_beam_sigma_x=sigma_x,
        other_beam_sigma_y=sigma_y,
        min_sigma_diff=1e-12,
        _context=test_context)

    probe_slots = np.append(slots, 20)
    particles = xp.Particles(
        _context=test_context, p0c=P0C, q0=1,
        mass0=xp.PROTON_MASS_EV,
        x=np.array([1.1e-3, -0.2e-3, 0.6e-3, 1.4e-3]),
        y=np.array([0.3e-3, 1.0e-3, -0.8e-3, 0.5e-3]),
        zeta=probe_slots * DZ)
    particles_initial = particles.copy()
    bb.track(particles)

    expected_px = np.zeros(len(probe_slots))
    expected_py = np.zeros(len(probe_slots))
    for ii in range(len(slots)):
        particle_ref = xp.Particles(
            _context=test_context, p0c=P0C, q0=1,
            mass0=xp.PROTON_MASS_EV,
            x=float(test_context.nparray_from_context_array(
                particles_initial.x)[ii]),
            y=float(test_context.nparray_from_context_array(
                particles_initial.y)[ii]))
        reference = xf.BeamBeamBiGaussian2D(
            other_beam_q0=1,
            other_beam_beta0=BETA0,
            other_beam_num_particles=populations[ii],
            other_beam_Sigma_11=sigma_x[ii]**2,
            other_beam_Sigma_33=sigma_y[ii]**2,
            other_beam_shift_x=centroids_x[ii],
            other_beam_shift_y=centroids_y[ii],
            min_sigma_diff=1e-12,
            _context=test_context)
        reference.track(particle_ref)
        expected_px[ii] = test_context.nparray_from_context_array(
            particle_ref.px)[0]
        expected_py[ii] = test_context.nparray_from_context_array(
            particle_ref.py)[0]

    ctx2np = test_context.nparray_from_context_array
    xo.assert_allclose(
        ctx2np(particles.px), expected_px, rtol=2e-13, atol=1e-30)
    xo.assert_allclose(
        ctx2np(particles.py), expected_py, rtol=2e-13, atol=1e-30)

    restored = xf.BeamBeamBiGaussianRigidBunch2D.from_dict(
        bb.to_dict(), _context=test_context)
    particles_restored = particles_initial.copy()
    restored.track(particles_restored)
    xo.assert_allclose(
        ctx2np(particles_restored.px), ctx2np(particles.px),
        rtol=0, atol=0)
    xo.assert_allclose(
        ctx2np(particles_restored.py), ctx2np(particles.py),
        rtol=0, atol=0)

    bb_scaled = bb.copy(_context=test_context)
    bb_scaled.scale_strength = 0.37
    particles_scaled = particles_initial.copy()
    bb_scaled.track(particles_scaled)
    xo.assert_allclose(
        ctx2np(particles_scaled.px), 0.37 * ctx2np(particles.px),
        rtol=2e-13, atol=1e-30)
    xo.assert_allclose(
        ctx2np(particles_scaled.py), 0.37 * ctx2np(particles.py),
        rtol=2e-13, atol=1e-30)

    # The three active representative particles determine arrays of length
    # three. A different bunch count is a filling change and requires the
    # caller to construct a newly sized element.
    assert len(bb.other_beam_zeta) == len(slots)
    opposing_with_different_filling = xp.Particles(
        _context=test_context, p0c=P0C, q0=1,
        mass0=xp.PROTON_MASS_EV,
        x=centroids_x[:2], y=centroids_y[:2], zeta=slots[:2] * DZ,
        weight=populations[:2])
    with pytest.raises(ValueError, match='reconfigure the element'):
        bb.update_from_other_beam(opposing_with_different_filling)


def test_rigid_bunch_covariance_api_and_coupling_warning():
    opposing = xp.Particles(
        p0c=P0C, q0=1, mass0=xp.PROTON_MASS_EV,
        zeta=[0, DZ], weight=INTENSITY)
    own_sigma_x = np.array([0.8, 1.2]) * SIGMA
    own_sigma_y = np.array([1.1, 1.4]) * SIGMA
    other_sigma_x = np.array([0.9, 1.3]) * SIGMA
    other_sigma_y = np.array([1.5, 0.7]) * SIGMA

    from_sigmas = xf.BeamBeamBiGaussianRigidBunch2D(
        other_particles=opposing,
        other_beam_q0=1,
        other_beam_beta0=BETA0,
        coherent=True,
        own_beam_zeta=[0, DZ],
        own_beam_sigma_x=own_sigma_x,
        own_beam_sigma_y=own_sigma_y,
        other_beam_sigma_x=other_sigma_x,
        other_beam_sigma_y=other_sigma_y)
    from_covariance = xf.BeamBeamBiGaussianRigidBunch2D(
        other_particles=opposing,
        other_beam_q0=1,
        other_beam_beta0=BETA0,
        coherent=True,
        own_beam_zeta=[0, DZ],
        own_beam_Sigma_11=own_sigma_x**2,
        own_beam_Sigma_13=0,
        own_beam_Sigma_33=own_sigma_y**2,
        other_beam_Sigma_11=other_sigma_x**2,
        other_beam_Sigma_13=0,
        other_beam_Sigma_33=other_sigma_y**2)

    for field in (
            'own_beam_Sigma_11', 'own_beam_Sigma_13',
            'own_beam_Sigma_33', 'other_beam_Sigma_11',
            'other_beam_Sigma_13', 'other_beam_Sigma_33'):
        xo.assert_allclose(
            getattr(from_sigmas, field), getattr(from_covariance, field),
            rtol=0, atol=0)

    probe_from_sigmas = xp.Particles(
        p0c=P0C, q0=1, mass0=xp.PROTON_MASS_EV,
        x=[0.7e-3, -0.4e-3], y=[0.2e-3, 0.5e-3], zeta=[0, DZ])
    probe_from_covariance = probe_from_sigmas.copy()
    from_sigmas.track(probe_from_sigmas)
    from_covariance.track(probe_from_covariance)
    xo.assert_allclose(
        probe_from_sigmas.px, probe_from_covariance.px, rtol=0, atol=0)
    xo.assert_allclose(
        probe_from_sigmas.py, probe_from_covariance.py, rtol=0, atol=0)

    serialized = from_covariance.to_dict()
    for field in (
            'own_beam_Sigma_11', 'own_beam_Sigma_13',
            'own_beam_Sigma_33', 'other_beam_Sigma_11',
            'other_beam_Sigma_13', 'other_beam_Sigma_33'):
        assert field in serialized
    assert 'sigma_x' not in serialized
    assert 'other_beam_sigma_x' not in serialized

    with pytest.raises(ValueError, match='not both'):
        xf.BeamBeamBiGaussianRigidBunch2D(
            other_particles=opposing,
            other_beam_Sigma_11=SIGMA**2,
            other_beam_Sigma_33=SIGMA**2,
            other_beam_sigma_x=SIGMA,
            other_beam_sigma_y=SIGMA)
    with pytest.raises(ValueError, match='must be provided together'):
        xf.BeamBeamBiGaussianRigidBunch2D(
            other_particles=opposing,
            other_beam_sigma_x=SIGMA)
    large_correlation = 2e-2
    with pytest.warns(RuntimeWarning, match='has not yet been validated'):
        coupled = xf.BeamBeamBiGaussianRigidBunch2D(
            other_particles=opposing,
            other_beam_q0=1,
            other_beam_beta0=BETA0,
            other_beam_Sigma_11=SIGMA**2,
            other_beam_Sigma_13=large_correlation * SIGMA**2,
            other_beam_Sigma_33=SIGMA**2)
    xo.assert_allclose(
        coupled.other_beam_Sigma_13,
        large_correlation * SIGMA**2, rtol=0, atol=0)
    uncoupled = coupled.copy()
    uncoupled.other_beam_Sigma_13[:] = 0
    coupled_probe = xp.Particles(
        p0c=P0C, q0=1, mass0=xp.PROTON_MASS_EV,
        x=0.7 * SIGMA, y=0.4 * SIGMA, zeta=0)
    uncoupled_probe = coupled_probe.copy()
    coupled.track(coupled_probe)
    uncoupled.track(uncoupled_probe)
    xo.assert_allclose(coupled_probe.px, uncoupled_probe.px, rtol=0, atol=0)
    xo.assert_allclose(coupled_probe.py, uncoupled_probe.py, rtol=0, atol=0)

    with pytest.warns(RuntimeWarning, match='has not yet been validated'):
        from_covariance.update_from_own_beam(
            own_beam_Sigma_11=own_sigma_x**2,
            own_beam_Sigma_13=(
                large_correlation * own_sigma_x * own_sigma_y),
            own_beam_Sigma_33=own_sigma_y**2)
    with pytest.warns(RuntimeWarning, match='has not yet been validated'):
        from_covariance.update_from_other_beam(
            opposing,
            other_beam_Sigma_11=other_sigma_x**2,
            other_beam_Sigma_13=(
                large_correlation * other_sigma_x * other_sigma_y),
            other_beam_Sigma_33=other_sigma_y**2)
