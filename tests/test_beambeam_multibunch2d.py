# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np

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
    return xf.BeamBeamBiGaussianMultibunch2D(
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
    bb = xf.BeamBeamBiGaussianMultibunch2D(
        other_particles=opp, zeta_offset=0.0,
        zeta_match_tol=0.4 * DZ, zeta_period=N_SLOTS * DZ,
        other_beam_q0=1.0, other_beam_beta0=BETA0,
        coherent=True, sigma_x=sig_own_x, sigma_y=sig_own_y,
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
        xf.BeamBeamBiGaussianMultibunch2D(
            num_bunches=1, other_beam_q0=1.0, other_beam_beta0=BETA0,
            coherent=True, _context=test_context)
        raise AssertionError('coherent=True without sigma_x/y must raise')
    except ValueError:
        pass


@for_all_test_contexts
def test_multibunch_coherent_per_bunch_own_size(test_context):
    # coherent=True with PER-BUNCH own sizes: sigma_x/sigma_y are indexed by THIS
    # beam (own_beam_zeta), other_beam_sigma_x/y by the opposing beam; the kernel
    # matches the tracked particle to its own bunch AND to its opposing partner
    # independently. The own- and opposing-bunch INDEXING differs here: 2 own
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
    bb = xf.BeamBeamBiGaussianMultibunch2D(
        other_particles=opp, zeta_offset=off * DZ,
        zeta_match_tol=0.4 * DZ, zeta_period=N_SLOTS * DZ,
        other_beam_q0=1.0, other_beam_beta0=BETA0, coherent=True,
        own_beam_zeta=own_slots * DZ, sigma_x=own_sx, sigma_y=own_sy,
        other_beam_sigma_x=oth_sx, other_beam_sigma_y=oth_sy,
        _context=test_context)
    assert bb.num_own_bunches == 2
    assert np.allclose(bb.sigma_x, own_sx, rtol=1e-15)
    assert np.allclose(bb.sigma_y, own_sy, rtol=1e-15)

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

    # 5) unsorted opposing bunches WITH per-bunch sizes:
    #    update_from_other_beam must sort bunches AND their sigmas in zeta
    #    (the kernel partner search is a binary search); pairing and sizes
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
    # stored sizes follow the zeta ordering
    assert np.allclose(bb_shuffled.other_beam_sigma_x, sig_arr, rtol=1e-15)
    assert np.allclose(bb_shuffled.other_beam_sigma_y, sig_arr[::-1], rtol=1e-15)
