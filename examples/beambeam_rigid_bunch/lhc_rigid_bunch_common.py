# copyright ############################### #
# This file is part of the Xfields Package. #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
LHC-specific glue for the rigid-bunch beam-beam examples.

The rigid-bunch beam-beam machinery itself is machine-independent and lives in
Xfields. The standard beam-beam install/configure workflow with
``mode='rigid_bunch'`` returns a ``BeamBeamRigidBunchStudy``; all
further operations are methods on it (``rigid_bunch_study.twiss()``,
``rigid_bunch_study.solve()``,
``rigid_bunch_study.get_study_with_second_order_maps()``,
``rigid_bunch_study.load_solution(...)``, and
``rigid_bunch_study.apply_filling_pattern(...)``). The examples use these
methods when changing an already configured rigid-bunch study; this module only
holds the LHC-specific bits the generic tools cannot know about:

* :func:`load_lhc` and the ``SCENARIOS`` presets (sequence, optics files, beam
  parameters for injection / collision);
* the filling-scheme helpers (:func:`load_scheme`, :func:`all_filled_slots`,
  :func:`filling_pattern_from_slots`, :func:`windowed_slots`
  -- which reuses the ``rigid_bunch_study.ip_offsets`` derived from geometry);
* the DataFrame / plotting utilities (:func:`results_dataframe`,
  :func:`plot_results`, :func:`plot_global_quantities`).

Model (following pytrain / TRAIN): head-on and long-range 2D beam-beam elements
(``xfields.BeamBeamBiGaussianRigidBunch2D``) at IP1/2/5/8; the per-IP head-on
bunch-pairing offsets are derived from the ring geometry
(``round(2 * (s_ip - s_ip1) / bunch_spacing_zeta)`` -> 0 at IP1/IP5, ~891 at
IP2, ~2670 at IP8); coherent (rigid-bunch) convolved-size kicks; beam
separation = live
closed-orbit difference + geometric survey separation of the two rings.
"""

import os
import json
import numpy as np

import xobjects as xo
import xtrack as xt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'test_data', 'lhc_2024')
FILLING_FILE = os.path.join(DATA, 'filling_25ns_2460b.json')

# LHC RF / slot layout: 3564 25-ns slots (h = 35640, 10 buckets per slot)
N_SLOTS = 3564
HARMONIC_NUMBER = 35640
BUNCH_SPACING_BUCKETS = 10

SCENARIOS = {
    # 450 GeV injection: nominal injection optics, separation bumps ON (so the
    # beams do NOT collide head-on -- long-range only), 1.8e11 p/b, 1.5 um.
    'injection': dict(p0c=450e9, bunch_intensity=1.8e11, nemitt=1.5e-6,
                      optics='injection_optics.madx'),
    # 6.8 TeV collision: fully squeezed R2025aRP 15 cm flat optics with
    # end-of-levelling knobs (flattened MAD-X state in
    # collision_optics_15cm_flat_2026.madx), 1.1e11 p/b, 2.3 um, head-on at
    # IP1/5 + BBLR.
    'collision': dict(p0c=6800e9, bunch_intensity=1.1e11, nemitt=2.3e-6,
                      optics='collision_optics_15cm_flat_2026.madx'),
}


def wrap_frac_tune(v):
    """Tune difference on the fractional-tune circle, wrapped to (-0.5, 0.5]
    (fast-mode twiss returns fractional tunes while the no-BB reference may
    carry an integer part)."""
    return (np.asarray(v) + 0.5) % 1.0 - 0.5


def print_solve_status(result):
    """Print the convergence metadata of a rigid-bunch solve result."""
    print(f'  converged={result.converged}, '
          f'iterations={result.num_iterations}, '
          f'max orbit change={result.max_orbit_change:.3e} sigma')


# ----------------------------------------------------------------------------
# Environment-variable defaults shared by the examples
# ----------------------------------------------------------------------------
def default_context():
    """CPU context from ``LHC_OMP`` (unset/``0``/``serial`` -> serial; a thread
    count or ``auto`` -> OpenMP kernels). Prebuilt kernels exist for both."""
    omp = os.environ.get('LHC_OMP', '0')
    if omp in ('0', '', 'serial'):
        return xo.ContextCpu()
    return xo.ContextCpu(omp_num_threads=('auto' if omp == 'auto' else int(omp)))


def default_ips():
    """IP element names from ``LHC_IPS`` (default ``1,2,5,8`` -> ip1/2/5/8)."""
    return [f'ip{v.strip()}' for v in
            os.environ.get('LHC_IPS', '1,2,5,8').split(',')]


def default_nparasitic():
    """Long-range encounters per IP side from ``LHC_NPAR`` (default 45)."""
    return int(os.environ.get('LHC_NPAR', '45'))


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_lhc(scenario=None, *, p0c=None, bunch_intensity=None, nemitt=None,
             optics_file=None, ips=None, nparasitic=None, context=None):
    """Load both LHC beams (sequence + optics) for a scenario.

    Pass a preset name (``'injection'`` / ``'collision'``) or the explicit beam
    parameters (``p0c``, ``bunch_intensity``, ``nemitt``, ``optics_file``) for a
    custom configuration. ``ips`` / ``nparasitic`` / ``context`` default from
    the ``LHC_IPS`` / ``LHC_NPAR`` / ``LHC_OMP`` environment variables.

    Returns ``(env, line_b1, line_b2, par)``; ``line_b2`` is the reversed line
    and ``par`` collects the beam parameters (``p0c``, ``bunch_intensity``,
    ``nemitt``, ``gamma0``, ``ips``, ``nparasitic``, ``context``) for the install
    / solve calls.
    """
    if scenario is not None:
        sc = SCENARIOS[scenario]
        p0c, bunch_intensity, nemitt = (sc['p0c'], sc['bunch_intensity'],
                                        sc['nemitt'])
        optics_file = os.path.join(DATA, sc['optics'])
    if context is None:
        context = default_context()
    if ips is None:
        ips = default_ips()
    if nparasitic is None:
        nparasitic = default_nparasitic()

    env = xt.load(os.path.join(DATA, 'lhc.seq'), format='madx',
                  reverse_lines=['lhcb2'])
    for ln in (env.lhcb1, env.lhcb2):
        ln.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=p0c)
    env.vars.load(optics_file)
    for ln in (env.lhcb1, env.lhcb2):
        ln.twiss_default['method'] = '4d'
        ln.cycle(name_first_element='ip3', inplace=True)  # no IP at s=0
        ln.build_tracker(_context=context)
    par = dict(p0c=p0c, bunch_intensity=bunch_intensity, nemitt=nemitt,
               gamma0=float(env.lhcb1.particle_ref.gamma0[0]),
               ips=list(ips), nparasitic=nparasitic, context=context)
    return env, env.lhcb1, env.lhcb2, par


# ----------------------------------------------------------------------------
# Filling pattern
# ----------------------------------------------------------------------------
def load_scheme():
    with open(FILLING_FILE) as fid:
        filling = json.load(fid)
    if filling['num_slots'] != N_SLOTS:
        raise ValueError(f'The filling must describe {N_SLOTS} slots.')
    return (
        filling_pattern_from_slots(filling['filled_slots_cw']),
        filling_pattern_from_slots(filling['filled_slots_acw']),
    )


def all_filled_slots(scheme_b1, scheme_b2):
    return (sorted(np.where(scheme_b1 > 0)[0].tolist()),
            sorted(np.where(scheme_b2 > 0)[0].tolist()))


def filling_pattern_from_slots(slots, n_slots=N_SLOTS):
    """Slot-indexed occupancy pattern populated at physical ``slots``."""
    filling_pattern = np.zeros(n_slots, dtype=int)
    filling_pattern[np.asarray(slots, dtype=int)] = 1
    return filling_pattern


def windowed_slots(ho_offsets, scheme_b1, scheme_b2, window, n_slots=N_SLOTS):
    """A bounded subset of the filling: a reference window (the longest
    contiguous filled run of beam 1) plus the windows it collides with at every
    distinct head-on offset (so all IPs get realistic PACMAN pairings).
    ``ho_offsets`` is the ``{ip: offset}`` mapping the tools derive from the
    ring geometry, i.e. ``rigid_bunch_study.ip_offsets`` after rigid-bunch
    configuration.
    """
    offsets = sorted(set(ho_offsets.values()))
    filled = scheme_b1 > 0
    best_len = best_start = cur_len = cur_start = 0
    for s in range(n_slots):
        if filled[s]:
            cur_start = s if cur_len == 0 else cur_start
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    window = min(window, best_len)
    ref_start = best_start
    cand = set()
    for o in offsets:
        for shift in (o, -o):
            for k in range(window):
                cand.add((ref_start + shift + k) % n_slots)
    return (sorted(s for s in cand if scheme_b1[s]),
            sorted(s for s in cand if scheme_b2[s]))


# ----------------------------------------------------------------------------
# Measured per-bunch emittances
# ----------------------------------------------------------------------------
def set_per_bunch_sizes(rigid_bunch_study, nemitt_cw, nemitt_acw):
    """Replace the uniform design beam sizes of the beam-beam elements by
    PER-BUNCH ones, from measured per-bunch normalised emittances.

    ``nemitt_cw`` / ``nemitt_acw`` are ``(nemitt_x, nemitt_y)`` pairs of
    arrays aligned with ``rigid_bunch_study.filled_slots_cw`` /
    ``rigid_bunch_study.filled_slots_acw``. The sizes follow the tools' own
    convention,
    ``sigma = sqrt(beta nemitt / gamma)`` with the beta functions computed
    without beam-beam and cached internally by the study;
    only the single design emittance is replaced by the per-bunch one, so the
    kick between bunch ``i`` and bunch ``j`` uses the convolved size
    ``sqrt(eps_i beta_1 / gamma + eps_j beta_2 / gamma)``, as in pytrain.

    ``BeamBeamRigidBunchStudy`` carries one design emittance per plane, so this has
    to reach into the elements (which do support per-bunch sizes: the own
    sizes are indexed by this beam, the opposing ones by the other beam, and
    the kernel convolves the matched pair). It must be called AFTER the
    install / any ``apply_filling_pattern`` / any geometry recomputation (all
    of which
    re-register the uniform design sizes) and holds through ``solve`` /
    ``load_solution`` as long as ``dynamic_beta`` is False -- those keep the
    stored sizes.
    """
    line = {'cw': rigid_bunch_study.cw_line,
            'acw': rigid_bunch_study.acw_line}
    gamma = {beam: float(line[beam].particle_ref.gamma0[0])
             for beam in ('cw', 'acw')}
    emit = {'cw': nemitt_cw, 'acw': nemitt_acw}
    n_bunches = {
        'cw': len(rigid_bunch_study.filled_slots_cw),
        'acw': len(rigid_bunch_study.filled_slots_acw),
    }
    for base in rigid_bunch_study.enc_names:
        config = rigid_bunch_study._encounter_config[base]
        sigma = {}
        for beam in ('cw', 'acw'):
            sigma[beam] = (
                np.sqrt(config[f'betx_no_bb_{beam}']
                        * emit[beam][0] / gamma[beam]),
                np.sqrt(config[f'bety_no_bb_{beam}']
                        * emit[beam][1] / gamma[beam]))
        for beam, other_beam in (('cw', 'acw'), ('acw', 'cw')):
            # The raw element (and hence `rigid_bunch_study.bb_cw[...]`) is
            # an expression view, whose array slices are not assignable
            bb_name = rigid_bunch_study.bb_name(base, beam=beam)
            bb = line[beam].element_dict[bb_name]
            own, other = sigma[beam], sigma[other_beam]
            n_other = n_bunches[other_beam]
            bb.update_from_own_beam(
                zeta=rigid_bunch_study.bunch_zeta(beam=beam),
                own_beam_sigma_x=own[0], own_beam_sigma_y=own[1])
            other_order = np.argsort(
                rigid_bunch_study.bunch_zeta(beam=other_beam), kind='stable')
            bb.other_beam_Sigma_11[:n_other] = other[0][other_order]**2
            bb.other_beam_Sigma_13[:n_other] = 0
            bb.other_beam_Sigma_33[:n_other] = other[1][other_order]**2


# ----------------------------------------------------------------------------
# Results as a DataFrame
# ----------------------------------------------------------------------------
def results_dataframe(rigid_bunch_study, mbtw, slots, qx_no_bb, qy_no_bb, beam,
                      ip='ip1'):
    """Per-bunch results as a pandas DataFrame, indexed by 25 ns slot.

    Columns: qx, qy (per-bunch tunes), dqx, dqy (beam-beam tune shift relative
    to the tune without beam-beam), x, y (closed orbit at the head-on marker of
    ``ip`` in the physical frame selected by ``beam``), and dx, dy (per-bunch
    orbit deviations from the beam average).
    """
    import pandas as pd
    marker = rigid_bunch_study.bb_name(f'bb_{ip}_ho', beam=beam)
    twiss_at_marker = mbtw.at_element(marker)
    x = twiss_at_marker.x * (-1.0 if beam == 'acw' else 1.0)
    y = twiss_at_marker.y

    df = pd.DataFrame({
        'slot': np.asarray(slots),
        'qx': mbtw.qx, 'qy': mbtw.qy,
        'dqx': wrap_frac_tune(mbtw.qx - qx_no_bb),
        'dqy': wrap_frac_tune(mbtw.qy - qy_no_bb),
        'x': x, 'y': y,
        'dx': x - x.mean(), 'dy': y - y.mean(),
    }).set_index('slot')
    return df


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def plot_results(rigid_bunch_study, slots_cw, mbtw_cw, qx_no_bb, qy_no_bb,
                 title_suffix=''):
    import matplotlib.pyplot as plt
    mk = rigid_bunch_study.bb_name('bb_ip1_ho', beam='cw')
    twiss_at_marker = mbtw_cw.at_element(mk)
    co_x = twiss_at_marker.x
    co_y = twiss_at_marker.y
    # per-bunch orbit deviation from the bunch-averaged orbit (removes the common
    # crossing/separation-bump orbit, leaving the bunch-by-bunch beam-beam part)
    dco_x = (co_x - co_x.mean()) * 1e6
    dco_y = (co_y - co_y.mean()) * 1e6
    fig, axs = plt.subplots(2, 1, figsize=(9, 7))
    axs[0].plot(slots_cw, wrap_frac_tune(mbtw_cw.qx - qx_no_bb) * 1e3, '.',
                label=r'$\Delta q_x$')
    axs[0].plot(slots_cw, wrap_frac_tune(mbtw_cw.qy - qy_no_bb) * 1e3, '.',
                label=r'$\Delta q_y$')
    axs[0].set_xlabel('25 ns slot')
    axs[0].set_ylabel(r'beam-beam tune shift [$10^{-3}$]')
    axs[0].set_title('LHC: per-bunch beam-beam tune shift (CW)'
                     + title_suffix)
    axs[0].legend()
    axs[1].plot(slots_cw, dco_x, '.', label='x')
    axs[1].plot(slots_cw, dco_y, '.', label='y')
    axs[1].set_xlabel('25 ns slot')
    axs[1].set_ylabel('orbit dev. from mean at IP1 [$\\mu$m]')
    axs[1].set_title('Per-bunch beam-beam closed-orbit deviation at IP1 (CW)')
    axs[1].legend()
    plt.tight_layout()
    return fig


def plot_global_quantities(rigid_bunch_study, slots_cw, mbtw_cw,
                           slots_acw, mbtw_acw):
    """Bunch-by-bunch orbit at IP1, beta* at IP1, tunes, chromaticity and
    coupling |C-| of both beams, from mode='fast' MultiBunchTwiss results
    (which carry per-bunch optics and global quantities)."""
    import matplotlib.pyplot as plt
    mk = {beam: rigid_bunch_study.bb_name('bb_ip1_ho', beam=beam)
          for beam in ('cw', 'acw')}
    twiss_at_ip1 = {
        'cw': mbtw_cw.at_element(mk['cw']),
        'acw': mbtw_acw.at_element(mk['acw']),
    }

    def at_ip1(col, beam):
        return twiss_at_ip1[beam][col]

    fig, axs = plt.subplots(3, 2, figsize=(13, 10), sharex=True)

    ax = axs[0, 0]   # orbit deviation at IP1 (physical frame for both beams)
    for slots, beam, lab in [(slots_cw, 'cw', 'CW'),
                             (slots_acw, 'acw', 'ACW')]:
        sgn = -1.0 if beam == 'acw' else 1.0
        x = sgn * at_ip1('x', beam)
        y = at_ip1('y', beam)
        ax.plot(slots, (x - x.mean()) * 1e6, '.', ms=3, label=f'{lab} x')
        ax.plot(slots, (y - y.mean()) * 1e6, '.', ms=3, label=f'{lab} y')
    ax.set_ylabel(r'orbit dev. at IP1 [$\mu$m]')
    ax.set_title('Per-bunch closed-orbit deviation at IP1')
    ax.legend(ncol=2, fontsize=8)

    ax = axs[0, 1]   # beta* at IP1
    for slots, beam, lab in [(slots_cw, 'cw', 'CW'),
                             (slots_acw, 'acw', 'ACW')]:
        ax.plot(slots, at_ip1('betx', beam), '.', ms=3,
                label=fr'{lab} $\beta_x^*$')
        ax.plot(slots, at_ip1('bety', beam), '.', ms=3,
                label=fr'{lab} $\beta_y^*$')
    ax.set_ylabel(r'$\beta^*$ at IP1 [m]')
    ax.set_title('Per-bunch $\\beta^*$ at IP1 (dynamic beta)')
    ax.legend(ncol=2, fontsize=8)

    ax = axs[1, 0]   # fractional tunes
    for slots, mbtw, lab in [(slots_cw, mbtw_cw, 'CW'),
                             (slots_acw, mbtw_acw, 'ACW')]:
        ax.plot(slots, mbtw.qx_frac, '.', ms=3, label=f'{lab} $q_x$')
        ax.plot(slots, mbtw.qy_frac, '.', ms=3, label=f'{lab} $q_y$')
    ax.set_ylabel('fractional tune')
    ax.set_title('Per-bunch tunes')
    ax.legend(ncol=2, fontsize=8)

    ax = axs[1, 1]   # chromaticity
    for slots, mbtw, lab in [(slots_cw, mbtw_cw, 'CW'),
                             (slots_acw, mbtw_acw, 'ACW')]:
        ax.plot(slots, mbtw.dqx, '.', ms=3, label=f"{lab} $q'_x$")
        ax.plot(slots, mbtw.dqy, '.', ms=3, label=f"{lab} $q'_y$")
    ax.set_ylabel("chromaticity $q'$")
    ax.set_title('Per-bunch chromaticity')
    ax.legend(ncol=2, fontsize=8)

    ax = axs[2, 0]   # coupling
    for slots, mbtw, lab in [(slots_cw, mbtw_cw, 'CW'),
                             (slots_acw, mbtw_acw, 'ACW')]:
        ax.plot(slots, mbtw.c_minus, '.', ms=3, label=lab)
    ax.set_xlabel('25 ns slot')
    ax.set_ylabel('$|C^-|$')
    ax.set_title('Per-bunch coupling (closest tune approach)')
    ax.legend(fontsize=8)

    axs[2, 1].axis('off')
    axs[1, 1].set_xlabel('25 ns slot')
    axs[1, 1].tick_params(labelbottom=True)
    plt.suptitle('Per-bunch optics & global quantities '
                 '(mode="fast" rigid-bunch Twiss)')
    plt.tight_layout()
    return fig
