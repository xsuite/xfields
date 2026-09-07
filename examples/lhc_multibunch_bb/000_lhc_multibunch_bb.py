# copyright ############################### #
# This file is part of the Xfields Package. #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""Rigid-bunch beam-beam on the full, thick LHC lattice.

This example shows the complete Xsuite workflow explicitly: load and prepare
the two LHC lines, install the beam-beam elements, configure the rigid-bunch
study, apply a filling, and solve for the self-consistent per-bunch closed
orbit and tunes.

The default run uses a bounded subset of the real LHC filling because every
iteration twisses the full thick lattice once per beam. Set ``LHC_ALL=1`` to
use the complete filling, or change ``LHC_WINDOW`` (default 48) to control the
bounded subset. The lattice itself is never reduced; see
``002_multibunch_sectormaps_collisions.py`` for the faster second-order-map
workflow.
"""

import json
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xtrack as xt


# LHC collision configuration ------------------------------------------------
THIS_DIR = Path(__file__).parent
LHC_DATA = THIS_DIR / '..' / '..' / 'test_data' / 'lhc_2024'
FILLING_FILE = LHC_DATA / '25ns_2460b_2448_2092_2239_144bpi_20inj.json'
OPTICS_FILE = LHC_DATA / 'collision_optics_15cm_flat_2026.madx'

P0C = 6.8e12
BUNCH_INTENSITY = 1.1e11
NEMITT_X = 2.3e-6
NEMITT_Y = 2.3e-6
HARMONIC_NUMBER = 35640
BUNCH_SPACING_BUCKETS = 10
N_SLOTS = HARMONIC_NUMBER // BUNCH_SPACING_BUCKETS

IP_NAMES = [f'ip{ip.strip()}'
            for ip in os.environ.get('LHC_IPS', '1,2,5,8').split(',')]
N_LONG_RANGE = int(os.environ.get('LHC_NPAR', '45'))
N_ITERATIONS = int(os.environ.get('LHC_NITER', '3'))
ALL_BUNCHES = os.environ.get('LHC_ALL', '0') == '1'
WINDOW = int(os.environ.get('LHC_WINDOW', '48'))


def select_bounded_filling(ip_offsets, filling_cw, filling_acw, window):
    """Select a train window and all regions paired with it at the IPs.

    This is only a runtime policy for this full-lattice example. It preserves
    representative PACMAN pairings without making filling selection part of
    the rigid-bunch beam-beam API.
    """
    filled_cw = np.asarray(filling_cw) > 0

    # Find the longest contiguous train in the clockwise beam.
    best_length = best_start = current_length = current_start = 0
    for slot, is_filled in enumerate(filled_cw):
        if is_filled:
            if current_length == 0:
                current_start = slot
            current_length += 1
            if current_length > best_length:
                best_length = current_length
                best_start = current_start
        else:
            current_length = 0

    window = min(window, best_length)
    candidate_slots = set()
    for offset in set(ip_offsets.values()):
        for signed_offset in (offset, -offset):
            candidate_slots.update(
                (best_start + signed_offset + ii) % N_SLOTS
                for ii in range(window))

    slots_cw = sorted(slot for slot in candidate_slots if filling_cw[slot])
    slots_acw = sorted(slot for slot in candidate_slots if filling_acw[slot])
    return slots_cw, slots_acw


def wrap_tune_difference(value):
    """Wrap a tune difference to ``[-0.5, 0.5)``."""
    return (np.asarray(value) + 0.5) % 1.0 - 0.5


# Load and prepare the two LHC lines ------------------------------------------
context = xo.ContextCpu()
env = xt.load(
    str(LHC_DATA / 'lhc.seq'), format='madx', reverse_lines=['lhcb2'])

for line_name in ('lhcb1', 'lhcb2'):
    env[line_name].particle_ref = xt.Particles(
        mass0=xt.PROTON_MASS_EV, p0c=P0C)

env.vars.load(str(OPTICS_FILE))
for line_name in ('lhcb1', 'lhcb2'):
    line = env[line_name]
    line.twiss_default['method'] = '4d'
    line.cycle(name_first_element='ip3', inplace=True)
    line.build_tracker(_context=context)

with open(FILLING_FILE) as fid:
    filling_data = json.load(fid)
filling_cw = np.asarray(filling_data['schemebeam1'])
filling_acw = np.asarray(filling_data['schemebeam2'])
if len(filling_cw) != N_SLOTS or len(filling_acw) != N_SLOTS:
    raise ValueError(f'The filling patterns must contain {N_SLOTS} slots.')


# Install and configure rigid-bunch beam-beam --------------------------------
env.xfields.install_beambeam_interactions(
    clockwise_line='lhcb1',
    anticlockwise_line='lhcb2',
    ip_names=IP_NAMES,
    num_long_range_encounters_per_side=N_LONG_RANGE,
    harmonic_number=HARMONIC_NUMBER,
    bunch_spacing_buckets=BUNCH_SPACING_BUCKETS,
    mode='rigid_bunch')

# Configuration computes the bare optics, encounter geometry, and IP pairing
# offsets. The filling is applied below, once the desired subset is known.
study = env.xfields.configure_beambeam_interactions(
    num_particles=BUNCH_INTENSITY,
    nemitt_x=NEMITT_X,
    nemitt_y=NEMITT_Y)

if ALL_BUNCHES:
    slots_cw = np.flatnonzero(filling_cw)
    slots_acw = np.flatnonzero(filling_acw)
    print('Filling selection: complete LHC filling (LHC_ALL=1)')
else:
    slots_cw, slots_acw = select_bounded_filling(
        study.ip_offsets, filling_cw, filling_acw, WINDOW)
    print(f'Filling selection: bounded subset (LHC_WINDOW={WINDOW}; '
          'set LHC_ALL=1 for the complete filling)')

study.apply_filling_pattern(
    filled_slots_cw=slots_cw,
    filled_slots_acw=slots_acw)

slots_cw = study.filled_slots_cw
slots_acw = study.filled_slots_acw
for ip in IP_NAMES:
    print(f'  {ip}: head-on pairing offset = {study.ip_offsets[ip]} slots')
print(f'  populated bunches: CW={len(slots_cw)}, ACW={len(slots_acw)}')


# Solve the self-consistent two-beam closed-orbit problem --------------------
print('Self-consistent solve on the full thick lattice:')
start_time = time.time()
solution = study.solve(max_iterations=N_ITERATIONS)
print(f'  converged={solution.converged}, '
      f'iterations={solution.num_iterations}, '
      f'max orbit change={solution.max_orbit_change:.3e} sigma')
print(f'  solve time: {time.time() - start_time:.1f} s')


# Inspect and plot the clockwise-beam result ---------------------------------
# The reference-tune interface and the structure of the solution object will
# be reviewed separately. For now, the configured bare tunes live in `meta`.
bare_qx = study.meta['qx_cw']
bare_qy = study.meta['qy_cw']
dqx = wrap_tune_difference(solution.cw.qx - bare_qx)
dqy = wrap_tune_difference(solution.cw.qy - bare_qy)

ip1_element = study.bb_name('bb_ip1_ho', beam='cw')
x_ip1 = solution.cw['x', ip1_element]
y_ip1 = solution.cw['y', ip1_element]
dx_ip1 = (x_ip1 - np.mean(x_ip1)) * 1e6
dy_ip1 = (y_ip1 - np.mean(y_ip1)) * 1e6

print(f'  CW horizontal tune shift: [{dqx.min():.2e}, {dqx.max():.2e}]')

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
axes[0].plot(slots_cw, dqx * 1e3, '.', label=r'$\Delta q_x$')
axes[0].plot(slots_cw, dqy * 1e3, '.', label=r'$\Delta q_y$')
axes[0].set_ylabel(r'beam-beam tune shift [$10^{-3}$]')
axes[0].legend()

axes[1].plot(slots_cw, dx_ip1, '.', label='x')
axes[1].plot(slots_cw, dy_ip1, '.', label='y')
axes[1].set_xlabel('25 ns slot')
axes[1].set_ylabel(r'orbit deviation at IP1 [$\mu$m]')
axes[1].legend()

fig.suptitle('Rigid-bunch beam-beam on the full thick LHC lattice (CW)')
fig.tight_layout()
plt.show()
