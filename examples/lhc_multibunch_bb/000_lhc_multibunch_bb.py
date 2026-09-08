# copyright ############################### #
# This file is part of the Xfields Package. #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""Rigid-bunch beam-beam on the full, thick LHC lattice.

This example shows the complete Xsuite workflow explicitly: load and prepare
the two LHC lines, install the beam-beam elements, configure the rigid-bunch
study, apply a filling, and solve for the self-consistent per-bunch closed
orbit and tunes.

The default run uses a prepared 104-bunch subset of the real LHC filling
because every iteration twisses the full thick lattice once per beam. Edit
``FILLING_FILE`` below to use another prepared filling, for example the
complete 2460-bunch pattern. The lattice itself is never reduced; see
``002_multibunch_sectormaps_collisions.py`` for the faster second-order-map
workflow.
"""

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt


# LHC collision configuration ------------------------------------------------
P0C = 6.8e12
BUNCH_INTENSITY = 1.1e11
NEMITT_X = 2.3e-6
NEMITT_Y = 2.3e-6
HARMONIC_NUMBER = 35640
BUNCH_SPACING_BUCKETS = 10

IP_NAMES = ['ip1', 'ip2', 'ip5', 'ip8']
N_LONG_RANGE = 45
N_ITERATIONS = 6
FILLING_FILE = '../../test_data/lhc_2024/filling_25ns_104b.json'


def wrap_tune_difference(value):
    """Wrap a tune difference to ``[-0.5, 0.5)``."""
    return (np.asarray(value) + 0.5) % 1.0 - 0.5


# Load and prepare the two LHC lines ------------------------------------------
env = xt.load('../../test_data/lhc_2024/lhc.seq', reverse_lines=['lhcb2'])
env.vars.load('../../test_data/lhc_2024/collision_optics_15cm_flat_2026.madx')

env.lhcb1.set_particle_ref('proton', p0c=P0C)
env.lhcb2.set_particle_ref('proton', p0c=P0C)

for lname in ('lhcb1', 'lhcb2'):
    env[lname].twiss_default['method'] = '4d'
    env[lname].cycle(name_first_element='ip3', inplace=True)

filling_data = xt.json.load(FILLING_FILE)
filled_slots_cw = np.asarray(filling_data['filled_slots_cw'])
filled_slots_acw = np.asarray(filling_data['filled_slots_acw'])

# Install and configure rigid-bunch beam-beam --------------------------------
env.xfields.install_beambeam_interactions(
    clockwise_line='lhcb1',
    anticlockwise_line='lhcb2',
    ip_names=IP_NAMES,
    num_long_range_encounters_per_side=N_LONG_RANGE,
    harmonic_number=HARMONIC_NUMBER,
    bunch_spacing_buckets=BUNCH_SPACING_BUCKETS,
    mode='rigid_bunch')

study = env.xfields.configure_beambeam_interactions(
    num_particles=BUNCH_INTENSITY,
    nemitt_x=NEMITT_X,
    nemitt_y=NEMITT_Y,
    filled_slots_cw=filled_slots_cw,
    filled_slots_acw=filled_slots_acw)

# Compute the reference optics with the configured beam-beam elements disabled.
env['beambeam_scale'] = 0
twiss_no_bb_cw = env.lhcb1.twiss()
env['beambeam_scale'] = 1

slots_cw = study.filled_slots_cw
slots_acw = study.filled_slots_acw
print(f'Filling: {filling_data["name"]}')
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
qx_no_bb = twiss_no_bb_cw.qx
qy_no_bb = twiss_no_bb_cw.qy
dqx = wrap_tune_difference(solution.cw.qx - qx_no_bb)
dqy = wrap_tune_difference(solution.cw.qy - qy_no_bb)

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
