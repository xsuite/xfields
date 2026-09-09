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
``004_lhc_rigid_bunch_collision.py`` for the faster second-order-map
workflow.
"""

import time
import numpy as np
import xtrack as xt

##############
# Parameters #
##############

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

###########################################
# Load lines, set ref. particle and cycle #
###########################################

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

###############################################
# Install and configure rigid-bunch beam-beam #
###############################################

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

###################################################
# Solve the self-consistent two-beam closed-orbit #
###################################################

print('Self-consistent solve on the full thick lattice:')
start_time = time.time()
solution = study.solve(max_iterations=N_ITERATIONS)
print(f'  converged={solution.converged}, '
      f'iterations={solution.num_iterations}, '
      f'max orbit change={solution.max_orbit_change:.3e} sigma')
print(f'  solve time: {time.time() - start_time:.1f} s')

##################
# Inspect result #
##################

# Compute the reference optics with the configured beam-beam elements disabled.
env['beambeam_scale'] = 0
twiss_no_bb_cw = env.lhcb1.twiss()
env['beambeam_scale'] = 1

# Inspect the filled slots for both beams
slots_cw = study.filled_slots_cw
slots_acw = study.filled_slots_acw

# The per-beam result behaves like a table with one row per filled slot. Global
# quantities such as qx and qy are columns of this bunch-indexed table.
solution.cw.cols['slot zeta qx qy']
# is:
# Table: 104 rows, 5 cols
# name    slot          zeta            qx            qy
# slot_63   63      -471.243      0.304725      0.309594
# slot_64   64      -478.723      0.304467      0.309408
# slot_65   65      -486.203      0.304205      0.309222
# slot_66   66      -493.683      0.303939      0.309042
# slot_67   67      -501.163       0.30367      0.308867
# slot_68   68      -508.643      0.303398      0.308695
# slot_69   69      -516.123      0.303125      0.308525
# ...

# Select one bunch by its physical filling slot. The returned object is the
# ordinary lattice-indexed xt.TwissTable for that bunch.
selected_slot = int(solution.cw.slot[0])
selected_bunch_twiss = solution.cw.bunch(slot=selected_slot)
# is:
# TwissTable: 14434 rows, 6 cols
# name                     s             x            px             y ...
# ip3                      0   -1.9417e-05   4.74682e-07   4.88005e-05
# ||drift_1665             0   -1.9417e-05   4.74682e-07   4.88005e-05
# mcbwv.4r3.b1         20.18  -9.83787e-06   4.74682e-07   5.73105e-05
# ||drift_1666         21.88  -9.03091e-06   4.74682e-07   5.80274e-05
# bpmw.4r3.b1        22.5205  -8.72688e-06   4.74682e-07   5.82975e-05
# ||drift_1667       22.5205  -8.72688e-06   4.74682e-07   5.82975e-05
# mqwa.a4r3.b1        23.081  -8.46082e-06   4.74682e-07   5.85339e-05
# ...

qx_no_bb = twiss_no_bb_cw.qx % 1 # only fractional part
qy_no_bb = twiss_no_bb_cw.qy % 1 # only fractional part
tune_shift_x = solution.cw.qx - qx_no_bb
tune_shift_y = solution.cw.qy - qy_no_bb

# Conversely, at_element returns an xt.Table with one row per bunch at the
# selected lattice element. Its rows can be selected with the standard table
# API using the slot-based bunch names.
ip1_element = study.bb_name('bb_ip1_ho', beam='cw')
twiss_at_ip1 = solution.cw.at_element(ip1_element)
twiss_at_ip1.cols['qx qy x y']
# is:
# Table: 104 rows, 5 cols
# name               qx            qy             x             y
# slot_63      0.304725      0.309594    2.5768e-06   4.29897e-07
# slot_64      0.304467      0.309408   2.45128e-06   4.39456e-07
# slot_65      0.304205      0.309222   2.33489e-06   4.48042e-07
# slot_66      0.303939      0.309042   2.22487e-06   4.56571e-07
# slot_67       0.30367      0.308867   2.11765e-06    4.6558e-07
# slot_68      0.303398      0.308695   2.01109e-06   4.74507e-07
# ...

# Compute positions relative to the mean orbit at IP1
x_ip1 = twiss_at_ip1.x
y_ip1 = twiss_at_ip1.y
dx_ip1 = (x_ip1 - np.mean(x_ip1)) * 1e6
dy_ip1 = (y_ip1 - np.mean(y_ip1)) * 1e6

print(f'  CW horizontal tune shift: [{tune_shift_x.min():.2e}, {tune_shift_x.max():.2e}]')

#########
# Plots #
#########

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
axes[0].plot(slots_cw, tune_shift_x * 1e3, '.', label=r'$\Delta q_x$')
axes[0].plot(slots_cw, tune_shift_y * 1e3, '.', label=r'$\Delta q_y$')
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
