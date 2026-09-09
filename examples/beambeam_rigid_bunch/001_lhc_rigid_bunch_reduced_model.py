# copyright ############################### #
# This file is part of the Xfields Package. #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""Rigid-bunch beam-beam on a reduced model of the LHC lattice.

This example repeats the workflow of ``000_lhc_rigid_bunch_full_lattice.py``
using the same collision optics and filling pattern. After installing and
configuring the beam-beam interactions on the full lattice, it replaces the
lattice regions between consecutive encounters with second-order maps. The
beam-beam elements remain exact.

The self-consistent solution is computed on the reduced model and then loaded
into the original study, making it available on the full lattice for subsequent
studies.
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

###############################
# Build the reduced LHC model #
###############################

start_time = time.time()
reduced_study = study.get_study_with_second_order_maps()
print(f'Reduced-model build time: {time.time() - start_time:.1f} s')
print(f'  CW elements: {len(study.cw_line.element_names)} full, '
      f'{len(reduced_study.cw_line.element_names)} reduced')
print(f'  ACW elements: {len(study.acw_line.element_names)} full, '
      f'{len(reduced_study.acw_line.element_names)} reduced')

###################################################
# Solve the self-consistent two-beam closed-orbit #
###################################################

print('Self-consistent solve on the reduced model:')
start_time = time.time()
solution = reduced_study.solve(max_iterations=N_ITERATIONS)
print(f'  converged={solution.converged}, '
      f'iterations={solution.num_iterations}, '
      f'max orbit change={solution.max_orbit_change:.3e} sigma')
print(f'  solve time: {time.time() - start_time:.1f} s')

############################################
# Load the solution into the full lattice #
############################################

study.load_solution(solution)

##################
# Inspect result #
##################

# Compute the reference optics with the configured beam-beam elements disabled.
env['beambeam_scale'] = 0
twiss_no_bb_cw = env.lhcb1.twiss()
env['beambeam_scale'] = 1

# Inspect the filled slots for both beams
slots_cw = reduced_study.filled_slots_cw
slots_acw = reduced_study.filled_slots_acw

# The per-beam result behaves like a table with one row per filled slot. Global
# quantities such as qx and qy are columns of this bunch-indexed table.
solution.cw.cols['slot zeta qx qy']

# Select one bunch by its physical filling slot. The returned object is the
# ordinary lattice-indexed xt.TwissTable for that bunch on the reduced line.
selected_slot = int(solution.cw.slot[0])
selected_bunch_twiss = solution.cw.bunch(slot=selected_slot)

qx_no_bb = twiss_no_bb_cw.qx % 1  # only fractional part
qy_no_bb = twiss_no_bb_cw.qy % 1  # only fractional part
tune_shift_x = solution.cw.qx - qx_no_bb
tune_shift_y = solution.cw.qy - qy_no_bb

# Conversely, at_element returns an xt.Table with one row per bunch at the
# selected lattice element. Its rows can be selected with the standard table
# API using the slot-based bunch names.
ip1_element = reduced_study.bb_name('bb_ip1_ho', beam='cw')
twiss_at_ip1 = solution.cw.at_element(ip1_element)
twiss_at_ip1.cols['qx qy x y']

# Compute positions relative to the mean orbit at IP1
x_ip1 = twiss_at_ip1.x
y_ip1 = twiss_at_ip1.y
dx_ip1 = (x_ip1 - np.mean(x_ip1)) * 1e6
dy_ip1 = (y_ip1 - np.mean(y_ip1)) * 1e6

print(f'  CW horizontal tune shift: '
      f'[{tune_shift_x.min():.2e}, {tune_shift_x.max():.2e}]')

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

fig.suptitle('Rigid-bunch beam-beam on the reduced LHC model (CW)')
fig.tight_layout()
plt.show()
