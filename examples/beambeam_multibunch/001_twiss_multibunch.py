# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

"""Per-bunch rigid-beam optics through ``BeamBeamRigidBunchStudy``.

The rigid-bunch study owns both the filling scheme and the beam-beam state. Its ``twiss``
method therefore needs no separate bunch-position argument, while ``solve``
iterates the two beams to a self-consistent closed orbit.
"""

import matplotlib.pyplot as plt
import numpy as np

import xtrack as xt


N_CELLS = 8
N_SLOTS = 8


def make_ring(suffix, shared_ip):
    elements = []
    names = []
    for ii in range(N_CELLS):
        marker = shared_ip if ii == 0 else xt.Marker()
        elements.extend([
            marker,
            xt.Multipole(knl=[0, 0.05]),
            xt.Drift(length=5),
            xt.Multipole(knl=[0, -0.05]),
            xt.Drift(length=5),
        ])
        names.extend([
            'ip1' if ii == 0 else f'cell{ii}_{suffix}',
            f'qf{ii}_{suffix}', f'da{ii}_{suffix}',
            f'qd{ii}_{suffix}', f'db{ii}_{suffix}',
        ])
    line = xt.Line(
        elements=elements,
        element_names=names,
        particle_ref=xt.Particles(p0c=7e12),
    )
    line.twiss_default['method'] = '4d'
    return line


shared_ip = xt.Marker()
env = xt.Environment(lines={
    'cw': make_ring('cw', shared_ip),
    'acw': make_ring('acw', shared_ip),
})

filling_cw = np.zeros(N_SLOTS, dtype=int)
filling_acw = np.zeros(N_SLOTS, dtype=int)
filling_cw[[0, 1, 3, 6]] = 1
filling_acw[[0, 2, 3, 7]] = 1

intensity_cw = np.zeros(N_SLOTS)
intensity_acw = np.zeros(N_SLOTS)
intensity_cw[filling_cw > 0] = [1.0e11, 1.2e11, 1.4e11, 1.6e11]
intensity_acw[filling_acw > 0] = [1.1e11, 1.3e11, 1.5e11, 1.7e11]

env.xfields.install_beambeam_interactions(
    clockwise_line='cw', anticlockwise_line='acw',
    ip_names=['ip1'], num_long_range_encounters_per_side=0,
    harmonic_number=N_SLOTS, bunch_spacing_buckets=1,
    mode='rigid_bunch')
rigid_bunch_study = env.xfields.configure_beambeam_interactions(
    num_particles={'cw': intensity_cw, 'acw': intensity_acw},
    nemitt_x=2.0e-6, nemitt_y=2.5e-6,
    filling_pattern_cw=filling_cw,
    filling_pattern_acw=filling_acw)

# ``twiss`` observes the current frozen opposing-beam state. ``solve`` updates
# that state until the two per-bunch closed orbits are self-consistent.
initial_twiss = rigid_bunch_study.twiss(mode='fast')
solution = rigid_bunch_study.solve(max_iterations=6)
print(f'converged={solution.converged}, '
      f'iterations={solution.num_iterations}, '
      f'max orbit change={solution.max_orbit_change:.3e} sigma')

for beam, slots, twiss in (
        ('cw', rigid_bunch_study.filled_slots_cw, solution.b1),
        ('acw', rigid_bunch_study.filled_slots_acw, solution.b2)):
    print(f'{beam} per-bunch tunes:')
    for slot, qx, qy in zip(slots, twiss.qx, twiss.qy):
        print(f'  slot {slot}: qx={qx:.8f}, qy={qy:.8f}')

plt.plot(rigid_bunch_study.filled_slots_cw, solution.b1.qx, 'o-', label='cw')
plt.plot(rigid_bunch_study.filled_slots_acw, solution.b2.qx, 's-', label='acw')
plt.xlabel('physical RF slot')
plt.ylabel('$q_x$')
plt.legend()
plt.tight_layout()
plt.show()
