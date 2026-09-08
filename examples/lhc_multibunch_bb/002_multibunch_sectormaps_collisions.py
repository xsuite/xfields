# copyright ############################### #
# This file is part of the Xfields Package. #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Multi-bunch beam-beam on the LHC in COLLISION (6.8 TeV, fully squeezed
R2025aRP 15 cm flat optics, levelling-style knobs), with second-order maps.

Scenario following LHC 2025/2026 physics at end of levelling: head-on collisions
at IP1/IP5 (flat optics, H crossing in 1, V crossing in 5, separations off),
levelling offsets at IP2/IP8, spectrometers/solenoids on, octupoles powered,
tunes/chromaticity matched to 62.316/60.322 and Q' = 10. 1.1e11 p/bunch,
2.3 um normalized emittance, full 2460-bunch filling pattern.

The standard beam-beam workflow in ``mode='rigid_bunch'`` installs and
configures the head-on + long-range lenses on the full lattice;
``rigid_bunch_study.get_study_with_second_order_maps``
then makes a fast copy where the arcs between the encounters are replaced by
second-order Taylor maps (the lenses stay exact), and ``study_red.solve`` finds
the per-bunch self-consistent closed solution on it. The flattened collision
optics and the pytrain reference are in ``test_data/lhc_2024`` (regenerate with
``test_data/lhc_2024/pytrain/regenerate_collision.py``).
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import lhc_mb_common as mb

N_ITER = int(os.environ.get('LHC_NITER', '6'))
ALL_BUNCHES = os.environ.get('LHC_ALL', '1') == '1'
WINDOW = int(os.environ.get('LHC_WINDOW', '48'))
COMPUTE_OPTICS_PARAMS = os.environ.get('COMPUTE_OPTICS_PARAMS', '1') == '1'

# ----------------------------------------------------------------------------
env, line_b1, line_b2, par = mb.load_lhc('collision')
scheme_b1, scheme_b2 = mb.load_scheme()

# Install the head-on + long-range lenses and compute the geometry on the full
# thick lattice.
env.xfields.install_beambeam_interactions(
    clockwise_line='lhcb1', anticlockwise_line='lhcb2', ip_names=par['ips'],
    num_long_range_encounters_per_side=par['nparasitic'],
    harmonic_number=mb.HARMONIC_NUMBER,
    bunch_spacing_buckets=mb.BUNCH_SPACING_BUCKETS,
    mode='rigid_bunch')
rigid_bunch_study = env.xfields.configure_beambeam_interactions(
    num_particles=par['bunch_intensity'],
    nemitt_x=par['nemitt'], nemitt_y=par['nemitt'],
    filling_pattern_cw=scheme_b1, filling_pattern_acw=scheme_b2)

# Compute the reference optics with the configured beam-beam elements disabled.
env['beambeam_scale'] = 0
twiss_no_bb_cw = line_b1.twiss()
twiss_no_bb_acw = line_b2.twiss()
env['beambeam_scale'] = 1
print(f'  tunes without BB, CW '
      f'{twiss_no_bb_cw.qx:.5f}/{twiss_no_bb_cw.qy:.5f}  '
      f'ACW {twiss_no_bb_acw.qx:.5f}/{twiss_no_bb_acw.qy:.5f}')

if not ALL_BUNCHES:
    # restrict to a bounded window with all-IP pairings (offsets from geometry)
    s1, s2 = mb.windowed_slots(rigid_bunch_study.ip_offsets, scheme_b1, scheme_b2, WINDOW)
    rigid_bunch_study.apply_filling_pattern(
        filled_slots_cw=s1, filled_slots_acw=s2)

# Fast sector-map copy: the arcs between the encounters become second-order maps
# (the beam-beam elements stay exact). Solving the reduced study is much faster.
print('  building second-order maps between the beam-beam elements...')
study_red = rigid_bunch_study.get_study_with_second_order_maps(
    context=par['context'])
slots_cw, slots_acw = study_red.filled_slots_cw, study_red.filled_slots_acw
print(f'  populated bunches: CW = {len(slots_cw)}, ACW = {len(slots_acw)}')

print('Self-consistent solve (head-on + long-range):')
t0 = time.time()
solution = study_red.solve(max_iterations=N_ITER)
mb.print_solve_status(solution)
mbtw_cw, mbtw_acw = solution.cw, solution.acw
print(f'  solve time ({len(slots_cw)}+{len(slots_acw)} bunches, '
      f'{solution.num_iterations} iters): {time.time() - t0:.1f} s')

if COMPUTE_OPTICS_PARAMS:
    print('Final mode="fast" twiss (per-bunch optics + global quantities):')
    t0 = time.time()
    final_twiss = study_red.twiss(mode='fast')
    mbtw_cw, mbtw_acw = final_twiss.cw, final_twiss.acw
    print(f'  final twiss (both beams): {time.time() - t0:.1f} s')

# Reference the tune shift to the optics without beam-beam.
dqx_cw = mb.wrap_frac_tune(mbtw_cw.qx - twiss_no_bb_cw.qx)
print(f"\nCW tune shift: dqx in [{dqx_cw.min():.2e}, {dqx_cw.max():.2e}]")

df_cw = mb.results_dataframe(study_red, mbtw_cw, slots_cw,
                             twiss_no_bb_cw.qx, twiss_no_bb_cw.qy,
                             beam='cw')
df_acw = mb.results_dataframe(study_red, mbtw_acw, slots_acw,
                              twiss_no_bb_acw.qx, twiss_no_bb_acw.qy,
                              beam='acw')
# Keep the established comparison filenames used by the PyTRAIN workflow.
df_cw.to_pickle(os.path.join(mb.HERE, 'results_b1_coll.pkl'))
df_acw.to_pickle(os.path.join(mb.HERE, 'results_b2_coll.pkl'))
print('saved results_b1_coll.pkl / results_b2_coll.pkl')

mb.plot_results(study_red, slots_cw, mbtw_cw,
                twiss_no_bb_cw.qx, twiss_no_bb_cw.qy,
                title_suffix='  [collision, 6.8 TeV]')
if COMPUTE_OPTICS_PARAMS:
    mb.plot_global_quantities(
        study_red, slots_cw, mbtw_cw, slots_acw, mbtw_acw)
plt.show()
