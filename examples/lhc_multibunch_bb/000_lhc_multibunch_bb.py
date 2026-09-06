# copyright ############################### #
# This file is part of the Xfields Package. #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Multi-bunch beam-beam on the FULL (thick) LHC lattice in collisions (6.8 TeV,
fully squeezed R2025aRP 15 cm flat optics, end-of-levelling knobs).

Head-on and long-range beam-beam elements (BeamBeamBiGaussianRigidBunch2D) are
installed at IP1/2/5/8 with the rigid-bunch mode of the standard beam-beam
install/configure workflow, and the per-bunch closed solution (closed orbit
+ tunes) of the two multi-bunch beams is found self-consistently with
``rigid_bunch_study.solve()``. The per-IP head-on bunch-pairing offsets are derived from the
ring geometry (the IPs are passed as a list of names); the beams collide head-on
at IP1/IP5 (levelling offsets at IP2/IP8), so the effect is head-on + BBLR.

This "direct" variant twisses the full thick line (no sector-map reduction).
The companion example ``002_multibunch_sectormaps_collisions.py`` replaces the
arcs by second-order maps (much faster) with ``rigid_bunch_study.second_order_maps()``.
"""
import os
import time

import matplotlib.pyplot as plt

import lhc_mb_common as mb

N_ITER = int(os.environ.get('LHC_NITER', '3'))
ALL_BUNCHES = os.environ.get('LHC_ALL', '0') == '1'  # False -> bounded subset
WINDOW = int(os.environ.get('LHC_WINDOW', '48'))

# ----------------------------------------------------------------------------
env, line_b1, line_b2, par = mb.load_lhc('collision')
scheme_b1, scheme_b2 = mb.load_scheme()

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

if not ALL_BUNCHES:
    # restrict to a bounded window (the pairing offsets are now known from the
    # geometry, so we can pick the colliding sub-set) to keep the thick-lattice
    # solve fast
    s1, s2 = mb.windowed_slots(rigid_bunch_study.ip_offsets, scheme_b1, scheme_b2, WINDOW)
    rigid_bunch_study.apply_filling_pattern(
        filling_pattern_cw=mb.filling_scheme_from_slots(s1),
        filling_pattern_acw=mb.filling_scheme_from_slots(s2))

slots_cw = rigid_bunch_study.filled_slots_cw
slots_acw = rigid_bunch_study.filled_slots_acw
for ip in par['ips']:
    print(f'  {ip}: head-on offset = {rigid_bunch_study.geom[f"bb_{ip}_ho"]["offset"]} slots')
print(f'  populated bunches: CW = {len(slots_cw)}, ACW = {len(slots_acw)}')

print('Self-consistent solve on the full thick lattice:')
t0 = time.time()
solution = rigid_bunch_study.solve(max_iterations=N_ITER)
mb.print_solve_status(solution)
mbtw_cw, mbtw_acw = solution.cw, solution.acw
print(f'  solve time ({len(slots_cw)}+{len(slots_acw)} bunches): '
      f'{time.time() - t0:.1f} s')

bare = rigid_bunch_study.meta
dqx_cw = mb.wrap_frac_tune(mbtw_cw.qx - bare['qx_cw'])
print(f"\nCW tune shift: dqx in [{dqx_cw.min():.2e}, {dqx_cw.max():.2e}]")

df_cw = mb.results_dataframe(rigid_bunch_study, mbtw_cw, slots_cw,
                             bare['qx_cw'], bare['qy_cw'], mirror=False)
df_acw = mb.results_dataframe(rigid_bunch_study, mbtw_acw, slots_acw,
                             bare['qx_acw'], bare['qy_acw'], mirror=True)
# Keep the established comparison filenames used by the PyTRAIN workflow.
out_b1 = os.path.join(mb.HERE, 'results_b1_coll_full.pkl')
out_b2 = os.path.join(mb.HERE, 'results_b2_coll_full.pkl')
df_cw.to_pickle(out_b1)
df_acw.to_pickle(out_b2)
print(f'saved {out_b1}\nsaved {out_b2}')

mb.plot_results(rigid_bunch_study, slots_cw, mbtw_cw, bare['qx_cw'], bare['qy_cw'],
                title_suffix='  [full thick lattice, collision]')
plt.show()
