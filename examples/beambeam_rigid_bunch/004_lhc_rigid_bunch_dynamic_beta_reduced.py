# copyright ############################### #
# This file is part of the Xfields Package. #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Impact of DYNAMIC BETA on the rigid-bunch beam-beam closed solution in the
LHC collision scenario (6.8 TeV squeezed flat optics, head-on + BBLR).

The beam-beam elements take the effective (convolved) transverse sizes of
the colliding bunch pairs. By default these are STATIC, computed once from
the optics without beam-beam:
``sigma^2 = (beta_b1 + beta_b2) * nemitt / gamma0``. But
head-on beam-beam changes the per-bunch beta functions at the encounters
(dynamic beta, ~10% spread of beta* in this scenario), which in turn changes
the sizes, the kicks, and hence the per-bunch closed solution.

This script solves the same machine twice:

1. static sizes (as ``002a_lhc_rigid_bunch_collision_reduced.py``);
2. ``dynamic_beta=True``: at every iteration the per-bunch effective sizes
   of all encounters are recomputed from the LIVE per-bunch betas of both
   beams (requires the optics-carrying mode='fast' twiss in the loop).

and compares per-bunch tunes, orbits and beta* at IP1.

By default the MULTI-THREADED CPU kernels (OpenMP) are used (set
``LHC_OMP=0`` for serial, ``LHC_OMP=<n>`` for a specific thread count; see
``005_lhc_rigid_bunch_openmp_reduced.py`` for the speed-up measurement).
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import lhc_rigid_bunch_common as rb

N_ITER = int(os.environ.get('LHC_NITER', '6'))

# multi-threaded CPU kernels by default in this demo
os.environ.setdefault('LHC_OMP', 'auto')
env, line_b1, line_b2, par = rb.load_lhc('collision')

# ----------------------------------------------------------------------------
# Build the machine (as in 002a): install + geometry on the full lattice, then
# a fast second-order-map copy
# ----------------------------------------------------------------------------
scheme_b1, scheme_b2 = rb.load_scheme()
env.xfields.install_beambeam_interactions(
    clockwise_line='lhcb1', anticlockwise_line='lhcb2', ip_names=par['ips'],
    num_long_range_encounters_per_side=par['nparasitic'],
    harmonic_number=rb.HARMONIC_NUMBER,
    bunch_spacing_buckets=rb.BUNCH_SPACING_BUCKETS,
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

print('  building second-order maps between the beam-beam elements...')
study_red = rigid_bunch_study.get_study_with_second_order_maps(
    context=par['context'])
slots_cw, slots_acw = study_red.filled_slots_cw, study_red.filled_slots_acw
print(f'  populated bunches: CW = {len(slots_cw)}, ACW = {len(slots_acw)}')

# ----------------------------------------------------------------------------
# Solve with static sizes, then with dynamic beta
# ----------------------------------------------------------------------------
results = {}
for label, dynamic_beta in (('static', False), ('dynamic beta', True)):
    print(f'Self-consistent solve ({label}):')
    t0 = time.time()
    # 'fast' twiss (per-bunch optics) in both cases so the returned tables carry
    # betx/bety for the static-vs-dynamic beta* comparison below (dynamic_beta
    # requires it; the static solve would otherwise default to fast_orbit).
    results[label] = study_red.solve(
        max_iterations=N_ITER, tol_sigma=0.0,
        twiss_mode='fast', dynamic_beta=dynamic_beta,
        require_convergence=False)
    rb.print_solve_status(results[label])
    print(f'  solve time ({N_ITER} iters): {time.time() - t0:.1f} s')

# ----------------------------------------------------------------------------
# Compare per-bunch tunes, orbit and beta* at IP1 (CW)
# ----------------------------------------------------------------------------
mk_b1 = study_red.bb_name('bb_ip1_ho', beam='cw')


def extract(mbtw):
    twiss_at_ip1 = mbtw.at_element(mk_b1)
    return dict(
        qx=np.asarray(mbtw.qx_frac), qy=np.asarray(mbtw.qy_frac),
        x=twiss_at_ip1.x,
        betx=twiss_at_ip1.betx,
        bety=twiss_at_ip1.bety,
    )


stat = extract(results['static'].cw)
dyn = extract(results['dynamic beta'].cw)

df_cw = rb.results_dataframe(study_red, results['dynamic beta'].cw, slots_cw,
                             twiss_no_bb_cw.qx, twiss_no_bb_cw.qy,
                             beam='cw')
df_acw = rb.results_dataframe(study_red, results['dynamic beta'].acw, slots_acw,
                              twiss_no_bb_acw.qx, twiss_no_bb_acw.qy,
                              beam='acw')
# Keep the established comparison filenames used by the PyTRAIN workflow.
df_cw.to_pickle(os.path.join(rb.HERE, 'results_b1_coll_dynbeta.pkl'))
df_acw.to_pickle(os.path.join(rb.HERE, 'results_b2_coll_dynbeta.pkl'))
print('saved results_b1_coll_dynbeta.pkl / results_b2_coll_dynbeta.pkl')

dqx = rb.wrap_frac_tune(dyn['qx'] - stat['qx'])
dqy = rb.wrap_frac_tune(dyn['qy'] - stat['qy'])
print(f"\nDynamic-beta impact (CW):")
print(f"  tune change dqx in [{dqx.min():+.2e}, {dqx.max():+.2e}] "
      f"(rms {dqx.std():.2e})")
print(f"  tune change dqy in [{dqy.min():+.2e}, {dqy.max():+.2e}] "
      f"(rms {dqy.std():.2e})")
print(f"  orbit change at IP1: rms {np.std(dyn['x'] - stat['x'])*1e9:.1f} nm")
print(f"  betx* at IP1: static [{stat['betx'].min():.4f}, "
      f"{stat['betx'].max():.4f}] m, dynamic [{dyn['betx'].min():.4f}, "
      f"{dyn['betx'].max():.4f}] m")

fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axs[0].plot(slots_cw, dqx * 1e3, '.', ms=3, label=r'$\Delta q_x$')
axs[0].plot(slots_cw, dqy * 1e3, '.', ms=3, label=r'$\Delta q_y$')
axs[0].set_ylabel(r'tune change [$10^{-3}$]')
axs[0].set_title('Per-bunch effect of dynamic beta on the closed solution '
                 '(CW, collision)')
axs[0].legend()

axs[1].plot(slots_cw, (dyn['x'] - stat['x']) * 1e9, '.', ms=3, label='x')
axs[1].set_ylabel('orbit change at IP1 [nm]')
axs[1].legend()

axs[2].plot(slots_cw, stat['betx'], '.', ms=3, label=r'$\beta_x^*$ static')
axs[2].plot(slots_cw, dyn['betx'], '.', ms=3,
            label=r'$\beta_x^*$ dynamic beta')
axs[2].plot(slots_cw, stat['bety'], '.', ms=3, label=r'$\beta_y^*$ static')
axs[2].plot(slots_cw, dyn['bety'], '.', ms=3,
            label=r'$\beta_y^*$ dynamic beta')
axs[2].set_ylabel(r'$\beta^*$ at IP1 [m]')
axs[2].set_xlabel('25 ns slot')
axs[2].legend(ncol=2, fontsize=8)

plt.tight_layout()
plt.show()
