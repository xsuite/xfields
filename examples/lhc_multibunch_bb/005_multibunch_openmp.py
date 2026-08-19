# copyright ############################### #
# This file is part of the Xfields Package. #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Multi-threaded (OpenMP) CPU kernels for the multi-bunch beam-beam machinery.

Builds the collision machine on second-order maps (as
``002_multibunch_sectormaps_collisions.py``), populates the beam-beam
elements with one self-consistent iteration, then times the batched
rigid-bunch Twiss of both beams -- both the orbit-only ``fast_orbit`` mode (used
in the solver loop) and the optics-carrying ``fast`` mode -- with the SERIAL
and the MULTI-THREADED CPU kernels, and reports the speed-up.

All the multibunch examples accept the environment variable ``LHC_OMP``
(unset/``0`` -> serial kernels; ``auto`` or a thread count -> OpenMP
kernels); prebuilt kernels exist for both flavours, so no compilation is
triggered either way. Here ``LHC_OMP`` defaults to ``auto``.
"""

import os
import time
import numpy as np

import xobjects as xo
import lhc_mb_common as mb

os.environ.setdefault('LHC_OMP', 'auto')
env, line_b1, line_b2, par = mb.load_lhc('collision')
ctx_mt = par['context']
n_threads = os.cpu_count() if ctx_mt.omp_num_threads == 'auto' \
    else ctx_mt.omp_num_threads
print(f'multi-threaded context: {ctx_mt.omp_num_threads} '
      f'({n_threads} threads)')

# ----------------------------------------------------------------------------
# Build the machine (as in 002) and populate the beam-beam elements
# ----------------------------------------------------------------------------
scheme_b1, scheme_b2 = mb.load_scheme()
env.xfields.install_beambeam_interactions(
    clockwise_line='lhcb1', anticlockwise_line='lhcb2', ip_names=par['ips'],
    num_long_range_encounters_per_side=par['nparasitic'],
    harmonic_number=mb.HARMONIC_NUMBER,
    bunch_spacing_buckets=mb.BUNCH_SPACING_BUCKETS,
    mode='rigid_bunch')
study = env.xfields.configure_beambeam_interactions(
    num_particles=par['bunch_intensity'],
    nemitt_x=par['nemitt'], nemitt_y=par['nemitt'],
    filling_pattern_cw=scheme_b1, filling_pattern_acw=scheme_b2)
print('  building second-order maps between the beam-beam elements...')
study_red = study.second_order_maps(context=ctx_mt)
slots_b1, slots_b2 = study_red.filled_slots_cw, study_red.filled_slots_acw
print(f'  populated bunches: B1 = {len(slots_b1)}, B2 = {len(slots_b2)}')

print('Populating the beam-beam elements (one solve iteration):')
study_red.solve(max_iterations=1, tol_sigma=0.0)

# ----------------------------------------------------------------------------
# Timing: batched rigid-bunch Twiss, serial vs multi-threaded kernels
# ----------------------------------------------------------------------------
contexts = (('serial', xo.ContextCpu()),
            (f'openmp ({n_threads} threads)', ctx_mt))

print(f'\nBatched rigid-bunch Twiss ({len(slots_b1)}+{len(slots_b2)} bunches):')
for mode in ('fast_orbit', 'fast'):
    timings = []
    for label, ctx in contexts:
        for line in (study_red.cw_line, study_red.acw_line):
            line.discard_tracker()
            line.build_tracker(_context=ctx)
        study_red.twiss(mode=mode, show_progress=False)  # warm-up
        t0 = time.time()
        study_red.twiss(mode=mode, show_progress=False)
        timings.append(time.time() - t0)
        print(f'  mode={mode!r:13s} {label:22s}: {timings[-1]:7.1f} s')
    print(f'  mode={mode!r:13s} speed-up: x{timings[0] / timings[1]:.2f}')
