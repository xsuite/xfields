# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #
"""
Touschek simulation split over independent jobs, merged afterwards.

1. Prepare: install and configure the Touschek markers, and save the line,
   which stores the whole configuration.
2. Jobs: each job loads the line, generates the particles of one marker and
   one replica, tracks them and saves them. Here the jobs run one after the
   other; on a cluster each one is a separate process, e.g. with
   marker = job_id % n_markers and replica = job_id // n_markers.
3. Merge: the particles of all the jobs, merged, give the loss rate and the
   lifetime of the whole ring; analysed marker by marker, they also give the
   Monte Carlo error, from the spread between the replicas.

The particles of a job do not depend on the other jobs: a single job
generating all the markers gives the same particles.
"""
import json
from pathlib import Path

import numpy as np

import xobjects as xo
import xtrack as xt

output_dir = Path('touschek_jobs')
output_dir.mkdir(exist_ok=True)

######################################################
# 1. Prepare and save the configured line
######################################################
line = xt.load('../../test_data/four_cell_ring/line.json')
env = line.env

# Touschek markers in the middle of each magnet, and at the end of the line
tab = line.get_table()
tab_bends_quads = tab.rows[
    (tab.element_type == 'Bend') | (tab.element_type == 'Quadrupole')
]
line.xfields.touschek.install_markers(at=list(tab_bends_quads.name))

# Apertures around the magnets
tab = line.get_table()
needs_aperture = tab.rows.match_not(element_type='Drift.*|Marker|').name
env.new('aper', xt.LimitRect, min_x=-0.04, max_x=0.04, min_y=-0.04, max_y=0.04)
placements = []
for nn in needs_aperture:
    env.new(f'{nn}_aper_entry', 'aper')
    env.new(f'{nn}_aper_exit', 'aper')
    placements.append(env.place(f'{nn}_aper_entry', at=f'{nn}@start'))
    placements.append(env.place(f'{nn}_aper_exit', at=f'{nn}@end'))
line.insert(placements)

# A fixed +/-0.6% local momentum acceptance, for brevity (see
# 000_touschek_toy_ring.py for its computation by tracking). It is smaller
# than the actual one: only a part of the scattered particles is lost.
markers = line.xfields.touschek.marker_names
tab = line.get_table()
lma = xt.Table({
    'name': np.array(markers),
    's': np.array([tab['s', nn] for nn in markers]),
    'delta_neg': np.full(len(markers), -0.006),
    'delta_pos': np.full(len(markers), 0.006),
})

line.xfields.touschek.configure(
    local_momentum_acceptance=lma,
    local_momentum_acceptance_scale=0.85,
    nemitt_x=1e-5,
    nemitt_y=1e-7,
    sigma_z=4e-3,
    sigma_delta=1e-3,
    bunch_intensity=4e9,
    n_scattering_events=int(1e5),
    weight_retention_fraction=0.99,
    seed=1997,
    method='4d',
)

# The configuration is stored on the markers, i.e. in the saved line
line.to_json(output_dir / 'line.json')

######################################################
# 2. Jobs: one marker and one replica each
######################################################
n_turns = 100
n_replicas = 2
n_markers = len(markers)
n_jobs = n_markers * n_replicas

# Each job would load the configured line; here it is loaded once, to build
# the tracker only once
line = xt.load(output_dir / 'line.json')
line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

for job_id in range(n_jobs):
    marker = markers[job_id % n_markers]
    replica = job_id // n_markers

    # The weights are divided by n_replicas: merged, the replicas of a
    # marker represent its section once
    part = line.xfields.touschek.generate(
        at=marker, replica=replica, n_replicas=n_replicas)
    line.track(part, ele_start=marker, ele_stop=marker, num_turns=n_turns)
    part.to_json(output_dir / f'job.{job_id}.json')

######################################################
# 3. Merge and analyse
######################################################
samples = {nn: [] for nn in markers}
for job_id in range(n_jobs):
    with open(output_dir / f'job.{job_id}.json') as fid:
        part = xt.Particles.from_dict(json.load(fid))
    samples[markers[job_id % n_markers]].append(part)

# All the particles: the loss rate and the lifetime of the ring
particles = xt.Particles.merge([pp for vv in samples.values() for pp in vv])
result = line.xfields.touschek.analyse(particles)
print(f'Touschek scattering rate: {result.rate_scattering*1e-3:.3f} kHz')
print(f'Touschek tracking rate:   {result.rate_tracking*1e-3:.3f} kHz')

# Marker by marker: the contribution of each section, and the Monte Carlo
# error from the spread between the replicas
result = line.xfields.touschek.analyse(samples)
print(result.local_rates)
print(f'Scattering lifetime: {result.lifetime_scattering/60:.2f} min')
print(f'Tracking lifetime:   {result.lifetime_tracking/60:.2f} '
      f'+- {result.lifetime_tracking_error/60:.2f} min')
