# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
import numpy as np
import matplotlib.pyplot as plt

import xobjects as xo
import xtrack as xt

######################################################
# Beam parameters
######################################################
nemitt_x = 1e-5
nemitt_y = 1e-7

sigma_z = 4e-3
sigma_delta = 1e-3

bunch_intensity = 4e9

######################################################
# Load a toy ring
######################################################
line = xt.load('../../test_data/four_cell_ring/line.json')
env = line.env

######################################################
# Install Touschek scattering markers
######################################################
# Each marker represents the section of the line from the previous marker to
# itself. They are installed in the middle of each magnet, to follow the
# variations of the optical functions. A marker is also added at the end of
# the line, so that the sections cover the whole ring.
tab = line.get_table()
tab_bends_quads = tab.rows[
    (tab.element_type == 'Bend') | (tab.element_type == 'Quadrupole')
]
line.xfields.touschek.install_markers(at=list(tab_bends_quads.name))

######################################################
# Install apertures
######################################################
tab = line.get_table()
needs_aperture = tab.rows.match_not(element_type='Drift.*|Marker|').name

aper_size = 0.040

env.new('aper', xt.LimitRect,
        min_x=-aper_size, max_x=aper_size,
        min_y=-aper_size, max_y=aper_size)

placements = []
for nn in needs_aperture:
    env.new(f'{nn}_aper_entry', 'aper')
    env.new(f'{nn}_aper_exit', 'aper')
    placements.append(env.place(f'{nn}_aper_entry', at=f'{nn}@start'))
    placements.append(env.place(f'{nn}_aper_exit', at=f'{nn}@end'))

line.insert(placements)

######################################################
# Evaluate local momentum acceptance profile
######################################################
touschek_markers = line.xfields.touschek.marker_names

lma = line.get_local_momentum_acceptance(
    elements=touschek_markers,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    y_offset=1e-9,
    delta_negative_limit=-0.012,
    delta_positive_limit=0.012,
    delta_step_size=1e-4,
    n_turns=1000,
    method='4d',
)

plt.plot(lma.s, lma.delta_neg*100, c='r')
plt.plot(lma.s, lma.delta_pos*100, c='r')
plt.title('Toy ring: local momentum acceptance profile')
plt.xlabel('s [m]')
plt.ylabel(r'$\delta$ [%]')
plt.grid()
plt.show()

######################################################
# Configure the Touschek markers
######################################################
local_momentum_acceptance_scale = 0.85
n_scattering_events = int(5e6)
nturns = 1000

line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

# Computes the Piwinski scattering rate integrated over the section of each
# marker, and stores it on the marker together with the local optics, the
# local momentum acceptance, the beam parameters and the Monte Carlo
# settings. The configured line can be saved here (e.g. with line.to_json):
# once loaded again, it generates the same particles (see
# 003_touschek_jobs.py).
line.xfields.touschek.configure(
    local_momentum_acceptance=lma,
    local_momentum_acceptance_scale=local_momentum_acceptance_scale,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    sigma_delta=sigma_delta,
    bunch_intensity=bunch_intensity,
    n_scattering_events=n_scattering_events,
    nx=3, ny=3, nz=3,
    weight_retention_fraction=0.99,
    seed=1997,
    rng='elegant',
    method='4d',
)

print(line.xfields.touschek.analyse().local_rates)

######################################################
# Generate and track the Touschek-scattered particles
######################################################
# For each marker:
#   1. Generate the particles scattered in its section, beyond the local
#      momentum acceptance
#   2. Track them around the ring for `nturns` turns, starting and ending at
#      the marker
particles_list = []
for name in touschek_markers:
    particles = line.xfields.touschek.generate(at=name)
    print(f'\nTracking particles scattered at {name}')
    line.track(particles, ele_start=name, ele_stop=name, num_turns=nturns,
               with_progress=1)
    particles_list.append(particles)

# The weights are rates [1/s]: merged, the samples of all the markers
# represent the Touschek scattering of the whole ring
particles = xt.Particles.merge(particles_list)

######################################################
# Optional: refine loss locations
######################################################
loss_loc_refinement = xt.LossLocationRefinement(
    line,
    n_theta=360, # Angular resolution of the polygonal aperture approximation
    r_max=0.5,   # Maximum transverse aperture [m]
    dr=50e-6,    # Transverse loss refinement accuracy [m]
    ds=0.1,      # Longitudinal loss refinement accuracy [m]
)

loss_loc_refinement.refine_loss_location(particles)

######################################################
# Touschek loss rate and lifetime
######################################################
result = line.xfields.touschek.analyse(particles)

# All of the above in one call:
# result = line.xfields.touschek.run(n_turns=nturns)

print(f'Touschek scattering rate: {result.rate_scattering*1e-3:.3f} kHz')
print(f'Touschek tracking rate:   {result.rate_tracking*1e-3:.3f} kHz')
print(f'Scattering lifetime:      {result.lifetime_scattering/60:.2f} min')
print(f'Tracking lifetime:        {result.lifetime_tracking/60:.2f} min')

######################################################
# Plot: Toy ring Touschek loss map
######################################################
lost_particles = particles.filter(particles.state == 0) # lost on an aperture

circumference = line.get_length()
binwidth = 0.1

plt.title(
    f'Toy ring Touschek loss map '
    f'(Touschek lifetime: {result.lifetime_tracking/60:.2f} min)'
)
plt.hist(
    lost_particles.s,
    bins=np.arange(0, circumference + binwidth, binwidth),
    weights=lost_particles.weight*1e-3,
)
plt.xlabel('s [m]')
plt.ylabel('Loss rate [kHz]')
plt.grid()
plt.show()
