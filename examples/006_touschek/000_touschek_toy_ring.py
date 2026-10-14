# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
import numpy as np
import matplotlib.pyplot as plt

import xobjects as xo
import xtrack as xt
import xfields as xf

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
# Insert Touschek scattering centers
######################################################
tab = line.get_table()
tab_bends_quads = tab.rows[
    (tab.element_type == 'Bend') | (tab.element_type == 'Quadrupole')
]

placements = []
for ii, nn in enumerate(tab_bends_quads.name):
    tscatter_name = f'TScatter.{ii}'
    env.elements[tscatter_name] = xf.TouschekScattering()
    placements.append(env.place(tscatter_name, at=0.0, from_=nn))

tscatter_name = f'TScatter.{ii+1}'
env.elements[tscatter_name] = xf.TouschekScattering()
placements.append(env.place(tscatter_name, at=tab.s[-1]))

line.insert(placements)

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
tab = line.get_table()
touschek_elements = tab.rows.match(element_type='TouschekScattering').name

lma = line.get_local_momentum_acceptance(
    elements=touschek_elements,
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
# Touschek simulation with the line.xfields facade
######################################################
local_momentum_acceptance_scale = 0.85
n_scattering_events = int(5e6)
nturns = 1000

line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

touschek = line.xfields.touschek_configure(
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

print(touschek.local_rates())

result = touschek.run(
    track=True,
    n_turns=nturns,
    keep_particles=True,
    with_progress=1,
)

print(f'Touschek scattering rate: {result.rate_scattering*1e-3:.3f} kHz')
print(f'Touschek tracking rate:   {result.rate_tracking*1e-3:.3f} kHz')
print(f'Scattering lifetime:      {result.lifetime_scattering/60:.2f} min')
print(f'Tracking lifetime:        {result.lifetime_tracking/60:.2f} min')

######################################################
# Optional: refine loss locations
######################################################
loss_loc_refinement = xt.LossLocationRefinement(
    line,
    n_theta=360,
    r_max=0.5,
    dr=50e-6,
    ds=0.1,
)

loss_loc_refinement.refine_loss_location(result.particles)
lost_particles = result.particles.filter(result.particles.state == 0)

######################################################
# Plot: Toy ring Touschek loss map
######################################################
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
