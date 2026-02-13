# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
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

bunch_population = 4e9

######################################################
# Build a toy ring
######################################################
lbend = 3
angle = np.pi / 2

lquad = 0.3
k1qf = 0.1
k1qd = 0.7

# Create environment
env = xt.Environment()

# Define the line (toy ring)
line = env.new_line(components=[
    env.new('mqf.1', xt.Quadrupole, length=lquad, k1=k1qf),
    env.new('d1.1',  xt.Drift, length=1),
    env.new('mb1.1', xt.Bend, length=lbend, angle=angle),
    env.new('d2.1',  xt.Drift, length=1),

    env.new('mqd.1', xt.Quadrupole, length=lquad, k1=-k1qd),
    env.new('d3.1',  xt.Drift, length=1),
    env.new('mb2.1', xt.Bend, length=lbend, angle=angle),
    env.new('d4.1',  xt.Drift, length=1),

    env.new('mqf.2', xt.Quadrupole, length=lquad, k1=k1qf),
    env.new('d1.2',  xt.Drift, length=1),
    env.new('mb1.2', xt.Bend, length=lbend, angle=angle),
    env.new('d2.2',  xt.Drift, length=1),

    env.new('mqd.2', xt.Quadrupole, length=lquad, k1=-k1qd),
    env.new('d3.2',  xt.Drift, length=1),
    env.new('mb2.2', xt.Bend, length=lbend, angle=angle),
    env.new('d4.2',  xt.Drift, length=1),
])

# Set the reference particle
line.set_particle_ref('electron', p0c=1e9)

# Configure the bend model
line.configure_bend_model(core='full', edge=None)

######################################################
# Insert Touschek scattering centers
######################################################
# We insert Touschek scattering centers in the middle of each magnet
# to have good coverage of variations of the optical functions
tab = line.get_table()
tab_bends_quads = tab.rows[(tab.element_type == 'Bend') | (tab.element_type == 'Quadrupole')]

# Would be good to have env.new for xf.TouschekScattering
for ii, nn in enumerate(tab_bends_quads.name):
    tscatter_name = f'TScatter_{ii}'
    env.elements[tscatter_name] = xf.TouschekScattering()
    line.insert(tscatter_name, at=0.0, from_=nn)

# The last TouschekScattering element has to be placed at the end of the line
tscatter_name = f'TScatter_{ii+1}'
env.elements[tscatter_name] = xf.TouschekScattering()
line.insert(tscatter_name, at=tab.s[-1])


######################################################
# Install apertures
######################################################
tab = line.get_table()
needs_aperture = np.unique(tab.element_type)[
    ~np.isin(np.unique(tab.element_type), ["", "Drift", "Marker"])
]

aper_size = 0.040 # m

placements = []
for nn, ee in zip(tab.name, tab.element_type):
    if ee not in needs_aperture:
        continue

    env.new(
        f'{nn}_aper_entry', xt.LimitRect,
        min_x=-aper_size, max_x=aper_size,
        min_y=-aper_size, max_y=aper_size
    )
    placements.append(env.place(f'{nn}_aper_entry', at=f'{nn}@start'))

    env.new(
        f'{nn}_aper_exit', xt.LimitRect,
        min_x=-aper_size, max_x=aper_size,
        min_y=-aper_size, max_y=aper_size
    )
    placements.append(env.place(f'{nn}_aper_exit', at=f'{nn}@end'))

line.insert(placements)

######################################################
# Evaluate momentum aperture profile
######################################################
# Norlamized emittance
nemitt_x = 1e-5
nemitt_y = 1e-7

# Evaluate local momentum aperture at the touschek scattering centers
momentum_aperture = line.momentum_aperture(
    # twiss=tw,
    include_type_pattern="TouschekScattering",
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    y_offset=1e-9,
    delta_negative_limit=-0.012,
    delta_positive_limit=0.012,
    delta_step_size=1e-4,
    n_turns=1000,
    method="4d"
)

df_momentum_aperture = momentum_aperture.to_pandas()

######################################################
# Plot
######################################################
plt.plot(momentum_aperture.s, momentum_aperture.deltan*100, c='r')
plt.plot(momentum_aperture.s, momentum_aperture.deltap*100, c='r')
plt.title('Toy ring: local momentum aperture profile')
plt.xlabel('s [m]')
plt.ylabel(r'$\delta$ [%]')
plt.grid()
plt.show()

######################################################
# Touschek simulation
######################################################
# Parameters
momentum_aperture_scale = 0.85 # scaling factor for momentum aperture
n_simulated = 5e6 # number of simulated scattering events with delta > delta_min
nturns = 1000 # number of turns to simulate

touschek_manager = xf.TouschekManager(
    line,
    momentum_aperture=df_momentum_aperture,
    momentum_aperture_scale=momentum_aperture_scale,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    sigma_delta=sigma_delta,
    bunch_population=bunch_population,
    n_simulated=n_simulated,
    nx=3, ny=3, nz=3,
    ignored_portion=0.01,
    seed=1997,
    method='4d'
)

touschek_manager.initialise_touschek()

touschek_elements = tab.rows[tab.element_type == 'TouschekScattering'].name

line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

particles_list = []
for ii in range(len(touschek_elements)):
    element = touschek_elements[ii] # xf.TouschekScattering
    s_start_elem = tab.rows[tab.name == element].s[0]

    # Touschek!
    particles = line[element].scatter()

    # Track!
    print(f"\nTrack particles scattered at {element} at  s = {s_start_elem}")
    line.track(particles, ele_start=element, ele_stop=element, num_turns=nturns, with_progress=1)

    particles_list.append(particles)

particles = xt.Particles.merge(particles_list)

# Refine loss location
loss_loc_refinement = xt.LossLocationRefinement(line,
    n_theta = 360, # Angular resolution in the polygonal approximation of the aperture
    r_max = 0.5, # Maximum transverse aperture in m
    dr = 50e-6, # Transverse loss refinement accuracy [m]
    ds = 0.1, # Longitudinal loss refinement accuracy [m]
    )

loss_loc_refinement.refine_loss_location(particles)

######################################################
# Compute Touschek lifetime
######################################################
# Keep lost particles only
particles = particles.filter(particles.state == 0)
# Compute total Touschek loss rate
loss_rate = sum(particles.weight)
# Compute Touschek lifetime
touschek_lifetime = bunch_population / loss_rate

######################################################
# Plot: Toy ring Touschek loss map
######################################################
circumference = line.get_length()
binwidth = 0.1 # m

plt.title(f'Toy ring Touschek loss map (Touschek lifetime: {touschek_lifetime/60:.2f} min)')
plt.hist(particles.s, bins=np.arange(0, circumference + binwidth, binwidth), weights=particles.weight*1e-3)
plt.xlabel('s [m]')
plt.ylabel('Loss rate [kHz]')
plt.grid()
plt.show()