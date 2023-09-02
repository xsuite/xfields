# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
from matplotlib import pyplot as plt
import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp

context = xo.ContextCpu(omp_num_threads=0)

# Machine and beam parameters
n_macroparticles = int(1e4)
bunch_intensity = 2.3e11
nemitt_x = 2E-6
nemitt_y = 2E-6
p0c = 7e12
gamma = p0c/xp.PROTON_MASS_EV
physemit_x = nemitt_x/gamma
physemit_y = nemitt_y/gamma
beta_x = 1.0
beta_y = 1.0
sigma_z = 0.08
sigma_delta = 1E-4
beta_s = sigma_z/sigma_delta
Qx = 0.31
Qy = 0.32
Qs = 2.1E-3

##############################
# Plot the beam-beam force   #
##############################

# Particles algined on the x axis
particles = xp.Particles(_context=context,
    p0c=p0c,
    x=np.sqrt(physemit_x*beta_x)*np.linspace(-6,6,n_macroparticles),
    px=np.zeros(n_macroparticles),
    y=np.zeros(n_macroparticles),
    py=np.zeros(n_macroparticles),
    zeta=np.zeros(n_macroparticles),
    delta=np.zeros(n_macroparticles),
)
# Definition of the beam-beam force based on the other beam's
# properties (here assumed identical to the tracked beam)
# Note that the arguments 'Sigma' are sigma**2 
bbeam = xf.BeamBeamBiGaussian2D(
            _context=context,
            other_beam_q0 = particles.q0,
            other_beam_beta0 = particles.beta0[0],
            other_beam_num_particles = bunch_intensity,
            other_beam_Sigma_11 = physemit_x*beta_x,
            other_beam_Sigma_33 = physemit_y*beta_y)

# Build line and tracker with only the beam-beam element
elements = [bbeam]
line = xt.Line(elements=elements)
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7e12)
line.build_tracker()
# track on turn
line.track(particles,num_turns=1)
#plot the resulting change of px
plt.figure()
plt.plot(particles.x/np.sqrt(physemit_x*beta_x),particles.px)
plt.xlabel(r'x [$\sigma_x$]')
plt.ylabel(r'$\Delta p_x$')

##############################
# Tune footprint             #
##############################

# Build linear lattice
arc = xt.LineSegmentMap(
        betx = beta_x,bety = beta_y,
        qx = Qx, qy = Qy,bets = beta_s, qs=Qs)
# Build line and tracker with a beam-beam element and linear lattice
elements = [bbeam,arc]
line = xt.Line(elements=elements)
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7e12)
line.build_tracker()
# plot footprint
plt.figure()
fp0 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y)
fp0.plot(color='k')

##############################
# Phase space distorion      #
##############################

# Initialise particles randomly in 6D phase space
particles = xp.Particles(_context=context,
    p0c=p0c,
    x=np.sqrt(physemit_x*beta_x)*(np.random.randn(n_macroparticles)),
    px=np.sqrt(physemit_x/beta_x)*np.random.randn(n_macroparticles),
    y=np.sqrt(physemit_y*beta_y)*(np.random.randn(n_macroparticles)),
    py=np.sqrt(physemit_y/beta_y)*np.random.randn(n_macroparticles),
    zeta=sigma_z*np.random.randn(n_macroparticles),
    delta=sigma_delta*np.random.randn(n_macroparticles),
)

# Change the tune to better visualise the 4th order resonance
arc.qx=0.255
# Track
line.track(particles,num_turns=10000)
# Plot phase space
plt.figure()
plt.plot(particles.x/np.sqrt(physemit_x*beta_x),particles.px/np.sqrt(physemit_x/beta_x),'.')
plt.xlabel('$x$ [$\sigma_x$]')
plt.ylabel('$p_x$ [$\sigma_{px}$]')
plt.show()


