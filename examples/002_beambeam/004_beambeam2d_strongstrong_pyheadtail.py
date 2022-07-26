# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import time

import numpy as np
from matplotlib import pyplot as plt

import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp
from PyHEADTAIL.trackers.transverse_tracking import TransverseSegmentMap

context = xo.ContextCpu(omp_num_threads=0)
xp.enable_pyheadtail_interface()

#################################
# Generate particles            #
#################################

n_macroparticles = int(1e4)
bunch_intensity_b1 = 2.3e11
bunch_intensity_b2 = 2.3e11
physemit_x = 2E-6*0.938/7E3
physemit_y = 2E-6*0.938/7E3
betastar_x = 1.0
betastar_y = 1.0
sigma_z = 0.08
sigma_delta = 1E-4
Qx = 0.31
Qy = 0.32

#Offsets in sigma
mean_x_b1 = 1E-2
mean_y_b1 = 0.0
mean_x_b2 = -1E-2
mean_y_b2 = 0.0

p0c = 7000e9

print('Initialising particles')
particles_b1 = xp.Particles(_context=context,
    p0c=p0c,
    x=np.sqrt(physemit_x*betastar_x)*(np.random.randn(n_macroparticles)+mean_x_b1),
    px=np.sqrt(physemit_x/betastar_x)*np.random.randn(n_macroparticles),
    y=np.sqrt(physemit_y*betastar_y)*(np.random.randn(n_macroparticles)+mean_y_b1),
    py=np.sqrt(physemit_y/betastar_y)*np.random.randn(n_macroparticles),
    zeta=sigma_z*np.random.randn(n_macroparticles),
    delta=sigma_delta*np.random.randn(n_macroparticles),
)

particles_b2 = xp.Particles(_context=context,
    p0c=p0c,
    x=np.sqrt(physemit_x*betastar_x)*(np.random.randn(n_macroparticles)+mean_x_b2),
    px=np.sqrt(physemit_x/betastar_x)*np.random.randn(n_macroparticles),
    y=np.sqrt(physemit_y*betastar_y)*(np.random.randn(n_macroparticles)+mean_y_b2),
    py=np.sqrt(physemit_y/betastar_y)*np.random.randn(n_macroparticles),
    zeta=sigma_z*np.random.randn(n_macroparticles),
    delta=sigma_delta*np.random.randn(n_macroparticles),
)

#############
# Beam-beam #
#############

print('build bb elements...')
bbeamIP1_b1 = xf.BeamBeamBiGaussian2D(
            _context=context,
            n_particles=bunch_intensity_b2,
            q0 = particles_b2.q0,
            beta0=particles_b2.beta0[0],
            sigma_x=1., # dummy
            sigma_y=1., # dummy
            mean_x=1., # dummy
            mean_y=1., # dummy
            min_sigma_diff=1e-10)

bbeamIP1_b2 = xf.BeamBeamBiGaussian2D(
            _context=context,
            n_particles=bunch_intensity_b1,
            q0 = particles_b1.q0,
            beta0=particles_b1.beta0[0],
            sigma_x=1., # dummy
            sigma_y=1., # dummy
            mean_x=1., # dummy
            mean_y=1., # dummy
            min_sigma_diff=1e-10)

bbeamIP2_b1 = xf.BeamBeamBiGaussian2D(
            _context=context,
            n_particles=bunch_intensity_b2,
            q0 = particles_b2.q0,
            beta0=particles_b2.beta0[0],
            sigma_x=1., # dummy
            sigma_y=1., # dummy
            mean_x=1., # dummy
            mean_y=1., # dummy
            min_sigma_diff=1e-10)

bbeamIP2_b2 = xf.BeamBeamBiGaussian2D(
            _context=context,
            n_particles=bunch_intensity_b1,
            q0 = particles_b1.q0,
            beta0=particles_b1.beta0[0],
            sigma_x=1., # dummy
            sigma_y=1., # dummy
            mean_x=1., # dummy
            mean_y=1., # dummy
            min_sigma_diff=1e-10)

#################################################################
# arcs (here they are all the same with half the phase advance) #
#################################################################

arc12_b1 = TransverseSegmentMap(alpha_x_s0 = 0.0, beta_x_s0 = betastar_x, D_x_s0 = 0.0,
                           alpha_x_s1 = 0.0, beta_x_s1 = betastar_x, D_x_s1 = 0.0,
                           alpha_y_s0 = 0.0, beta_y_s0 = betastar_y, D_y_s0 = 0.0,
                           alpha_y_s1 = 0.0, beta_y_s1 = betastar_y, D_y_s1 = 0.0,
                           dQ_x = Qx/2, dQ_y=Qy/2)
arc21_b1 = TransverseSegmentMap(alpha_x_s0 = 0.0, beta_x_s0 = betastar_x, D_x_s0 = 0.0,
                           alpha_x_s1 = 0.0, beta_x_s1 = betastar_x, D_x_s1 = 0.0,
                           alpha_y_s0 = 0.0, beta_y_s0 = betastar_y, D_y_s0 = 0.0,
                           alpha_y_s1 = 0.0, beta_y_s1 = betastar_y, D_y_s1 = 0.0,
                           dQ_x = Qx/2, dQ_y=Qy/2)
arc12_b2 = TransverseSegmentMap(alpha_x_s0 = 0.0, beta_x_s0 = betastar_x, D_x_s0 = 0.0,
                           alpha_x_s1 = 0.0, beta_x_s1 = betastar_x, D_x_s1 = 0.0,
                           alpha_y_s0 = 0.0, beta_y_s0 = betastar_y, D_y_s0 = 0.0,
                           alpha_y_s1 = 0.0, beta_y_s1 = betastar_y, D_y_s1 = 0.0,
                           dQ_x = Qx/2, dQ_y=Qy/2)
arc21_b2 = TransverseSegmentMap(alpha_x_s0 = 0.0, beta_x_s0 = betastar_x, D_x_s0 = 0.0,
                           alpha_x_s1 = 0.0, beta_x_s1 = betastar_x, D_x_s1 = 0.0,
                           alpha_y_s0 = 0.0, beta_y_s0 = betastar_y, D_y_s0 = 0.0,
                           alpha_y_s1 = 0.0, beta_y_s1 = betastar_y, D_y_s1 = 0.0,
                           dQ_x = Qx/2, dQ_y=Qy/2)

#################################################################
# Tracking                                                      #
#################################################################
print('Track...')
nTurn = 1024
positions_x_b1 = np.zeros(nTurn,dtype=float)
positions_y_b1 = np.zeros(nTurn,dtype=float)
positions_x_b2 = np.zeros(nTurn,dtype=float)
positions_y_b2 = np.zeros(nTurn,dtype=float)
for turn in range(nTurn):
    time0 = time.time()
    # Measure beam properties at IP1
    mean_x_meas_b1, sigma_x_meas_b1 = xf.mean_and_std(particles_b1.x)
    mean_y_meas_b1, sigma_y_meas_b1 = xf.mean_and_std(particles_b1.y)
    mean_x_meas_b2, sigma_x_meas_b2 = xf.mean_and_std(particles_b2.x)
    mean_y_meas_b2, sigma_y_meas_b2 = xf.mean_and_std(particles_b2.y)
    #Record positions for post-processing
    positions_x_b1[turn] = mean_x_meas_b1
    positions_y_b1[turn] = mean_y_meas_b1
    positions_x_b2[turn] = mean_x_meas_b2
    positions_y_b2[turn] = mean_y_meas_b2
    # Update bb lens with measured properties of the other beam
    bbeamIP1_b1.sigma_x = sigma_x_meas_b2
    bbeamIP1_b1.mean_x = mean_x_meas_b2
    bbeamIP1_b1.sigma_y = sigma_y_meas_b2
    bbeamIP1_b1.mean_y = mean_y_meas_b2
    bbeamIP1_b2.sigma_x = sigma_x_meas_b1
    bbeamIP1_b2.mean_x = mean_x_meas_b1
    bbeamIP1_b2.sigma_y = sigma_y_meas_b1
    bbeamIP1_b2.mean_y = mean_y_meas_b1
    #track beam-beam at IP1
    bbeamIP1_b1.track(particles_b1)
    bbeamIP1_b2.track(particles_b2)
    #track both bunches from IP1 to IP2
    arc12_b1.track(particles_b1)
    arc12_b2.track(particles_b2)
    # Measure beam properties at IP2
    mean_x_meas_b1, sigma_x_meas_b1 = xf.mean_and_std(particles_b1.x)
    mean_y_meas_b1, sigma_y_meas_b1 = xf.mean_and_std(particles_b1.y)
    mean_x_meas_b2, sigma_x_meas_b2 = xf.mean_and_std(particles_b2.x)
    mean_y_meas_b2, sigma_y_meas_b2 = xf.mean_and_std(particles_b2.y)
    # Update bb lens with measured properties of the other beam
    bbeamIP2_b1.sigma_x = sigma_x_meas_b2
    bbeamIP2_b1.mean_x = mean_x_meas_b2
    bbeamIP2_b1.sigma_y = sigma_y_meas_b2
    bbeamIP2_b1.mean_y = mean_y_meas_b2
    bbeamIP2_b2.sigma_x = sigma_x_meas_b1
    bbeamIP2_b2.mean_x = mean_x_meas_b1
    bbeamIP2_b2.sigma_y = sigma_y_meas_b1
    bbeamIP2_b2.mean_y = mean_y_meas_b1
    #track beam-beam at IP2
    bbeamIP2_b1.track(particles_b1)
    bbeamIP2_b2.track(particles_b2)
    #track both bunches from IP2 to IP1
    arc21_b1.track(particles_b1)
    arc21_b2.track(particles_b2)
    if turn%100 == 0:
        print(f'Time for turn {turn}: {time.time()-time0}s')

#################################################################
# Post-processing: raw data and spectrum                        #
#################################################################

plt.figure(0)
plt.plot(np.arange(nTurn),positions_x_b1/np.sqrt(physemit_x*betastar_x),'x')
plt.plot(np.arange(nTurn),positions_y_b1/np.sqrt(physemit_y*betastar_y),'x')
plt.figure(1)
freqs = np.fft.fftshift(np.fft.fftfreq(nTurn))
mask = freqs > 0
myFFT = np.fft.fftshift(np.fft.fft(positions_x_b1))
plt.semilogy(freqs[mask], (np.abs(myFFT[mask])))
myFFT = np.fft.fftshift(np.fft.fft(positions_y_b1))
plt.semilogy(freqs[mask], (np.abs(myFFT[mask])))
plt.show()



