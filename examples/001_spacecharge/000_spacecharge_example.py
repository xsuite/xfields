# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import time

import numpy as np

from xobjects import ContextCpu, ContextCupy, ContextPyopencl
import xtrack as xt
import xpart as xp

###################
# Choose context #
###################

context = ContextCpu(omp_num_threads=0) # no omp
#context = ContextCpu(omp_num_threads=48) # omp
#context = ContextCupy(default_block_size=256)
#context = ContextPyopencl('0.0')

print(repr(context))

#################################
# Generate particles and probes #
#################################

n_macroparticles = int(1e6)
bunch_intensity = 2.5e11
sigma_x = 3e-3
sigma_y = 2e-3
sigma_z = 30e-2
p0c = 25.92e9
mass = xp.pmass
theta_probes = 30 * np.pi/180
r_max_probes = 2e-2
z_probes = 1.2*sigma_z
n_probes = 1000

from xfields.test_support.temp_makepart import generate_particles_object
(particles_dtk, r_probes, x_probes,
        y_probes, z_probes) = generate_particles_object(
                            n_macroparticles,
                            bunch_intensity,
                            sigma_x,
                            sigma_y,
                            sigma_z,
                            p0c,
                            mass,
                            n_probes,
                            r_max_probes,
                            z_probes,
                            theta_probes)
particles = xp.Particles(
        _context=context, **particles_dtk.to_dict())


######################
# Space charge (PIC) #
######################

x_lim = 5.*sigma_x
y_lim = 5.*sigma_y
z_lim = 5.*sigma_z

from xfields import SpaceCharge3D
spcharge = SpaceCharge3D(
        _context=context,
        length=1, update_on_track=True, apply_z_kick=False,
        x_range=(-x_lim, x_lim),
        y_range=(-y_lim, y_lim),
        z_range=(-z_lim, z_lim),
        nx=256, ny=256, nz=100,
        # solver='FFTSolver2p5D',
        solver='FFTSolver2p5DAveraged',
        gamma0=particles_dtk.gamma0[0])

spcharge.track(particles)


#############################
# Compare against ducktrack #
#############################


p2np = context.nparray_from_context_array

from ducktrack import SCQGaussProfile, TestParticles
scdtk = SCQGaussProfile(
        number_of_particles = bunch_intensity,
        bunchlength_rms=sigma_z,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        length=spcharge.length,
        x_co=0.,
        y_co=0.)

p_dtk = TestParticles(p0c=p0c,
        mass=mass,
        x=x_probes.copy(),
        y=y_probes.copy(),
        zeta=z_probes.copy())

scdtk.track(p_dtk)

mask_inside_grid = ((np.abs(x_probes)<0.9*x_lim) &
                    (np.abs(y_probes)<0.9*y_lim))


import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.subplot(211)
plt.plot(r_probes, p_dtk.px, color='red')
plt.plot(r_probes, p2np(particles.px[:n_probes]), color='blue',
        linestyle='--')
plt.subplot(212)
plt.plot(r_probes, p_dtk.py, color='red')
plt.plot(r_probes, p2np(particles.py[:n_probes]), color='blue',
        linestyle='--')

plt.figure(2)
plt.subplot(111)
tr_profile = spcharge.fieldmap.rho.sum(axis=0)
for ii in range(tr_profile.shape[1]):
    tr_profile[:,ii] = tr_profile[:,ii]/tr_profile[:,ii].max()
tr_profile = context.nparray_from_context_array(tr_profile)
plt.plot(tr_profile)

###########
# Time it #
###########

n_rep = 5

for _ in range(n_rep):
    t1 = time.time()
    spcharge.track(particles)
    context.synchronize()
    t2 = time.time()
    print(f'Time (full PIC): {(t2-t1)*1e3:.2f} ms')

spcharge.update_on_track = False
for _ in range(n_rep):
    t1 = time.time()
    spcharge.track(particles)
    context.synchronize()
    t2 = time.time()
    print(f'Time (interp only): {(t2-t1)*1e3:.2f} ms')

plt.show()
