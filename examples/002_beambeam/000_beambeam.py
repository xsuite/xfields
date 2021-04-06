import time

import numpy as np

from pysixtrack.particles import Particles
from xobjects.context import ContextCpu, ContextCupy, ContextPyopencl

###################
# Choose context #
###################

#context = ContextCpu(omp_num_threads=0) # no omp
#context = ContextCpu(omp_num_threads=1) # omp
context = ContextCpu(omp_num_threads=48) # omp
context = ContextCupy(default_block_size=256)
#context = ContextPyopencl('0.0')

print(repr(context))

#################################
# Generate particles and probes #
#################################

n_macroparticles = int(1e6)

bunch_intensity_b1 = 2.5e11
sigma_x_b1 = 3e-3
sigma_y_b1 = 2e-3
mean_x_b1 = 1.3e-3
mean_y_b1 = -1.2e-3

bunch_intensity_b2 = 3e11
sigma_x_b2 = 1.7e-3
sigma_y_b2 = 2.1e-3
mean_x_b2 = -1e-3
mean_y_b2 = 1.4e-3

sigma_z = 30e-2
p0c = 25.92e9
mass = Particles.pmass,
theta_probes = 30 * np.pi/180
r_max_probes = 2e-2
z_probes = 1.2*sigma_z
n_probes = 1000

from xfields.test_support.temp_makepart import generate_particles_object
(particles_b1, r_probes, _, _, _
        ) =  generate_particles_object(context,
                            n_macroparticles,
                            bunch_intensity_b1,
                            sigma_x_b1,
                            sigma_y_b1,
                            sigma_z,
                            p0c,
                            mass,
                            n_probes,
                            r_max_probes,
                            z_probes,
                            theta_probes)
particles_b1.x += mean_x_b1
particles_b1.y += mean_y_b1

(particles_b2, r_probes, _, _, _
        ) =  generate_particles_object(context,
                            n_macroparticles,
                            bunch_intensity_b2,
                            sigma_x_b2,
                            sigma_y_b2,
                            sigma_z,
                            p0c,
                            mass,
                            n_probes,
                            r_max_probes,
                            z_probes,
                            theta_probes)

particles_b2.x += mean_x_b2
particles_b2.y += mean_y_b2

from xfields import BeamBeamBiGaussian2D

bbeam_b1 = BeamBeamBiGaussian2D(
            context=context,
            n_particles=bunch_intensity_b2,
            q0 = particles_b2.q0,
            beta0=particles_b2.beta0,
            sigma_x=sigma_x_b2,
            sigma_y=sigma_y_b2,
            mean_x=mean_x_b2,
            mean_y=mean_y_b2,
            min_sigma_diff=1e-10)

bbeam_b1.track(particles_b1)

##############################
# Compare against pysixtrack #
##############################

p2np = context.nparray_from_context_array
x_probes = p2np(particles_b1.x[:n_probes])
y_probes = p2np(particles_b1.y[:n_probes])
z_probes = p2np(particles_b1.zeta[:n_probes])

from pysixtrack.elements import BeamBeam4D
bb_b1_pyst= BeamBeam4D(
        charge = bunch_intensity_b2,
        sigma_x=sigma_x_b2,
        sigma_y=sigma_y_b2,
        x_bb=mean_x_b2,
        y_bb=mean_y_b2,
        beta_r=np.float64(particles_b2.beta0))

p_pyst = Particles(p0c=p0c,
        mass=mass,
        x=x_probes.copy(),
        y=y_probes.copy(),
        zeta=z_probes.copy())

bb_b1_pyst.track(p_pyst)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure()
plt.subplot(211)
plt.plot(r_probes, p_pyst.px, color='red')
plt.plot(r_probes, p2np(particles_b1.px[:n_probes]), color='blue',
        linestyle='--')
plt.subplot(212)
plt.plot(r_probes, p_pyst.py, color='red')
plt.plot(r_probes, p2np(particles_b1.py[:n_probes]), color='blue',
        linestyle='--')

plt.show()
