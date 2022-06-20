# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np

from xpart import Particles

def generate_particles_object(
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
                            theta_probes):

    x_part = sigma_x * np.random.normal(size=(n_macroparticles,))
    y_part = sigma_y * np.random.normal(size=(n_macroparticles,))
    z_part = sigma_z * np.random.normal(size=(n_macroparticles,))
    weights_part = 0*x_part + bunch_intensity/n_macroparticles

    # insert probes
    r_probes= np.linspace(-r_max_probes, r_max_probes, n_probes)
    x_probes = r_probes * np.cos(theta_probes)
    y_probes = r_probes * np.sin(theta_probes)
    z_probes = 0 * x_probes +z_probes

    x_part = np.concatenate([x_probes, x_part])
    y_part = np.concatenate([y_probes, y_part])
    z_part = np.concatenate([z_probes, z_part])
    weights_part = np.concatenate([0*x_probes, weights_part])


    particles = Particles(
            p0c=p0c,
            mass0 = mass,
            x=x_part,
            y=y_part,
            zeta=z_part,
            )
    particles.weight = weights_part

    return particles, r_probes, x_probes, y_probes, z_probes

