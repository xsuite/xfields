import numpy as np

from xline.particles import Particles

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

    px_part = 0*x_part
    py_part = 0*x_part
    pt_part = 0*x_part

    pyst_particles = Particles(
            p0c=p0c,
            mass = mass,
            x=x_part,
            y=y_part,
            zeta=z_part,
            px=px_part,
            py=py_part,
            ptau=pt_part)
    pyst_particles.weight = weights_part

    return pyst_particles, r_probes, x_probes, y_probes, z_probes

