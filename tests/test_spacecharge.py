
import numpy as np

from pysixtrack.particles import Particles
import xobjects as xo

def test_spacecharge():
    for solver in ['FFTSolver2p5D', 'FFTSolver3D']:
        for CTX in xo.ContextCpu, xo.ContextPyopencl, xo.ContextCupy:
            if CTX not in xo.context.available:
                continue

            context = CTX()

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
            mass = Particles.pmass,
            theta_probes = 30 * np.pi/180
            r_max_probes = 2e-2
            z_probes = 1.2*sigma_z
            n_probes = 1000

            from xfields.test_support.temp_makepart import generate_particles_object
            (particles, r_probes, x_probes,
                    y_probes, z_probes) = generate_particles_object(context,
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

            ######################
            # Space charge (PIC) #
            ######################

            x_lim = 4.*sigma_x
            y_lim = 4.*sigma_y
            z_lim = 4.*sigma_z

            from xfields import SpaceCharge3D

            spcharge = SpaceCharge3D(
                    length=1, update_on_track=True, apply_z_kick=False,
                    x_range=(-x_lim, x_lim),
                    y_range=(-y_lim, y_lim),
                    z_range=(-z_lim, z_lim),
                    nx=128, ny=128, nz=25,
                    solver=solver,
                    gamma0=particles.gamma0,
                    context=context)

            spcharge.track(particles)

            ##############################
            # Compare against pysixtrack #
            ##############################

            p2np = context.nparray_from_context_array

            from pysixtrack.elements import SCQGaussProfile
            scpyst = SCQGaussProfile(
                    number_of_particles = bunch_intensity,
                    bunchlength_rms=sigma_z,
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                    length=spcharge.length,
                    x_co=0.,
                    y_co=0.)

            p_pyst = Particles(p0c=p0c,
                    mass=mass,
                    x=x_probes.copy(),
                    y=y_probes.copy(),
                    zeta=z_probes.copy())

            scpyst.track(p_pyst)

            mask_inside_grid = ((np.abs(x_probes)<0.9*x_lim) &
                                (np.abs(y_probes)<0.9*y_lim))

            assert np.allclose(
                    p2np(particles.px[:n_probes])[mask_inside_grid],
                    p_pyst.px[mask_inside_grid])
            assert np.allclose(
                    p2np(particles.py[:n_probes])[mask_inside_grid],
                    p_pyst.py[mask_inside_grid])
