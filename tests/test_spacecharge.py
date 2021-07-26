import numpy as np

from xline.particles import Particles
import xobjects as xo
import xtrack as xt

def test_spacecharge_gauss_qgauss():
    for frozen in [True, False]:
        for CTX in xo.ContextCpu, xo.ContextPyopencl, xo.ContextCupy:
            if CTX not in xo.context.available:
                continue

            context = CTX()
            print(context)

            #################################
            # Generate particles and probes #
            #################################

            n_macroparticles = int(1e6)
            bunch_intensity = 2.5e11
            sigma_x = 3e-3
            sigma_y = 2e-3
            sigma_z = 30e-2
            x0 = 1e-3
            y0 = -4e-3
            p0c = 25.92e9
            mass = Particles.pmass,
            theta_probes = 30 * np.pi/180
            r_max_probes = 2e-2
            z_probes = 1.2*sigma_z
            n_probes = 1000

            from xfields.test_support.temp_makepart import generate_particles_object
            (particles_pyst, r_probes, _, _, _) = generate_particles_object(
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
            part_dict = xt.pyparticles_to_xtrack_dict(particles_pyst)
            particles = xt.Particles(
                    _context=context, **part_dict)

            particles.x += x0
            particles.y += y0

            ################
            # Space charge #
            ################

            from xfields import LongitudinalProfileQGaussian
            lprofile = LongitudinalProfileQGaussian(
                    _context=context,
                    number_of_particles=bunch_intensity,
                    sigma_z=sigma_z,
                    z0=0.,
                    q_parameter=1. # there is a bug in xline, 
                                   # only q=1 can be tested 
                    )

            from xfields import SpaceChargeBiGaussian
            # Just not to fool myself in the test
            if frozen:
                x0_init = x0
                y0_init = y0
                sx_init = sigma_x
                sy_init = sigma_y
            else:
                x0_init = None
                y0_init = None
                sx_init = None
                sy_init = None

            scgauss = SpaceChargeBiGaussian(
                            _context=context,
                            update_on_track=not(frozen),
                            length=1.,
                            apply_z_kick=False,
                            longitudinal_profile=lprofile,
                            mean_x=x0_init,
                            mean_y=y0_init,
                            sigma_x=sx_init,
                            sigma_y=sy_init,
                            min_sigma_diff=1e-10)

            scgauss.track(particles)

            ##############################
            # Compare against xline #
            ##############################

            p2np = context.nparray_from_context_array
            x_probes = p2np(particles.x[:n_probes])
            y_probes = p2np(particles.y[:n_probes])
            z_probes = p2np(particles.zeta[:n_probes])

            from xline.elements import SCQGaussProfile
            scpyst = SCQGaussProfile(
                    number_of_particles = bunch_intensity,
                    bunchlength_rms=sigma_z,
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                    length=scgauss.length,
                    q_parameter=scgauss.longitudinal_profile.q_parameter,
                    x_co=x0,
                    y_co=y0)

            p_pyst = Particles(p0c=p0c,
                    mass=mass,
                    x=x_probes.copy(),
                    y=y_probes.copy(),
                    zeta=z_probes.copy())

            scpyst.track(p_pyst)

            assert np.allclose(
                    p2np(particles.px[:n_probes]),
                    p_pyst.px,
                    atol={True:1e-7, False:1e2}[frozen]
                      * np.max(np.abs(p_pyst.px)))
            assert np.allclose(
                    p2np(particles.py[:n_probes]),
                    p_pyst.py,
                    atol={True:1e-7, False:1e2}[frozen]
                      * np.max(np.abs(p_pyst.py)))

# TODO: re-enable when switch for longitudinal profiles will be introduced
#def test_spacecharge_gauss_coast():
#    for frozen in [True, False]:
#        for CTX in xo.ContextCpu, xo.ContextPyopencl, xo.ContextCupy:
#            if CTX not in xo.context.available:
#                continue
#
#            context = CTX()
#
#            #################################
#            # Generate particles and probes #
#            #################################
#
#            n_macroparticles = int(1e6)
#            bunch_intensity = 2.5e11
#            sigma_x = 3e-3
#            sigma_y = 2e-3
#            sigma_z = 30e-2
#            x0 = 1e-3
#            y0 = -4e-3
#            p0c = 25.92e9
#            mass = Particles.pmass,
#            theta_probes = 30 * np.pi/180
#            r_max_probes = 2e-2
#            z_probes = 1.2*sigma_z
#            n_probes = 1000
#
#            from xfields.test_support.temp_makepart import generate_particles_object
#            (particles, r_probes, _, _, _) = generate_particles_object(
#                                        context,
#                                        n_macroparticles,
#                                        bunch_intensity,
#                                        sigma_x,
#                                        sigma_y,
#                                        sigma_z,
#                                        p0c,
#                                        mass,
#                                        n_probes,
#                                        r_max_probes,
#                                        z_probes,
#                                        theta_probes)
#
#            particles.x += x0
#            particles.y += y0
#
#            ################
#            # Space charge #
#            ################
#
#            beam_line_density = 1e13
#            from xfields import LongitudinalProfileCoasting
#            lprofile = LongitudinalProfileCoasting(
#                            context=context,
#                            beam_line_density=beam_line_density)
#
#            from xfields import SpaceChargeBiGaussian
#            # Just not to fool myself in the test
#            if frozen:
#                x0_init = x0
#                y0_init = y0
#                sx_init = sigma_x
#                sy_init = sigma_y
#            else:
#                x0_init = None
#                y0_init = None
#                sx_init = None
#                sy_init = None
#
#            scgauss = SpaceChargeBiGaussian(
#                            context=context,
#                            update_on_track=not(frozen),
#                            length=1.,
#                            apply_z_kick=False,
#                            longitudinal_profile=lprofile,
#                            mean_x=x0_init,
#                            mean_y=y0_init,
#                            sigma_x=sx_init,
#                            sigma_y=sy_init,
#                            min_sigma_diff=1e-10)
#
#            scgauss.track(particles)
#
#            ##############################
#            # Compare against xline #
#            ##############################
#
#            p2np = context.nparray_from_context_array
#            x_probes = p2np(particles.x[:n_probes])
#            y_probes = p2np(particles.y[:n_probes])
#            z_probes = p2np(particles.zeta[:n_probes])
#
#            from xline.elements import SCCoasting
#            scpyst = SCCoasting(
#                    number_of_particles = beam_line_density,
#                    circumference = 1.,
#                    sigma_x=sigma_x,
#                    sigma_y=sigma_y,
#                    length=scgauss.length,
#                    x_co=x0,
#                    y_co=y0)
#
#            p_pyst = Particles(p0c=p0c,
#                    mass=mass,
#                    x=x_probes.copy(),
#                    y=y_probes.copy(),
#                    zeta=z_probes.copy())
#
#            scpyst.track(p_pyst)
#
#
#            assert np.allclose(
#                    p2np(particles.px[:n_probes]),
#                    p_pyst.px, atol=1e-2*np.max(np.abs(p_pyst.px)))
#            assert np.allclose(
#                    p2np(particles.py[:n_probes]),
#                    p_pyst.py, atol=1e-2*np.max(np.abs(p_pyst.py)))


def test_spacecharge_pic():
    for solver in ['FFTSolver2p5D', 'FFTSolver3D']:
        for CTX in xo.ContextCpu, xo.ContextPyopencl, xo.ContextCupy:
            if CTX not in xo.context.available:
                continue

            context = CTX()

            print(repr(context))

            #################################
            # Generate particles and probes #
            #################################

            n_macroparticles = int(5e6)
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
            (particles_pyst, r_probes, x_probes,
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
            part_dict = xt.pyparticles_to_xtrack_dict(particles_pyst)
            particles = xt.Particles(
                    _context=context, **part_dict)

            ######################
            # Space charge (PIC) #
            ######################

            x_lim = 4.*sigma_x
            y_lim = 4.*sigma_y
            z_lim = 4.*sigma_z

            from xfields import SpaceCharge3D

            spcharge = SpaceCharge3D(
                    _context=context,
                    length=1, update_on_track=True, apply_z_kick=False,
                    x_range=(-x_lim, x_lim),
                    y_range=(-y_lim, y_lim),
                    z_range=(-z_lim, z_lim),
                    nx=128, ny=128, nz=25,
                    solver=solver,
                    gamma0=particles_pyst.gamma0,
                    )

            spcharge.track(particles)

            ##############################
            # Compare against xline #
            ##############################

            p2np = context.nparray_from_context_array

            from xline.elements import SCQGaussProfile
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
                    p_pyst.px[mask_inside_grid],
                    atol=3e-2*np.max(np.abs(p_pyst.px[mask_inside_grid])))
            assert np.allclose(
                    p2np(particles.py[:n_probes])[mask_inside_grid],
                    p_pyst.py[mask_inside_grid],
                    atol=3e-2*np.max(np.abs(p_pyst.py[mask_inside_grid])))
