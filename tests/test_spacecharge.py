# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
import pytest

from scipy.constants import m_p as pmass_kg
from scipy.constants import e as qe
from scipy.constants import c as clight

import xpart as xp

import ducktrack as dtk

import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts


pmass = pmass_kg*clight**2/qe


@pytest.mark.parametrize('frozen', [True, False])
@for_all_test_contexts
def test_spacecharge_gauss_qgauss(frozen, test_context):
    n_macroparticles = int(1e6)
    bunch_intensity = 2.5e11
    sigma_x = 3e-3
    sigma_y = 2e-3
    sigma_z = 30e-2
    x0 = 1e-3
    y0 = -4e-3
    p0c = 25.92e9
    mass = pmass
    theta_probes = 30 * np.pi/180
    r_max_probes = 2e-2
    z_probes = 1.2*sigma_z
    n_probes = 1000

    from xfields.test_support.temp_makepart import generate_particles_object
    (particles_dtk, r_probes, _, _, _) = generate_particles_object(
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
            _context=test_context, **particles_dtk.to_dict())

    particles.x += x0
    particles.y += y0

    ################
    # Space charge #
    ################

    from xfields import LongitudinalProfileQGaussian
    lprofile = LongitudinalProfileQGaussian(
            _context=test_context,
            number_of_particles=bunch_intensity,
            sigma_z=sigma_z,
            z0=0.,
            q_parameter=1. # there is a bug in ducktrack,
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
                    _context=test_context,
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

    #############################
    # Compare against ducktrack #
    #############################

    p2np = test_context.nparray_from_context_array
    x_probes = p2np(particles.x[:n_probes])
    y_probes = p2np(particles.y[:n_probes])
    z_probes = p2np(particles.zeta[:n_probes])

    scdtk = dtk.SCQGaussProfile(
            number_of_particles = bunch_intensity,
            bunchlength_rms=sigma_z,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            length=scgauss.length,
            q_parameter=scgauss.longitudinal_profile.q_parameter,
            x_co=x0,
            y_co=y0)

    p_dtk = dtk.TestParticles(p0c=p0c,
            mass=mass,
            x=x_probes.copy(),
            y=y_probes.copy(),
            zeta=z_probes.copy())

    scdtk.track(p_dtk)

    assert np.allclose(
            p2np(particles.px[:n_probes]),
            p_dtk.px,
            atol={True:1e-7, False:1e2}[frozen]
              * np.max(np.abs(p_dtk.px)))
    assert np.allclose(
            p2np(particles.py[:n_probes]),
            p_dtk.py,
            atol={True:1e-7, False:1e2}[frozen]
              * np.max(np.abs(p_dtk.py)))


@pytest.mark.parametrize('solver',
        ['FFTSolver2p5D', 'FFTSolver2p5DAveraged', 'FFTSolver3D'])
@for_all_test_contexts
def test_spacecharge_pic(solver, test_context):

    if (isinstance(test_context, xo.ContextPyopencl)
                and solver=='FFTSolver2p5DAveraged'):
        pytest.skip('Not implemented')

    #################################
    # Generate particles and probes #
    #################################

    n_macroparticles = int(5e6)
    bunch_intensity = 2.5e11
    sigma_x = 3e-3
    sigma_y = 2e-3
    sigma_z = 30e-2
    p0c = 25.92e9
    mass = pmass
    theta_probes = 30 * np.pi/180
    r_max_probes = 2e-2
    z_probes = 1.2*sigma_z
    n_probes = 1000

    from xfields.test_support.temp_makepart import generate_particles_object
    (particles_gen, r_probes, x_probes,
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
    # Transfer particles to context
    particles = xp.Particles(
            _context=test_context, **particles_gen.to_dict())

    ######################
    # Space charge (PIC) #
    ######################

    x_lim = 4.*sigma_x
    y_lim = 4.*sigma_y
    z_lim = 4.*sigma_z

    from xfields import SpaceCharge3D

    spcharge = SpaceCharge3D(
            _context=test_context,
            length=1, update_on_track=True, apply_z_kick=False,
            x_range=(-x_lim, x_lim),
            y_range=(-y_lim, y_lim),
            z_range=(-z_lim, z_lim),
            nx=128, ny=128, nz=25,
            solver=solver,
            gamma0=particles_gen.gamma0[0],
            )

    spcharge.track(particles)

    #############################
    # Compare against ducktrack #
    #############################

    p2np = test_context.nparray_from_context_array

    scdtk = dtk.SCQGaussProfile(
            number_of_particles = bunch_intensity,
            bunchlength_rms=sigma_z,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            length=spcharge.length,
            x_co=0.,
            y_co=0.)

    p_dtk = dtk.TestParticles(p0c=p0c,
            mass=mass,
            x=x_probes.copy(),
            y=y_probes.copy(),
            zeta=z_probes.copy())

    scdtk.track(p_dtk)

    mask_inside_grid = ((np.abs(x_probes)<0.9*x_lim) &
                        (np.abs(y_probes)<0.9*y_lim))

    assert np.allclose(
            p2np(particles.px[:n_probes])[mask_inside_grid],
            p_dtk.px[mask_inside_grid],
            atol=3e-2*np.max(np.abs(p_dtk.px[mask_inside_grid])))
    assert np.allclose(
            p2np(particles.py[:n_probes])[mask_inside_grid],
            p_dtk.py[mask_inside_grid],
            atol=3e-2*np.max(np.abs(p_dtk.py[mask_inside_grid])))
