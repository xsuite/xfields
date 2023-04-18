# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np

import xpart as xp

import ducktrack as dtk

from xobjects.test_helpers import for_all_test_contexts

from scipy.constants import m_p as pmass_kg
from scipy.constants import e as qe
from scipy.constants import c as clight
pmass = pmass_kg*clight**2/qe


@for_all_test_contexts
def test_beambeam(test_context):
    #################################
    # Generate particles and probes #
    #################################

    n_macroparticles_b1 = int(1e6)
    bunch_intensity_b1 = 2.5e11
    sigma_x_b1 = 3e-3
    sigma_y_b1 = 2e-3
    mean_x_b1 = 1.3e-3
    mean_y_b1 = -1.2e-3

    n_macroparticles_b2 = int(1e6)
    bunch_intensity_b2 = 3e11
    sigma_x_b2 = 1.7e-3
    sigma_y_b2 = 2.1e-3
    mean_x_b2 = -1e-3
    mean_y_b2 = 1.4e-3

    sigma_z = 30e-2
    p0c = 25.92e9
    mass = pmass
    theta_probes = 30 * np.pi/180
    r_max_probes = 2e-2
    z_probes = 1.2*sigma_z
    n_probes = 1000

    from xfields.test_support.temp_makepart import generate_particles_object
    (particles_b1, r_probes, _, _, _
            ) =  generate_particles_object(
                                n_macroparticles_b1,
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
    # Move to right context
    particles_b1 = xp.Particles(_context=test_context,
                                **particles_b1.to_dict())
    particles_b1.x += mean_x_b1
    particles_b1.y += mean_y_b1

    (particles_b2, r_probes, _, _, _
            ) =  generate_particles_object(
                                n_macroparticles_b2,
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
    particles_b2 = xp.Particles(_context=test_context,
                                **particles_b2.to_dict())
    particles_b2.x += mean_x_b2
    particles_b2.y += mean_y_b2

    #############
    # Beam-beam #
    #############

    from xfields import BeamBeamBiGaussian2D, mean_and_std

    # if beta0 is array I just take the first
    beta0_b2 = test_context.nparray_from_context_array(particles_b2.beta0)[0]
    beta0_b1 = test_context.nparray_from_context_array(particles_b1.beta0)[0]

    bbeam_b1 = BeamBeamBiGaussian2D(
                _context=test_context,
                n_particles=bunch_intensity_b2,
                q0 = particles_b2.q0,
                beta0=beta0_b2,
                sigma_x=1., # dummy
                sigma_y=1., # dummy
                mean_x=1.,  # dummy
                mean_y=1.,  # dummy
                min_sigma_diff=1e-10)

    # Measure beam properties
    mean_x_meas, sigma_x_meas = mean_and_std(particles_b2.x)
    mean_y_meas, sigma_y_meas = mean_and_std(particles_b2.y)
    # Update bb lens
    bbeam_b1.sigma_x = sigma_x_meas
    bbeam_b1.sigma_y = sigma_y_meas
    bbeam_b1.mean_x = mean_x_meas
    bbeam_b1.mean_y = mean_y_meas
    #Track
    bbeam_b1.track(particles_b1)

    #############################
    # Compare against ducktrack #
    #############################

    p2np = test_context.nparray_from_context_array
    x_probes = p2np(particles_b1.x[:n_probes])
    y_probes = p2np(particles_b1.y[:n_probes])
    z_probes = p2np(particles_b1.zeta[:n_probes])

    from ducktrack.elements import BeamBeam4D
    bb_b1_dtk= BeamBeam4D(
            charge = bunch_intensity_b2,
            sigma_x=sigma_x_b2,
            sigma_y=sigma_y_b2,
            x_bb=mean_x_b2,
            y_bb=mean_y_b2,
            beta_r=np.float64(beta0_b2))

    p_dtk = dtk.TestParticles(p0c=p0c,
            mass=mass,
            x=x_probes.copy(),
            y=y_probes.copy(),
            zeta=z_probes.copy())

    bb_b1_dtk.track(p_dtk)

    assert np.allclose(p_dtk.px,
        p2np(particles_b1.px[:n_probes]),
        atol=2e-2*np.max(np.abs(p_dtk.px)))
    assert np.allclose(p_dtk.py,
        p2np(particles_b1.py[:n_probes]),
        atol=2e-2*np.max(np.abs(p_dtk.py)))


    # Some test on scale_strength
    bbeam_b1.scale_strength = 0
    p_before = particles_b1.copy()

    bbeam_b1.track(particles_b1)

    assert np.allclose(p2np(p_before.px), p2np(particles_b1.px), atol=1e-14)
    assert np.allclose(p2np(p_before.py), p2np(particles_b1.py), atol=1e-14)
