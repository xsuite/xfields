# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
from scipy.integrate import trapezoid

from xfields import LongitudinalProfileQGaussian
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_qgauss(test_context):
    z0 = 0.1
    sigma_z = 0.2
    npart = 1e11

    for qq in [0, 0.5, 0.95, 1.05, 1.3]:
        lprofile = LongitudinalProfileQGaussian(
                _context=test_context,
                number_of_particles=npart,
                sigma_z=sigma_z,
                z0=z0,
                q_parameter=qq)

        z = np.linspace(-10., 10., 10000)
        z_dev = test_context.nparray_to_context_array(z)
        lden_dev = lprofile.line_density(z_dev)
        lden = test_context.nparray_from_context_array(lden_dev)

        area = trapezoid(lden, z)
        z_mean = trapezoid(lden*z/area, z)
        z_std = np.sqrt(trapezoid(lden*(z-z_mean)**2/area, z))
        assert np.isclose(area, npart)
        assert np.isclose(z_mean, z0)
        assert np.isclose(z_std, sigma_z)

@for_all_test_contexts
def test_qgauss_derivative(test_context):
    z0 = 0.1
    sigma_z = 0.2
    npart = 1e11

    for qq in [0, 0.5, 0.95, 1.05, 1.3]:
        lprofile = LongitudinalProfileQGaussian(
                _context=test_context,
                number_of_particles=npart,
                sigma_z=sigma_z,
                z0=z0,
                q_parameter=qq)

        # Select range relevant for derivative continuity
        if qq < 1:
            z = np.linspace(-0.347, 0.5, 1_000_000)
        else:
            z = np.linspace(-3., 3., 1_000_000)
        z_dev = test_context.nparray_to_context_array(z)
        lden_dev = lprofile.line_density(z_dev)
        lden = test_context.nparray_from_context_array(lden_dev)

        lderivative_dev = lprofile.line_derivative(z_dev)
        lderivative = test_context.nparray_from_context_array(lderivative_dev)

        numerical_derivative = np.gradient(lden, z)

        assert np.isclose(lderivative, numerical_derivative, rtol=1e-5, atol=1e3).all()
