# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
from xfields import LongitudinalProfileQGaussian
import xobjects as xo
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 15,
        "figure.titlesize": 20,
    }
)

# Longitudinal example parameters
z0 = 0.1
sigma_z = 0.2
npart = 1e11

test_context = xo.ContextCpu()
#test_context = xo.ContextCupy()

for qq in [0.0, 0.5, 0.95, 1.05, 1.3]:
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

    print('q = {:.2f} max relative difference anal. vs num. : {:.3e}'.format(qq, max((lderivative - numerical_derivative) / numerical_derivative)))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(z, lderivative, label='Analytical derivative')
    ax.plot(z, numerical_derivative, ls='--', label='Numerical derivative')
    ax.text(0.83, 0.92, 'q={:.2f}'.format(qq), color='k', fontsize=15.5, transform=ax.transAxes)
    ax.set_ylabel('Function value')
    ax.set_xlabel('Zeta [m]')
    ax.legend(loc='lower left')
    plt.show()