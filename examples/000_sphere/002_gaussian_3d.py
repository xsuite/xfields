# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import time

import numpy as np
from numpy.random import rand
from numpy import pi

from xobjects import ContextCpu, ContextCupy, ContextPyopencl
from xfields import TriLinearInterpolatedFieldMap


sigma_x = 0.001
sigma_y = 0.002
sigma_z = 0.1

x_lim = (-3*sigma_x, 3*sigma_x)
y_lim = (-3*sigma_y, 3*sigma_y)
z_lim = (-3*sigma_z, 3*sigma_z)

nx = 102
ny = 104
nz = 106

# Build fieldmap object
fmap = TriLinearInterpolatedFieldMap(
        x_range=x_lim, nx=nx,
        y_range=y_lim, ny=ny,
        z_range=z_lim, nz=nz,
        solver='FFTSolver3D')

X, Y, Z = np.meshgrid(
        fmap.x_grid, fmap.y_grid, fmap.z_grid,
        indexing='ij')

# Fill rho with Gaussian
fmap.update_rho(1/(2*pi*sigma_x*sigma_y*sigma_z)*np.exp(
    X**2/(-2*sigma_x**2) + Y**2/(-2*sigma_y**2) + Z**2/(-2*sigma_z**2)))

phi = fmap.solver.solve(fmap.rho)
fmap.update_phi(phi)

x_list = np.linspace(x_lim[0], x_lim[1], 11)
z_plot = np.linspace(z_lim[0], z_lim[1], 1000)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)

for x in x_list:

    x_plot = x * np.ones_like(z_plot)
    y_plot = np.zeros_like(z_plot)

    rho, phi, dphi_dx, dphi_dy, dphi_dz = fmap.get_values_at_points(x_plot, y_plot, z_plot)
    plt.plot(z_plot, dphi_dz, label=f'x = {x}')

plt.legend()
plt.xlabel('z [m]')
plt.ylabel('dphi/dz [V/m]')


z_list = np.linspace(z_lim[0], z_lim[1], 11)
x_plot = np.linspace(x_lim[0], x_lim[1], 1000)

plt.figure(2)

for z in z_list:

    z_plot = z * np.ones_like(x_plot)
    y_plot = np.zeros_like(x_plot)

    rho, phi, dphi_dx, dphi_dy, dphi_dz = fmap.get_values_at_points(x_plot, y_plot, z_plot)
    plt.plot(x_plot, dphi_dz, label=f'z = {z}')

plt.legend()
plt.xlabel('x [m]')
plt.ylabel('dphi/dz [V/m]')

plt.show()
