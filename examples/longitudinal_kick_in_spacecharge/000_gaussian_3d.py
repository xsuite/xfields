# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import time

import numpy as np
from numpy import pi
from scipy.integrate import cumulative_trapezoid
from xfields import TriLinearInterpolatedFieldMap
from scipy.constants import epsilon_0


sigma_x = 0.001
sigma_y = 0.002
sigma_z = 1.

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
sp_phi = plt.subplot(3, 1, 1)
sp_dphi_dz = plt.subplot(3, 1, 2, sharex=sp_phi)
sp_simpl_corr = plt.subplot(3, 1, 3, sharex=sp_phi)

rho_on_axis, _, _, _, dphi_dz_on_axis = fmap.get_values_at_points(0*z_plot, 0*z_plot, z_plot)

lam = np.sum(fmap.rho, axis=(0, 1)) * fmap.dx * fmap.dy
lam_prime = (lam[2:] - lam[:-2]) / (2*fmap.dz)
lam_prime = np.concatenate([lam_prime[:1], lam_prime, lam_prime[-1:]])

lam0 = np.max(lam)



for x in x_list:

    x_plot = x * np.ones_like(z_plot)
    y_plot = np.zeros_like(z_plot)

    rho, phi, dphi_dx, dphi_dy, dphi_dz = fmap.get_values_at_points(x_plot, y_plot, z_plot)
    sp_dphi_dz.plot(z_plot, dphi_dz, label=f'x = {x}')
    sp_phi.plot(z_plot, phi, label=f'x = {x}')
    sp_simpl_corr.plot(z_plot, dphi_dz-dphi_dz_on_axis, label=f'x = {x}')


plt.legend()
sp_dphi_dz.set_xlabel('z [m]')
sp_dphi_dz.set_ylabel('dphi/dz [V/m]')
sp_phi.set_ylabel('phi [V]')

x_corr = 0.001
y_corr = 0.002

n_integ = 1000
x_integ = np.linspace(0, x_corr, n_integ)
y_integ = np.linspace(0, y_corr, n_integ)
dx_integ = x_integ[1] - x_integ[0]
dy_integ = y_integ[1] - y_integ[0]
# r_integ = np.sqrt(x_integ**2 + y_integ**2)

_, _, dphi_dx_integ, dphi_dy_integ, _ = fmap.get_values_at_points(x_integ, y_integ, 0*x_integ)
# dphi_dr_integ = np.sqrt(dphi_dx_integ**2 + dphi_dy_integ**2) / lam0 # need to go to normalized potentional

phi_corr = np.cumsum(dphi_dx_integ * dx_integ + dphi_dy_integ * dy_integ) / lam0
# phi_corr[-1] *= 0.5
dphi_dz_corr = lam_prime * phi_corr[-1]

_, _, _, _, dphi_dz_check1 = fmap.get_values_at_points(x_corr + 0*fmap.z_grid, y_corr + 0*fmap.z_grid, fmap.z_grid)
_, _, _, _, dphi_dz_check0 = fmap.get_values_at_points(0*fmap.z_grid, 0*fmap.z_grid, fmap.z_grid)
dphi_dz_corr_check = dphi_dz_check1 - dphi_dz_check0

# z_list = np.linspace(z_lim[0], z_lim[1], 11)
# x_plot = np.linspace(x_lim[0], x_lim[1], 1000)

# plt.figure(2)

# for z in z_list:

#     z_plot = z * np.ones_like(x_plot)
#     y_plot = np.zeros_like(x_plot)

#     rho, phi, dphi_dx, dphi_dy, dphi_dz = fmap.get_values_at_points(x_plot, y_plot, z_plot)
#     plt.plot(x_plot, dphi_dz, label=f'z = {z}')

plt.legend()
plt.xlabel('x [m]')
plt.ylabel('dphi/dz [V/m]')

plt.figure(100)
sp_lam = plt.subplot(2, 1, 1)
sp_lam_prime = plt.subplot(2, 1, 2, sharex=sp_lam)

sp_lam.plot(fmap.z_grid, lam, label='lam')
sp_lam_prime.plot(fmap.z_grid, lam_prime, label='lam_prime')

sp_lam.set_ylabel(r'$\lambda$(z)')
sp_lam_prime.set_ylabel(r'$\lambda^{\prime}$(z)')
sp_lam_prime.set_xlabel('z [m]')

plt.figure(101)
sp_dphi_dz_corr = plt.subplot(2, 1, 1, sharex=sp_lam)
sp_dphi_dz_ratio = plt.subplot(2, 1, 2, sharex=sp_lam)
sp_dphi_dz_corr.plot(fmap.z_grid, dphi_dz_corr, label='dphi_dz_corr')
sp_dphi_dz_corr.plot(fmap.z_grid, dphi_dz_corr_check, label='dphi_dz_check')

sp_dphi_dz_ratio.plot(fmap.z_grid, dphi_dz_corr / dphi_dz_corr_check, label='dphi_dz_corr / dphi_dz_check1')
sp_dphi_dz_ratio.set_ylim(0.8, 1.2)


plt.show()
