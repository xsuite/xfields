# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp

import numpy as np
import matplotlib.pyplot as plt


outer_radius = 3.e-3
inner_radius = 1.5e-3

x_center = 0.
y_center = 0.

x_range = (-1.e-2, 1.e-2)
y_range = (-1.1e-2, 1.1e-2)
nx = 201
ny = 301
x_grid = np.linspace(x_range[0], x_range[1], nx)
y_grid = np.linspace(y_range[0], y_range[1], ny)
dx=x_grid[1] - x_grid[0]
dy=y_grid[1] - y_grid[0]


X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
rho = np.zeros_like(X)
rho[:] = 0
R = np.sqrt( (X - x_center)**2 + (Y - y_center)**2)
rho[ (R > inner_radius) & (R < outer_radius) ] = 1.
#rho[ X < 0 ] = 0
norm_rho = np.sum(rho[:,:])*dx*dy
rho[:] /= norm_rho

elens = xf.ElectronLensInterpolated(current=1, length=1, voltage=15e3, 
                                    x_grid=x_grid, y_grid=y_grid, rho=rho)

elens_ideal = xt.Elens(current=1, elens_length=1, voltage=15e3, 
                                    inner_radius=inner_radius, outer_radius=outer_radius)

npart = 10000
part = xp.Particles(x=np.linspace(-1.e-2, 1.e-2, npart),
                    y=[y_center], zeta=[0], p0c=450e9
                   )

part_ideal = xp.Particles(x=np.linspace(-1.e-2, 1.e-2, npart),
                    y=[0], zeta=[0], p0c=450e9
                   )

elens.track(part)
elens_ideal.track(part_ideal)

plt.figure(1)
plt.plot(part.x*1000., part.px, 'r.')
plt.plot(part_ideal.x*1000., part_ideal.px, 'k.')


plt.figure(2)
plt.pcolormesh(X, Y, rho)
plt.show()
# fieldmap = xf.TriLinearInterpolatedFieldMap(x_range=x_range, y_range=y_range, 
#                             z_range=z_range, nx=nx, ny=ny, nz=FFTSolver2p5D")
# 
# 
# X, Y = np.meshgrid(fieldmap.x_grid, fieldmap.y_grid, indexing="ij")
# 
# fieldmap.rho[:] = 0
# R = np.sqrt( (X - x_center)**2 + (Y - y_center)**2)
# fieldmap.rho[ (R > inner_r) & (R < outer_r) ] = 1.
# 
# norm_rho = np.sum(fieldmap.rho[:,:,0])*fieldmap.dx*fieldmap.dy
# fieldmap.rho[:] /= norm_rho
# 
# fieldmap.update_phi_from_rho()
# 
# x_particles = np.linspace(x_range[0]*1.05, x_range[1]*1.05, 10000)
# y_particles = 0*x_particles + y_center
# z_particles = 0*x_particles 
# 
# rho, phi, dphi_dx, dphi_dy, dphi_dz = fieldmap.get_values_at_points(x=x_particles, y=y_particles, z=z_particles)
# 
# plt.pcolormesh(fieldmap.x_grid, fieldmap.y_grid, fieldmap.rho[:, :, 0].T)
# plt.show()
