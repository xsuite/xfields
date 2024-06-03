# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import time

import numpy as np

import xpart as xp
import xfields as xf
import xobjects as xo

num_particles = int(10e6)
bunch_intensity = 2.5e11
sigma_x = 3e-3
sigma_y = 2e-3
sigma_z = 30e-2
p0c = 25.92e9

p = xp.Particles(
        p0c=p0c,
        x=np.random.normal(0, sigma_x, num_particles),
        y=np.random.normal(0, sigma_y, num_particles),
        zeta=np.random.normal(0, sigma_z, num_particles),
        )


######################
# Space charge (PIC) #
######################

x_lim = 5.*sigma_x
y_lim = 5.*sigma_y
z_lim = 5.*sigma_z

spch_pic = xf.SpaceCharge3D(
        length=100, update_on_track=True, apply_z_kick=True,
        x_range=(-x_lim, x_lim),
        y_range=(-y_lim, y_lim),
        z_range=(-z_lim, z_lim),
        nx=256, ny=256, nz=100,
        solver='FFTSolver3D',
        gamma0=float(p.gamma0[0]))

spch_pic.track(p)

spch_pic.update_on_track = False

x_test = 1.1 * sigma_x
y_test = -0.4 * sigma_y
z_test = np.linspace(-z_lim, z_lim, 1000)

p_test_pic = xp.Particles(
        p0c=p0c,
        x=x_test,
        y=y_test,
        zeta=z_test,
        )
p_zero_pic = p_test_pic.copy()
p_zero_pic.x = 0
p_zero_pic.y = 0
p_test_frozen = p_test_pic.copy()
spch_pic.track(p_test_pic)
spch_pic.track(p_zero_pic)

spch_frozen = xf.SpaceChargeBiGaussian(
    length=spch_pic.length,
    longitudinal_profile=xf.LongitudinalProfileQGaussian(
        number_of_particles=num_particles,
        sigma_z=sigma_z,
    ),
    sigma_x=sigma_x,
    sigma_y=sigma_y,
    z_kick_num_integ_per_sigma=10,
)
spch_frozen.track(p_test_frozen)

xo.assert_allclose(p_test_pic.px, p_test_frozen.px,
                   rtol=0, atol=0.2*np.max(np.abs(p_test_frozen.px)))
xo.assert_allclose(p_test_pic.py, p_test_frozen.py,
                   rtol=0, atol=0.2*np.max(np.abs(p_test_frozen.py)))
xo.assert_allclose(p_test_pic.pzeta - p_zero_pic.pzeta, p_test_frozen.pzeta,
                   rtol=0, atol=0.2*np.max(np.abs(p_test_frozen.pzeta)))
xo.assert_allclose(p_test_frozen.pzeta.max(), 1.3e-13, rtol=0.2, atol=0)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(z_test, p_test_pic.pzeta - p_zero_pic.pzeta, label='PIC')
plt.plot(z_test, p_test_frozen.pzeta, label='Frozen')
plt.legend()
plt.xlabel('z [m]')
plt.ylabel(r'$\Delta p_z$')

plt.show()