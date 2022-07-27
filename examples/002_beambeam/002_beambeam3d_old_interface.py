# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
from scipy.constants import e as qe

import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp

import ducktrack as dtk

context = xo.ContextCpu()

# crossing plane
alpha = 0.7

# crossing angle
phi = 0.8

# separations
x_bb_co=5e-3
y_bb_co=-4e-3
charge_slices=np.array([1e16, 2e16, 5e16])
z_slices=np.array([-60., 2., 55.])

# Single particle properties
q_part = qe

# strong beam shape at the IP (decoupled round beam)
(Sig_11_0, Sig_12_0, Sig_13_0,
Sig_14_0, Sig_22_0, Sig_23_0,
Sig_24_0, Sig_33_0, Sig_34_0, Sig_44_0) = (
20e-06,  0.,  0.,
0., 0., 0.,
0., 20e-6, 0., 0.)

# strong beam shape at the IP (decoupled tall beam)
(Sig_11_0, Sig_12_0, Sig_13_0,
Sig_14_0, Sig_22_0, Sig_23_0,
Sig_24_0, Sig_33_0, Sig_34_0, Sig_44_0) = (
20e-06,  0.,  0.,
0., 0., 0.,
0., 40e-6, 0., 0.)

# strong beam shape at the IP (decoupled tall beam)
(Sig_11_0, Sig_12_0, Sig_13_0,
Sig_14_0, Sig_22_0, Sig_23_0,
Sig_24_0, Sig_33_0, Sig_34_0, Sig_44_0) = (
40e-06,  0.,  0.,
0., 0., 0.,
0., 20e-6, 0., 0.)

# strong beam shape at the IP (coupled beam)
(Sig_11_0, Sig_12_0, Sig_13_0,
Sig_14_0, Sig_22_0, Sig_23_0,
Sig_24_0, Sig_33_0, Sig_34_0, Sig_44_0) = (
  8.4282060230000004e-06,  1.8590458800000001e-07,  -3.5512334410000001e-06,
 -3.8254462239999997e-08, 4.101510281e-09, -7.5517657920000006e-08,
 -8.1134615060000002e-10, 1.031446898e-05, 1.177863077e-07, 1.3458251810000001e-09)


x_co = 2e-3
px_co= 1e-6
y_co=-3e-3
py_co=-2e-6
zeta_co=0.01
delta_co=1.2e-3

d_x=1.5e-3
d_px=1.6e-6
d_y=-1.7e-3
d_py=-1.8e-6
d_zeta=0.019
d_delta=3e-4

bb_dtk = dtk.elements.BeamBeam6D(
        phi=phi, alpha=alpha,
        x_bb_co=x_bb_co,
        y_bb_co=y_bb_co,
        charge_slices=charge_slices,
        zeta_slices=z_slices,
        sigma_11=Sig_11_0,
        sigma_12=Sig_12_0,
        sigma_13=Sig_13_0,
        sigma_14=Sig_14_0,
        sigma_22=Sig_22_0,
        sigma_23=Sig_23_0,
        sigma_24=Sig_24_0,
        sigma_33=Sig_33_0,
        sigma_34=Sig_34_0,
        sigma_44=Sig_44_0,
        x_co=x_co,
        px_co=px_co,
        y_co=y_co,
        py_co=py_co,
        zeta_co=zeta_co,
        delta_co=delta_co,
        d_x=d_x,
        d_px=d_px,
        d_y=d_y,
        d_py=d_py,
        d_zeta=d_zeta,
        d_delta=d_delta
        )

bb = xf.BeamBeamBiGaussian3D(old_interface=bb_dtk.to_dict(), _context=context)

dtk_part = dtk.TestParticles(
        p0c=6500e9,
        x=-1.23e-3,
        px = 50e-3,
        y = 2e-3,
        py = 27e-3,
        sigma = 3.,
        delta = 2e-4)

part= xp.Particles(_context=context, **dtk_part.to_dict())

bb.track(part)

print('------------------------')

bb_dtk.track(dtk_part)

for cc in 'x px y py zeta delta'.split():
    val_test = getattr(part, cc)[0]
    val_ref = getattr(dtk_part, cc)
    print('')
    print(f'ducktrack: {cc} = {val_ref:.12e}')
    print(f'xsuite:    {cc} = {val_test:.12e}')
    assert np.isclose(val_test, val_ref, rtol=1e-11, atol=5e-12)

