import numpy as np
from scipy.constants import e as qe
import xfields as xf
import pysixtrack

# TODO: change q0 from Coulomb to elementary charges

# crossing plane
alpha = 0.7

# crossing angle
phi = 0.8

# separations
x_bb_co=5e-3
y_bb_co=-4e-3
charge_slices=np.array([1e-14, 2e-14, 5e-14])
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
d_delta=2e-4

bb_pyst = pysixtrack.elements.BeamBeam6D(
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
        d_delta=d_delta)



bb = xf.BeamBeamBiGaussian3D(
        N_part_per_slice=5, x_slices_star=5,
        y_slices_star=5, sigma_slices_star=5)

