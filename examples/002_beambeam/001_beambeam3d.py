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

params = bb_pyst.to_dict(keepextra=True)
bb6d_data = pysixtrack.BB6Ddata.BB6D_init(
                q_part=qe, # the pysixtrack input has the charge
                           # of the slices in elementary charges 
                phi=params["phi"],
                alpha=params["alpha"],
                delta_x=params["x_bb_co"],
                delta_y=params["y_bb_co"],
                N_part_per_slice=params["charge_slices"],
                z_slices=params["zeta_slices"],
                Sig_11_0=params["sigma_11"],
                Sig_12_0=params["sigma_12"],
                Sig_13_0=params["sigma_13"],
                Sig_14_0=params["sigma_14"],
                Sig_22_0=params["sigma_22"],
                Sig_23_0=params["sigma_23"],
                Sig_24_0=params["sigma_24"],
                Sig_33_0=params["sigma_33"],
                Sig_34_0=params["sigma_34"],
                Sig_44_0=params["sigma_44"],
                x_CO=params["x_co"],
                px_CO=params["px_co"],
                y_CO=params["y_co"],
                py_CO=params["py_co"],
                sigma_CO=params["zeta_co"],
                delta_CO=params["delta_co"],
                min_sigma_diff=params["min_sigma_diff"],
                threshold_singular=params["threshold_singular"],
                Dx_sub=params["d_x"],
                Dpx_sub=params["d_px"],
                Dy_sub=params["d_y"],
                Dpy_sub=params["d_py"],
                Dsigma_sub=params["d_zeta"],
                Ddelta_sub=params["d_delta"],
                enabled=params["enabled"],
            )
assert(
    len(bb6d_data.N_part_per_slice) ==
    len(bb6d_data.x_slices_star) ==
    len(bb6d_data.y_slices_star) ==
    len(bb6d_data.sigma_slices_star))

bb = xf.BeamBeamBiGaussian3D(
    q0 = bb6d_data.q_part,
    boost_parameters = {
        'sphi': bb6d_data.parboost.sphi,
        'cphi': bb6d_data.parboost.cphi,
        'tphi': bb6d_data.parboost.tphi,
        'salpha': bb6d_data.parboost.salpha,
        'calpha': bb6d_data.parboost.calpha},
    Sigmas_0_star = {
        'Sig_11': bb6d_data.Sigmas_0_star.Sig_11_0,
        'Sig_12': bb6d_data.Sigmas_0_star.Sig_12_0,
        'Sig_13': bb6d_data.Sigmas_0_star.Sig_13_0,
        'Sig_14': bb6d_data.Sigmas_0_star.Sig_14_0,
        'Sig_22': bb6d_data.Sigmas_0_star.Sig_22_0,
        'Sig_23': bb6d_data.Sigmas_0_star.Sig_23_0,
        'Sig_24': bb6d_data.Sigmas_0_star.Sig_24_0,
        'Sig_33': bb6d_data.Sigmas_0_star.Sig_33_0,
        'Sig_34': bb6d_data.Sigmas_0_star.Sig_34_0,
        'Sig_44': bb6d_data.Sigmas_0_star.Sig_44_0},
    min_sigma_diff = bb6d_data.min_sigma_diff,
    threshold_singular = bb6d_data.threshold_singular,
    delta_x = bb6d_data.delta_x,
    delta_y = bb6d_data.delta_y,
    x_CO = bb6d_data.x_CO,
    px_CO = bb6d_data.px_CO,
    y_CO = bb6d_data.y_CO,
    py_CO = bb6d_data.py_CO,
    sigma_CO = bb6d_data.sigma_CO,
    delta_CO = bb6d_data.delta_CO,
    Dx_sub = bb6d_data.Dx_sub,
    Dpx_sub = bb6d_data.Dpx_sub,
    Dy_sub = bb6d_data.Dy_sub,
    Dpy_sub = bb6d_data.Dpy_sub,
    Dsigma_sub = bb6d_data.Dsigma_sub,
    Ddelta_sub = bb6d_data.Ddelta_sub,
    num_slices = len(bb6d_data.N_part_per_slice),
    N_part_per_slice = bb6d_data.N_part_per_slice,
    x_slices_star = bb6d_data.x_slices_star,
    y_slices_star = bb6d_data.y_slices_star,
    sigma_slices_star = bb6d_data.sigma_slices_star,
    )
