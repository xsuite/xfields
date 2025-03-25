import pathlib
import pytest

import numpy as np
from scipy import constants as cst

import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp

import itertools

from xobjects.test_helpers import for_all_test_contexts

@for_all_test_contexts(excluding="ContextPyopencl")
def test_beambeam3d_lumi_ws_no_config(test_context):

    if isinstance(test_context, xo.ContextCupy):
        import cupy as cp

    print(repr(test_context))

    n_slices = 100
    phi = 0
    alpha = 0
    slicer = xf.TempSlicer(n_slices=100, sigma_z=1e-3, mode="shatilov")

    bbeam = xf.BeamBeamBiGaussian3D(
        other_beam_q0       = -1,
        phi                 = phi,
        alpha               = alpha,
        config_for_update   = None,
        # slice intensity [num. real particles] 101 inferred from length of this
        slices_other_beam_num_particles = slicer.bin_weights * 1e10,
        # unboosted strong beam moments
        slices_other_beam_zeta_center   = slicer.bin_centers,
        slices_other_beam_Sigma_11      = n_slices * [1e-6],
        slices_other_beam_Sigma_22      = n_slices * [1e-9],
        slices_other_beam_Sigma_33      = n_slices * [1e-7],
        slices_other_beam_Sigma_44      = n_slices * [1e-8],
        # only if BS on
        slices_other_beam_zeta_bin_width_star_beamstrahlung = \
        slicer.bin_widths_beamstrahlung / np.cos(phi),  # boosted dz
        # has to be set
        slices_other_beam_Sigma_12      = n_slices * [0],
        slices_other_beam_Sigma_34      = n_slices * [0],
    )

    #########
    # tests #
    #########

    phi_arr = np.hstack([0, np.logspace(-7, 0, 11), 0])
    alpha_arr = np.hstack([np.arange(9)*np.pi/4, 0])
    grid = itertools.product(phi_arr, alpha_arr)

    for phi_i, alpha_i in grid:
        bbeam.phi = phi_i
        bbeam.alpha = alpha_i
        run_asserts(bbeam)

# deferred expr. from beambeam3d.py
def run_asserts(el, rtol=1e-14):
    
    assert np.allclose(np.sin(el.alpha), el.sin_alpha, rtol=rtol)
    assert np.allclose(np.cos(el.alpha), el.cos_alpha, rtol=rtol)
    
    assert np.allclose(np.sin(el.phi),  el.sin_phi, rtol=rtol)
    assert np.allclose(np.cos(el.phi),  el.cos_phi, rtol=rtol)
    assert np.allclose(np.tan(el.phi),  el.tan_phi, rtol=rtol)
    
    (
    x_slices_star,
    px_slices_star,
    y_slices_star,
    py_slices_star,
    zeta_slices_star,
    pzeta_slices_star,
    ) = _python_boost(
        x=el.slices_other_beam_x_center,
        px=el.slices_other_beam_px_center,
        y=el.slices_other_beam_y_center,
        py=el.slices_other_beam_py_center,
        zeta=el.slices_other_beam_zeta_center,
        pzeta=el.slices_other_beam_pzeta_center,
        sphi=el.sin_phi,
        cphi=el.cos_phi,
        tphi=el.tan_phi,
        salpha=el.sin_alpha,
        calpha=el.cos_alpha,
    )

    assert np.allclose(x_slices_star    ,     el.slices_other_beam_x_center_star, rtol=rtol)
    assert np.allclose(px_slices_star   ,    el.slices_other_beam_px_center_star, rtol=rtol)
    assert np.allclose(y_slices_star    ,     el.slices_other_beam_y_center_star, rtol=rtol)
    assert np.allclose(py_slices_star   ,    el.slices_other_beam_py_center_star, rtol=rtol)
    assert np.allclose(zeta_slices_star ,  el.slices_other_beam_zeta_center_star, rtol=rtol)
    assert np.allclose(pzeta_slices_star, el.slices_other_beam_pzeta_center_star, rtol=rtol)
    
    assert np.all(el.slices_other_beam_Sigma_11                      == el.slices_other_beam_Sigma_11_star)
    assert np.all(el.slices_other_beam_Sigma_12 / np.cos(el.phi)     == el.slices_other_beam_Sigma_12_star)
    assert np.all(el.slices_other_beam_Sigma_13                      == el.slices_other_beam_Sigma_13_star)
    assert np.all(el.slices_other_beam_Sigma_14 / np.cos(el.phi)     == el.slices_other_beam_Sigma_14_star)
    assert np.all(el.slices_other_beam_Sigma_22 / np.cos(el.phi)**2  == el.slices_other_beam_Sigma_22_star)
    assert np.all(el.slices_other_beam_Sigma_23 / np.cos(el.phi)     == el.slices_other_beam_Sigma_23_star)
    assert np.all(el.slices_other_beam_Sigma_24 / np.cos(el.phi)**2  == el.slices_other_beam_Sigma_24_star)
    assert np.all(el.slices_other_beam_Sigma_33                      == el.slices_other_beam_Sigma_33_star)
    assert np.all(el.slices_other_beam_Sigma_34  / np.cos(el.phi)    == el.slices_other_beam_Sigma_34_star)
    assert np.all(el.slices_other_beam_Sigma_44  / np.cos(el.phi)**2 == el.slices_other_beam_Sigma_44_star)
    
    assert np.all(el.slices_other_beam_zeta_bin_width_star_beamstrahlung ==\
                  el.slices_other_beam_zeta_bin_width_beamstrahlung/np.cos(el.phi))
    
    
def _python_boost_scalar(x, px, y, py, zeta, pzeta,
                  sphi, cphi, tphi, salpha, calpha):

    h = (
        pzeta
        + 1.0
        - np.sqrt((1.0 + pzeta) * (1.0 + pzeta) - px * px - py * py)
    )

    px_st = px / cphi - h * calpha * tphi / cphi
    py_st = py / cphi - h * salpha * tphi / cphi
    pzeta_st = (
        pzeta - px * calpha * tphi - py * salpha * tphi + h * tphi * tphi
    )

    pz_st = np.sqrt(
        (1.0 + pzeta_st) * (1.0 + pzeta_st) - px_st * px_st - py_st * py_st
    )
    hx_st = px_st / pz_st
    hy_st = py_st / pz_st
    hzeta_st = 1.0 - (pzeta_st + 1) / pz_st

    L11 = 1.0 + hx_st * calpha * sphi
    L12 = hx_st * salpha * sphi
    L13 = calpha * tphi

    L21 = hy_st * calpha * sphi
    L22 = 1.0 + hy_st * salpha * sphi
    L23 = salpha * tphi

    L31 = hzeta_st * calpha * sphi
    L32 = hzeta_st * salpha * sphi
    L33 = 1.0 / cphi

    x_st = L11 * x + L12 * y + L13 * zeta
    y_st = L21 * x + L22 * y + L23 * zeta
    zeta_st = L31 * x + L32 * y + L33 * zeta

    return x_st, px_st, y_st, py_st, zeta_st, pzeta_st

_python_boost = np.vectorize(_python_boost_scalar,
    excluded=("sphi", "cphi", "tphi", "salpha", "calpha"))

