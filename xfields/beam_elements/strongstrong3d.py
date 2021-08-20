from scipy.constants import e as qe
import xobjects as xo
import xtrack as xt
import numpy as np

from ..general import _pkg_root

"""
18/08/21: add slicing by index
"""

# covariance matrix is symmetric
class Sigmas(xo.Struct):
    Sig_11 = xo.Float64
    Sig_12 = xo.Float64
    Sig_13 = xo.Float64
    Sig_14 = xo.Float64
    Sig_22 = xo.Float64
    Sig_23 = xo.Float64
    Sig_24 = xo.Float64
    Sig_33 = xo.Float64
    Sig_34 = xo.Float64
    Sig_44 = xo.Float64


class StrongStrong3D(xt.BeamElement):

    # xofields element member vars
    _xofields = {
        'q0_bb': xo.Float64,
        'n_macroparts_bb': xo.Float64,     
        'min_sigma_diff': xo.Float64,
        'threshold_singular': xo.Float64,
        'sigma_matrix_ip': Sigmas, # pass these as a dict
        'sigma_matrix_cp': Sigmas,
        'is_sliced': xo.Int64,
        'slice_id': xo.Int64,
     }

    def update(self, **kwargs):
        for kk in kwargs.keys():
            if not hasattr(self, kk):
                raise NameError(f'Unknown parameter: {kk}')
            setattr(self, kk, kwargs[kk])

srcs = []
srcs.append(_pkg_root.joinpath('headers/constants.h'))
srcs.append(_pkg_root.joinpath('fieldmaps/bigaussian_src/complex_error_function.h'))
srcs.append('#define NOFIELDMAP') #TODO Remove this workaound
srcs.append(_pkg_root.joinpath('fieldmaps/bigaussian_src/bigaussian.h'))
srcs.append(Sigmas._gen_c_api()[0]) #TODO This shouldnt be needed
srcs.append(_pkg_root.joinpath('beam_elements/beambeam_src/strongstrong3d.h'))

StrongStrong3D.XoStruct.extra_sources = srcs
