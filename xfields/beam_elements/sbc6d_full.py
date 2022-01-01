from scipy.constants import e as qe
import xobjects as xo
import xtrack as xt
import numpy as np

from ..general import _pkg_root

"""
18/08/21: add slicing by index
25/08/21: add beamstrahlung flag
22/12/21: add sbc6d element to merge transport and kick
"""

class Sbc6D_full(xt.BeamElement):

    # xofields element member vars
    _xofields = {
        'min_sigma_diff': xo.Float64,
        'threshold_singular': xo.Float64,
        'use_strongstrong': xo.UInt8,
        'verbose_info': xo.Int64,
        'q0_bb': xo.Float64[:],
        'mean_x'   : xo.Float64[:],
        'mean_xp'  : xo.Float64[:],
        'mean_y'   : xo.Float64[:],
        'mean_yp'  : xo.Float64[:],
        'mean_z'   : xo.Float64[:],
        'var_x'    : xo.Float64[:],
        'cov_x_xp' : xo.Float64[:],
        'cov_x_y'  : xo.Float64[:],
        'cov_x_yp' : xo.Float64[:],
        'var_xp'   : xo.Float64[:],
        'cov_xp_y' : xo.Float64[:],
        'cov_xp_yp': xo.Float64[:],
        'var_y'    : xo.Float64[:],  
        'cov_y_yp' : xo.Float64[:],
        'var_yp'   : xo.Float64[:],   
        'x_full_bb_centroid': xo.Float64,
        'y_full_bb_centroid': xo.Float64,
        'timestep': xo.Int64, 
        'n_slices': xo.Int64, 
        'do_beamstrahlung': xo.Int64,
     }
  #  """

    def update(self, **kwargs):
        for kk in kwargs.keys():
            if not hasattr(self, kk):
                raise NameError(f'Unknown parameter: {kk}')
            setattr(self, kk, kwargs[kk])

srcs = []
srcs.append(_pkg_root.joinpath('headers/constants.h'))
srcs.append(_pkg_root.joinpath('headers/power_n.h'))
srcs.append(_pkg_root.joinpath('headers/sincos.h'))
srcs.append(_pkg_root.joinpath('fieldmaps/bigaussian_src/complex_error_function.h'))
srcs.append('#define NOFIELDMAP') #TODO Remove this workaound
srcs.append(_pkg_root.joinpath('fieldmaps/bigaussian_src/bigaussian.h'))
srcs.append(_pkg_root.joinpath('beam_elements/beambeam_src/sbc6d_full.h'))

Sbc6D_full.XoStruct.extra_sources = srcs
