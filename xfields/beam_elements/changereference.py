from scipy.constants import e as qe
import xobjects as xo
import xtrack as xt
import numpy as np

from ..general import _pkg_root


class ChangeReference(xt.BeamElement):

    # xofields element member vars
    _xofields = {
        'delta_x': xo.Float64,
        'delta_y': xo.Float64,
        'delta_px': xo.Float64,
        'delta_py': xo.Float64,
        'x_CO': xo.Float64,
        'px_CO': xo.Float64,
        'y_CO': xo.Float64,
        'py_CO': xo.Float64,
        'z_CO': xo.Float64,
        'delta_CO': xo.Float64,
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
srcs.append('#define NOFIELDMAP') #TODO Remove this workaound
srcs.append(_pkg_root.joinpath('beam_elements/beambeam_src/changereference.h'))

ChangeReference.XoStruct.extra_sources = srcs
