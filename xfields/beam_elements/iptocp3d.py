from scipy.constants import e as qe
import xobjects as xo
import xtrack as xt

from ..general import _pkg_root

"""
18/08/21: add slicing by index
02/09/21: add z_centroid and z_bb_centroid (COMBI)
06/09/2021: add xy_full_bb_centroid (since slice and beam centroids are different)
"""

class IPToCP3D(xt.BeamElement):

    _xofields = {
        'x_bb_centroid': xo.Float64,
        'px_bb_centroid': xo.Float64,
        'y_bb_centroid': xo.Float64,
        'py_bb_centroid': xo.Float64,
        'z_bb_centroid': xo.Float64,
        'z_centroid': xo.Float64,
        'x_full_bb_centroid': xo.Float64,
        'y_full_bb_centroid': xo.Float64,
        'is_sliced': xo.Int64,
        'slice_id': xo.Int64,
        'use_strongstrong': xo.UInt8, 
    }

    def update(self, **kwargs):
        for kk in kwargs.keys():
            if not hasattr(self, kk):
                raise NameError(f'Unknown parameter: {kk}')
            setattr(self, kk, kwargs[kk])


srcs = []
srcs.append(_pkg_root.joinpath('headers/constants.h'))
srcs.append('#define NOFIELDMAP') #TODO Remove this workaound
srcs.append(_pkg_root.joinpath('beam_elements/beambeam_src/iptocp3d.h'))

IPToCP3D.XoStruct.extra_sources = srcs
