from scipy.constants import e as qe
import xobjects as xo
import xtrack as xt
import numpy as np

from ..general import _pkg_root

class BoostParameters(xo.Struct):
    sphi = xo.Float64
    cphi = xo.Float64
    tphi = xo.Float64
    salpha = xo.Float64
    calpha = xo.Float64

class Boost3D(xt.BeamElement):

    # xofields element member vars
    _xofields = {
        'boost_parameters': BoostParameters,
	'alpha': xo.Float64,
	'phi': xo.Float64,
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
        'change_to_CO': xo.UInt8
    }

    def __init__(self,
            _context=None,
            _buffer=None,
            _offset=None,
	    alpha=0.,
            phi=0.,
            delta_x=0.,
            delta_y=0.,
            delta_px=0.,
            delta_py=0.,
            x_CO=0.,
            px_CO=0.,
            y_CO=0.,
            py_CO=0.,
            z_CO=0.,
            delta_CO=0.,
 	    change_to_CO=0): # not the same as above the xobject vars, only if set below

        if _context is None:
            _context = context_default

        self.xoinitialize(
                 _context=_context,
                 _buffer=_buffer,
                 _offset=_offset)

        # element member variables, input arg. can be named anyhow, the member var. name matters
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_px = delta_px
        self.delta_py = delta_py
        self.x_CO = x_CO
        self.px_CO = px_CO
        self.y_CO = y_CO
        self.py_CO = py_CO
        self.z_CO = z_CO
        self.delta_CO = delta_CO
        self.change_to_CO = change_to_CO
        self.boost_parameters = {
                'sphi': np.sin(phi),
                'cphi': np.cos(phi),
                'tphi': np.tan(phi),
                'salpha': np.sin(alpha),
                'calpha': np.cos(alpha)
                }
    
srcs = []
srcs.append(_pkg_root.joinpath('headers/constants.h'))
srcs.append('#define NOFIELDMAP') #TODO Remove this workaound
srcs.append(BoostParameters._gen_c_api()[0]) #TODO This shouldnt be needed
srcs.append(_pkg_root.joinpath('beam_elements/beambeam_src/boost3d.h'))

Boost3D.XoStruct.extra_sources = srcs
