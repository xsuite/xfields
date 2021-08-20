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

class BoostInv3D(xt.BeamElement):
    # C getters will be generated with these names
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
        'Dx_sub': xo.Float64,
        'Dpx_sub': xo.Float64,
        'Dy_sub': xo.Float64,
        'Dpy_sub': xo.Float64,
        'Dz_sub': xo.Float64,
        'Ddelta_sub': xo.Float64,
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
            Dx_sub=0.,
            Dpx_sub=0.,
            Dy_sub=0.,
            Dpy_sub=0.,
            Dz_sub=0.,
            Ddelta_sub=0.,
	    change_to_CO=0):
 
        if _context is None:
            _context = context_default

        self.xoinitialize(
                 _context=_context,
                 _buffer=_buffer,
                 _offset=_offset)

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
        self.Dx_sub = Dx_sub
        self.Dpx_sub = Dpx_sub
        self.Dy_sub = Dy_sub
        self.Dpy_sub= Dpy_sub
        self.Dz_sub = Dz_sub
        self.Ddelta_sub = self.Ddelta_sub
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
srcs.append(_pkg_root.joinpath('beam_elements/beambeam_src/boostinv3d.h'))

BoostInv3D.XoStruct.extra_sources = srcs
