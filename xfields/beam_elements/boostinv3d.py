from scipy.constants import e as qe
import xobjects as xo
import xtrack as xt
import numpy as np

from ..general import _pkg_root

"""
14/10/2021: add x2_bc for ws case. Remove x_CO and delta_x as closed orbit is not explicit here. "x2_bc = x_CO+delta_x"
28/03/2022: add x_co and delta_x for convenience
- add swap_x as part of this element
"""


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
        'x2_CO': xo.Float64,
        'y2_CO': xo.Float64,
        'px2_CO': xo.Float64,
        'py2_CO': xo.Float64,
        'zeta2_CO': xo.Float64,
        'delta2_CO': xo.Float64,       
        'Dx_sub': xo.Float64,
        'Dpx_sub': xo.Float64,
        'Dy_sub': xo.Float64,
        'Dpy_sub': xo.Float64,
        'Dz_sub': xo.Float64,
        'Ddelta_sub': xo.Float64,
        'delta_x': xo.Float64,
        'delta_y': xo.Float64,
        'swap_x': xo.Int64,
 
    }

    
    def __init__(self,
            _context=None,
            _buffer=None,
            _offset=None,
            _xobject=None,
       	    alpha=0.,
            phi=0., 
            x2_CO=0.,
            y2_CO=0.,
            px2_CO=0.,
            py2_CO=0.,
            zeta2_CO=0.,
            delta2_CO=0.,
            Dx_sub=0.,
            Dpx_sub=0.,
            Dy_sub=0.,
            Dpy_sub=0.,
            Dz_sub=0.,
            Ddelta_sub=0.,
            delta_x=0.,
            delta_y=0.,
            swap_x=0,
	    ):
 
        self.xoinitialize(
                 _context=_context,
                 _buffer=_buffer,
                 _offset=_offset)

        self.x2_CO      = x2_CO
        self.y2_CO      = y2_CO
        self.px2_CO     = px2_CO
        self.py2_CO     = py2_CO
        self.zeta2_CO   = zeta2_CO
        self.delta2_CO  = delta2_CO
        self.Dx_sub     = Dx_sub
        self.Dpx_sub    = Dpx_sub
        self.Dy_sub     = Dy_sub
        self.Dpy_sub    = Dpy_sub
        self.Dz_sub     = Dz_sub
        self.Ddelta_sub = Ddelta_sub
        self.phi        = phi,
        self.alpha      = alpha,
        self.delta_x    = delta_x,
        self.delta_y    = delta_y,
        self.swap_x     = swap_x,
        self.boost_parameters = {
                'sphi': np.sin(phi),
                'cphi': np.cos(phi),
                'tphi': np.tan(phi),
                'salpha': np.sin(alpha),
                'calpha': np.cos(alpha)
                }

    def to_dict(self):
        dct = super().to_dict()
        dct[ "x2_CO"]     = self.x2_CO      
        dct[ "y2_CO"]     = self.y2_CO  
        dct[ "px2_CO"]    = self.px2_CO 
        dct[ "py2_CO"]    = self.py2_CO 
        dct[ "zeta2_CO"]  = self.zeta2_CO
        dct[ "delta2_CO"] = self.delta2_CO
        dct["Dx_sub"]     = self.Dx_sub
        dct["Dpx_sub"]    = self.Dpx_sub
        dct["Dy_sub"]     = self.Dy_sub
        dct["Dpy_sub"]    = self.Dpy_sub
        dct["Dz_sub"]     = self.Dz_sub
        dct["Ddelta_sub"] = self.Ddelta_sub
        dct[ "delta_x"]   = self.delta_x
        dct[ "delta_y"]   = self.delta_y
        dct[ "phi"]       = self.phi   
        dct[ "alpha"]     = self.alpha  
        dct[ "swap_x"]    = self.swap_x 
        return dct

    def update(self, **kwargs):
        for kk in kwargs.keys():
            if not hasattr(self, kk):
                raise NameError(f'Unknown parameter: {kk}')
            setattr(self, kk, kwargs[kk])


 
srcs = []
srcs.append(_pkg_root.joinpath('headers/constants.h'))
srcs.append('#define NOFIELDMAP') #TODO Remove this workaound
srcs.append(BoostParameters._gen_c_api()[0]) #TODO This shouldnt be needed
srcs.append(_pkg_root.joinpath('beam_elements/beambeam_src/boostinv3d.h'))

BoostInv3D.XoStruct.extra_sources = srcs
