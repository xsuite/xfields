from scipy.constants import e as qe
import xobjects as xo
import xtrack as xt
import xpart as xp
import numpy as np

from ..general import _pkg_root

"""
18/08/21: add slicing by index
25/08/21: add beamstrahlung flag
22/12/21: add sbc6d element to merge transport and kick
25/05/22: add record data
"""


class Sbc6D_fullRecord(xo.DressedStruct):
    _xofields = {
        '_index': xt.RecordIndex,
        'generated_rr': xo.Float64[:],
        'at_element': xo.Int64[:],
        'at_turn': xo.Int64[:],
        'particle_id': xo.Int64[:]
        }

class Sbc6D_full(xt.BeamElement):

    # xofields element member vars
    _xofields = {
        'min_sigma_diff': xo.Float64,
        'threshold_singular': xo.Float64,
        'q0_bb': xo.Float64,
        'n_bb'     : xo.Float64[:], 
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
        'var_z'    : xo.Float64[:],
        'timestep': xo.Int64, 
        'n_slices': xo.Int64, 
        'do_beamstrahlung': xo.Int64,
        'dz': xo.Float64[:],  # slice z widths for beamstrahlung
        'n_macroparts_bb': xo.Float64[:],
     }

    _internal_record_class = Sbc6D_fullRecord

    def __init__(self,
            _context=None,
            _buffer=None,
            _offset=None,
            _xobject=None,
            **kwargs,
 	    ): # not the same as above the xobject vars, only if set below

        # this is none when i first init, but then gets built
        if _xobject is not None:
               self.xoinitialize(_xobject=_xobject)
               return  

        assert "n_slices" in kwargs.keys(), "First provide ´n_slices´" 
        n_slices = kwargs["n_slices"]
        assert n_slices is not None, "´n_slices´ should be a positive integer"

        # array malloc happens here, context handled here
        self.xoinitialize(
                 _context        = _context,
                 _buffer         = _buffer,
                 _offset         = _offset,
                 mean_x          = n_slices, 
                 mean_xp         = n_slices, 
                 mean_y          = n_slices, 
                 mean_yp         = n_slices, 
                 mean_z          = n_slices, 
                 var_x           = n_slices, 
                 cov_x_xp        = n_slices, 
                 cov_x_y         = n_slices, 
                 cov_x_yp        = n_slices, 
                 var_xp          = n_slices, 
                 cov_xp_y        = n_slices, 
                 cov_xp_yp       = n_slices, 
                 var_y           = n_slices, 
                 cov_y_yp        = n_slices, 
                 var_yp          = n_slices, 
                 var_z           = n_slices, 
                 dz              = n_slices, 
                 n_bb            = n_slices, 
                 n_macroparts_bb = n_slices, 
)
       
        self.mean_x             = kwargs.get("mean_x"         , np.zeros(n_slices))
        self.mean_xp            = kwargs.get("mean_xp"        , np.zeros(n_slices))
        self.mean_y             = kwargs.get("mean_y"         , np.zeros(n_slices))
        self.mean_yp            = kwargs.get("mean_yp"        , np.zeros(n_slices))
        self.mean_z             = kwargs.get("mean_z"         , np.zeros(n_slices))
        self.var_x              = kwargs.get("var_x"          , np.zeros(n_slices))
        self.cov_x_xp           = kwargs.get("cov_x_xp"       , np.zeros(n_slices))
        self.cov_x_y            = kwargs.get("cov_x_y"        , np.zeros(n_slices))
        self.cov_x_yp           = kwargs.get("cov_x_yp"       , np.zeros(n_slices))
        self.var_xp             = kwargs.get("var_xp"         , np.zeros(n_slices))
        self.cov_xp_y           = kwargs.get("cov_xp_y"       , np.zeros(n_slices))
        self.cov_xp_yp          = kwargs.get("cov_xp_yp"      , np.zeros(n_slices)) 
        self.var_y              = kwargs.get("var_y"          , np.zeros(n_slices))
        self.cov_y_yp           = kwargs.get("cov_y_yp"       , np.zeros(n_slices))
        self.var_yp             = kwargs.get("var_yp"         , np.zeros(n_slices))
        self.var_z              = kwargs.get("var_z"          , np.zeros(n_slices))
        self.dz                 = kwargs.get("dz"             , np.zeros(n_slices))
        self.n_bb               = kwargs.get("n_bb"           , np.zeros(n_slices))
        self.n_macroparts_bb    = kwargs.get("n_macroparts_bb", np.zeros(n_slices))    

 
        self.min_sigma_diff     = kwargs.get("min_sigma_diff", 1e-10)    
        self.threshold_singular = kwargs.get("threshold_singular", 1e-28)
        self.use_strongstrong   = kwargs.get("use_strongstrong", 0)
        self.q0_bb              = kwargs.get("q0_bb", 1)
        self.timestep           = kwargs.get("timestep", 0)          
        self.n_slices           = kwargs.get("n_slices", 1)         
        self.do_beamstrahlung   = kwargs.get("do_beamstrahlung", 0) 
   
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

# beamstrahlung / record data
srcs.append(xp.general._pkg_root.joinpath('random_number_generator/rng_src/base_rng.h'))
srcs.append(xp.general._pkg_root.joinpath('random_number_generator/rng_src/local_particle_rng.h'))
srcs.append(_pkg_root.joinpath('headers/beamstrahlung_spectrum.h'))

# this has to be the last so that all functions are declared before a call happens
srcs.append(_pkg_root.joinpath('beam_elements/beambeam_src/sbc6d_full.h'))

Sbc6D_full.XoStruct.extra_sources = srcs
