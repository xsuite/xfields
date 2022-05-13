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
"""

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
  #  """
    """
    def __init__(self,
            _context=None,
            _buffer=None,
            _offset=None,
            min_sigma_diff     = 0.0,
            threshold_singular = 0.0,
            use_strongstrong   = 0,
            verbose_info       = 0,
            q0_bb              =0,# np.array([], dtype=float),
            mean_x             =0,# np.array([], dtype=float),
            mean_xp            =0,# np.array([], dtype=float), 
            mean_y             =0,# np.array([], dtype=float),
            mean_yp            =0,# np.array([], dtype=float),
            mean_z             =0,# np.array([], dtype=float),
            var_x              =0,# np.array([], dtype=float),
            cov_x_xp           =0,# np.array([], dtype=float), 
            cov_x_y            =0,# np.array([], dtype=float),
            cov_x_yp           =0,# np.array([], dtype=float),
            var_xp             =0,# np.array([], dtype=float),
            cov_xp_y           =0,# np.array([], dtype=float),
            cov_xp_yp          =0,# np.array([], dtype=float),
            var_y              =0,# np.array([], dtype=float),  
            cov_y_yp           =0,# np.array([], dtype=float),
            var_yp             =0,# np.array([], dtype=float),   
            x_full_bb_centroid =0,# 0.0,
            y_full_bb_centroid =0,# 0.0,
            timestep           =0,# 0, 
            n_slices           =0,# 1, 
            do_beamstrahlung   =0,# 0,
            dz                 =0,# np.array([], dtype=float),
 	    ): # not the same as above the xobject vars, only if set below

        if _context is None:
            _context = context_default

        self.xoinitialize(
                 _context=_context,
                 _buffer=_buffer,
                 _offset=_offset)
        self.min_sigma_diff     = min_sigma_diff    
        self.threshold_singular = threshold_singular
        self.use_strongstrong   = use_strongstrong  
        self.verbose_info       = verbose_info      
        self.q0_bb              = q0_bb             
        self.mean_x             = mean_x            
        self.mean_xp            = mean_xp           
        self.mean_y             = mean_y            
        self.mean_yp            = mean_yp           
        self.mean_z             = mean_z            
        self.var_x              = var_x             
        self.cov_x_xp           = cov_x_xp          
        self.cov_x_y            = cov_x_y           
        self.cov_x_yp           = cov_x_yp          
        self.var_xp             = var_xp            
        self.cov_xp_y           = cov_xp_y          
        self.cov_xp_yp          = cov_xp_yp         
        self.var_y              = var_y             
        self.cov_y_yp           = cov_y_yp          
        self.var_yp             = var_yp            
        self.x_full_bb_centroid = x_full_bb_centroid
        self.y_full_bb_centroid = y_full_bb_centroid
        self.timestep           = timestep          
        self.n_slices           = n_slices          
        self.do_beamstrahlung   = do_beamstrahlung  
        self.dz                 = dz                
    """

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
# beamstrahlung
srcs.append(xp.general._pkg_root.joinpath('random_number_generator/rng_src/base_rng.h'))
srcs.append(xp.general._pkg_root.joinpath('random_number_generator/rng_src/local_particle_rng.h'))
srcs.append(_pkg_root.joinpath('headers/beamstrahlung_spectrum.h'))

# this has to be the last so that all functions are declared before a call happens
srcs.append(_pkg_root.joinpath('beam_elements/beambeam_src/sbc6d_full.h'))

Sbc6D_full.XoStruct.extra_sources = srcs
