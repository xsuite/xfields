# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np

import xobjects as xo
import xtrack as xt

from ..general import _pkg_root

class BeamBeamBiGaussian2D(xt.BeamElement):

    _xofields = {

        'scale_strength': xo.Float64,

        'ref_shift_x': xo.Float64,
        'ref_shift_y': xo.Float64,

        'other_beam_shift_x': xo.Float64,
        'other_beam_shift_y': xo.Float64,

        'post_subtract_px': xo.Float64,
        'post_subtract_py': xo.Float64,

        # TODO this could become other_beam_q0, other_beam_beta0 (to be done also in 6D)
        'other_beam_q0': xo.Float64,
        'other_beam_beta0': xo.Float64,

        'other_beam_num_particles': xo.Float64,

        'other_beam_Sigma_11': xo.Float64,
        'other_beam_Sigma_13': xo.Float64,
        'other_beam_Sigma_33': xo.Float64,

        'min_sigma_diff': xo.Float64,

    }

    _extra_c_sources= [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('headers/sincos.h'),
        _pkg_root.joinpath('headers/power_n.h'),
        _pkg_root.joinpath('headers','particle_states.h'),
        _pkg_root.joinpath('fieldmaps/bigaussian_src/faddeeva.h'),
        _pkg_root.joinpath('fieldmaps/bigaussian_src/bigaussian.h'),
        _pkg_root.joinpath('beam_elements/beambeam_src/beambeam2d.h'),
    ]


    def __init__(self,
                    scale_strength=1.,

                    other_beam_q0=None,
                    other_beam_beta0=None,

                    other_beam_num_particles=None,

                    other_beam_Sigma_11=None,
                    other_beam_Sigma_13=None,
                    other_beam_Sigma_33=None,

                    ref_shift_x=0,
                    ref_shift_y=0,

                    other_beam_shift_x=0,
                    other_beam_shift_y=0,

                    post_subtract_px=0,
                    post_subtract_py=0,

                    min_sigma_diff=1e-10,

                    config_for_update=None,

                    **kwargs):

        if '_xobject' in kwargs.keys():
            self.xoinitialize(**kwargs)
            return

        # Collective mode (pipeline update)
        if config_for_update is not None:
            raise NotImplementedError
            # To be implemented based on 6d implementation

        params = self._handle_init_old_interface(kwargs)

        self.xoinitialize(**kwargs)

        if self.iscollective:
            if not isinstance(self._buffer.context, xo.ContextCpu):
                raise NotImplementedError(
                    'BeamBeamBiGaussian3D only works with CPU context for now')

        # Handle old interface
        if 'other_beam_num_particles' in params.keys(): other_beam_num_particles = params['other_beam_num_particles']
        if 'other_beam_q0' in params.keys(): other_beam_q0 = params['other_beam_q0']
        if 'other_beam_beta0' in params.keys(): other_beam_beta0 = params['other_beam_beta0']
        if 'other_beam_shift_x' in params.keys(): other_beam_shift_x = params['other_beam_shift_x']
        if 'other_beam_shift_y' in params.keys(): other_beam_shift_y = params['other_beam_shift_y']
        if 'other_beam_Sigma_11' in params.keys(): other_beam_Sigma_11 = params['other_beam_Sigma_11']
        if 'other_beam_Sigma_33' in params.keys(): other_beam_Sigma_33 = params['other_beam_Sigma_33']
        if 'post_subtract_px' in params.keys(): post_subtract_px = params['post_subtract_px']
        if 'post_subtract_py' in params.keys(): post_subtract_py = params['post_subtract_py']

        
        # Mandatory sigmas
        assert other_beam_Sigma_11 is not None, ("`other_beam_Sigma_11` must be provided")
        assert other_beam_Sigma_33 is not None, ("`other_beam_Sigma_33` must be provided")

        # Coupling between transverse planes
        if other_beam_Sigma_13 is None:
            other_beam_Sigma_13 = 0

        if np.abs(other_beam_Sigma_13) > 0:
            raise NotImplementedError("Coupled case not tested yet.")

        assert other_beam_num_particles is not None, ("`other_beam_num_particles` must be provided")
        self.other_beam_num_particles = other_beam_num_particles

        self.other_beam_Sigma_11 = other_beam_Sigma_11
        self.other_beam_Sigma_13 = other_beam_Sigma_13
        self.other_beam_Sigma_33 = other_beam_Sigma_33

        assert other_beam_q0 is not None
        self.other_beam_q0 = other_beam_q0

        assert other_beam_beta0 is not None
        self.other_beam_beta0 = other_beam_beta0

        self.ref_shift_x = ref_shift_x
        self.ref_shift_y = ref_shift_y

        self.other_beam_shift_x = other_beam_shift_x
        self.other_beam_shift_y = other_beam_shift_y

        self.post_subtract_px = post_subtract_px
        self.post_subtract_py = post_subtract_py

        self.min_sigma_diff = min_sigma_diff

        self.scale_strength = scale_strength

    def _handle_init_old_interface(self, kwargs):

        params = {}

        if 'n_particles' in kwargs.keys():
            params['other_beam_num_particles'] = kwargs['n_particles']
            del kwargs['n_particles']

        if 'q0' in kwargs.keys():
            params['other_beam_q0'] = kwargs['q0']
            del kwargs['q0']

        if 'beta0' in kwargs.keys():
            params['other_beam_beta0'] = kwargs['beta0']
            del kwargs['beta0']

        if 'mean_x' in kwargs.keys():
            params['other_beam_shift_x'] = kwargs['mean_x']
            del kwargs['mean_x']

        if 'mean_y' in kwargs.keys():
            params['other_beam_shift_y'] = kwargs['mean_y']
            del kwargs['mean_y']

        if 'sigma_x' in kwargs.keys():
            params['other_beam_Sigma_11'] = kwargs['sigma_x']**2
            del kwargs['sigma_x']

        if 'sigma_y' in kwargs.keys():
            params['other_beam_Sigma_33'] = kwargs['sigma_y']**2
            del kwargs['sigma_y']

        if 'd_px' in kwargs.keys():
            params['post_subtract_px'] = kwargs['d_px']
            del kwargs['d_px']

        if 'd_py' in kwargs.keys():
            params['post_subtract_py'] = kwargs['d_py']
            del kwargs['d_py']

        return params

    # Properties to mimic the old interfece (to be removed)
    @property
    def n_particles(self):
        return self.other_beam_num_particles

    @n_particles.setter
    def n_particles(self, value):
        self.other_beam_num_particles = value

    @property
    def q0(self):
        return self.other_beam_q0

    @q0.setter
    def q0(self, value):
        self.other_beam_q0 = value

    @property
    def beta0(self):
        return self.other_beam_beta0

    @beta0.setter
    def beta0(self, value):
        self.other_beam_beta0 = value

    @property
    def mean_x(self):
        return self.other_beam_shift_x

    @mean_x.setter
    def mean_x(self, value):
        self.other_beam_shift_x = value

    @property
    def mean_y(self):
        return self.other_beam_shift_y

    @mean_y.setter
    def mean_y(self, value):
        self.other_beam_shift_y = value

    @property
    def sigma_x(self):
        return np.sqrt(self.other_beam_Sigma_11)

    @sigma_x.setter
    def sigma_x(self, value):
        self.other_beam_Sigma_11 = value**2

    @property
    def sigma_y(self):
        return np.sqrt(self.other_beam_Sigma_33)

    @sigma_y.setter
    def sigma_y(self, value):
        self.other_beam_Sigma_33 = value**2

    @property
    def d_px(self):
        return self.post_subtract_px

    @d_px.setter
    def d_px(self, value):
        self.post_subtract_px = value

    @property
    def d_py(self):
        return self.post_subtract_py

    @d_py.setter
    def d_py(self, value):
        self.post_subtract_py = value





