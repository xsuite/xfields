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

        'ref_shift_x': xo.Float64,
        'ref_shift_y': xo.Float64,

        'other_beam_shift_x': xo.Float64,
        'other_beam_shift_y': xo.Float64,

        'post_subtract_px': xo.Float64,
        'post_subtract_py': xo.Float64,

        'q0_other_beam': xo.Float64,
        'beta0_other_beam': xo.Float64,

        'other_beam_num_particles': xo.Float64,

        'other_beam_Sigma_11_star': xo.Float64,
        'other_beam_Sigma_13_star': xo.Float64,
        'other_beam_Sigma_33_star': xo.Float64,

        'min_sigma_diff': xo.Float64,

    }

    extra_sources= [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('headers/sincos.h'),
        _pkg_root.joinpath('headers/power_n.h'),
        _pkg_root.joinpath('fieldmaps/bigaussian_src/complex_error_function.h'),
        '#define NOFIELDMAP', #TODO Remove this workaround
        _pkg_root.joinpath('fieldmaps/bigaussian_src/bigaussian.h'),
        _pkg_root.joinpath('beam_elements/beambeam_src/beambeam2d.h'),
    ]


    def __init__(self,
                    q0_other_beam=None,

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

            self.config_for_update = config_for_update
            self.iscollective = True
            self.track = self._track_collective # switch to specific track method

            # Some dummy values just to initialize the object
            if other_beam_Sigma_11 is None: other_beam_Sigma_11 = 1.
            if other_beam_Sigma_33 is None: other_beam_Sigma_33 = 1.
            if other_beam_num_particles is None:
                other_beam_num_particles = 0.

        self.xoinitialize(**kwargs)

        if self.iscollective:
            if not isinstance(self._buffer.context, xo.ContextCpu):
                raise NotImplementedError(
                    'BeamBeamBiGaussian3D only works with CPU context for now')

        # Mandatory sigmas
        assert other_beam_Sigma_11 is not None, ("`other_beam_Sigma_11` must be provided")
        assert other_beam_Sigma_33 is not None, ("`other_beam_Sigma_33` must be provided")

        # Coupling between transverse planes
        if other_beam_Sigma_13 is None:
            other_beam_Sigma_13 = 0

        assert other_beam_num_particles is not None, ("`other_beam_num_particles` must be provided")
        self.other_beam_num_particles = other_beam_num_particles

        self.other_beam_Sigma_11 = other_beam_Sigma_11
        self.other_beam_Sigma_13 = other_beam_Sigma_13
        self.other_beam_Sigma_33 = other_beam_Sigma_33

        assert q0_other_beam is not None
        self.q0_other_beam = q0_other_beam

        self.ref_shift_x = ref_shift_x
        self.ref_shift_y = ref_shift_y

        self.other_beam_shift_x = other_beam_shift_x
        self.other_beam_shift_y = other_beam_shift_y

        self.post_subtract_px = post_subtract_px
        self.post_subtract_py = post_subtract_py

        self.min_sigma_diff = min_sigma_diff




