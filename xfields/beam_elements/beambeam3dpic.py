# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
import time

import xobjects as xo
import xtrack as xt

from ..general import _pkg_root
from .beambeam3d import _init_alpha_phi
from xfields import TriLinearInterpolatedFieldMap

class BeamBeamPIC3D(xt.BeamElement):

    _xofields = {

        '_sin_phi': xo.Float64,
        '_cos_phi': xo.Float64,
        '_tan_phi': xo.Float64,
        '_sin_alpha': xo.Float64,
        '_cos_alpha': xo.Float64,

        'ref_shift_x': xo.Float64,
        'ref_shift_px': xo.Float64,
        'ref_shift_y': xo.Float64,
        'ref_shift_py': xo.Float64,
        'ref_shift_zeta': xo.Float64,
        'ref_shift_pzeta': xo.Float64,

        'other_beam_shift_x': xo.Float64,
        'other_beam_shift_px': xo.Float64,
        'other_beam_shift_y': xo.Float64,
        'other_beam_shift_py': xo.Float64,
        'other_beam_shift_zeta': xo.Float64,
        'other_beam_shift_pzeta': xo.Float64,

        'post_subtract_x': xo.Float64,
        'post_subtract_px': xo.Float64,
        'post_subtract_y': xo.Float64,
        'post_subtract_py': xo.Float64,
        'post_subtract_zeta': xo.Float64,
        'post_subtract_pzeta': xo.Float64,

        'fieldmap_self': xo.Ref(TriLinearInterpolatedFieldMap),
        'fieldmap_other': xo.Ref(TriLinearInterpolatedFieldMap),

    }
    iscollective = True

    _extra_c_sources= [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('headers/sincos.h'),
        _pkg_root.joinpath('headers/power_n.h'),
        _pkg_root.joinpath('beam_elements/beambeam_src/beambeam3d_ref_frame_changes.h'),

        # beamstrahlung
        _pkg_root.joinpath(
            'beam_elements/beambeam_src/beambeampic_methods.h'),

   ]

    _per_particle_kernels={
        'change_ref_frame': xo.Kernel(
            c_name='BeamBeamPIC3D_change_ref_frame_local_particle',
            args=[]),
        'change_back_ref_frame_and_subtract_dipolar': xo.Kernel(
            c_name='BeamBeamPIC3D_change_back_ref_frame_and_subtract_dipolar_local_particle',
            args=[]),
    }

    def __init__(self, phi=None, alpha=None,
                 x_range=None, y_range=None, z_range=None,
                 nx=None, ny=None, nz=None,
                 dx=None, dy=None, dz=None,
                 x_grid=None, y_grid=None, z_grid=None,
                 solver=None,
                 _context=None, _buffer=None,
                 **kwargs):

        if '_xobject' in kwargs.keys():
            self.xoinitialize(**kwargs)
            return

        if _buffer is None:
            if _context is None:
                _context = xo.context_default
            _buffer = _context.new_buffer(capacity=64)

        fieldmap_self = TriLinearInterpolatedFieldMap(
            _buffer=_buffer,
            x_grid=x_grid, y_grid=y_grid, z_grid=z_grid,
            x_range=x_range, y_range=y_range, z_range=z_range,
            dx=dx, dy=dy, dz=dz,
            nx=nx, ny=ny, nz=nz,
            solver='FFTSolver2p5D',
            scale_coordinates_in_solver=(1,1,1))

        fieldmap_other = TriLinearInterpolatedFieldMap(
            _buffer=_buffer,
            x_grid=x_grid, y_grid=y_grid, z_grid=z_grid,
            x_range=x_range, y_range=y_range, z_range=z_range,
            dx=dx, dy=dy, dz=dz,
            nx=nx, ny=ny, nz=nz,
            solver='FFTSolver2p5D',
            scale_coordinates_in_solver=(1,1,1))

        self.xoinitialize(_buffer=_buffer,
                          fieldmap_self=fieldmap_self,
                          fieldmap_other=fieldmap_other,
                          **kwargs)

        _init_alpha_phi(self, phi=phi, alpha=alpha,
                _sin_phi=kwargs.get('sin_phi', None),
                _cos_phi=kwargs.get('cos_phi', None),
                _tan_phi=kwargs.get('tan_phi', None),
                _sin_alpha=kwargs.get('sin_alpha', None),
                _cos_alpha=kwargs.get('cos_alpha', None))

    @property
    def sin_phi(self):
        return self._sin_phi

    @property
    def cos_phi(self):
        return self._cos_phi

    @property
    def tan_phi(self):
        return self._tan_phi

    @property
    def sin_alpha(self):
        return self._sin_alpha

    @property
    def cos_alpha(self):
        return self._cos_alpha

    @property
    def phi(self):
        return np.arctan2(self.sin_phi, self.cos_phi)

    @phi.setter
    def phi(self, value):
        raise NotImplementedError("Setting phi is not implemented yet")

    @property
    def alpha(self):
        return np.arctan2(self.sin_alpha, self.cos_alpha)

    @alpha.setter
    def alpha(self, value):
        raise NotImplementedError("Setting alpha is not implemented yet")

