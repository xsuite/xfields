# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
from scipy.constants import e as qe
from scipy.constants import c as clight

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

        'fieldmap_self': TriLinearInterpolatedFieldMap,
        'fieldmap_other': TriLinearInterpolatedFieldMap,

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
        'change_ref_frame_bbpic': xo.Kernel(
            c_name='BeamBeamPIC3D_change_ref_frame_local_particle',
            args=[]),
        'change_back_ref_frame_and_subtract_dipolar_bbpic': xo.Kernel(
            c_name='BeamBeamPIC3D_change_back_ref_frame_and_subtract_dipolar_local_particle',
            args=[]),
        'propagate_transverse_coords_at_step': xo.Kernel(
            c_name='BeamBeamPIC3D_propagate_transverse_coords_at_step',
            args=[xo.Arg(xo.Float64, name='z_step_other')]),
        'kick_and_propagate_transverse_coords_back': xo.Kernel(
            c_name='BeamBeamPIC3D_kick_and_propagate_transverse_coords_back',
            args=[
                xo.Arg(xo.Float64, name='dphi_dx', pointer=True),
                xo.Arg(xo.Float64, name='dphi_dy', pointer=True),
                xo.Arg(xo.Float64, name='dphi_dz', pointer=True),
                xo.Arg(xo.Float64, name='z_step_other'),
            ]),
    }

    def __init__(self, phi=None, alpha=None,
                 x_range=None, y_range=None, z_range=None,
                 nx=None, ny=None, nz=None,
                 dx=None, dy=None, dz=None,
                 x_grid=None, y_grid=None, z_grid=None,
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

        self._working_on_bunch = None

    def track(self, particles):

        pp = particles
        mask_alive = pp.state > 0

        if self._working_on_bunch is None:
            # Starting a new interaction
            self._working_on_bunch = pp

            # Move particles to computation reference frame
            self.change_ref_frame_bbpic(pp)

            self._i_step = 0
            self._z_steps_self = self.fieldmap_self.z_grid[::-1].copy() # earlier time first
            self._z_steps_other = self.fieldmap_other.z_grid[::-1].copy() # earlier time first
            self._sent_rho_to_partner = False

            assert len(self._z_steps_other) == len(self._z_steps_self)

        assert self._working_on_bunch is pp

        if not self._sent_rho_to_partner:
            # Propagate transverse coordinates to the position at the time step
            z_step_other = self._z_steps_other[self._i_step]
            self.propagate_transverse_coords_at_step(pp, z_step_other=z_step_other)

            # Compute charge density
            self.fieldmap_self.update_from_particles(particles=pp,
                                                    update_phi=False)

            at_turn = pp.at_turn[mask_alive][0]

            # Pass charge density to partner
            communication_send_id_data = dict(
                    element_name=self.name,
                    sender_name=pp.name,
                    receiver_name=self.partner_name,
                    turn=at_turn,
                    internal_tag=self._i_step)
            if self.pipeline_manager.is_ready_to_send(**communication_send_id_data):
                if isinstance(self._context, xo.ContextCpu):
                    buffer = self.fieldmap_self.rho.flatten().copy()
                else:
                    buffer = self._context.nparray_from_context_array(self.fieldmap_self.rho.flatten())
                self.pipeline_manager.send_message(buffer, **communication_send_id_data)
            self._sent_rho_to_partner = True

        # Try to receive rho from partner
        communication_recv_id_data = dict(
                element_name=self.name,
                sender_name=self.partner_name,
                receiver_name=pp.name,
                internal_tag=self._i_step)
        if self.pipeline_manager.is_ready_to_receive(**communication_recv_id_data):
            buffer_receive = np.zeros(np.prod(self.fieldmap_other.rho.shape),
                                      dtype=float)

            self.pipeline_manager.receive_message(buffer_receive, **communication_recv_id_data)
            buffer_receive = self._context.nparray_to_context_array(buffer_receive)
            rho = buffer_receive.reshape(self.fieldmap_other.rho.shape)
            self.fieldmap_other.update_rho(rho, reset=True)
        else:
            return xt.PipelineStatus(on_hold=True,
                        info=f'waiting for rho for step {self._i_step}')

        # Restarting after receiving rho
        self._sent_rho_to_partner = False # Clear flag

        # Compute potential
        self.fieldmap_other.update_phi_from_rho()

        # Compute particles coordinates in the reference system of the other beam
        z_step_self = self._z_steps_self[self._i_step]
        mask_alive = pp.state > 0
        z_step_other = self._z_steps_other[self._i_step]
        # For now assuming symmetric ultra-relativistic beams
        z_other = (-pp.zeta[mask_alive] + z_step_other + z_step_self)
        x_other = -pp.x[mask_alive]
        y_other = pp.y[mask_alive]

        # Get fields in the reference system of the other beam
        dphi_dx, dphi_dy, dphi_dz = self.fieldmap_other.get_values_at_points(
            x=x_other, y=y_other, z=z_other,
            return_rho=False,
            return_phi=False,
            return_dphi_dx=True,
            return_dphi_dy=True,
            return_dphi_dz=True,
        )

        # Transform fields to self reference frame (dphi_dy is unchanged)
        dphi_dx *= -1
        dphi_dz *= -1

        # The kernel relies on the contiguity of pp for element-wise multiplying
        # with dphi_d{x,y,z}
        pp.reorganize()
        self.kick_and_propagate_transverse_coords_back(
            pp, dphi_dx=dphi_dx, dphi_dy=dphi_dy, dphi_dz=dphi_dz,
            z_step_other=z_step_other)

        self._i_step += 1
        if self._i_step < len(self._z_steps_other):
            return xt.PipelineStatus(on_hold=True,
                    info=f'ready to start step {self._i_step}')
        else:
            self.change_back_ref_frame_and_subtract_dipolar_bbpic(pp)
            self._working_on_bunch = None
            return None # Interaction done!

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

