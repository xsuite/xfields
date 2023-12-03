# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np

from xfields import BiGaussianFieldMap, mean_and_std
from xfields import TriLinearInterpolatedFieldMap
from ..longitudinal_profiles import LongitudinalProfileQGaussian
from ..fieldmaps import BiGaussianFieldMap
from ..general import _pkg_root

import xobjects as xo
import xtrack as xt


class SpaceCharge3D(xt.BeamElement):

    """
    Simulates the effect of space charge on a bunch.

    Args:
        context (XfContext): identifies the :doc:`context <contexts>`
            on which the computation is executed.
        update_on_track (bool): If ``True`` the beam field map is update
            at each interaction. If ``False`` the initial field map is
            used at each interaction (frozen model). The default is
            ``True``.
        length (float): the length of the space-charge interaction in
            meters.
        apply_z_kick (bool): If ``True``, the longitudinal kick on the
            particles is applied.
        x_range (tuple): Horizontal extent (in meters) of the
            computing grid.
        y_range (tuple): Vertical extent (in meters) of the
            computing grid.
        z_range (tuple): Longitudina extent  (in meters) of
            the computing grid.
        nx (int): Number of cells in the horizontal direction.
        ny (int): Number of cells in the vertical direction.
        nz (int): Number of cells in the vertical direction.
        dx (float): Horizontal cell size in meters. It can be
            provided alternatively to ``nx``.
        dy (float): Vertical cell size in meters. It can be
            provided alternatively to ``ny``.
        dz (float): Longitudinal cell size in meters.It can be
            provided alternatively to ``nz``.
        x_grid (np.ndarray): Equispaced array with the horizontal grid points
            (cell centers).
            It can be provided alternatively to ``x_range``, ``dx``/``nx``.
        y_grid (np.ndarray): Equispaced array with the horizontal grid points
            (cell centers).
            It can be provided alternatively to ``y_range``, ``dy``/``ny``.
        z_grid (np.ndarray): Equispaced array with the horizontal grid points
            (cell centers).
            It can be provided alternatively to ``z_range``, ``dz``/``nz``.
        rho (np.ndarray): initial charge density at the grid points in
            Coulomb/m^3.
        phi (np.ndarray): initial electric potential at the grid points in
            Volts. If not provided the ``phi`` is calculated from ``rho``
            using the Poisson solver (if available).
        solver (str or solver object): Defines the Poisson solver to be used
            to compute phi from rho. Accepted values are ``FFTSolver3D`` and
            ``FFTSolver2p5D``. A Xfields solver object can also be provided.
            In case ``update_on_track``is ``False`` and ``phi`` is provided
            by the user, this argument can be omitted.
        gamma0 (float): Relativistic gamma factor of the beam. This is required
            only if the solver is ``FFTSolver3D``.
    Returns:
        (SpaceCharge3D): A space-charge 3D beam element.
    """
    _xofields = {
        'fieldmap': xo.Ref(TriLinearInterpolatedFieldMap),
        'length': xo.Float64,
        }

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('headers','particle_states.h'),
        _pkg_root.joinpath('fieldmaps/interpolated_src/linear_interpolators.h'),
        _pkg_root.joinpath('beam_elements/spacecharge_src/spacecharge3d.h'),
    ]

    def copy(self, _context=None, _buffer=None, _offset=None):
        if _buffer is not self._buffer:
            raise NotImplementedError
        return SpaceCharge3D(_context=_context,
                _buffer=_buffer, _offset=_offset,
                update_on_track=self.update_on_track,
                length=self.length,
                apply_z_kick=self.apply_z_kick,
                fieldmap=self.fieldmap)

    def __init__(self,
                 _context=None,
                 _buffer=None,
                 _offset=None,
                 update_on_track=True,
                 length=None,
                 apply_z_kick=True,
                 fieldmap=None,
                 x_range=None, y_range=None, z_range=None,
                 nx=None, ny=None, nz=None,
                 dx=None, dy=None, dz=None,
                 x_grid=None, y_grid=None, z_grid=None,
                 rho=None, phi=None,
                 solver=None,
                 gamma0=None,
                 fftplan=None):

        self.update_on_track = update_on_track
        self.apply_z_kick = apply_z_kick

        if solver=='FFTSolver3D':
            assert gamma0 is not None, ('To use FFTSolver3D '
                                        'gamma0 must be provided')

        if gamma0 is not None:
            if not np.isscalar(gamma0):
                raise ValueError('gamma0 needs to be a scalar')
            scale_coordinates_in_solver=(1.,1., float(gamma0))
        else:
            scale_coordinates_in_solver=(1.,1.,1.)

        if fieldmap is not None:
            if _buffer is not None:
                assert _buffer is fieldmap._buffer, (
                    'The buffer of the fieldmap and the buffer of the '
                    'SpaceCharge3D object must be the same')
            if _context is not None:
                assert _context is fieldmap._context, (
                    'The context of the fieldmap and the context of the '
                    'SpaceCharge3D object must be the same')
            _buffer = fieldmap._buffer
        else:
            if _buffer is None:
                if _context is None:
                    _context = xo.context_default
                _buffer = _context.new_buffer(capacity=64)

        if fieldmap is None:
            fieldmap = TriLinearInterpolatedFieldMap(
                        _buffer=_buffer,
                        rho=rho, phi=phi,
                        x_grid=x_grid, y_grid=y_grid, z_grid=z_grid,
                        x_range=x_range, y_range=y_range, z_range=z_range,
                        dx=dx, dy=dy, dz=dz,
                        nx=nx, ny=ny, nz=nz,
                        solver=solver,
                        scale_coordinates_in_solver=scale_coordinates_in_solver,
                        updatable=update_on_track,
                        fftplan=fftplan)

        self.xoinitialize(
                 _buffer=_buffer,
                 _offset=_offset,
                 fieldmap=fieldmap,
                 length=length)

        # temp_buff is deallocate here

    @property
    def iscollective(self):
        return self.update_on_track


    def track(self, particles):

        """
        Computes and applies the space-charge forces for the provided set of
        particles.

        Args:
            particles (Particles Object): Particles to be tracked.
        """

        if self.update_on_track:
            self.fieldmap.update_from_particles(
                particles=particles)

        # call C tracking kernel
        super().track(particles)

class SpaceChargeBiGaussian(xt.BeamElement):

    _xofields = {
        'longitudinal_profile': LongitudinalProfileQGaussian, # TODO: Will become unionref
        'fieldmap': BiGaussianFieldMap,
        'length': xo.Float64,
        }

    _extra_c_sources = [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('headers/sincos.h'),
        _pkg_root.joinpath('headers/power_n.h'),
        _pkg_root.joinpath('fieldmaps/bigaussian_src/faddeeva.h'),
        _pkg_root.joinpath('fieldmaps/bigaussian_src/bigaussian.h'),
        _pkg_root.joinpath('fieldmaps/bigaussian_src/bigaussian_fieldmap.h'),
        _pkg_root.joinpath('longitudinal_profiles/qgaussian_src/qgaussian.h'),
        _pkg_root.joinpath('beam_elements/spacecharge_src/spacechargebigaussian.h'),
    ]

    def to_dict(self):
        dct = super().to_dict()
        # To be loaded by ducktrack:
        dct['number_of_particles'] = self.longitudinal_profile.number_of_particles
        dct['bunchlength_rms'] = self.longitudinal_profile.sigma_z
        dct['sigma_x'] = self.fieldmap.sigma_x
        dct['sigma_y'] = self.fieldmap.sigma_y
        dct['x_co'] = self.fieldmap.mean_x
        dct['y_co'] = self.fieldmap.mean_y
        return dct

    def __init__(self,
                 _context=None,
                 _buffer=None,
                 _offset=None,
                 _xobject=None,
                 update_on_track=False,
                 length=None,
                 apply_z_kick=False,
                 longitudinal_profile=None,
                 mean_x=0.,
                 mean_y=0.,
                 sigma_x=None,
                 sigma_y=None,
                 fieldmap=None,
                 min_sigma_diff=1e-10,
                 **kwargs # to avoid issues when building form dict
                 ):

        if _xobject is not None:
            self.xoinitialize(
                     _context=_context,
                     _buffer=_buffer,
                     _offset=_offset,
                     _xobject=_xobject)
        else:
            self.xoinitialize(
                     _context=_context,
                     _buffer=_buffer,
                     _offset=_offset)

            if apply_z_kick:
                raise NotImplementedError

            assert longitudinal_profile is not None, (
                'Longitudinal profile must be provided')

            self.length = length
            self.longitudinal_profile = longitudinal_profile
            self.apply_z_kick = apply_z_kick
            self._init_update_on_track(update_on_track)

            if fieldmap is None:
                self.fieldmap = BiGaussianFieldMap(
                         _context=self._buffer.context,
                         mean_x=mean_x,
                         mean_y=mean_y,
                         sigma_x=sigma_x,
                         sigma_y=sigma_y,
                         min_sigma_diff=min_sigma_diff,
                         updatable=True)
            else:
                self.fieldmap=fieldmap

        self.iscollective = None # Inferred from _update_flag

    def track(self, particles):

        if self._update_flag:
            self.longitudinal_profile.number_of_particles = (
                (particles.weight * (particles.state > 0)).sum()
            )
            mean_x, sigma_x = mean_and_std(
                    particles.x,
                    weights=particles.weight * (particles.state>0))
            mean_y, sigma_y = mean_and_std(
                    particles.y,
                    weights=particles.weight * (particles.state>0))
            if self.update_mean_x_on_track:
                self.mean_x = mean_x
            if self.update_mean_y_on_track:
                self.mean_y = mean_y
            if self.update_sigma_x_on_track:
                self.sigma_x = sigma_x
            if self.update_sigma_y_on_track:
                self.sigma_y = sigma_y

        super().track(particles)


    def _init_update_on_track(self, update_on_track):
        self.update_mean_x_on_track = False
        self.update_mean_y_on_track = False
        self.update_sigma_x_on_track = False
        self.update_sigma_y_on_track = False
        if update_on_track == True:
            self.update_mean_x_on_track = True
            self.update_mean_y_on_track = True
            self.update_sigma_x_on_track = True
            self.update_sigma_y_on_track = True
        elif update_on_track == False:
            pass
        else:
            for nn in update_on_track:
                assert nn in ['mean_x', 'mean_y',
                              'sigma_x', 'sigma_y']
                setattr(self, f'update_{nn}_on_track', True)

    @property
    def _update_flag(self):
        return (self.update_mean_x_on_track or
                self.update_mean_y_on_track or
                self.update_sigma_x_on_track or
                self.update_sigma_y_on_track)

    @property
    def iscollective(self):
        if self._iscollective is not None:
            return self._iscollective
        else:
            return self._update_flag

    @iscollective.setter
    def iscollective(self, value):
        self._iscollective = value

    @property
    def mean_x(self):
        return self.fieldmap.mean_x

    @ mean_x.setter
    def mean_x(self, value):
        self.fieldmap.mean_x = value

    @property
    def mean_y(self):
        return self.fieldmap.mean_y

    @ mean_y.setter
    def mean_y(self, value):
        self.fieldmap.mean_y = value

    @property
    def sigma_x(self):
        return self.fieldmap.sigma_x

    @ sigma_x.setter
    def sigma_x(self, value):
        self.fieldmap.sigma_x = value

    @property
    def sigma_y(self):
        return self.fieldmap.sigma_y

    @ sigma_y.setter
    def sigma_y(self, value):
        self.fieldmap.sigma_y = value



