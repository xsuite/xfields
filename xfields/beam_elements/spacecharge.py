import numpy as np
from scipy.constants import e as qe
from scipy.constants import c as clight

from xfields import BiGaussianFieldMap, mean_and_std
from xfields import TriLinearInterpolatedFieldMap
from ..longitudinal_profiles import LongitudinalProfileQGaussianData
from ..longitudinal_profiles import LongitudinalProfileQGaussian
from ..fieldmaps import BiGaussianFieldMapData
from ..fieldmaps import TriLinearInterpolatedFieldMapData
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
        'fieldmap': TriLinearInterpolatedFieldMapData,
        'length': xo.Float64,
        }

    def __init__(self,
                 _context=None,
                 _buffer=None,
                 _offset=None,
                 update_on_track=True,
                 length=None,
                 apply_z_kick=True,
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

        if _buffer is not None:
            _context = _buffer.context
        if _context is None:
            _context = xo.context_default
        # I build the fieldmap on a temporary buffer
        temp_buff = _context.new_buffer()
        fieldmap = TriLinearInterpolatedFieldMap(
                    _buffer=temp_buff,
                    rho=rho, phi=phi,
                    x_grid=z_grid, y_grid=y_grid, z_grid=z_grid,
                    x_range=x_range, y_range=y_range, z_range=z_range,
                    dx=dx, dy=dy, dz=dz,
                    nx=nx, ny=ny, nz=nz,
                    solver=solver,
                    scale_coordinates_in_solver=scale_coordinates_in_solver,
                    updatable=update_on_track,
                    fftplan=fftplan)

        self.xoinitialize(
                 _context=_context,
                 _buffer=_buffer,
                 _offset=_offset,
                 fieldmap=fieldmap,
                 length=length)

        # temp_buff is deallocate here


    def track(self, particles):

        """
        Computes and applies the space-charge forces for the provided set of
        particles.

        Args:
            particles (Particles Object): Particles to be tracked.
        """

        if self.update_on_track:
            self.fieldmap.update_from_particles(
                    x_p=particles.x,
                    y_p=particles.y,
                    z_p=particles.zeta,
                    ncharges_p=particles.weight,
                    q0_coulomb=particles.q0*qe)

        # call C tracking kernel
        super().track(particles)


srcs = []
srcs.append(_pkg_root.joinpath('headers/constants.h'))
srcs.append(_pkg_root.joinpath('fieldmaps/interpolated_src/linear_interpolators.h'))
srcs.append(_pkg_root.joinpath('beam_elements/spacecharge_src/spacecharge3d.h'))

SpaceCharge3D.XoStruct.extra_sources = srcs



class SpaceChargeBiGaussian(xt.BeamElement):

    _xofields = {
        'longitudinal_profile': LongitudinalProfileQGaussianData, # TODO: Will become unionref
        'fieldmap': BiGaussianFieldMapData,
        'length': xo.Float64,
        }

    def __init__(self,
                 _context=None,
                 _buffer=None,
                 _offset=None,
                 update_on_track=False,
                 length=None,
                 apply_z_kick=False,
                 longitudinal_profile=None,
                 mean_x=0.,
                 mean_y=0.,
                 sigma_x=None,
                 sigma_y=None,
                 min_sigma_diff=1e-10):

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

        self.fieldmap = BiGaussianFieldMap(
                     _context=self._buffer.context,
                     mean_x=mean_x,
                     mean_y=mean_y,
                     sigma_x=sigma_x,
                     sigma_y=sigma_y,
                     min_sigma_diff=min_sigma_diff,
                     updatable=True)

        self.iscollective = None # Inferred from _update_flag

    def track(self, particles):

        if self._update_flag:
            mean_x, sigma_x = mean_and_std(
                    particles.x, weights=particles.weight)
            mean_y, sigma_y = mean_and_std(
                    particles.y, weights=particles.weight)
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

    @classmethod
    def from_xline(cls, xline_spacecharge=None,
            _context=None, _buffer=None, _offset=None):

        assert xline_spacecharge.__class__.__name__ == 'SCQGaussProfile'
        xlsc = xline_spacecharge
        assert np.isclose(xlsc.q_parameter, 1, atol=1e-13) # TODO Bug to be sorted out in pysixtrack (see issue), for now gaussian only!

        lprofile = LongitudinalProfileQGaussian(
                _context=_context,
                _buffer=_buffer,
                number_of_particles=xlsc.number_of_particles,
                sigma_z=xlsc.bunchlength_rms,
                z0=0.,
                q_parameter=xlsc.q_parameter)

        sc = cls(
            _context=_context,
            _buffer=_buffer,
            _offset=_offset,
            length=xlsc.length,
            apply_z_kick=False,
            longitudinal_profile=lprofile,
            mean_x=xlsc.x_co,
            mean_y=xlsc.y_co,
            sigma_x=xlsc.sigma_x,
            sigma_y=xlsc.sigma_y,
            min_sigma_diff=xlsc.min_sigma_diff)

        return sc

srcs = []
srcs.append(_pkg_root.joinpath('headers/constants.h'))
srcs.append(_pkg_root.joinpath('fieldmaps/bigaussian_src/complex_error_function.h'))
srcs.append(_pkg_root.joinpath('fieldmaps/bigaussian_src/bigaussian.h'))
srcs.append(_pkg_root.joinpath('longitudinal_profiles/qgaussian_src/qgaussian.h'))
srcs.append(_pkg_root.joinpath('beam_elements/spacecharge_src/spacechargebigaussian.h'))

SpaceChargeBiGaussian.XoStruct.extra_sources = srcs
