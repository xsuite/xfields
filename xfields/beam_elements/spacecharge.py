from xfields import BiGaussianFieldMap, mean_and_std
from xfields import TriLinearInterpolatedFieldMap

from xobjects.context import ContextDefault

class SpaceCharge3D(object):
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

    def __init__(self,
                 context=None,
                 update_on_track=True,
                 length=None,
                 apply_z_kick=True,
                 x_range=None, y_range=None, z_range=None,
                 nx=None, ny=None, nz=None,
                 dx=None, dy=None, dz=None,
                 x_grid=None, y_grid=None, z_grid=None,
                 rho=None, phi=None,
                 solver=None,
                 gamma0=None):

        if context is None:
            context = ContextDefault()

        self.length = length
        self.update_on_track = update_on_track
        self.apply_z_kick = apply_z_kick
        self.context=context

        if solver=='FFTSolver3D':
            assert gamma0 is not None, ('To use FFTSolver3D '
                                        'gamma0 must be provided')

        if gamma0 is not None:
            scale_coordinates_in_solver=(1.,1., float(gamma0))
        else:
            scale_coordinates_in_solver=(1.,1.,1.)

        fieldmap = TriLinearInterpolatedFieldMap(
                    rho=rho, phi=phi,
                    x_grid=z_grid, y_grid=y_grid, z_grid=z_grid,
                    x_range=x_range, y_range=y_range, z_range=z_range,
                    dx=dx, dy=dy, dz=dz,
                    nx=nx, ny=ny, nz=nz,
                    solver=solver,
                    scale_coordinates_in_solver=scale_coordinates_in_solver,
                    updatable=update_on_track,
                    context=context)

        self.fieldmap = fieldmap

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
                    q0_coulomb=particles.q0*particles.echarge)


        res = self.fieldmap.get_values_at_points(
                            x=particles.x, y=particles.y, z=particles.zeta,
                            return_rho=False, return_phi=False,
                            return_dphi_dz=self.apply_z_kick)
        # res = [dphi_dx, dphi_dy, (dphi_z)]

        #Build factor
        beta0 = particles.beta0
        clight = float(particles.clight)
        charge_mass_ratio = (particles.chi*particles.echarge*particles.q0
                                /(particles.mass0*particles.echarge/(clight*clight)))
        gamma0 = particles.gamma0
        beta0 = particles.beta0
        factor = -(charge_mass_ratio*self.length*(1.-beta0*beta0)
                    /(gamma0*beta0*beta0*clight*clight))

        # Kick particles
        particles.px += factor*res[0]
        particles.py += factor*res[1]
        if self.apply_z_kick:
            particles.delta += factor*res[2]

class SpaceChargeBiGaussian(object):

    def __init__(self,
                 context=None,
                 update_on_track=True,
                 length=None,
                 apply_z_kick=False,
                 longitudinal_profile=None,
                 mean_x=0.,
                 mean_y=0.,
                 sigma_x=None,
                 sigma_y=None,
                 min_sigma_diff=1e-10):

        if context is None:
            context = ContextDefault()

        if apply_z_kick:
            raise NotImplementedError

        self.context = context
        self.length = length
        self.longitudinal_profile = longitudinal_profile
        self.apply_z_kick = apply_z_kick
        self._init_update_on_track(update_on_track)

        self.fieldmap = BiGaussianFieldMap(
                     context=context,
                     mean_x=mean_x,
                     mean_y=mean_y,
                     sigma_x=sigma_x,
                     sigma_y=sigma_y,
                     min_sigma_diff=min_sigma_diff,
                     updatable=True)

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

        dphi_dx, dphi_dy = self.fieldmap.get_values_at_points(
                            x=particles.x, y=particles.y,
                            return_rho=False, return_phi=False)

        lambda_z = self.longitudinal_profile.line_density(particles.zeta)

        #Build factor
        beta0 = particles.beta0
        clight = float(particles.clight)
        charge_mass_ratio = (particles.chi*particles.echarge*particles.q0
                                /(particles.mass0*particles.echarge/(clight*clight)))
        gamma0 = particles.gamma0
        beta0 = particles.beta0
        factor = -(charge_mass_ratio*particles.q0*particles.echarge
                   *self.length*(1.-beta0*beta0)
                   /(gamma0*beta0*beta0*clight*clight))

        # Kick particles
        particles.px += factor*lambda_z*dphi_dx
        particles.py += factor*lambda_z*dphi_dy

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


