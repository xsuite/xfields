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



class SpaceCharge2D(object):

    def __init__(self,
                 update_on_track=False, # Decides if frozen or soft-gaussian
                 apply_z_kick=True,
                 transverse_field_map=None,
                 longitudinal_profile=None,
                 context=None,
                 ):
        pass

class SpaceCharge2DBiGaussian(SpaceCharge2D):

    def __init__(self,
                 update_on_track=False, # Decides if frozen or soft-gaussian
                 apply_z_kick=True,
                 sigma_x=None, sigma_y=None,
                 longitudinal_mode='Gaussian',
                 sigma_z=None,
                 z_grid=None, dz=None,
                 z_interp_method='linear',
                 context=None,
                 ):
        pass

class SpaceCharge2DInterpMap(SpaceCharge2D):

    def __init__(self,
                 update_on_track=False, # Decides if frozen or kick
                 apply_z_kick=True,
                 rho=None, phi=None,
                 x_grid=None, y_grid=None,
                 dx=None, dy=None,
                 x_range=None, y_range=None,
                 xy_interp_method='linear',
                 longitudinal_mode='Gaussian',
                 sigma_z=None,
                 z_grid=None, dz=None,
                 z_interp_method='linear',
                 context=None,
                 ):
        pass


