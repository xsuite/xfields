from xfields.platforms import XfCpuPlatform
from xfields import TriLinearInterpolatedFieldMap

class SpaceCharge3D(object):
    """
    Simulates the effect of space charge on the bunch.

    Args:
        platform (XfPlatform): identifies the :doc:`platform <platforms>`
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
        dx (float): Horizontal cell size in meters.
        dy (float): Vertical cell size in meters.
        dz (float): Longitudinal cell size in meters.
    """

    def __init__(self,
                 platform=None,
                 update_on_track=True,
                 length=None,
                 apply_z_kick=True,
                 x_range=None, y_range=None, z_range=None,
                 dx=None, dy=None, dz=None,
                 nx=None, ny=None, nz=None,
                 x_grid=None, y_grid=None, z_grid=None,
                 rho=None, phi=None,
                 solver=None,
                 gamma0=None):

        if platform is None:
            platform = XfCpuPlatform()

        self.length = length
        self.update_on_track = update_on_track
        self.apply_z_kick = apply_z_kick
        self.platform=platform

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
                    platform=platform)

        self.fieldmap = fieldmap

    def track(self, particles):

        if self.update_on_track:
            self.fieldmap.update_from_particles(
                    x_p=particles.x,
                    y_p=particles.y,
                    z_p=particles.zeta,
                    ncharges_p=particles.weight,
                    q0=particles.q0*particles.echarge)


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
                 platform=None,
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
                 platform=None,
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


