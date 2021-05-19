from xfields import BiGaussianFieldMap
from xobjects.context import context_default
import xobjects as xo
import xtrack as xt
from xfields.fieldmaps.bigaussian import BiGaussianFieldMapData

class BeamBeamBiGaussian2DData(xo.Struct):
    n_particles = xo.Float64
    q0 = xo.Float64
    beta0 = xo.Float64
    fieldmap = BiGaussianFieldMapData


class BeamBeamBiGaussian2D(xt.dress(BeamBeamBiGaussian2DData)):
    """
    Simulates the effect of beam-beam on a bunch.

    Args:
        context (xobjects context): identifies the :doc:`context <contexts>`
            on which the computation is executed.
        n_particles (float64): Number of particles in the colliding bunch.
        q0 (float64): Number of particles in the colliding bunch.
        beta0 (float64): Relativistic beta of the colliding bunch.
        mean_x (float64): Horizontal position (in meters) of the colliding
            bunch. It can be updated after the object creation.
            Default is ``0.``.
        mean_y (float64): Vertical position (in meters) of the Gaussian
            distribution. It can be updated after the object creation.
            Default is ``0.``.
        sigma_x (float64): Horizontal r.m.s. size (in meters) of the colliding
            bunch. It can be updated after the object creation.
            Default is ``None``.
        sigma_y (float64): Vertical r.m.s. size (in meters) of the colliding
            bunch. It can be updated after the object creation.
            Default is ``None``.
    Returns:
        (BeamBeamBiGaussian2D): A beam-beam element.
    """

    def __init__(self,
            context=None,
            n_particles=None,
            q0=None,
            beta0=None,
            mean_x=0.,
            mean_y=0.,
            sigma_x=None,
            sigma_y=None,
            min_sigma_diff=1e-10):

        if context is None:
            context = context_default

        self.context = context
        self.n_particles = n_particles
        self.q0 = q0
        self.beta0 = beta0

        self.fieldmap = BiGaussianFieldMap(
                     context=context,
                     mean_x=mean_x,
                     mean_y=mean_y,
                     sigma_x=sigma_x,
                     sigma_y=sigma_y,
                     min_sigma_diff=min_sigma_diff,
                     updatable=True)

    def update(self, **kwargs):
        for kk in kwargs.keys():
            if not hasattr(self, kk):
                raise NameError(f'Unknown parameter: {kk}')
            setattr(self, kk, kwargs[kk])

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

    def track(self, particles):
        """
        Computes and applies the beam-beam forces for the provided set of
        particles.

        Args:
            particles (Particles Object): Particles to be tracked.
        """

        dphi_dx, dphi_dy = self.fieldmap.get_values_at_points(
                            x=particles.x, y=particles.y,
                            return_rho=False, return_phi=False)

        clight = float(particles.clight)
        charge_mass_ratio = (particles.chi*particles.echarge*particles.q0
                    /(particles.mass0*particles.echarge/(clight*clight)))

        factor = -(charge_mass_ratio*self.n_particles*self.q0
                    * particles.echarge
                    /(particles.gamma0*particles.beta0*
                        clight*clight)
                    *(1+self.beta0*particles.beta0)
                    /(self.beta0 + particles.beta0))

        # Kick particles
        particles.px += factor*dphi_dx
        particles.py += factor*dphi_dy

