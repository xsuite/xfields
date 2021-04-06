from xfields import BiGaussianFieldMap
from xobjects.context import ContextDefault

class BeamBeamBiGaussian2D(object):

    def __init__(self,
            context=None,
            n_particles=None,
            q0=None,
            beta0=None,
            sigma_x=None,
            sigma_y=None,
            mean_x=0.,
            mean_y=0.,
            min_sigma_diff=1e-10):

        if context is None:
            context = ContextDefault()

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

    @property
    def mean_x(self):
        return self.fieldmap.mean_x

    @property.setter
    def mean_x(self, value):
        self.fieldmap.mean_x = value

    @property
    def mean_y(self):
        return self.fieldmap.mean_y

    @property.setter
    def mean_y(self, value):
        self.fieldmap.mean_y = value

    @property
    def sigma_x(self):
        return self.fieldmap.sigma_x

    @property.setter
    def sigma_x(self, value):
        self.fieldmap.sigma_x = value

    @property
    def sigma_y(self):
        return self.fieldmap.sigma_y

    @property.setter
    def sigma_y(self, value):
        self.fieldmap.sigma_y = value

    def track(self, particles):

        dphi_dx, dphi_dy = self.fieldmap.get_values_at_points(
                            x=particles.x, y=particles.y
                            return_rho=False, return_phi=False)

        clight = float(particles.clight)
        charge_mass_ratio = (particles.chi*particles.echarge*particles.q0
                    /(particles.mass0*particles.echarge/(clight*clight)))

        factor = -(charge_mass_ratio*self.n_particles*self.q0
                    /(particles.gamma0*particles.beta0*
                        clight*clight)
                    *(1+self.beta0*particles.beta0)
                    /(self.beta0 + particles.beta0))

        # Kick particles
        particles.px += factor*dphi_dx
        particles.py += factor*dphi_dy

