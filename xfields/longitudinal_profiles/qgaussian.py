from numpy import sqrt, pi
from scipy.special import gamma

from ..contexts import add_default_kernels

class LongitudinalProfileQGaussian(object):

    @staticmethod
    def cq_from_q(q, q_tol):
        cq = sqrt(np.pi)
        if q >= (1 + q_tol):
            cq *= gamma((3 - q) / (2 * q - 2))
            cq /= sqrt((q - 1)) * gamma(1 / (q - 1))
        elif q <= (1 - q_tol):
            cq *= 2 * gamma(1 / (1 - q))
            cq /= (
                (3 - q)
                * sqrt(1 - q)
                * gamma((3 - q) / (2 - 2 * q))
            )

    def __init__(self,
            context=None,
            number_of_particles=None,
            q0=1,
            sigma_z=None,
            z0=0.,
            q_parameter=1.,
            q_tol=1e-6):

        if context is None:
            context = ContextDefault()

        add_default_kernels(context)

        assert number_of_particles is not None
        assert sigma_z is not None

        self.number_of_particles = number_of_particles
        self.q0 = q0
        self.sigma_z = sigma_z
        self.z0 = z0
        self.q_parameter = q_parameter
        self.q_tol = q_tol
        self.z_min = z_min
        self.z_max = z_max

    @property
    def beta_param(self):
        return 1./(self.sigma_z*self.sigma_z*(5.-3.*self.q_parameter))

    @property
    def q_parameter(self):
        return self._q_param

    @q_parameter.setter
    def q_parameter(self, value):
        if value >= 5./3.:
            raise NotImplementedError
        self._q_param = value
        self._cq_param = self.__class__.cq_from_q(value, self.q_tol)

    def line_density(self, z):
        res = context.zeros(len(z), dtype=np.float64)
        support_min = self.z_min
        support_max = self.z_max

        # Handle limited support
        if self.q_parameter < (1. + q_tol):
            rng = 1./sqrt(self.beta_param*(1-self.q_parameter))
            allowed_min = z0 - rng
            allowed_max = z0 + rng
            if support_min < allowed_min:
                support_min = allowed_min
            if support_max > allowed_max:
                support_max = allowed_max

        self.context.kernels.q_gaussian_profile(
                n=len(z),
                z=z,
                z0=self.z0,
                z_min=support_min,
                z_max=support_max,
                beta=self._beta_param,
                q=self.q_parameter,
                q_tol=self.q_tol,
                factor=factor,
                res=res)


