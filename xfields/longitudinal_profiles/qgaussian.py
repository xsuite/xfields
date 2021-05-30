import xobjects as xo
import xtrack as xt

import numpy as np
from numpy import sqrt, pi
from scipy.special import gamma

from ..general import _pkg_root

class LongitudinalProfileQGaussianData(xo.Struct):
    number_of_particles = xo.Float64
    _q_tol = xo.Float64
    _z0 = xo.Float64
    _sigma_z = xo.Float64
    _q_param = xo.Float64
    _cq_param = xo.Float64
    _beta_param = xo.Float64
    _sqrt_beta_param = xo.Float64
    _support_min = xo.Float64
    _support_max = xo.Float64

LongitudinalProfileQGaussianData.extra_sources = [
    _pkg_root.joinpath('longitudinal_profiles/qgaussian_src/qgaussian.h')
    ]
LongitudinalProfileQGaussianData.custom_kernels = {'line_density_qgauss':
        xo.Kernel(args=[xo.Arg(LongitudinalProfileQGaussianData, name='prof'),
                        xo.Arg(xo.Int64, name='n'),
                        xo.Arg(xo.Float64, pointer=True, name='z'),
                        xo.Arg(xo.Float64, pointer=True, name='res')],
                  n_threads='n')}




class LongitudinalProfileQGaussian(xt.dress(LongitudinalProfileQGaussianData)):

    @staticmethod
    def cq_from_q(q, q_tol):
        cq = sqrt(pi)
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
        return cq

    def __init__(self,
            _context=None,
            _buffer=None,
            _offset=None,
            number_of_particles=None,
            sigma_z=None,
            z0=0.,
            q_parameter=1.,
            z_min = -1e10,
            z_max = 1e10,
            q_tol=1e-6):


        assert number_of_particles is not None
        assert sigma_z is not None

        self.xoinitialize(_context=_context, _buffer=_buffer, _offset=_offset)

        self._z_min = z_min
        self._z_max = z_max
        self.number_of_particles = number_of_particles
        self.sigma_z = sigma_z
        self.z0 = z0
        self.q_tol = q_tol
        self.q_parameter = q_parameter

    def _recompute_beta_param(self):
        self._beta_param = 1./(self.sigma_z*self.sigma_z*(5.-3.*self.q_parameter))
        self._sqrt_beta_param = np.sqrt(self._beta_param)

    def _recompute_support(self):
        support_min = self._z_min
        support_max = self._z_max
        # Handle limited support
        if self.q_parameter < (1. - self.q_tol):
            rng = 1./sqrt(self.beta_param*(1-self.q_parameter))
            allowed_min = self.z0 - rng
            allowed_max = self.z0 + rng
            if support_min < allowed_min:
                support_min = allowed_min
            if support_max > allowed_max:
                support_max = allowed_max
        self._support_min = support_min
        self._support_max = support_max

    @property
    def sigma_z(self):
        return self._sigma_z

    @sigma_z.setter
    def sigma_z(self, value):
        self._sigma_z = value
        self._recompute_beta_param()
        self._recompute_support()

    @property
    def z0(self):
        return self._z0

    @z0.setter
    def z0(self, value):
        self._z0 = value
        self._recompute_support()

    @property
    def q_parameter(self):
        return self._q_param

    @q_parameter.setter
    def q_parameter(self, value):
        if value >= 5./3.:
            raise NotImplementedError
        self._q_param = value
        self._cq_param = self.__class__.cq_from_q(value, self.q_tol)
        self._recompute_beta_param()
        self._recompute_support()

    @property
    def q_tol(self):
        return self._q_tol

    @q_tol.setter
    def q_tol(self, value):
        self._q_tol = value
        self._recompute_support()

    @property
    def beta_param(self):
        return self._beta_param

    @property
    def z_min(self):
        return self._z_min

    @z_min.setter
    def z_min(self, value):
        self._z_min = value
        self._recompute_support()

    @property
    def z_max(self):
        return self._z_max

    @z_max.setter
    def z_max(self, value):
        self._z_max = value
        self._recompute_support()

    def line_density(self, z):
        context = self._buffer.context
        res = context.zeros(len(z), dtype=np.float64)

        if 'line_density_q_gauss' not in context.kernels.keys():
            self.compile_custom_kernels()

        context.kernels.line_density_qgauss(prof=self._xobject, n=len(z), z=z, res=res)

        return res

