import numpy as np
from xobjects.context import ContextDefault

from .base import FieldMap
from ..contexts import add_default_kernels

def mean_and_std(a, weights=None):
    if weights is None:
        mean = a.sum()/len(a)
        std = np.sqrt(((a-mean)**2).sum() / len(a))
    else:
        assert len(weights) == len(a)
        tot = weights.sum()
        mean = (a*weights).sum() / tot
        std = np.sqrt(((a-mean)**2 * weights).sum() / tot)

    return mean, std

class BiGaussianFieldMap(FieldMap):
    '''
    Bassetti-Erskine
    Must be 2D, no closed form for 3D poisson.
    We assume normalized charge density, see twx.
    '''

    def __init__(self,
                 context=None,
                 mean_x=0., mean_y=0.,
                 sigma_x=None, sigma_y=None,
                 min_sigma_diff=1e-10,
                 updatable=True):


        if context is None:
            context = ContextDefault()

        add_default_kernels(context)

        self.updatable = updatable
        self.context = context

        self.mean_x = mean_x
        self.mean_y = mean_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.min_sigma_diff=min_sigma_diff

    def get_values_at_points(self,
            x, y,
            return_rho=True,
            return_phi=True,
            return_dphi_dx=True,
            return_dphi_dy=True,
            ):

        if self.sigma_x is None:
            raise ValueError('sigma_x must be set')
        if self.sigma_y is None:
            raise ValueError('sigma_y must be set')

        assert len(x) == len(y)
        tobereturned = []

        if return_rho:
            raise notimplementederror('not yet implemented :-(')
        if return_phi:
            raise notimplementederror('not yet implemented :-(')

        if return_dphi_dx or return_dphi_dy:
            Ex = self.context.zeros(x.shape, dtype=np.float64)
            Ey = self.context.zeros(x.shape, dtype=np.float64)
            self.context.kernels.get_Ex_Ey_Gx_Gy_gauss(
                n_points=len(x),
                x_ptr=x-self.mean_x,
                y_ptr=y-self.mean_y,
                sigma_x=self.sigma_x,
                sigma_y=self.sigma_y,
                min_sigma_diff=self.min_sigma_diff,
                skip_Gs=1,
                Ex_ptr=Ex,
                Ey_ptr=Ey,
                Gx_ptr=Ex, # untouchd when skip_Gs is zero
                Gy_ptr=Ex, # untouchd when skip_Gs is zero
                )
            if return_dphi_dx:
                tobereturned.append(-Ex)
            if return_dphi_dy:
                tobereturned.append(-Ey)

        return tobereturned

    def update_from_particles(self, x_p, y_p, z_p, ncharges_p, q0_coulomb,
                reset=True, update_phi=True, solver=None, force=False):

        if not force:
            self._assert_updatable()

        raise NotImplementedError()

    def update_rho(self, rho, reset):
        raise ValueError('rho cannot be directly updated'
                         'for BiGaussianFieldMap')

    def update_phi(self, phi, reset=True, force=False):
        raise ValueError('phi cannot be directly updated'
                         'for BiGaussianFieldMap')

    def update_phi_from_rho(self, solver=None):
        raise ValueError('phi cannot be directly updated'
                         'for BiGaussianFieldMap')

    def generate_solver(self, solver):
        raise ValueError('solver cannot be generated'
                         'for BiGaussianFieldMap')
