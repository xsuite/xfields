# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #
from pathlib import Path

import numpy as np
from scipy.constants import epsilon_0
from numpy import pi

from .base import Solver

import xobjects as xo

_kernels_complex_prod = {
    'broadcast_complex_product_inplace': xo.Kernel(
        args=[
            xo.Arg(xo.scalar.Float64, pointer=True, name='big'),
            xo.Arg(xo.scalar.Float64, pointer=True, name='small'),
            xo.Arg(xo.UInt64, name='n0_big'),
            xo.Arg(xo.UInt64, name='n1_big'),
            xo.Arg(xo.UInt64, name='n2_big'),
            xo.Arg(xo.UInt64, name='nn'),
        ],
        n_threads='nn',
    )
}

class FFTSolver2D(Solver):

    def solve(self, rho):
        pass

class FFTSolver3D(xo.HybridClass):

    '''
    Creates a Poisson solver object that solves the full 3D Poisson
    equation using the FFT method (free space).

    Args:
        nx (int): Number of cells in the horizontal direction.
        ny (int): Number of cells in the vertical direction.
        nz (int): Number of cells in the vertical direction.
        dx (float): Horizontal cell size in meters.
        dy (float): Vertical cell size in meters.
        dz (float): Longitudinal cell size in meters.
        context (XfContext): identifies the :doc:`context <contexts>`
            on which the computation is executed.
    Returns:
        (FFTSolver3D): Poisson solver object.
    '''

    _xofields = {
        '_dummy': xo.Int8,
    }

    _kernels = _kernels_complex_prod

    _extra_c_sources = [Path(__file__).parent / 'src/broadcast_complex_product_inplace.h']

    def __init__(self, dx, dy, dz, nx, ny, nz, context=None, fftplan=None):

        if context is None:
            context = xo.context_default

        self.context = context

        # Prepare arrays
        workspace_dev = context.nparray_to_context_array(
                    np.zeros((2*nx, 2*ny, 2*nz), dtype=np.complex128, order='F'))


        # Build grid for primitive function
        xg_F = np.arange(0, nx+2) * dx - dx/2
        yg_F = np.arange(0, ny+2) * dy - dy/2
        zg_F = np.arange(0, nz+2) * dz - dz/2
        XX_F, YY_F, ZZ_F = np.meshgrid(xg_F, yg_F, zg_F, indexing='ij')

        # Compute primitive
        F_temp = primitive_func_3d(XX_F, YY_F, ZZ_F)

        # Integrated Green Function (I will transform inplace)
        gint_rep= np.zeros((2*nx, 2*ny, 2*nz), dtype=np.complex128, order='F')
        gint_rep[:nx+1, :ny+1, :nz+1] = (F_temp[ 1:,  1:,  1:]
                                       - F_temp[:-1,  1:,  1:]
                                       - F_temp[ 1:, :-1,  1:]
                                       + F_temp[:-1, :-1,  1:]
                                       - F_temp[ 1:,  1:, :-1]
                                       + F_temp[:-1,  1:, :-1]
                                       + F_temp[ 1:, :-1, :-1]
                                       - F_temp[:-1, :-1, :-1])

        # Replicate
        # To define how to make the replicas I have a look at:
        # np.abs(np.fft.fftfreq(10))*10
        # = [0., 1., 2., 3., 4., 5., 4., 3., 2., 1.]
        gint_rep[nx+1:, :ny+1, :nz+1] = gint_rep[nx-1:0:-1, :ny+1,     :nz+1    ]
        gint_rep[:nx+1, ny+1:, :nz+1] = gint_rep[:nx+1,     ny-1:0:-1, :nz+1    ]
        gint_rep[nx+1:, ny+1:, :nz+1] = gint_rep[nx-1:0:-1, ny-1:0:-1, :nz+1    ]
        gint_rep[:nx+1, :ny+1, nz+1:] = gint_rep[:nx+1,     :ny+1,     nz-1:0:-1]
        gint_rep[nx+1:, :ny+1, nz+1:] = gint_rep[nx-1:0:-1, :ny+1,     nz-1:0:-1]
        gint_rep[:nx+1, ny+1:, nz+1:] = gint_rep[:nx+1,     ny-1:0:-1, nz-1:0:-1]
        gint_rep[nx+1:, ny+1:, nz+1:] = gint_rep[nx-1:0:-1, ny-1:0:-1, nz-1:0:-1]

        self._gint_rep = gint_rep.copy()

        # Tranasfer to device
        gint_rep_dev = context.nparray_to_context_array(gint_rep)

        # Prepare fft plan
        if fftplan is None:
            fftplan = context.plan_FFT(workspace_dev, axes=(0,1,2))

        # Transform the green function (in place)
        fftplan.transform(gint_rep_dev)

        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self._workspace_dev = workspace_dev
        self._gint_rep_transf_dev = gint_rep_dev
        self.fftplan = fftplan

        self.xoinitialize(_context=context)

    #@profile
    def solve(self, rho):

        '''
        Solves Poisson's equation in free space for a given charge density.

        Args:
            rho (float64 array): charge density at the grid points in
                Coulomb/m^3.
        Returns:
            phi (float64 array): electric potential at the grid points in Volts.
        '''

        nz_alloc = self.nz
        if self._gint_rep_transf_dev.shape[2] > 1:
            nz_alloc = self._gint_rep_transf_dev.shape[2]
        _workspace_dev = self.context.zeros(
                (2*self.nx, 2*self.ny, nz_alloc), dtype=np.complex128, order='F')

        # The transposes make it faster in cupy (C-contigous arrays)
        _workspace_dev.T[:self.nz, :self.ny, :self.nx] = rho.T
        self.fftplan.transform(_workspace_dev) # rho_rep_hat

        print('a:' , _workspace_dev[120, 121, 15])
        print('b:' , self._gint_rep_transf_dev[120, 121, 0])

        if False:
            _workspace_dev.T[:,:,:] *= (
                        self._gint_rep_transf_dev.T) # phi_rep_hat
        # except Exception: # pyopencl does not support array broadcasting (used in 2.5D)
        if True:
            self.compile_kernels()
            self.context.kernels.broadcast_complex_product_inplace(
                big=_workspace_dev[:1, :1, :1].view(dtype=np.float64),
                small=self._gint_rep_transf_dev[:1, :1, :1].view(dtype=np.float64),
                n0_big=_workspace_dev.shape[0],
                n1_big=_workspace_dev.shape[1],
                n2_big=_workspace_dev.shape[2],
                nn=(_workspace_dev.shape[0]
                    * _workspace_dev.shape[1]
                    * _workspace_dev.shape[2])
            )

        print('Check number:', _workspace_dev[120, 121, 15])


        self.fftplan.itransform(_workspace_dev) #phi_rep
        return _workspace_dev.real[:self.nx, :self.ny, :self.nz]

class FFTSolver2p5D(xo.HybridClass):

    _xofields = {
        '_dummy': xo.Int8,
    }

    _kernels = _kernels_complex_prod

    _extra_c_sources = [Path(__file__).parent / 'src/broadcast_complex_product_inplace.h']

    '''
    Creates a Poisson solver object that solve's Poisson equation in
    the 2.5D aaoroximation equation the FFT method (free space).

    Args:
        nx (int): Number of cells in the horizontal direction.
        ny (int): Number of cells in the vertical direction.
        nz (int): Number of cells in the vertical direction.
        dx (float): Horizontal cell size in meters.
        dy (float): Vertical cell size in meters.
        dz (float): Longitudinal cell size in meters.
        context (XfContext): identifies the :doc:`context <contexts>`
            on which the computation is executed.
    Returns:
        (FFTSolver3D): Poisson solver object.
    '''

    def __init__(self, dx, dy, dz, nx, ny, nz, context=None, fftplan=None):

        if context is None:
            context = xo.context_default
        self.context = context

        # Build grid for primitive function
        xg_F = np.arange(0, nx+2) * dx - dx/2
        yg_F = np.arange(0, ny+2) * dy - dy/2
        XX_F, YY_F= np.meshgrid(xg_F, yg_F, indexing='ij')

        # Compute primitive
        F_temp = primitive_func_2p5d(XX_F, YY_F)

        # Integrated Green Function (I will transform inplace)
        gint_rep= np.zeros((2*nx, 2*ny), dtype=np.complex128, order='F')
        gint_rep[:nx+1, :ny+1] = (F_temp[ 1:,  1:]
                                - F_temp[:-1,  1:]
                                - F_temp[ 1:, :-1]
                                + F_temp[:-1, :-1])

        # Replicate
        # To define how to make the replicas I have a look at:
        # np.abs(np.fft.fftfreq(10))*10
        # = [0., 1., 2., 3., 4., 5., 4., 3., 2., 1.]
        gint_rep[nx+1:, :ny+1] = gint_rep[nx-1:0:-1, :ny+1]
        gint_rep[:nx+1, ny+1:] = gint_rep[:nx+1, ny-1:0:-1]
        gint_rep[nx+1:, ny+1:] = gint_rep[nx-1:0:-1, ny-1:0:-1]


        # Prepare fft plan
        if fftplan is None:
            temp_dev = context.zeros((2*nx, 2*ny, nz),
                                    dtype=np.complex128, order='F')
            fftplan = context.plan_FFT(temp_dev, axes=(0,1))
            del(temp_dev)

        # Transform the green function
        gint_rep_transf = np.fft.fftn(gint_rep, axes=(0,1))

        # Transfer to GPU (if needed)
        gint_rep_transf_dev = context.nparray_to_context_array(
                                       np.atleast_3d(gint_rep_transf))

        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self._gint_rep_transf_dev = gint_rep_transf_dev
        self.fftplan = fftplan
        self.xoinitialize(_context=context)

    def solve(self, *args, **kwargs):
        return FFTSolver3D.solve(self, *args, **kwargs)

class FFTSolver2p5DAveraged(Solver):

    def __init__(self, dx, dy, dz, nx, ny, nz, context=None, fftplan=None):

        if context is None:
            context = xo.context_default
        self.context = context

        # Build grid for primitive function
        xg_F = np.arange(0, nx+2) * dx - dx/2
        yg_F = np.arange(0, ny+2) * dy - dy/2
        XX_F, YY_F= np.meshgrid(xg_F, yg_F, indexing='ij')

        # Compute primitive
        F_temp = primitive_func_2p5d(XX_F, YY_F)

        # Integrated Green Function (I will transform inplace)
        gint_rep= np.zeros((2*nx, 2*ny), dtype=np.complex128, order='F')
        gint_rep[:nx+1, :ny+1] = (F_temp[ 1:,  1:]
                                - F_temp[:-1,  1:]
                                - F_temp[ 1:, :-1]
                                + F_temp[:-1, :-1])

        # Replicate
        # To define how to make the replicas I have a look at:
        # np.abs(np.fft.fftfreq(10))*10
        # = [0., 1., 2., 3., 4., 5., 4., 3., 2., 1.]
        gint_rep[nx+1:, :ny+1] = gint_rep[nx-1:0:-1, :ny+1]
        gint_rep[:nx+1, ny+1:] = gint_rep[:nx+1, ny-1:0:-1]
        gint_rep[nx+1:, ny+1:] = gint_rep[nx-1:0:-1, ny-1:0:-1]


        # Prepare fft plan
        if fftplan is None:
            temp_dev = context.zeros((2*nx, 2*ny),
                                    dtype=np.complex128, order='F')
            fftplan = context.plan_FFT(temp_dev, axes=(0,1))
            del(temp_dev)

        # Transform the green function
        gint_rep_transf = np.fft.fftn(gint_rep, axes=(0,1))

        # Transfer to GPU (if needed)
        gint_rep_transf_dev = context.nparray_to_context_array(
                                       np.atleast_2d(gint_rep_transf))

        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self._gint_rep_transf_dev = gint_rep_transf_dev
        self.fftplan = fftplan

    #@profile
    def solve(self, rho):

        '''
        Solves Poisson's equation in free space for a given charge density.

        Args:
            rho (float64 array): charge density at the grid points in
                Coulomb/m^3.
        Returns:
            phi (float64 array): electric potential at the grid points in Volts.
        '''

        _workspace_dev = self.context.zeros(
                (2*self.nx, 2*self.ny), dtype=np.complex128, order='F')

        sum_rho_xy = rho.sum(axis=0).sum(axis=0)
        sum_rho = sum_rho_xy.sum()
        _workspace_dev[:self.nx, :self.ny] = rho.sum(axis=2)
        self.fftplan.transform(_workspace_dev) # rho_rep_hat

        _workspace_dev[:,:] *= (
                        self._gint_rep_transf_dev) # phi_rep_hat

        self.fftplan.itransform(_workspace_dev) #phi_rep
        phi_sum = _workspace_dev.real[:self.nx, :self.ny]

        phi = 0 * rho
        for iz in range(self.nz):
            phi[:, :, iz] = phi_sum * sum_rho_xy[iz] / sum_rho

        self._sum_rho_xy = sum_rho_xy
        self._sum_rho = sum_rho

        return phi

def primitive_func_3d(x,y,z):
    abs_r = np.sqrt(x * x + y * y + z * z)
    inv_abs_r = 1./abs_r
    res = 1./(4*pi*epsilon_0)*(
            -0.5 * (z*z * np.arctan(x*y*inv_abs_r/z)
                    + y*y * np.arctan(x*z*inv_abs_r/y)
                    + x*x * np.arctan(y*z*inv_abs_r/x))
               + y*z*np.log(x+abs_r)
               + x*z*np.log(y+abs_r)
               + x*y*np.log(z+abs_r))
    return res

def primitive_func_2p5d(x,y):

    abs_r = np.sqrt(x * x + y * y)
    inv_abs_r = 1./abs_r
    res = 1./(4*pi*epsilon_0)*(3*x*y - x*x*np.arctan(y/x)
                     - y*y*np.arctan(x/y) - x*y*np.log(x*x + y*y))
    return res
