
import numpy as np

import xobjects as xo
import xpart as xp
import xtrack as xt

from .interpolated import _configure_grid
from ..general import _pkg_root

class TriCubicInterpolatedFieldMapData(xo.Struct):
    x_min = xo.Float64
    y_min = xo.Float64
    z_min = xo.Float64
    nx = xo.Int64
    ny = xo.Int64
    nz = xo.Int64
    mirror_x = xo.Int64
    mirror_y = xo.Int64
    mirror_z = xo.Int64
    dx = xo.Float64
    dy = xo.Float64
    dz = xo.Float64
    phi_taylor = xo.Float64[:]

TriCubicInterpolatedFieldMapData.extra_sources = [
    _pkg_root.joinpath('fieldmaps/interpolated_src/tricubic_coefficients.h'),
    _pkg_root.joinpath('fieldmaps/interpolated_src/cubic_interpolators.h'),
    ]

TriCubicInterpolatedFieldMapData.custom_kernels = {
    'TriCubicInterpolatedFieldMap_interpolate_derivatives': xo.Kernel(
        args=[
            xo.Arg(TriCubicInterpolatedFieldMapData, pointer=False, name='fmap'),
            xo.Arg(xo.Int64,   pointer=False, name='n_points'),
            xo.Arg(xo.Float64, pointer=True,  name='x'),
            xo.Arg(xo.Float64, pointer=True,  name='y'),
            xo.Arg(xo.Float64, pointer=True,  name='z'),
            xo.Arg(xo.Int8,    pointer=True,  name='buffer_mesh_quantities'),
            xo.Arg(xo.Int64,   pointer=True,  name='offsets_mesh_quantities'),
            xo.Arg(xo.Float64, pointer=True,  name='particles_quantities'),
            ],
        n_threads='n_points'
        ),
    }

# I add undescores in front of the names so that I can define custom properties
rename_tricubic = {ff.name:'_'+ff.name for ff
                in TriCubicInterpolatedFieldMapData._fields}
class TriCubicInterpolatedFieldMap(xo.dress(TriCubicInterpolatedFieldMapData,
                                             rename=rename_tricubic)):

    """
    Builds a cubic interpolator for a 3D field map.

    Args:
        context (xobjects context): identifies the :doc:`context <contexts>`
            on which the computation is executed.
        x_range (tuple): Horizontal extent (in meters) of the
            computing grid.
        y_range (tuple): Vertical extent (in meters) of the
            computing grid.
        z_range (tuple): Longitudinal extent  (in meters) of
            the computing grid.
        nx (int): Number of cells in the horizontal direction.
        ny (int): Number of cells in the vertical direction.
        nz (int): Number of cells in the longitudinal direction.
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
        mirror_x (int): if equal to 1, the map is mirrored along the x axis
            around x = 0. 
        mirror_y (int): if equal to 1, the map is mirrored along the y axis
            around y = 0. 
        mirror_z (int): if equal to 1, the map is mirrored along the z axis
            around z = 0. 
        phi_taylor (np.ndarray): Normalized scalar potential and its derivatives 
            at the grid points. Should be of dimension (nx, ny, nz, 8). For the 
            last index: 0 -> phi, 1 -> dphi/dx, 2 -> dphi/dy, 3 -> dphi/dz,
            4 -> d^2phi/dxdy, 5 -> d^2phi/dxdz, 6 -> d^2phi/dydz,
            7 -> d^3phi/dxdydz. The derivatives are normalized in the sense that
            they should be multiplied with the grid's step size, 
            e.g. (d^2phi/dxdy)* (Δx*Δy). Units are Volts.
    Returns:
        (TriCubicInterpolatedFieldMap): Interpolator object.
    """

    def __init__(self,
                 _context=None,
                 _buffer=None,
                 _offset=None,
                 x_range=None, y_range=None, z_range=None,
                 nx=None, ny=None, nz=None,
                 dx=None, dy=None, dz=None,
                 x_grid=None, y_grid=None, z_grid=None,
                 mirror_x=0, mirror_y=0, mirror_z=0,
                 phi_taylor=None
                 ):

        self._x_grid = _configure_grid('x', x_grid, dx, x_range, nx)
        self._y_grid = _configure_grid('y', y_grid, dy, y_range, ny)
        self._z_grid = _configure_grid('z', z_grid, dz, z_range, nz)

        nelem = self.nx*self.ny*self.nz*8
        self.xoinitialize(
                 _context=_context,
                 _buffer=_buffer,
                 _offset=_offset,
                 x_min = self._x_grid[0],
                 y_min = self._y_grid[0],
                 z_min = self._z_grid[0],
                 nx = self.nx,
                 ny = self.ny,
                 nz = self.nz,
                 dx = self.dx,
                 dy = self.dy,
                 dz = self.dz,
                 mirror_x = mirror_x,
                 mirror_y = mirror_y,
                 mirror_z = mirror_z,
                 phi_taylor = nelem
                 )

        #self.compile_custom_kernels(only_if_needed=True)
        if phi_taylor is not None:
            self.phi_taylor = phi_taylor

    # #@profile
    # def get_values_at_points(self,
    #         x, y, z,
    #         return_rho=True,
    #         return_phi=True,
    #         return_dphi_dx=True,
    #         return_dphi_dy=True,
    #         return_dphi_dz=True):

    #     """
    #     Returns the charge density, the field potential and its derivatives
    #     at the points specified by x, y, z. The output can be customized (see below).
    #     Zeros are returned for points outside the grid.

    #     Args:
    #         x (float64 array): Horizontal coordinates at which the field is evaluated.
    #         y (float64 array): Vertical coordinates at which the field is evaluated.
    #         z (float64 array): Longitudinal coordinates at which the field is evaluated.
    #         return_rho (bool): If ``True``, the charge density at the given points is
    #             returned.
    #         return_phi (bool): If ``True``, the potential at the given points is returned.
    #         return_dphi_dx (bool): If ``True``, the horizontal derivative of the potential
    #             at the given points is returned.
    #         return_dphi_dy: If ``True``, the vertical derivative of the potential
    #             at the given points is returned.
    #         return_dphi_dz: If ``True``, the longitudinal derivative of the potential
    #             at the given points is returned.
    #     Returns:
    #         (tuple of float64 array): The required quantities at the provided points.
    #     """

    #     assert len(x) == len(y) == len(z)

    #     pos_in_buffer_of_maps_to_interp = []
    #     if return_rho:
    #         pos_in_buffer_of_maps_to_interp.append(
    #                 self._xobject.rho._offset + self._xobject.rho._data_offset)
    #     if return_phi:
    #         pos_in_buffer_of_maps_to_interp.append(
    #                 self._xobject.phi._offset + self._xobject.phi._data_offset)
    #     if return_dphi_dx:
    #         pos_in_buffer_of_maps_to_interp.append(
    #                 self._xobject.dphi_dx._offset + self._xobject.dphi_dx._data_offset)
    #     if return_dphi_dy:
    #         pos_in_buffer_of_maps_to_interp.append(
    #                 self._xobject.dphi_dy._offset + self._xobject.dphi_dy._data_offset)
    #     if return_dphi_dz:
    #         pos_in_buffer_of_maps_to_interp.append(
    #                 self._xobject.dphi_dz._offset + self._xobject.dphi_dz._data_offset)

    #     context = self._buffer.context

    #     pos_in_buffer_of_maps_to_interp = context.nparray_to_context_array(
    #                     np.array(pos_in_buffer_of_maps_to_interp, dtype=np.int64))
    #     nmaps_to_interp = len(pos_in_buffer_of_maps_to_interp)
    #     buffer_out = context.zeros(
    #             shape=(nmaps_to_interp * len(x),), dtype=np.float64)
    #     if nmaps_to_interp > 0:
    #         context.kernels.TriLinearInterpolatedFieldMap_interpolate_3d_map_vector(
    #                 fmap=self._xobject,
    #                 n_points=len(x),
    #                 x=x, y=y, z=z,
    #                 n_quantities=nmaps_to_interp,
    #                 buffer_mesh_quantities=self._buffer.buffer,
    #                 offsets_mesh_quantities=pos_in_buffer_of_maps_to_interp,
    #                 particles_quantities=buffer_out)

    #     # Split buffer 
    #     particles_quantities = [buffer_out[ii*len(x):(ii+1)*len(x)]
    #                                     for ii in range(nmaps_to_interp)]

    #     return particles_quantities


    @property
    def x_grid(self):
        """
        Array with the horizontal grid points (cell centers).
        """
        return self._x_grid

    @property
    def y_grid(self):
        """
        Array with the vertical grid points (cell centers).
        """
        return self._y_grid

    @property
    def z_grid(self):
        """
        Array with the longitudinal grid points (cell centers).
        """
        return self._z_grid

    @property
    def nx(self):
        """
        Number of cells in the horizontal direction.
        """
        return len(self.x_grid)

    @property
    def ny(self):
        """
        Number of cells in the vertical direction.
        """
        return len(self.y_grid)

    @property
    def nz(self):
        """
        Number of cells in the longitudinal direction.
        """
        return len(self.z_grid)

    @property
    def dx(self):
        """
        Horizontal cell size in meters.
        """
        return self.x_grid[1] - self.x_grid[0]

    @property
    def dy(self):
        """
        Vertical cell size in meters.
        """
        return self.y_grid[1] - self.y_grid[0]

    @property
    def dz(self):
        """
        Longitudinal cell size in meters.
        """
        return self.z_grid[1] - self.z_grid[0]
