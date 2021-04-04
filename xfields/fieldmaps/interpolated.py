import numpy as np

from xobjects.context import ContextDefault
import xobjects as xo

from .base import FieldMap
from ..solvers.fftsolvers import FFTSolver3D, FFTSolver2p5D
from ..contexts import add_default_kernels


class TriLinearInterpolatedFieldMap(FieldMap):

    """
    Builds a linear interpolator for a 3D field map. The map can be updated
    using the Parcle In Cell method.

    Args:
        context (XfContext): identifies the :doc:`context <contexts>`
            on which the computation is executed.
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
        scale_coordinates_in_solver (tuple): Three coefficients used to rescale
            the grid coordinates in the definition of the solver. The default is
            (1.,1.,1.).
        updatable (bool): If ``True`` the field map can be updated after
            creation. Default is ``True``.
    Returns:
        (TriLinearInterpolatedFieldMap): Interpolator object.
    """

    def __init__(self,
                 context=None,
                 x_range=None, y_range=None, z_range=None,
                 nx=None, ny=None, nz=None,
                 dx=None, dy=None, dz=None,
                 x_grid=None, y_grid=None, z_grid=None,
                 rho=None, phi=None,
                 solver=None,
                 scale_coordinates_in_solver=(1.,1.,1.),
                 updatable=True,
                 ):


        if context is None:
            context = ContextDefault()

        add_default_kernels(context)

        self.updatable = updatable
        self.context = context
        self.scale_coordinates_in_solver = scale_coordinates_in_solver

        self._x_grid = _configure_grid('x', x_grid, dx, x_range, nx)
        self._y_grid = _configure_grid('y', y_grid, dy, y_range, ny)
        self._z_grid = _configure_grid('z', z_grid, dz, z_range, nz)

        # Prepare arrays (contiguous to use a single pointer in C/GPU)
        self._maps_buffer_dev = context.nparray_to_context_array(
                np.zeros((self.nx, self.ny, self.nz, 5),
                         dtype=np.float64, order='F'))

        # These are slices (they are are on the context)
        self._rho_dev = self._maps_buffer_dev[:, :, :, 0]
        self._phi_dev = self._maps_buffer_dev[:, :, :, 1]
        self._dphi_dx_dev = self._maps_buffer_dev[:, :, :, 2]
        self._dphi_dy_dev = self._maps_buffer_dev[:, :, :, 3]
        self._dphi_dz_dev = self._maps_buffer_dev[:, :, :, 4]


        if isinstance(solver, str):
            self.solver = self.generate_solver(solver)
        else:
            #TODO: consistency check to be added
            self.solver = solver

        # Set rho
        if rho is not None:
            self.update_rho(rho, force=True)

        # Set phi
        if phi is not None:
            self.update_phi(phi, force=True)
        else:
            if solver is not None:
                self.update_phi_from_rho()

    #@profile
    def get_values_at_points(self,
            x, y, z,
            return_rho=True,
            return_phi=True,
            return_dphi_dx=True,
            return_dphi_dy=True,
            return_dphi_dz=True):

        """
        Returns the charge density, the field potential and its derivatives
        at the points specified by x, y, z. The output can be customized (see below).
        Zeros are returned for points outside the grid.

        Args:
            x (float64 array): Horizontal coordinates at which the field is evaluated.
            y (float64 array): Vertical coordinates at which the field is evaluated.
            z (float64 array): Longitudinal coordinates at which the field is evaluated.
            return_rho (bool): If ``True``, the charge density at the given points is
                returned.
            return_phi (bool): If ``True``, the potential at the given points is returned.
            return_dphi_dx (bool): If ``True``, the horizontal derivative of the potential
                at the given points is returned.
            return_dphi_dy: If ``True``, the vertical derivative of the potential
                at the given points is returned.
            return_dphi_dz: If ``True``, the longitudinal derivative of the potential
                at the given points is returned.
        Returns:
            (tuple of float64 array): The required quantitie at the provided points.
        """

        assert len(x) == len(y) == len(z)

        pos_in_buffer_of_maps_to_interp = []
        mapsize = self.nx*self.ny*self.nz
        if return_rho:
            pos_in_buffer_of_maps_to_interp.append(0*mapsize)
        if return_phi:
            pos_in_buffer_of_maps_to_interp.append(1*mapsize)
        if return_dphi_dx:
            pos_in_buffer_of_maps_to_interp.append(2*mapsize)
        if return_dphi_dy:
            pos_in_buffer_of_maps_to_interp.append(3*mapsize)
        if return_dphi_dz:
            pos_in_buffer_of_maps_to_interp.append(4*mapsize)

        pos_in_buffer_of_maps_to_interp = self.context.nparray_to_context_array(
                        np.array(pos_in_buffer_of_maps_to_interp, dtype=np.int32))
        nmaps_to_interp = len(pos_in_buffer_of_maps_to_interp)
        buffer_out = self.context.zeros(
                shape=(nmaps_to_interp * len(x),), dtype=np.float64)
        if nmaps_to_interp > 0:
            self.context.kernels.m2p_rectmesh3d(
                    nparticles=len(x),
                    x=x, y=y, z=z,
                    x0=self.x_grid[0], y0=self.y_grid[0], z0=self.z_grid[0],
                    dx=self.dx, dy=self.dy, dz=self.dz,
                    nx=self.nx, ny=self.ny, nz=self.nz,
                    n_quantities=nmaps_to_interp,
                    offsets_mesh_quantities=pos_in_buffer_of_maps_to_interp,
                    mesh_quantity=self._maps_buffer_dev,
                    particles_quantity=buffer_out)

        # Split buffer 
        particles_quantities = [buffer_out[ii*len(x):(ii+1)*len(x)]
                                        for ii in range(nmaps_to_interp)]

        return particles_quantities

    #@profile
    def update_from_particles(self, x_p, y_p, z_p, ncharges_p, q0_coulomb,
                        reset=True, update_phi=True, solver=None, force=False):

        """
        Updates the charge density at the grid using a given set of particles.
        The potential can be optionally updated accordingly.

        Args:
            x_p (float64 array): Horizontal coordinates of the macroparticles.
            y_p (float64 array): Vertical coordinates of the macroparticles.
            z_p (float64 array): Longitudinal coordinates of the macroparticles.
            ncharges_p (float64 array): Number of reference charges in the
                macroparticles.
            q0_coulomb (float64): Reference charge in Coulomb.
            reset (bool): If ``True`` the stored charge density is overwritten
                with the provided one. If ``False`` the provided charge density
                is added to the stored one. The default is ``True``.
            update_phi (bool): If ``True`` the stored potential is recalculated
                from the stored charge density.
            solver (Solver object): solver object to be used to solve Poisson's
                equation (compute phi from rho). If ``None`` is provided the solver
                attached to the fieldmap is used (if any). The default is ``None``.
            force (bool): If ``True`` the potential is updated even if the
                map is declared as not updateable. The default is ``False``.
        """

        if not force:
            self._assert_updatable()

        if reset:
            self._rho_dev[:,:,:] = 0.

        assert len(x_p) == len(y_p) == len(z_p) == len(ncharges_p)

        self.context.kernels.p2m_rectmesh3d(
                nparticles=len(x_p),
                x=x_p, y=y_p, z=z_p,
                part_weights=q0_coulomb*ncharges_p,
                x0=self.x_grid[0], y0=self.y_grid[0], z0=self.z_grid[0],
                dx=self.dx, dy=self.dy, dz=self.dz,
                nx=self.nx, ny=self.ny, nz=self.nz,
                grid1d=self._rho_dev)

        if update_phi:
            self.update_phi_from_rho(solver=solver)

    def update_rho(self, rho, reset=True, force=False):
        """
        Updates the charge density on the grid.

        Args:
            rho (float64 array): Charge density at the grid points in C/m^3.
            reset (bool): If ``True`` the stored charge density is overwritten
                with the provided one. If ``False`` the provided charge density
                is added to the stored one. The default is ``True``.
            force (bool): If ``True`` the charge density is updated even if the
                map is declared as not updateable. The default is ``False``.
        """

        if not force:
            self._assert_updatable()

        if reset:
            self._rho_dev[:,:,:] = rho
        else:
            raise ValueError('Not implemented!')

    #@profile
    def update_phi(self, phi, reset=True, force=False):

        """
        Updates the potential on the grid. The stored derivatives are also
        updated.

        Args:
            rho (float64 array): Potential at the grid points.
            reset (bool): If ``True`` the stored potential is overwritten
                with the provided one. If ``False`` the provided potential
                is added to the stored one. The default is ``True``.
            force (bool): If ``True`` the potential is updated even if the
                map is declared as not updateable. The default is ``False``.
        """

        if not force:
            self._assert_updatable()

        if reset:
            self._phi_dev[:,:,:] = phi
        else:
            raise ValueError('Not implemented!')

        # Compute gradient
        if isinstance(self.context, xo.ContextPyopencl):
            # Copies are needed only for pyopencl
            self._dphi_dx_dev[1:self.nx-1,:,:] = 1/(2*self.dx)*(
                    self._phi_dev[2:,:,:].copy()-self._phi_dev[:-2,:,:].copy())
            self._dphi_dy_dev[:,1:self.ny-1,:] = 1/(2*self.dy)*(
                    self._phi_dev[:,2:,:].copy()-self._phi_dev[:,:-2,:].copy())
            self._dphi_dz_dev[:,:,1:self.nz-1] = 1/(2*self.dz)*(
                    self._phi_dev[:,:,2:].copy()-self._phi_dev[:,:,:-2].copy())
        else:
            self._dphi_dx_dev[1:self.nx-1,:,:] = 1/(2*self.dx)*(
                    self._phi_dev[2:,:,:]-self._phi_dev[:-2,:,:])
            self._dphi_dy_dev[:,1:self.ny-1,:] = 1/(2*self.dy)*(
                    self._phi_dev[:,2:,:]-self._phi_dev[:,:-2,:])
            self._dphi_dz_dev[:,:,1:self.nz-1] = 1/(2*self.dz)*(
                    self._phi_dev[:,:,2:]-self._phi_dev[:,:,:-2])


    #@profile
    def update_phi_from_rho(self, solver=None):

        """
        Updates the potential on the grid (phi) from the charge density on the
        grid (phi). It requires a :doc:`Poisson solver object <solvers>`. If
        none is provided the one attached to the fieldmap is used (if any).

        Args:
            solver (Solver object): solver object to be used to solve Poisson's
                equation. If ``None`` is provided the solver attached to the fieldmap
                is used (if any). The default is ``None``.
        """

        self._assert_updatable()

        if solver is None:
            if hasattr(self, 'solver'):
                solver = self.solver
            else:
                raise ValueError('I have no solver to compute phi!')

        new_phi = solver.solve(self._rho_dev)
        self.update_phi(new_phi)

    def generate_solver(self, solver):

        """
        Generates a Poisson solver associated to the defined grid.

        Args:
            solver (str): Defines the Poisson solver to be used
            to compute phi from rho. Accepted values are ``FFTSolver3D`` and
            ``FFTSolver2p5D``.
        Returns:
            (Solver): Solver object associated to the defined grid.
        """

        scale_dx, scale_dy, scale_dz = self.scale_coordinates_in_solver

        if solver == 'FFTSolver3D':
            solver = FFTSolver3D(
                    dx=self.dx*scale_dx,
                    dy=self.dy*scale_dy,
                    dz=self.dz*scale_dz,
                    nx=self.nx, ny=self.ny, nz=self.nz,
                    context=self.context)
        elif solver == 'FFTSolver2p5D':
            solver = FFTSolver2p5D(
                    dx=self.dx*scale_dx,
                    dy=self.dy*scale_dy,
                    dz=self.dz*scale_dz,
                    nx=self.nx, ny=self.ny, nz=self.nz,
                    context=self.context)
        else:
            raise ValueError(f'solver name {solver} not recognized')

        return solver

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

    @property
    def rho(self):
        """
        Charge density at the grid points in Coulomb/m^3.
        """
        return self._rho_dev

    @property
    def phi(self):
        """
        Electric potential at the grid points in Volts.
        """
        return self._phi_dev

def _configure_grid(vname, v_grid, dv, v_range, nv):

    # Check input consistency
    if v_grid is not None:
        assert dv is None, (f'd{vname} cannot be given '
                            f'if {vname}_grid is provided ')
        assert nv is None, (f'n{vname} cannot be given '
                            f'if {vname}_grid is provided ')
        assert v_range is None, (f'{vname}_range cannot be given '
                                 f'if {vname}_grid is provided')
        ddd = np.diff(v_grid)
        assert np.allclose(ddd,ddd[0]), (f'{vname}_grid must be '
                                          'unifirmly spaced')
    else:
        assert v_range is not None, (f'{vname}_grid or {vname}_range '
                                     f'must be provided')
        assert len(v_range)==2, (f'{vname}_range must be in the form '
                                 f'({vname}_min, {vname}_max)')
        if dv is not None:
            assert nv is None, (f'n{vname} cannot be given '
                                    f'if d{vname} is provided ')
            v_grid = np.arange(v_range[0], v_range[1]+0.1*dv, dv)
        else:
            assert nv is not None, (f'n{vname} must be given '
                                    f'if d{vname} is not provided ')
            v_grid = np.linspace(v_range[0], v_range[1], nv)

    return v_grid

# ## First sketch ##
#
# class InterpolatedFieldMap(FieldMap):
# 
#     def __init__(self, rho=None, phi=None,
#                  x_grid=None, y_grid=None, z_grid=None,
#                  dx=None, dy=None, dz=None,
#                  x_range=None, y_range=None, z_range=None,
#                  xy_interp_method='linear',
#                  z_interp_method='linear',
#                  context=None):
#         '''
#         interp_methods can be 'linear' or 'cubic'
#         '''
# 
#         # 1D, 2D or 3D is inferred from the matrix size 
#         pass
# 
#     def get_values_at_points(self,
#             x, y, z=0,
#             return_rho=False,
#             return_phi=False,
#             return_dphi_dx=False,
#             return_dphi_dy=False,
#             return_dphi_dz=False):
#         pass
# 
# 
# class InterpolatedFieldMapWithBoundary(FieldMap):
# 
#     def __init__(self, rho=None, phi=None,
#                  x_grid=None, y_grid=None, z_grid=None,
#                  dx=None, dy=None, dz=None,
#                  xy_interp_method='linear',
#                  z_interp_method='linear',
#                  boundary=None,
#                  context=None):
#         '''
#         Does the Shortley-Weller interpolation close to the boundary.
#         Might need to force 2D and linear for now.
#         '''
# 
#         pass
# 
#     def get_values_at_points(self,
#             x, y, z=0,
#             return_rho=False,
#             return_phi=False,
#             return_dphi_dx=False,
#             return_dphi_dy=False,
#             return_dphi_dz=False):
#         pass
