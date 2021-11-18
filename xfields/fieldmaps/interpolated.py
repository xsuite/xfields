import numpy as np

import xobjects as xo
import xpart as xp
import xtrack as xt

from ..solvers.fftsolvers import FFTSolver3D, FFTSolver2p5D
from ..general import _pkg_root

class TriLinearInterpolatedFieldMapData(xo.Struct):
    x_min = xo.Float64
    y_min = xo.Float64
    z_min = xo.Float64
    nx = xo.Int64
    ny = xo.Int64
    nz = xo.Int64
    dx = xo.Float64
    dy = xo.Float64
    dz = xo.Float64
    rho = xo.Float64[:]
    phi = xo.Float64[:]
    dphi_dx = xo.Float64[:]
    dphi_dy = xo.Float64[:]
    dphi_dz = xo.Float64[:]

TriLinearInterpolatedFieldMapData.extra_sources = [
    _pkg_root.joinpath('headers/constants.h'),
    _pkg_root.joinpath('fieldmaps/interpolated_src/central_diff.h'),
    _pkg_root.joinpath('fieldmaps/interpolated_src/linear_interpolators.h'),
    _pkg_root.joinpath('fieldmaps/interpolated_src/charge_deposition.h'),
    ]

TriLinearInterpolatedFieldMapData.custom_kernels = {
    'central_diff': xo.Kernel(
        args=[
            xo.Arg(xo.Int32,   pointer=False, name='nelem'),
            xo.Arg(xo.Int32,   pointer=False, name='row_size'),
            xo.Arg(xo.Int32,   pointer=False, name='stride_in_dbl'),
            xo.Arg(xo.Float64, pointer=False, name='factor'),
            xo.Arg(xo.Int8,    pointer=True,  name='matrix_buffer'),
            xo.Arg(xo.Int64,   pointer=False, name='matrix_offset'),
            xo.Arg(xo.Int8,    pointer=True,  name='res_buffer'),
            xo.Arg(xo.Int64,   pointer=False, name='res_offset'),
            ],
        n_threads='nelem'
        ),
    'p2m_rectmesh3d_xparticles': xo.Kernel(
        args=[
            xo.Arg(xo.Int32,   pointer=False, name='nparticles'),
            xo.Arg(xp.Particles.XoStruct, pointer=False, name='particles'),
            xo.Arg(xo.Float64, pointer=False, name='x0'),
            xo.Arg(xo.Float64, pointer=False, name='y0'),
            xo.Arg(xo.Float64, pointer=False, name='z0'),
            xo.Arg(xo.Float64, pointer=False, name='dx'),
            xo.Arg(xo.Float64, pointer=False, name='dy'),
            xo.Arg(xo.Float64, pointer=False, name='dz'),
            xo.Arg(xo.Int32,   pointer=False, name='nx'),
            xo.Arg(xo.Int32,   pointer=False, name='ny'),
            xo.Arg(xo.Int32,   pointer=False, name='nz'),
            xo.Arg(xo.Int8,    pointer=True,  name='grid1d_buffer'),
            xo.Arg(xo.Int64,   pointer=False, name='grid1d_offset'),
            ],
        n_threads='nparticles'
        ),
    'p2m_rectmesh3d': xo.Kernel(
        args=[
            xo.Arg(xo.Int32,   pointer=False, name='nparticles'),
            xo.Arg(xo.Float64, pointer=True, name='x'),
            xo.Arg(xo.Float64, pointer=True, name='y'),
            xo.Arg(xo.Float64, pointer=True, name='z'),
            xo.Arg(xo.Float64, pointer=True, name='part_weights'),
            xo.Arg(xo.Int64,   pointer=True, name='part_state'),
            xo.Arg(xo.Float64, pointer=False, name='x0'),
            xo.Arg(xo.Float64, pointer=False, name='y0'),
            xo.Arg(xo.Float64, pointer=False, name='z0'),
            xo.Arg(xo.Float64, pointer=False, name='dx'),
            xo.Arg(xo.Float64, pointer=False, name='dy'),
            xo.Arg(xo.Float64, pointer=False, name='dz'),
            xo.Arg(xo.Int32,   pointer=False, name='nx'),
            xo.Arg(xo.Int32,   pointer=False, name='ny'),
            xo.Arg(xo.Int32,   pointer=False, name='nz'),
            xo.Arg(xo.Int8,    pointer=True,  name='grid1d_buffer'),
            xo.Arg(xo.Int64,   pointer=False, name='grid1d_offset'),
            ],
        n_threads='nparticles'
        ),
    'TriLinearInterpolatedFieldMap_interpolate_3d_map_vector': xo.Kernel(
        args=[
            xo.Arg(TriLinearInterpolatedFieldMapData, pointer=False, name='fmap'),
            xo.Arg(xo.Int64,   pointer=False, name='n_points'),
            xo.Arg(xo.Float64, pointer=True,  name='x'),
            xo.Arg(xo.Float64, pointer=True,  name='y'),
            xo.Arg(xo.Float64, pointer=True,  name='z'),
            xo.Arg(xo.Int64,   pointer=False, name='n_quantities'),
            xo.Arg(xo.Int8,    pointer=True,  name='buffer_mesh_quantities'),
            xo.Arg(xo.Int64,   pointer=True,  name='offsets_mesh_quantities'),
            xo.Arg(xo.Float64, pointer=True,  name='particles_quantities'),
            ],
        n_threads='n_points'
        ),
    }

# I add undescores in front of the names so that I can define custom properties
rename_trilinear = {ff.name:'_'+ff.name for ff
                in TriLinearInterpolatedFieldMapData._fields}
class TriLinearInterpolatedFieldMap(xo.dress(TriLinearInterpolatedFieldMapData,
                                             rename=rename_trilinear)):

    """
    Builds a linear interpolator for a 3D field map. The map can be updated
    using the Parcle In Cell method.

    Args:
        context (xobjects context): identifies the :doc:`context <contexts>`
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
                 _context=None,
                 _buffer=None,
                 _offset=None,
                 x_range=None, y_range=None, z_range=None,
                 nx=None, ny=None, nz=None,
                 dx=None, dy=None, dz=None,
                 x_grid=None, y_grid=None, z_grid=None,
                 rho=None, phi=None,
                 solver=None,
                 scale_coordinates_in_solver=(1.,1.,1.),
                 updatable=True,
                 fftplan=None
                 ):


        self.updatable = updatable
        self.scale_coordinates_in_solver = scale_coordinates_in_solver

        self._x_grid = _configure_grid('x', x_grid, dx, x_range, nx)
        self._y_grid = _configure_grid('y', y_grid, dy, y_range, ny)
        self._z_grid = _configure_grid('z', z_grid, dz, z_range, nz)

        nelem = self.nx*self.ny*self.nz
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
                 rho = nelem,
                 phi = nelem,
                 dphi_dx = nelem,
                 dphi_dy = nelem,
                 dphi_dz = nelem)

        self.compile_custom_kernels(only_if_needed=True)

        if isinstance(solver, str):
            self.solver = self.generate_solver(solver, fftplan)
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
            if solver is not None and rho is not None:
                self.update_phi_from_rho()

    def _assert_updatable(self):
        assert self.updatable, 'This FieldMap is not updatable!'

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
            (tuple of float64 array): The required quantities at the provided points.
        """

        assert len(x) == len(y) == len(z)

        pos_in_buffer_of_maps_to_interp = []
        if return_rho:
            pos_in_buffer_of_maps_to_interp.append(
                    self._xobject.rho._offset + self._xobject.rho._data_offset)
        if return_phi:
            pos_in_buffer_of_maps_to_interp.append(
                    self._xobject.phi._offset + self._xobject.phi._data_offset)
        if return_dphi_dx:
            pos_in_buffer_of_maps_to_interp.append(
                    self._xobject.dphi_dx._offset + self._xobject.dphi_dx._data_offset)
        if return_dphi_dy:
            pos_in_buffer_of_maps_to_interp.append(
                    self._xobject.dphi_dy._offset + self._xobject.dphi_dy._data_offset)
        if return_dphi_dz:
            pos_in_buffer_of_maps_to_interp.append(
                    self._xobject.dphi_dz._offset + self._xobject.dphi_dz._data_offset)

        context = self._buffer.context

        pos_in_buffer_of_maps_to_interp = context.nparray_to_context_array(
                        np.array(pos_in_buffer_of_maps_to_interp, dtype=np.int64))
        nmaps_to_interp = len(pos_in_buffer_of_maps_to_interp)
        buffer_out = context.zeros(
                shape=(nmaps_to_interp * len(x),), dtype=np.float64)
        if nmaps_to_interp > 0:
            context.kernels.TriLinearInterpolatedFieldMap_interpolate_3d_map_vector(
                    fmap=self._xobject,
                    n_points=len(x),
                    x=x, y=y, z=z,
                    n_quantities=nmaps_to_interp,
                    buffer_mesh_quantities=self._buffer.buffer,
                    offsets_mesh_quantities=pos_in_buffer_of_maps_to_interp,
                    particles_quantities=buffer_out)

        # Split buffer 
        particles_quantities = [buffer_out[ii*len(x):(ii+1)*len(x)]
                                        for ii in range(nmaps_to_interp)]

        return particles_quantities

    #@profile
    def update_from_particles(self,
                        particles=None,
                        x_p=None, y_p=None, z_p=None,
                        ncharges_p=None, state_p=None, q0_coulomb=None,
                        reset=True, update_phi=True, solver=None, force=False):

        """
        Updates the charge density at the grid using a given set of particles,
        which can be provided by a particles object or by individual arrays.
        The potential can be optionally updated accordingly.

        Args:
            particles (xtrack.Particles): xtrack particle object.
            x_p (float64 array): Horizontal coordinates of the macroparticles.
            y_p (float64 array): Vertical coordinates of the macroparticles.
            z_p (float64 array): Longitudinal coordinates of the macroparticles.
            ncharges_p (float64 array): Number of reference charges in the
                macroparticles.
            state_p (int64, array): particle state (>0 active, lost otherwise)
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
            self.rho[:,:,:] = 0.

        context = self._buffer.context

        if particles is None:
            assert (len(x_p) == len(y_p) == len(z_p) == len(ncharges_p))
            if state_p is None:
                state_p = context.zeros(shape=x_p.shape, dtype=np.int64) + 1
            else:
                assert len(state_p) == len(x_p)

            context.kernels.p2m_rectmesh3d(
                    nparticles=len(x_p),
                    x=x_p, y=y_p, z=z_p,
                    part_weights=q0_coulomb*ncharges_p,
                    part_state=state_p,
                    x0=self.x_grid[0], y0=self.y_grid[0], z0=self.z_grid[0],
                    dx=self.dx, dy=self.dy, dz=self.dz,
                    nx=self.nx, ny=self.ny, nz=self.nz,
                    grid1d_buffer=self._xobject.rho._buffer.buffer,
                    grid1d_offset=self._xobject.rho._offset
                                 +self._xobject.rho._data_offset)
        else:
            assert (x_p is None and y_p is None and z_p is None
                    and ncharges_p is None and state_p is None)
            context.kernels.p2m_rectmesh3d_xparticles(
                    nparticles=particles._capacity,
                    particles=particles,
                    x0=self.x_grid[0], y0=self.y_grid[0], z0=self.z_grid[0],
                    dx=self.dx, dy=self.dy, dz=self.dz,
                    nx=self.nx, ny=self.ny, nz=self.nz,
                    grid1d_buffer=self._xobject.rho._buffer.buffer,
                    grid1d_offset=self._xobject.rho._offset
                                 +self._xobject.rho._data_offset)

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
            self.rho[:,:,:] = rho
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
            self.phi.T[:,:,:] = phi.T
        else:
            raise ValueError('Not implemented!')

        context = self._buffer.context

        # Compute gradient
        context.kernels.central_diff(
                nelem = self.phi.size,
                row_size = self.nx,
                stride_in_dbl = self.phi.strides[0]/8,
                factor = 1/(2*self.dx),
                matrix_buffer = self._xobject.phi._buffer.buffer,
                matrix_offset = (self._xobject.phi._offset
                               + self._xobject.phi._data_offset),
                res_buffer = self._xobject.dphi_dx._buffer.buffer,
                res_offset = (self._xobject.dphi_dx._offset
                            + self._xobject.dphi_dx._data_offset))
        context.kernels.central_diff(
                nelem = self.phi.size,
                row_size = self.ny,
                stride_in_dbl = self.phi.strides[1]/8,
                factor = 1/(2*self.dy),
                matrix_buffer = self._xobject.phi._buffer.buffer,
                matrix_offset = (self._xobject.phi._offset
                               + self._xobject.phi._data_offset),
                res_buffer = self._xobject.dphi_dy._buffer.buffer,
                res_offset = (self._xobject.dphi_dy._offset
                            + self._xobject.dphi_dy._data_offset))
        context.kernels.central_diff(
                nelem = self.phi.size,
                row_size = self.nz,
                stride_in_dbl = self.phi.strides[2]/8,
                factor = 1/(2*self.dz),
                matrix_buffer = self._xobject.phi._buffer.buffer,
                matrix_offset = (self._xobject.phi._offset
                               + self._xobject.phi._data_offset),
                res_buffer = self._xobject.dphi_dz._buffer.buffer,
                res_offset = (self._xobject.dphi_dz._offset
                            + self._xobject.dphi_dz._data_offset))

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

        new_phi = solver.solve(self.rho)
        self.update_phi(new_phi)

    def generate_solver(self, solver, fftplan):

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
                    context=self._buffer.context,
                    fftplan=fftplan)
        elif solver == 'FFTSolver2p5D':
            solver = FFTSolver2p5D(
                    dx=self.dx*scale_dx,
                    dy=self.dy*scale_dy,
                    dz=self.dz*scale_dz,
                    nx=self.nx, ny=self.ny, nz=self.nz,
                    context=self._buffer.context,
                    fftplan=fftplan)
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

    # TODO: these reshapes can be avoided by allocating 3d arrays directly in the xobject
    @property
    def rho(self):
        return self._rho.reshape(
                (self.nx, self.ny, self.nz), order='F')

    @property
    def phi(self):
        """
        Electric potential at the grid points in Volts.
        """
        return self._phi.reshape(
                (self.nx, self.ny, self.nz), order='F')

    @property
    def dphi_dx(self):
        return self._dphi_dx.reshape(
                (self.nx, self.ny, self.nz), order='F')

    @property
    def dphi_dy(self):
        return self._dphi_dy.reshape(
                (self.nx, self.ny, self.nz), order='F')

    @property
    def dphi_dz(self):
        return self._dphi_dz.reshape(
                (self.nx, self.ny, self.nz), order='F')



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

