import numpy as np

import xobjects as xo
import xtrack as xt

from ..fieldmaps import TriLinearInterpolatedFieldMap
from ..fieldmaps import TriCubicInterpolatedFieldMap

from ..fieldmaps import TriCubicInterpolatedFieldMapData
from ..fieldmaps import TriLinearInterpolatedFieldMapData
from ..general import _pkg_root

class ElectronLensInterpolated(xt.BeamElement):

    _xofields={
               'current':  xo.Float64,
               'length':   xo.Float64,
               'voltage':  xo.Float64,
               "fieldmap": TriCubicInterpolatedFieldMapData,
              }

    def __init__(self,
                 _context=None,
                 _buffer=None,
                 _offset=None,
                 length=None,
                 fieldmap=None,
                 x_range=None, y_range=None,
                 nx=None, ny=None,
                 dx=None, dy=None,
                 x_grid=None, y_grid=None,
                 rho=None,
                 current=None, voltage=None,
                 ):

        if _buffer is not None:
            _context = _buffer.context
        if _context is None:
            _context = xo.context_default

        # Choose a small number of slices with a large range in zeta.
        # By copying the same potential on each slice, the 
        # interpolation becomes 2D, doesn't depend on the 
        # longitudinal variable.
        nz = 11 
        z_range=(-1,1) 

        fieldmap = TriLinearInterpolatedFieldMap(x_range=x_range, y_range=y_range, 
                                z_range=z_range, nx=nx, ny=ny, nz=nz, 
                                dx=dx, dy=dy, 
                                x_grid=x_grid, y_grid=y_grid,
                                solver="FFTSolver2p5D")

        for ii in range(nz):
            fieldmap.rho[:,:,ii] = rho
        fieldmap.update_phi_from_rho()

        tc_fieldmap = TriCubicInterpolatedFieldMap(x_grid=fieldmap._x_grid, 
                                                   y_grid=fieldmap._y_grid, 
                                                   z_grid=fieldmap._z_grid,
                                                  )

        nx = tc_fieldmap.nx
        ny = tc_fieldmap.ny
        nz = tc_fieldmap.nz
        dx = tc_fieldmap.dx 
        dy = tc_fieldmap.dy 
        dz = tc_fieldmap.dz 
        scale = [1., dx, dy, dz, 
                dx * dy, dx * dz, dy * dz, 
                dx * dy * dz]

        phi = fieldmap.phi
        #print(phi.shape)

        ##########################################################################
        # dphi_dx = np.zeros_like(phi)
        # dphi_dy = np.zeros_like(phi)
        # dphi_dxdy = np.zeros_like(phi)
        # dphi_dx[1:-1,:] = 0.5*(phi[2:,:] - phi[:-2,:])
        # dphi_dy[:,1:-1] = 0.5*(phi[:,2:] - phi[:,:-2])
        # dphi_dxdy[1:-1,:] = 0.5*(dphi_dy[2:,:] - dphi_dy[:-2,:])
        
        # print("Setting derivatives: ")
        # kk=0.
        # for ix in range(nx):
        #     if (ix)/nx > kk:
        #         while (ix)/nx > kk:
        #             kk += 0.1
        #         print(f"{int(np.round(100*kk)):d}%..")
        #     for iy in range(ny):
        #         for iz in range(nz):
        #             index = 8 * ix + 8 * nx * iy + 8 * nx * ny * iz
        #             tc_fieldmap._phi_taylor[index+0] = phi[ix, iy, 0]
        #             tc_fieldmap._phi_taylor[index+1] = dphi_dx[ix, iy, 0]
        #             tc_fieldmap._phi_taylor[index+2] = dphi_dy[ix, iy, 0]
        #             tc_fieldmap._phi_taylor[index+3] = 0.
        #             tc_fieldmap._phi_taylor[index+4] = dphi_dxdy[ix, iy, 0]
        #             tc_fieldmap._phi_taylor[index+5] = 0.
        #             tc_fieldmap._phi_taylor[index+6] = 0.
        #             tc_fieldmap._phi_taylor[index+7] = 0.
        ##########################################################################

        ## Optimized version of above block ##########################################
        phi_slice = np.zeros([phi.shape[0], phi.shape[1], 8])
        for iz in range(nz):
            phi_slice[:,:,0] = phi[:,:, iz]
            phi_slice[1:-1,:,1] =  0.5*(phi[2:,:,iz] - phi[:-2,:,iz])
            phi_slice[:,1:-1,2] = 0.5*(phi[:,2:,iz] - phi[:,:-2,iz])
            phi_slice[1:-1,:,4] = 0.5*(phi_slice[2:,:,2] - phi_slice[:-2,:,2])

            flat_slice = phi_slice.transpose(1,0,2).flatten()
            len_slice = len(flat_slice)
            index_offset = 8 * nx * ny * iz
            tc_fieldmap._phi_taylor[index_offset:index_offset+len_slice] = flat_slice
        ##############################################################################

        self.xoinitialize(
                 _context=_context,
                 _buffer=_buffer,
                 _offset=_offset,
                 current=current,
                 length=length,
                 voltage=voltage,
                 fieldmap=tc_fieldmap)


srcs = []
srcs.append(_pkg_root.joinpath('headers/constants.h'))
srcs.append(_pkg_root.joinpath('fieldmaps/interpolated_src/tricubic_coefficients.h'))
srcs.append(_pkg_root.joinpath('fieldmaps/interpolated_src/cubic_interpolators.h'))
srcs.append(_pkg_root.joinpath('beam_elements/electronlens_src/electronlens_interpolated.h'))

ElectronLensInterpolated.XoStruct.extra_sources = srcs