import numpy as np

import xobjects as xo
import xtrack as xt

from ..fieldmaps import TriLinearInterpolatedFieldMap

from ..fieldmaps import TriCubicInterpolatedFieldMapData
from ..fieldmaps import TriLinearInterpolatedFieldMapData
from ..general import _pkg_root

class ElectronLensInterpolated(xt.BeamElement):

    _xofields={
               'current':  xo.Float64,
               'length':   xo.Float64,
               'voltage':  xo.Float64,
               "fieldmap": TriLinearInterpolatedFieldMapData,
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
                 rho=None, phi=None,
                 solver=None,
                 fftplan=None,
                 current=None, voltage=None,
                 x_center=0., y_center=0.,
                 inner_radius=None, outer_radius=None,
                 ):

        if _buffer is not None:
            _context = _buffer.context
        if _context is None:
            _context = xo.context_default

        nz = 11
        z_range=(-1,1)
        z_grid=None
        dz=None

        fieldmap = TriLinearInterpolatedFieldMap(x_range=x_range, y_range=y_range, 
                                z_range=z_range, nx=nx, ny=ny, nz=nz, 
                                dx=dx, dy=dy, dz=dz, 
                                x_grid=x_grid, y_grid=y_grid, z_grid=z_grid,
                                solver="FFTSolver2p5D")

        for ii in range(nz):
            fieldmap.rho[:,:,ii] = rho
        fieldmap.update_phi_from_rho()

        self.xoinitialize(
                 _context=_context,
                 _buffer=_buffer,
                 _offset=_offset,
                 current=current,
                 length=length,
                 voltage=voltage,
                 fieldmap=fieldmap)

    

    def track(self, particles):

        """
        Computes and applies the electron lens forces for the provided set of
        particles.

        Args:
            particles (Particles Object): Particles to be tracked.
        """

        # call C tracking kernel
        super().track(particles)

srcs = []
srcs.append(_pkg_root.joinpath('headers/constants.h'))
srcs.append(_pkg_root.joinpath('fieldmaps/interpolated_src/linear_interpolators.h'))
#srcs.append(_pkg_root.joinpath('fieldmaps/interpolated_src/tricubic_coefficients.h'))
#srcs.append(_pkg_root.joinpath('fieldmaps/interpolated_src/cubic_interpolators.h'))
srcs.append(_pkg_root.joinpath('beam_elements/electronlens_src/electronlens_interpolated.h'))

ElectronLensInterpolated.XoStruct.extra_sources = srcs