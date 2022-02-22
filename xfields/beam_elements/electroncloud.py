# import numpy as np
# from scipy.constants import e as qe
# from scipy.constants import c as clight
# 
# from xfields import BiGaussianFieldMap, mean_and_std
# from xfields import TriLinearInterpolatedFieldMap
# from ..longitudinal_profiles import LongitudinalProfileQGaussianData
# from ..longitudinal_profiles import LongitudinalProfileQGaussian
# from ..fieldmaps import BiGaussianFieldMapData
from ..fieldmaps import TriCubicInterpolatedFieldMapData
from ..general import _pkg_root

import xobjects as xo
import xtrack as xt


class ElectronCloud(xt.BeamElement):

    """
    Simulates the effect of an electron cloud on a bunch.

    Args:
        context (XfContext): identifies the :doc:`context <contexts>`
            on which the computation is executed.
        x_shift (float): shifts the x coordinate. Should be set equal to
            the closed orbit if the fieldmap is defined with respect to 
            the closed orbit. Measured in meters.
        y_shift (float): shifts the y coordinate. Should be set equal to
            the closed orbit if the fieldmap is defined with respect to 
            the closed orbit. Measured in meters.
        tau_shift (float): shifts the tau coordinate. Should be set equal 
            to the closed orbit if the fieldmap is defined with respect to 
            the closed orbit. Measured in meters. (tau = s/beta_0 - c t)
        dipolar_px_kick (float): subtracts a constant value from the kick to px.
            Should be set equal to the field map's kick at the closed orbit to 
            remove closed orbit distortion effects.
        dipolar_py_kick (float): subtracts a constant value from the kick to py.
            Should be set equal to the field map's kick at the closed orbit to 
            remove closed orbit distortion effects.
        dipolar_ptau_kick (float): subtracts a constant value from the kick to 
            ptau. Should be set equal to the field map's kick at the closed 
            orbit to remove closed orbit distortion effects.
        length (float): the length of the electron-cloud interaction in
            meters.
        apply_z_kick (bool): If ``True``, the longitudinal kick on the
            particles is applied. The default is ``True``.
        fieldmap (xfields.TriCubicInterpolatedFieldMap): Field map of the 
            electron cloud forces.
    Returns:
        (ElectronCloud): An electron cloud beam element.
    """

    _xofields = {
        'x_shift': xo.Float64,
        'y_shift': xo.Float64,
        'tau_shift': xo.Float64,
        'dipolar_px_kick': xo.Float64,
        'dipolar_py_kick': xo.Float64,
        'dipolar_ptau_kick': xo.Float64,
        'length': xo.Float64,
        'fieldmap': TriCubicInterpolatedFieldMapData,
        #'fieldmap': xo.Ref(TriCubicInterpolatedFieldMapData),
        }

#     def copy(self, _context=None, _buffer=None, _offset=None):
#         if _buffer is not self._buffer:
#             raise NotImplementedError
#         return SpaceCharge3D(_context=_context,
#                 _buffer=_buffer, _offset=_offset,
#                 update_on_track=self.update_on_track,
#                 length=self.length,
#                 apply_z_kick=self.apply_z_kick,
#                 fieldmap=self.fieldmap)

    def __init__(self,
                 _context=None,
                 _buffer=None,
                 _offset=None,
                 x_shift=0.,
                 y_shift=0.,
                 tau_shift=0.,
                 dipolar_px_kick=0.,
                 dipolar_py_kick=0.,
                 dipolar_ptau_kick=0.,
                 length=None,
                 apply_z_kick=True,
                 fieldmap=None,
                 ):

        # To be implemented if false
        self.apply_z_kick = apply_z_kick

        if _buffer is not None:
            _context = _buffer.context
        if _context is None:
            _context = xo.context_default

        # if fieldmap is None:
        #     # I build the fieldmap on a temporary buffer
        #     temp_buff = _context.new_buffer()
        #     fieldmap = TriLinearInterpolatedFieldMap(
        #                 _buffer=temp_buff,
        #                 rho=rho, phi=phi,
        #                 x_grid=z_grid, y_grid=y_grid, z_grid=z_grid,
        #                 x_range=x_range, y_range=y_range, z_range=z_range,
        #                 dx=dx, dy=dy, dz=dz,
        #                 nx=nx, ny=ny, nz=nz,
        #                 solver=solver,
        #                 scale_coordinates_in_solver=scale_coordinates_in_solver,
        #                 updatable=update_on_track,
        #                 fftplan=fftplan)

        self.xoinitialize(
                 _context=_context,
                 _buffer=_buffer,
                 _offset=_offset,
                 x_shift=x_shift,
                 y_shift=y_shift,
                 tau_shift=tau_shift,
                 dipolar_px_kick=dipolar_px_kick,
                 dipolar_py_kick=dipolar_py_kick,
                 dipolar_ptau_kick=dipolar_ptau_kick,
                 length=length,
                 fieldmap=fieldmap)

        # temp_buff is deallocate here

    def track(self, particles):

        """
        Computes and applies the electron cloud forces for the provided set of
        particles.

        Args:
            particles (Particles Object): Particles to be tracked.
        """

        # call C tracking kernel
        super().track(particles)


srcs = []
#srcs.append(_pkg_root.joinpath('headers/constants.h'))
srcs.append(_pkg_root.joinpath('fieldmaps/interpolated_src/tricubic_coefficients.h'))
srcs.append(_pkg_root.joinpath('fieldmaps/interpolated_src/cubic_interpolators.h'))
srcs.append(_pkg_root.joinpath('beam_elements/electroncloud_src/electroncloud.h'))

ElectronCloud.XoStruct.extra_sources = srcs
