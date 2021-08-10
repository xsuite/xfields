import numpy as np
from scipy.constants import e as qe

from ..fieldmaps.bigaussian import BiGaussianFieldMap, BiGaussianFieldMapData
from ..general import _pkg_root

import xobjects as xo
import xtrack as xt


class BeamBeamBiGaussian2D(xt.BeamElement):
    """
    Simulates the effect of beam-beam on a bunch.

    Args:
        context (xobjects context): identifies the :doc:`context <contexts>`
            on which the computation is executed.
        n_particles (float64): Number of particles in the colliding bunch.
        q0 (float64): Number of particles in the colliding bunch.
        beta0 (float64): Relativistic beta of the colliding bunch.
        mean_x (float64): Horizontal position (in meters) of the colliding
            bunch. It can be updated after the object creation.
            Default is ``0.``.
        mean_y (float64): Vertical position (in meters) of the Gaussian
            distribution. It can be updated after the object creation.
            Default is ``0.``.
        sigma_x (float64): Horizontal r.m.s. size (in meters) of the colliding
            bunch. It can be updated after the object creation.
            Default is ``None``.
        sigma_y (float64): Vertical r.m.s. size (in meters) of the colliding
            bunch. It can be updated after the object creation.
            Default is ``None``.
    Returns:
        (BeamBeamBiGaussian2D): A beam-beam element.
    """
    _xofields={
        'n_particles': xo.Float64,
        'q0': xo.Float64,
        'beta0': xo.Float64,
        'fieldmap': BiGaussianFieldMapData,
        'd_px': xo.Float64,
        'd_py': xo.Float64,
        }

    def __init__(self,
            _context=None,
            _buffer=None,
            _offset=None,
            n_particles=None,
            q0=None,
            beta0=None,
            mean_x=0.,
            mean_y=0.,
            sigma_x=None,
            sigma_y=None,
            d_px=0.,
            d_py=0.,
            min_sigma_diff=1e-10):

        self.xoinitialize(
                 _context=_context,
                 _buffer=_buffer,
                 _offset=_offset)

        if not np.isscalar(beta0):
            raise ValueError('beta0 needs to be ascalar')

        self.n_particles = n_particles
        self.q0 = q0
        self.beta0 = beta0
        self.d_px = d_px
        self.d_py = d_py

        self.fieldmap = BiGaussianFieldMap(
                     _context=_context,
                     _buffer=_buffer,
                     _offset=_offset,
                     mean_x=mean_x,
                     mean_y=mean_y,
                     sigma_x=sigma_x,
                     sigma_y=sigma_y,
                     min_sigma_diff=min_sigma_diff,
                     updatable=True)

    def update(self, **kwargs):
        for kk in kwargs.keys():
            if not hasattr(self, kk):
                raise NameError(f'Unknown parameter: {kk}')
            setattr(self, kk, kwargs[kk])

    @property
    def mean_x(self):
        return self.fieldmap.mean_x

    @ mean_x.setter
    def mean_x(self, value):
        self.fieldmap.mean_x = value

    @property
    def mean_y(self):
        return self.fieldmap.mean_y

    @ mean_y.setter
    def mean_y(self, value):
        self.fieldmap.mean_y = value

    @property
    def sigma_x(self):
        return self.fieldmap.sigma_x

    @ sigma_x.setter
    def sigma_x(self, value):
        self.fieldmap.sigma_x = value

    @property
    def sigma_y(self):
        return self.fieldmap.sigma_y

    @ sigma_y.setter
    def sigma_y(self, value):
        self.fieldmap.sigma_y = value


    @classmethod
    def from_xline(cls, xline_beambeam=None,
            _context=None, _buffer=None, _offset=None):

        bb = cls(
            _context=_context,
            _buffer=_buffer,
            _offset=_offset,
            n_particles=xline_beambeam.charge/qe, # pysixtrak has it in coulumb
            q0=qe*float(xline_beambeam.enabled), # I implement the enable flag like this
            beta0=xline_beambeam.beta_r,
            mean_x=xline_beambeam.x_bb,
            mean_y=xline_beambeam.y_bb,
            sigma_x=xline_beambeam.sigma_x,
            sigma_y=xline_beambeam.sigma_y,
            d_px=xline_beambeam.d_px,
            d_py=xline_beambeam.d_py,
            min_sigma_diff=xline_beambeam.min_sigma_diff)

        return bb

srcs = []
srcs.append(_pkg_root.joinpath('headers/constants.h'))
srcs.append(_pkg_root.joinpath('fieldmaps/bigaussian_src/complex_error_function.h'))
srcs.append(_pkg_root.joinpath('fieldmaps/bigaussian_src/bigaussian.h'))
srcs.append(_pkg_root.joinpath('beam_elements/beambeam_src/beambeam.h'))

BeamBeamBiGaussian2D.XoStruct.extra_sources = srcs
