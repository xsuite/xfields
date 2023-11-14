from pathlib import Path
import numpy as np
from ..general import _pkg_root

import xfields as xf
import xobjects as xo
import xtrack as xt

_configure_grid = xf.fieldmaps.interpolated._configure_grid

COORDS = ['x', 'px', 'y', 'py', 'zeta', 'delta']
SECOND_MOMENTS={}
for cc1 in COORDS:
    for cc2 in COORDS:
        if cc1 + '_' + cc2 in SECOND_MOMENTS or cc2 + '_' + cc1 in SECOND_MOMENTS:
            continue
        SECOND_MOMENTS[cc1 + '_' + cc2] = (cc1, cc2)

_xof = {
    'z_min': xo.Float64,
    'num_slices': xo.Int64,
    'dzeta': xo.Float64,
    'i_bunch_0': xo.Int64,
    'num_bunches': xo.Int64,
    'bunch_spacing_zeta': xo.Float64,
    'particles_per_slice': xo.Float64[:],
}
for cc in COORDS:
    _xof['sum_'+cc] = xo.Float64[:]
for ss in SECOND_MOMENTS:
    _xof['sum_'+ss] = xo.Float64[:]

short_second_mom_names={}
for ss in SECOND_MOMENTS:
    short_second_mom_names[ss.replace('_','')] = ss
# Gives {'xx': 'x_x', 'xpx': 'x_px', ...}

_rnm = {}

for kk in _xof.keys():
    _rnm[kk] = '_' + kk

class UniformBinSlicer(xt.BeamElement):
    _xofields = _xof
    _rename = _rnm

    _extra_c_sources = [
        xt.general._pkg_root.joinpath('headers/atomicadd.h'),
        _pkg_root.joinpath('slicers/slicers_src/uniform_bin_slicer.h')
    ]

    _per_particle_kernels = {
            '_slice_kernel_all': xo.Kernel(
                c_name='UniformBinSlicer_slice',
                args=[
                    xo.Arg(xo.Int64, name='use_bunch_index_array'),
                    xo.Arg(xo.Int64, name='use_slice_index_array'),
                    xo.Arg(xo.Int64, pointer=True, name='i_slice_particles'),
                    xo.Arg(xo.Int64, pointer=True, name='i_bunch_particles')
                ]),
            '_slice_kernel_x_only': xo.Kernel(
                c_name='UniformBinSlicer_slice_x_only',
                args=[
                    xo.Arg(xo.Int64, name='use_bunch_index_array'),
                    xo.Arg(xo.Int64, name='use_slice_index_array'),
                    xo.Arg(xo.Int64, pointer=True, name='i_slice_particles'),
                    xo.Arg(xo.Int64, pointer=True, name='i_bunch_particles')
                ]),
        }

    def __init__(self, zeta_range=None, num_slices=None, dzeta=None, zeta_slices=None,
                 num_bunches=None, i_bunch_0=None, bunch_spacing_zeta=None,
                 moments='all', **kwargs):

        self._zeta_slices = _configure_grid('zeta', zeta_slices, dzeta, zeta_range, num_slices)
        num_bunches = num_bunches or 0
        i_bunch_0 = i_bunch_0 or 0
        bunch_spacing_zeta = bunch_spacing_zeta or 0

        all_moments = COORDS + list(SECOND_MOMENTS.keys())
        if moments == 'all':
            selected_moments = all_moments
        else:
            assert isinstance (moments, (list, tuple))
            selected_moments = []
            for mm in moments:
                if mm in COORDS:
                    selected_moments.append(mm)
                elif mm in SECOND_MOMENTS:
                    selected_moments.append(mm)
                    for cc in SECOND_MOMENTS[mm]:
                        if cc not in SECOND_MOMENTS:
                            selected_moments.append(cc)
                elif mm in short_second_mom_names:
                    selected_moments.append(short_second_mom_names[mm])
                    for cc in SECOND_MOMENTS[short_second_mom_names[mm]]:
                        if cc not in SECOND_MOMENTS:
                            selected_moments.append(cc)
                else:
                    raise ValueError(f'Unknown moment {mm}')

        allocated_sizes = {}
        for mm in all_moments:
            if mm in selected_moments:
                allocated_sizes['sum_' + mm] = (num_bunches or 1) * self.num_slices
            else:
                allocated_sizes['sum_' + mm] = 0

        self.xoinitialize(z_min=self._zeta_slices[0], num_slices=self.num_slices,
                          dzeta=self.dzeta,
                          num_bunches=num_bunches, i_bunch_0=i_bunch_0,
                          bunch_spacing_zeta=bunch_spacing_zeta,
                          particles_per_slice=(num_bunches or 1) * self.num_slices, # initialization with tuple not working
                          **allocated_sizes, **kwargs)

        self._slice_kernel = self._slice_kernel_all

    def slice(self, particles, i_slice_particles=None, i_bunch_particles=None):

        self.particles_per_slice[:] = 0

        if i_bunch_particles is not None:
            use_bunch_index_array = 1
        else:
            use_bunch_index_array = 0
            i_bunch_particles = particles.particle_id[:1] # Dummy
        if i_slice_particles is not None:
            use_slice_index_array = 1
        else:
            use_slice_index_array = 0
            i_slice_particles = particles.particle_id[:1] # Dummy

        for cc in COORDS:
            getattr(self, '_sum_' + cc)[:] = 0
        for ss in SECOND_MOMENTS:
            getattr(self, '_sum_' + ss)[:] = 0

        self._slice_kernel(particles=particles,
                    use_bunch_index_array=use_bunch_index_array,
                    use_slice_index_array=use_slice_index_array,
                    i_slice_particles=i_slice_particles,
                    i_bunch_particles=i_bunch_particles)

    def track(self, particles):
        self.slice(particles)

    @property
    def zeta_centers(self):
        """
        Array with the grid points (bin centers).
        """
        if self.num_bunches <= 0:
            return self._zeta_slices
        else:
            out = np.zeros((self.num_bunches, self.num_slices))
            for ii in range(self.num_bunches):
                out[ii, :] = (self._zeta_slices - (ii + self.i_bunch_0) * self.bunch_spacing_zeta)
            return out

    @property
    def num_slices(self):
        """
        Number of bins
        """
        return len(self._zeta_slices)

    @property
    def dzeta(self):
        """
        Bin size in meters.
        """
        return self._zeta_slices[1] - self._zeta_slices[0]

    @property
    def num_bunches(self):
        """
        Number of bunches
        """
        return self._num_bunches

    @property
    def i_bunch_0(self):
        """
        Index of the first bunch
        """
        return self._i_bunch_0

    @property
    def bunch_spacing_zeta(self):
        """
        Spacing between bunches in meters
        """
        return self._bunch_spacing_zeta

    @property
    def moments(self):
        """
        List of moments that are being recorded
        """
        out = []
        for cc in COORDS:
            if len(getattr(self._xobject, 'sum_' + cc)) > 0:
                out.append(cc)
        for ss in SECOND_MOMENTS:
            if len(getattr(self._xobject, 'sum_' + ss)) > 0:
                out.append(ss)

        return out

    @property
    def particles_per_slice(self):
        """
        Number of particles per slice
        """
        return self._reshape_for_multibunch(self._particles_per_slice)

    def sum(self, cc, cc2=None):
        """
        Sum of the quantity cc per slice
        """
        if cc in short_second_mom_names:
            cc = short_second_mom_names[cc]
        if cc2 is not None:
            cc = cc + '_' + cc2
        if len(getattr(self._xobject, 'sum_' + cc)) == 0:
            raise ValueError(f'Moment `{cc}` not recorded')
        return self._reshape_for_multibunch(getattr(self, '_sum_' + cc))

    def mean(self, cc, cc2=None):
        """
        Mean of the quantity cc per slice
        """
        out = 0 * self.particles_per_slice
        mask_nonzero = self.particles_per_slice > 0
        out[mask_nonzero] = (self.sum(cc, cc2)[mask_nonzero]
                             / self.particles_per_slice[mask_nonzero])
        return out

    def cov(self, cc1, cc2=None):
        """
        Covariance between cc1 and cc2 per slice
        """
        if cc2 is None:
            if cc1 in short_second_mom_names:
                cc1 = short_second_mom_names[cc1]
            cc1, cc2 = cc1.split('_')
        return self.mean(cc1, cc2) - self.mean(cc1) * self.mean(cc2)

    def var(self, cc):
        """
        Variance of the quantity cc per slice
        """
        return self.cov(cc, cc)

    def std(self, cc):
        """
        Standard deviation of the quantity cc per slice
        """
        return np.sqrt(self.var(cc))

    def _reshape_for_multibunch(self, data):
        if self.num_bunches <= 0:
            return data
        else:
            return data.reshape(self.num_bunches, self.num_slices)