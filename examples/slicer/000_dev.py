import xtrack as xt
import xpart as xp
import xobjects as xo
import xfields as xf

from pathlib import Path

import numpy as np

# line = xt.Line.from_json(
#     '../../../xtrack/test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json')
# line.particle_ref = xt.Particles(p0c=26e9, mass0=xt.PROTON_MASS_EV)
# line.build_tracker()
# tw = line.twiss()

# num_partilces_per_bunch = 100
# num_bunches = 3
# total_intensity_particles_bunch = 1e11

# beam = xp.generate_matched_gaussian_bunch(
#             num_particles=num_partilces_per_bunch * num_bunches,
#             total_intensity_particles=total_intensity_particles_bunch * num_bunches,
#             sigma_z=0.1, nemitt_x=2.5e-6, nemitt_y=2.5e-6, line=line)

# harmonic_number = 4620
# dz_bucket = tw.circumference / harmonic_number
# bunch_spacing_buckets = 5

# for ii in range(num_bunches):
#     beam.zeta[ii * num_partilces_per_bunch:(ii+1) * num_partilces_per_bunch] += (
#         ii * bunch_spacing_buckets * dz_bucket)


_configure_grid = xf.fieldmaps.interpolated._configure_grid

coords = ['x', 'px', 'y', 'py', 'zeta', 'delta']
second_moments={}
for cc1 in coords:
    for cc2 in coords:
        if cc1 + '_' + cc2 in second_moments or cc2 + '_' + cc1 in second_moments:
            continue
        second_moments[cc1 + '_' + cc2] = (cc1, cc2)

_xof = {
    'z_min': xo.Float64,
    'num_slices': xo.Int64,
    'dzeta': xo.Float64,
    'i_bunch_0': xo.Int64,
    'num_bunches': xo.Int64,
    'bunch_spacing_zeta': xo.Float64,
    'particles_per_slice': xo.Float64[:],
}
for cc in coords:
    _xof['sum_'+cc] = xo.Float64[:]
for ss in second_moments:
    _xof['sum_'+ss] = xo.Float64[:]

short_second_mom_names={}
for ss in second_moments:
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
        Path('uniform_bin_slicer.h')
    ]

    _per_particle_kernels = {
            '_slice_kernel': xo.Kernel(
                c_name='UniformBinSlicer_slice',
                args=[
                    xo.Arg(xo.Int64, name='use_bunch_index_array'),
                    xo.Arg(xo.Int64, name='use_slice_index_array'),
                    xo.Arg(xo.Int64, pointer=True, name='i_slice_particles'),
                    xo.Arg(xo.Int64, pointer=True, name='i_bunch_particles')
                ]),
        }

    def __init__(self, zeta_range=None, nbins=None, dzeta=None, zeta_slices=None,
                 num_bunches=None, i_bunch_0=None, bunch_spacing_zeta=None,
                 **kwargs):

        self._zeta_slices = _configure_grid('zeta', zeta_slices, dzeta, zeta_range, nbins)
        num_bunches = num_bunches or 0
        i_bunch_0 = i_bunch_0 or 0
        bunch_spacing_zeta = bunch_spacing_zeta or 0

        self.xoinitialize(z_min=self._zeta_slices[0], num_slices=self.num_slices,
                          dzeta=self.dzeta,
                          num_bunches=num_bunches, i_bunch_0=i_bunch_0,
                          bunch_spacing_zeta=bunch_spacing_zeta,
                          particles_per_slice=(num_bunches or 1) * self.num_slices, # initialization with tuple not working
                          **{'sum_' + cc: (num_bunches or 1) * self.num_slices for cc in coords + list(second_moments.keys())},
                          **kwargs)

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

        for cc in coords:
            getattr(self, '_sum_' + cc)[:] = 0
        for ss in second_moments:
            getattr(self, '_sum_' + ss)[:] = 0

        self._slice_kernel(particles=particles,
                    use_bunch_index_array=use_bunch_index_array,
                    use_slice_index_array=use_slice_index_array,
                    i_slice_particles=i_slice_particles,
                    i_bunch_particles=i_bunch_particles)

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
                out[ii, :] = (self._zeta_slices + ii * self.bunch_spacing_zeta)
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

###############################################
# Check slice attribution (single-bunch mode) #
###############################################

slicer = UniformBinSlicer(zeta_range=(-1, 1), nbins=3)
assert slicer.num_bunches == 0 # Single-bunch mode

p0 = xt.Particles(zeta  =[-2, -1.51, -1.49, -1, -0.51, -0.49, 0, 0.49, 0.51,  1, 1.49, 1.51,  2, 2.51],
                  weight=[10,  10,    10,    10, 10,    20,    20,  20,   30,  30,  30,   40, 40,   40],
                  x=     [0.,  1.,   2.,     3.,  4.,    5.,    6., 7.,    8.,  9.,  10.,  11., 12.,  13.],
                  y=     [13., 12.,  11.,    10., 9.,    8.,    7., 6.,    5.,  4.,  3.,   2.,  1.,   0.]
                  )
p0.state[-1] = 0

p = p0.copy()

i_slice_expected    = [-1, -1,    0,      0,  0,    1,     1,    1,    2, 2, 2,    -1,  -1, -999]

ss = 0 * p.x
ctx= xo.ContextCpu()
i_slice_particles = p.particle_id * 0 - 999
i_bunch_particles = p.particle_id * 0 - 9999
slicer.slice(particles=p, i_bunch_particles=i_bunch_particles,
                    i_slice_particles=i_slice_particles)

assert np.all(np.array(i_slice_expected) == i_slice_particles)
assert np.all(i_bunch_particles == -9999)


expected_particles_per_slice = np.array([30, 60, 90])
assert np.allclose(slicer.particles_per_slice, expected_particles_per_slice,
                     atol=1e-12, rtol=0)

##############################################
# Check slice attribution (multi-bunch mode) #
##############################################

bunch_spacing_zeta = 10.

p1 = p0.copy()
p2 = p0.copy()
p2.zeta += bunch_spacing_zeta
p2.weight *= 10
p3 = p0.copy()
p3.zeta += 2 * bunch_spacing_zeta
p3.weight *= 100
p4 = p0.copy()
p4.zeta += 3 * bunch_spacing_zeta
p4.weight *= 1000

p = xt.Particles.merge([p1, p2, p3, p4])

i_bunch_particles = p.particle_id * 0 - 999
i_slice_particles = p.particle_id * 0 - 999

slicer = UniformBinSlicer(zeta_range=(-1, 1), nbins=3, i_bunch_0=0,
                          num_bunches=4, bunch_spacing_zeta=bunch_spacing_zeta)
slicer.slice(particles=p,
                    i_slice_particles=i_slice_particles,
                    i_bunch_particles=i_bunch_particles)

i_slice_expected  = np.array([
    -1, -1,    0,      0,  0,    1,     1,    1,    2, 2, 2,    -1,  -1,
    -1, -1,    0,      0,  0,    1,     1,    1,    2, 2, 2,    -1,  -1,
    -1, -1,    0,      0,  0,    1,     1,    1,    2, 2, 2,    -1,  -1,
    -1, -1,    0,      0,  0,    1,     1,    1,    2, 2, 2,    -1,  -1,
    -999, -999, -999, -999
])
i_bunch_expected  = np.array([
    -1, -1,    0,      0,  0,    0,     0,    0,    0, 0, 0,     0,   0,
     0,  0,    1,      1,  1,    1,     1,    1,    1, 1, 1,     1,   1,
     1,  1,    2,      2,  2,    2,     2,    2,    2, 2, 2,     2,   2,
     2,  2,    3,      3,  3,    3,     3,    3,    3, 3, 3,     3,   3,
    -999, -999, -999, -999
])

expected_particles_per_slice = np.array([
    [30, 60, 90],
    [300, 600, 900],
    [3000, 6000, 9000],
    [30000, 60000, 90000],
])

assert np.all(i_slice_particles == i_slice_expected)
assert np.all(i_bunch_particles == i_bunch_expected)
assert np.allclose(slicer.particles_per_slice, expected_particles_per_slice,
                   atol=1e-12, rtol=0)


#####################################
# Check moments (single-bunch mode) #
#####################################

slicer_single_bunch = UniformBinSlicer(zeta_range=(-1, 1), nbins=3)

p = xt.Particles(zeta=[0.99, 1.0, 1.01],
                 weight=[1, 2, 1],
                 x = [99, 100, 101],
                 y = [201,200, 199])
slicer_single_bunch.slice(p)

assert slicer_single_bunch.bunch_spacing_zeta == 0

assert np.allclose(slicer_single_bunch.zeta_centers, np.array([-1, 0, 1]), rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.particles_per_slice, [0, 0, p.weight.sum()], rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.sum('x'), [0, 0, (p.x * p.weight).sum()], rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.sum('y'), [0, 0, (p.y * p.weight).sum()], rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.sum('zeta'), [0, 0, (p.zeta * p.weight).sum()], rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.sum('xx'), [0, 0, (p.x**2 * p.weight).sum()], rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.sum('yy'), [0, 0, (p.y**2 * p.weight).sum()], rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.sum('zetazeta'), [0, 0, (p.zeta**2 * p.weight).sum()], rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.sum('xy'), [0, 0, (p.x * p.y * p.weight).sum()], rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.sum('xzeta'), [0, 0, (p.x * p.zeta * p.weight).sum()], rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.mean('x'), [0, 0, (p.x * p.weight).sum() / p.weight.sum()], rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.mean('y'), [0, 0, (p.y * p.weight).sum() / p.weight.sum()], rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.mean('xx'), [0, 0, (p.x**2 * p.weight).sum() / p.weight.sum()], rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.mean('yy'), [0, 0, (p.y**2 * p.weight).sum() / p.weight.sum()], rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.mean('zetazeta'), [0, 0, (p.zeta**2 * p.weight).sum() / p.weight.sum()], rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.mean('xy'), [0, 0, (p.x * p.y * p.weight).sum() / p.weight.sum()], rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.mean('xzeta'), [0, 0, (p.x * p.zeta * p.weight).sum() / p.weight.sum()], rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.cov('x', 'y'),
    slicer_single_bunch.mean('xy') - slicer_single_bunch.mean('x') * slicer_single_bunch.mean('y'),
    rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.var('x'), slicer_single_bunch.cov('x', 'x'),
    rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.var('x'), slicer_single_bunch.mean('xx') - slicer_single_bunch.mean('x')**2,
    rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.var('zeta'), slicer_single_bunch.mean('zetazeta') - slicer_single_bunch.mean('zeta')**2,
    rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.std('x'), np.sqrt(slicer_single_bunch.var('x')),
    rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.std('y'), np.sqrt(slicer_single_bunch.var('y')),
    rtol=0, atol=1e-12)
assert np.allclose(slicer_single_bunch.std('zeta'), np.sqrt(slicer_single_bunch.var('zeta')),
    rtol=0, atol=1e-12)

assert np.all(slicer_single_bunch.sum('xy') == slicer_single_bunch.sum('x_y'))
assert np.all(slicer_single_bunch.sum('x', 'y') == slicer_single_bunch.sum('x_y'))
assert np.all(slicer_single_bunch.mean('xy') == slicer_single_bunch.mean('x_y'))
assert np.all(slicer_single_bunch.mean('x', 'y') == slicer_single_bunch.mean('x_y'))
assert np.all(slicer_single_bunch.cov('xy') == slicer_single_bunch.cov('x_y'))
assert np.all(slicer_single_bunch.cov('x', 'y') == slicer_single_bunch.cov('x_y'))

# # Same parametrized

moms = ['x_px', 'x_y', 'x_py', 'x_delta',
        'px_y', 'px_py', 'px_delta',
        'y_py', 'y_delta',
        'py_delta']
for mm in moms:
    c1_name, c2_name = mm.split('_')

    p = xt.Particles(zeta=[0.99, 1.0, 1.01],
                    weight=[1, 2, 1],
                    **{c1_name: [99, 100, 101],
                       c2_name: [201,200, 199]})
    c1 = getattr(p, c1_name)
    c2 = getattr(p, c2_name)

    slicer_single_bunch.slice(p)

    assert np.allclose(slicer_single_bunch.zeta_centers, np.array([-1, 0, 1]), rtol=0, atol=1e-12)
    assert np.allclose(slicer_single_bunch.particles_per_slice, [0, 0, p.weight.sum()], rtol=0, atol=1e-12)
    assert np.allclose(slicer_single_bunch.sum(c1_name), [0, 0, (c1 * p.weight).sum()], rtol=0, atol=1e-12)
    assert np.allclose(slicer_single_bunch.sum(c2_name), [0, 0, (c2 * p.weight).sum()], rtol=0, atol=1e-12)
    assert np.allclose(slicer_single_bunch.sum('zeta'), [0, 0, (p.zeta * p.weight).sum()], rtol=0, atol=1e-12)
    assert np.allclose(slicer_single_bunch.sum(c1_name + c1_name), [0, 0, (c1**2 * p.weight).sum()], rtol=0, atol=1e-12)
    assert np.allclose(slicer_single_bunch.sum(c2_name + c2_name), [0, 0, (c2**2 * p.weight).sum()], rtol=0, atol=1e-12)
    assert np.allclose(slicer_single_bunch.sum(c1_name + c2_name), [0, 0, (c1 * c2 * p.weight).sum()], rtol=0, atol=1e-12)
    assert np.allclose(slicer_single_bunch.sum('zetazeta'), [0, 0, (p.zeta**2 * p.weight).sum()], rtol=0, atol=1e-12)
    assert np.allclose(slicer_single_bunch.sum(c1_name + 'zeta'), [0, 0, (c1 * p.zeta * p.weight).sum()], rtol=0, atol=1e-12)
    assert np.allclose(slicer_single_bunch.mean(c1_name), [0, 0, (c1 * p.weight).sum() / p.weight.sum()], rtol=0, atol=1e-12)
    assert np.allclose(slicer_single_bunch.mean(c2_name), [0, 0, (c2 * p.weight).sum() / p.weight.sum()], rtol=0, atol=1e-12)
    assert np.allclose(slicer_single_bunch.mean(c1_name + c1_name), [0, 0, (c1**2 * p.weight).sum() / p.weight.sum()], rtol=0, atol=1e-12)
    assert np.allclose(slicer_single_bunch.mean(c2_name + c2_name), [0, 0, (c2**2 * p.weight).sum() / p.weight.sum()], rtol=0, atol=1e-12)
    assert np.allclose(slicer_single_bunch.mean(c1_name + c2_name), [0, 0, (c1 * c2 * p.weight).sum() / p.weight.sum()], rtol=0, atol=1e-12)
    assert np.allclose(slicer_single_bunch.mean(c1_name + 'zeta'), [0, 0, (c1 * p.zeta * p.weight).sum() / p.weight.sum()], rtol=0, atol=1e-12)
    assert np.allclose(slicer_single_bunch.cov(c1_name, c2_name),
        slicer_single_bunch.mean(c1_name + c2_name) - slicer_single_bunch.mean(c1_name) * slicer_single_bunch.mean(c2_name),
        rtol=0, atol=1e-12)
    assert np.allclose(slicer_single_bunch.var(c1_name), slicer_single_bunch.cov(c1_name, c1_name),
        rtol=0, atol=1e-12)
    assert np.allclose(slicer_single_bunch.var(c1_name), slicer_single_bunch.mean(c1_name + c1_name) - slicer_single_bunch.mean(c1_name)**2,
        rtol=0, atol=1e-12)
    assert np.allclose(slicer_single_bunch.var('zeta'), slicer_single_bunch.mean('zetazeta') - slicer_single_bunch.mean('zeta')**2,
        rtol=0, atol=1e-12)

    assert np.all(slicer_single_bunch.sum(c1_name + c2_name) == slicer_single_bunch.sum(c1_name + '_' + c2_name))
    assert np.all(slicer_single_bunch.sum(c1_name, c2_name) == slicer_single_bunch.sum(c1_name + '_' + c2_name))
    assert np.all(slicer_single_bunch.mean(c1_name + c2_name) == slicer_single_bunch.mean(c1_name + '_' + c2_name))
    assert np.all(slicer_single_bunch.mean(c1_name, c2_name) == slicer_single_bunch.mean(c1_name + '_' + c2_name))
    assert np.all(slicer_single_bunch.cov(c1_name + c2_name) == slicer_single_bunch.cov(c1_name + '_' + c2_name))
    assert np.all(slicer_single_bunch.cov(c1_name, c2_name) == slicer_single_bunch.cov(c1_name + '_' + c2_name))

####################################
# Check moments (multi-bunch mode) #
####################################

slicer_multi_bunch = UniformBinSlicer(zeta_range=(-1, 1), nbins=3,
                                        num_bunches=4, bunch_spacing_zeta=bunch_spacing_zeta)

p1 = xt.Particles(zeta=[0.99, 1.0, 1.01],
                 weight=[1, 2, 1],
                 x = [99, 100, 101],
                 y = [201,200, 199])
p2 = xt.Particles(zeta=np.array([-0.01, 0, 0.01]) + 2 * bunch_spacing_zeta,
                    weight=[1, 2, 1],
                    x = [99, 100, 101],
                    y = [201,200, 199])
p = xt.Particles.merge([p1, p2])

assert np.isclose(slicer_multi_bunch.bunch_spacing_zeta, 10, rtol=0, atol=1e-12)

slicer_multi_bunch.slice(p)

assert np.allclose(slicer_multi_bunch.zeta_centers, np.array([[-1, 0, 1], [9, 10, 11], [19, 20, 21], [29, 30, 31]]), rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.particles_per_slice, [[0, 0, p1.weight.sum()], [0, 0, 0], [0, p2.weight.sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.sum('x'), [[0, 0, (p1.x * p1.weight).sum()], [0, 0, 0], [0, (p2.x * p2.weight).sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.sum('y'), [[0, 0, (p1.y * p1.weight).sum()], [0, 0, 0], [0, (p2.y * p2.weight).sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.sum('zeta'), [[0, 0, (p1.zeta * p1.weight).sum()], [0, 0, 0], [0, (p2.zeta * p2.weight).sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.sum('xx'), [[0, 0, (p1.x**2 * p1.weight).sum()], [0, 0, 0], [0, (p2.x**2 * p2.weight).sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.sum('yy'), [[0, 0, (p1.y**2 * p1.weight).sum()], [0, 0, 0], [0, (p2.y**2 * p2.weight).sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.sum('zetazeta'), [[0, 0, (p1.zeta**2 * p1.weight).sum()], [0, 0, 0], [0, (p2.zeta**2 * p2.weight).sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.sum('xy'), [[0, 0, (p1.x * p1.y * p1.weight).sum()], [0, 0, 0], [0, (p2.x * p2.y * p2.weight).sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.sum('xzeta'), [[0, 0, (p1.x * p1.zeta * p1.weight).sum()], [0, 0, 0], [0, (p2.x * p2.zeta * p2.weight).sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.mean('x'), [[0, 0, (p1.x * p1.weight).sum() / p1.weight.sum()], [0, 0, 0], [0, (p2.x * p2.weight).sum() / p2.weight.sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.mean('y'), [[0, 0, (p1.y * p1.weight).sum() / p1.weight.sum()], [0, 0, 0], [0, (p2.y * p2.weight).sum() / p2.weight.sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.mean('xx'), [[0, 0, (p1.x**2 * p1.weight).sum() / p1.weight.sum()], [0, 0, 0], [0, (p2.x**2 * p2.weight).sum() / p2.weight.sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.mean('yy'), [[0, 0, (p1.y**2 * p1.weight).sum() / p1.weight.sum()], [0, 0, 0], [0, (p2.y**2 * p2.weight).sum() / p2.weight.sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.mean('xy'), [[0, 0, (p1.x * p1.y * p1.weight).sum() / p1.weight.sum()], [0, 0, 0], [0, (p2.x * p2.y * p2.weight).sum() / p2.weight.sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.mean('xzeta'), [[0, 0, (p1.x * p1.zeta * p1.weight).sum() / p1.weight.sum()], [0, 0, 0], [0, (p2.x * p2.zeta * p2.weight).sum() / p2.weight.sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.cov('x', 'y'),
    slicer_multi_bunch.mean('xy') - slicer_multi_bunch.mean('x') * slicer_multi_bunch.mean('y'),
    rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.var('x'), slicer_multi_bunch.cov('x', 'x'),
    rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.var('x'), slicer_multi_bunch.mean('xx') - slicer_multi_bunch.mean('x')**2,
    rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.var('zeta'), slicer_multi_bunch.mean('zetazeta') - slicer_multi_bunch.mean('zeta')**2,
    rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.std('x'), np.sqrt(slicer_multi_bunch.var('x')),
                    rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.std('y'), np.sqrt(slicer_multi_bunch.var('y')),
                    rtol=0, atol=1e-12)
assert np.allclose(slicer_multi_bunch.std('zeta'), np.sqrt(slicer_multi_bunch.var('zeta')),
                    rtol=0, atol=1e-12)

assert np.all(slicer_multi_bunch.sum('xy') == slicer_multi_bunch.sum('x_y'))
assert np.all(slicer_multi_bunch.sum('x', 'y') == slicer_multi_bunch.sum('x_y'))
assert np.all(slicer_multi_bunch.mean('xy') == slicer_multi_bunch.mean('x_y'))
assert np.all(slicer_multi_bunch.mean('x', 'y') == slicer_multi_bunch.mean('x_y'))
assert np.all(slicer_multi_bunch.cov('xy') == slicer_multi_bunch.cov('x_y'))
assert np.all(slicer_multi_bunch.cov('x', 'y') == slicer_multi_bunch.cov('x_y'))

# # Same parametrized

moms = ['x_px', 'x_y', 'x_py', 'x_delta',
        'px_y', 'px_py', 'px_delta',
        'y_py', 'y_delta',
        'py_delta']

for mm in moms:
    c1_name, c2_name = mm.split('_')

    p1 = xt.Particles(zeta=[0.99, 1.0, 1.01],
                    weight=[1, 2, 1],
                    **{c1_name: [99, 100, 101],
                       c2_name: [201,200, 199]})
    p2 = xt.Particles(zeta=np.array([-0.01, 0, 0.01]) + 2 * bunch_spacing_zeta,
                        weight=[1, 2, 1],
                        **{c1_name: [99, 100, 101],
                           c2_name: [201,200, 199]})

    p = xt.Particles.merge([p1, p2])

    slicer_multi_bunch.slice(p)

    c1_p1 = getattr(p1, c1_name)
    c2_p1 = getattr(p1, c2_name)
    c1_p2 = getattr(p2, c1_name)
    c2_p2 = getattr(p2, c2_name)

    assert np.allclose(slicer_multi_bunch.zeta_centers, np.array([[-1, 0, 1], [9, 10, 11], [19, 20, 21], [29, 30, 31]]), rtol=0, atol=1e-12)
    assert np.allclose(slicer_multi_bunch.particles_per_slice, [[0, 0, p1.weight.sum()], [0, 0, 0], [0, p2.weight.sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
    assert np.allclose(slicer_multi_bunch.sum(c1_name), [[0, 0, (c1_p1 * p1.weight).sum()], [0, 0, 0], [0, (c1_p2 * p2.weight).sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
    assert np.allclose(slicer_multi_bunch.sum(c2_name), [[0, 0, (c2_p1 * p1.weight).sum()], [0, 0, 0], [0, (c2_p2 * p2.weight).sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
    assert np.allclose(slicer_multi_bunch.sum('zeta'), [[0, 0, (p1.zeta * p1.weight).sum()], [0, 0, 0], [0, (p2.zeta * p2.weight).sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
    assert np.allclose(slicer_multi_bunch.sum(c1_name + c1_name), [[0, 0, (c1_p1**2 * p1.weight).sum()], [0, 0, 0], [0, (c1_p2**2 * p2.weight).sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
    assert np.allclose(slicer_multi_bunch.sum(c2_name + c2_name), [[0, 0, (c2_p1**2 * p1.weight).sum()], [0, 0, 0], [0, (c2_p2**2 * p2.weight).sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
    assert np.allclose(slicer_multi_bunch.sum(c1_name + c2_name), [[0, 0, (c1_p1 * c2_p1 * p1.weight).sum()], [0, 0, 0], [0, (c1_p2 * c2_p2 * p2.weight).sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
    assert np.allclose(slicer_multi_bunch.sum('zetazeta'), [[0, 0, (p1.zeta**2 * p1.weight).sum()], [0, 0, 0], [0, (p2.zeta**2 * p2.weight).sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
    assert np.allclose(slicer_multi_bunch.sum(c1_name + 'zeta'), [[0, 0, (c1_p1 * p1.zeta * p1.weight).sum()], [0, 0, 0], [0, (c1_p2 * p2.zeta * p2.weight).sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
    assert np.allclose(slicer_multi_bunch.mean(c1_name), [[0, 0, (c1_p1 * p1.weight).sum() / p1.weight.sum()], [0, 0, 0], [0, (c1_p2 * p2.weight).sum() / p2.weight.sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
    assert np.allclose(slicer_multi_bunch.mean(c2_name), [[0, 0, (c2_p1 * p1.weight).sum() / p1.weight.sum()], [0, 0, 0], [0, (c2_p2 * p2.weight).sum() / p2.weight.sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
    assert np.allclose(slicer_multi_bunch.mean(c1_name + c1_name), [[0, 0, (c1_p1**2 * p1.weight).sum() / p1.weight.sum()], [0, 0, 0], [0, (c1_p2**2 * p2.weight).sum() / p2.weight.sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
    assert np.allclose(slicer_multi_bunch.mean(c2_name + c2_name), [[0, 0, (c2_p1**2 * p1.weight).sum() / p1.weight.sum()], [0, 0, 0], [0, (c2_p2**2 * p2.weight).sum() / p2.weight.sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
    assert np.allclose(slicer_multi_bunch.mean(c1_name + c2_name), [[0, 0, (c1_p1 * c2_p1 * p1.weight).sum() / p1.weight.sum()], [0, 0, 0], [0, (c1_p2 * c2_p2 * p2.weight).sum() / p2.weight.sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
    assert np.allclose(slicer_multi_bunch.mean(c1_name + 'zeta'), [[0, 0, (c1_p1 * p1.zeta * p1.weight).sum() / p1.weight.sum()], [0, 0, 0], [0, (c1_p2 * p2.zeta * p2.weight).sum() / p2.weight.sum(), 0], [0, 0, 0]], rtol=0, atol=1e-12)
    assert np.allclose(slicer_multi_bunch.cov(c1_name, c2_name),
        slicer_multi_bunch.mean(c1_name + c2_name) - slicer_multi_bunch.mean(c1_name) * slicer_multi_bunch.mean(c2_name),
        rtol=0, atol=1e-12)
    assert np.allclose(slicer_multi_bunch.var(c1_name), slicer_multi_bunch.cov(c1_name, c1_name),
        rtol=0, atol=1e-12)
    assert np.allclose(slicer_multi_bunch.var(c1_name), slicer_multi_bunch.mean(c1_name + c1_name) - slicer_multi_bunch.mean(c1_name)**2,
        rtol=0, atol=1e-12)
    assert np.allclose(slicer_multi_bunch.var('zeta'), slicer_multi_bunch.mean('zetazeta') - slicer_multi_bunch.mean('zeta')**2,
        rtol=0, atol=1e-12)

    assert np.all(slicer_multi_bunch.sum(c1_name + c2_name) == slicer_multi_bunch.sum(c1_name + '_' + c2_name))
    assert np.all(slicer_multi_bunch.sum(c1_name, c2_name) == slicer_multi_bunch.sum(c1_name + '_' + c2_name))
    assert np.all(slicer_multi_bunch.mean(c1_name + c2_name) == slicer_multi_bunch.mean(c1_name + '_' + c2_name))
    assert np.all(slicer_multi_bunch.mean(c1_name, c2_name) == slicer_multi_bunch.mean(c1_name + '_' + c2_name))
    assert np.all(slicer_multi_bunch.cov(c1_name + c2_name) == slicer_multi_bunch.cov(c1_name + '_' + c2_name))
    assert np.all(slicer_multi_bunch.cov(c1_name, c2_name) == slicer_multi_bunch.cov(c1_name + '_' + c2_name))
