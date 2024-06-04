import xtrack as xt
import xpart as xp
import xobjects as xo

from xfields import UniformBinSlicer

import numpy as np

###############################################
# Check slice attribution (single-bunch mode) #
###############################################

slicer = UniformBinSlicer(zeta_range=(-1, 1), num_slices=3)
assert slicer.num_bunches == 1  # Single-bunch mode

p0 = xt.Particles(
    zeta=[-2, -1.51, -1.49, -1, -0.51, -0.49, 0, 0.49, 0.51, 1, 1.49, 1.51,  2,
          2.51],
    weight=[10, 10, 10, 10, 10, 20, 20, 20, 30, 30, 30, 40, 40, 40],
    x=[0.,  1.,   2.,     3.,  4.,    5.,    6., 7.,    8.,  9.,  10.,  11.,
       12., 13.],
    y=[13., 12.,  11.,    10., 9.,    8.,    7., 6.,    5.,  4.,  3.,   2.,
       1., 0.]
)

p0.state[-1] = 0

p = p0.copy()

i_slice_expected = [-1, -1, -1, 0, 0, 0, 1, 2, 2, -1, -1, -1, -1, -999]
i_bunch_expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -9999]
ss = 0 * p.x
ctx = xo.ContextCpu()
i_slice_particles = p.particle_id * 0 - 999
i_bunch_particles = p.particle_id * 0 - 9999
slicer.slice(particles=p, i_bunch_particles=i_bunch_particles,
             i_slice_particles=i_slice_particles)

assert np.all(np.array(i_slice_expected) == i_slice_particles)
assert np.all(i_bunch_particles == i_bunch_expected)


expected_num_particles = np.array([40, 20, 50])
assert np.allclose(slicer.num_particles, expected_num_particles,
                   atol=1e-12, rtol=0)

##############################################
# Check slice attribution (multi-bunch mode) #
##############################################

bunch_spacing_zeta = 10.

p1 = p0.copy()
p2 = p0.copy()
p2.zeta -= bunch_spacing_zeta
p2.weight *= 10
p3 = p0.copy()
p3.zeta -= 2 * bunch_spacing_zeta
p3.weight *= 100
p4 = p0.copy()
p4.zeta -= 3 * bunch_spacing_zeta
p4.weight *= 1000

p = xt.Particles.merge([p1, p2, p3, p4])

i_bunch_particles = p.particle_id * 0 - 999
i_slice_particles = p.particle_id * 0 - 999

slicer = UniformBinSlicer(zeta_range=(-1, 1), num_slices=3,
                          num_bunches=4, bunch_spacing_zeta=bunch_spacing_zeta)
slicer.slice(particles=p,
             i_slice_particles=i_slice_particles,
             i_bunch_particles=i_bunch_particles)

# when we merge the particles the ones with state zero end up at the end so
# the -999 are now at the end of the array
i_slice_expected_mb = i_slice_expected[0: 13]*4 + [-999, -999, -999, -999]

# the first three particles of each ensemble are outside of the slot (at the
# left side) so they end up being assigned to the next bunch
i_bunch_expected_mb = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                       -1, -1, -1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                       -999, -999, -999, -999]

expected_num_particles_mb = np.array([
    [40, 20, 50],
    [400, 200, 500],
    [4000, 2000, 5000],
    [40000, 20000, 50000],
])

assert np.all(i_slice_particles == i_slice_expected_mb)
assert np.all(i_bunch_particles == i_bunch_expected_mb)
assert np.allclose(slicer.num_particles, expected_num_particles_mb,
                   atol=1e-12, rtol=0)


#####################################
# Check moments (single-bunch mode) #
#####################################

slicer_single_bunch = UniformBinSlicer(zeta_range=(-1, 1), num_slices=3)
slicer_single_bunch_1 = slicer_single_bunch.copy()
slicer_single_bunch_2 = slicer_single_bunch.copy()

p = xt.Particles(zeta=[0.97, 0.98, 0.99],
                 weight=[1, 2, 1],
                 x=[99, 100, 101],
                 y=[201, 200, 199])
slicer_single_bunch.slice(p)

# Test copy
slicer_single_bunch_copy = slicer_single_bunch.copy()

# try round-trip with _to/_from_npbuffer
slicer_single_bunch_buffer = UniformBinSlicer._from_npbuffer(
                                    slicer_single_bunch._to_npbuffer())

# Test sum
pp1 = p.filter(p.zeta < 0.98)
pp2 = p.filter(p.zeta >= 0.98)

slicer_single_bunch_1.slice(pp1)
slicer_single_bunch_2.slice(pp2)
slicer_single_bunch_sum = sum([slicer_single_bunch_1, slicer_single_bunch_2])


for sl in [slicer_single_bunch, slicer_single_bunch_copy,
           slicer_single_bunch_buffer, slicer_single_bunch_sum]:
    assert sl.bunch_spacing_zeta == 0

    assert np.allclose(sl.zeta_centers, np.array([-2/3, 0, 2/3]), rtol=0,
                       atol=1e-12)
    assert np.allclose(sl.num_particles, [0, 0, p.weight.sum()], rtol=0,
                       atol=1e-12)
    assert np.allclose(sl.sum('x'), [0, 0, (p.x * p.weight).sum()], rtol=0,
                       atol=1e-12)
    assert np.allclose(sl.sum('y'), [0, 0, (p.y * p.weight).sum()], rtol=0,
                       atol=1e-12)
    assert np.allclose(sl.sum('zeta'), [0, 0, (p.zeta * p.weight).sum()],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('xx'), [0, 0, (p.x**2 * p.weight).sum()], rtol=0,
                       atol=1e-12)
    assert np.allclose(sl.sum('yy'), [0, 0, (p.y**2 * p.weight).sum()], rtol=0,
                       atol=1e-12)
    assert np.allclose(sl.sum('zetazeta'), [0, 0, (p.zeta**2 * p.weight).sum()],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('xy'), [0, 0, (p.x * p.y * p.weight).sum()],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('xzeta'), [0, 0, (p.x * p.zeta * p.weight).sum()],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.mean('x'),
                       [0, 0, (p.x * p.weight).sum() / p.weight.sum()], rtol=0,
                       atol=1e-12)
    assert np.allclose(sl.mean('y'),
                       [0, 0, (p.y * p.weight).sum() / p.weight.sum()], rtol=0,
                       atol=1e-12)
    assert np.allclose(sl.mean('xx'),
                       [0, 0, (p.x**2 * p.weight).sum() / p.weight.sum()],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.mean('yy'),
                       [0, 0, (p.y**2 * p.weight).sum() / p.weight.sum()],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.mean('zetazeta'),
                       [0, 0, (p.zeta**2 * p.weight).sum() / p.weight.sum()],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.mean('xy'),
                       [0, 0, (p.x * p.y * p.weight).sum() / p.weight.sum()],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.mean('xzeta'),
                       [0, 0, (p.x * p.zeta * p.weight).sum() / p.weight.sum()],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.cov('x', 'y'),
                       sl.mean('xy') - sl.mean('x') * sl.mean('y'),
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.var('x'), sl.cov('x', 'x'),
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.var('x'), sl.mean('xx') - sl.mean('x')**2,
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.var('zeta'), sl.mean('zetazeta') - sl.mean('zeta')**2,
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.std('x'), np.sqrt(sl.var('x')),
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.std('y'), np.sqrt(sl.var('y')),
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.std('zeta'), np.sqrt(sl.var('zeta')),
                       rtol=0, atol=1e-12)

    assert np.all(sl.sum('xy') == sl.sum('x_y'))
    assert np.all(sl.sum('x', 'y') == sl.sum('x_y'))
    assert np.all(sl.mean('xy') == sl.mean('x_y'))
    assert np.all(sl.mean('x', 'y') == sl.mean('x_y'))
    assert np.all(sl.cov('xy') == sl.cov('x_y'))
    assert np.all(sl.cov('x', 'y') == sl.cov('x_y'))

# # Same parametrized

moms = ['x_px', 'x_y', 'x_py', 'x_delta',
        'px_y', 'px_py', 'px_delta',
        'y_py', 'y_delta',
        'py_delta']
for mm in moms:
    c1_name, c2_name = mm.split('_')

    p = xt.Particles(zeta=[0.97, 0.98, 0.99],
                     weight=[1, 2, 1],
                     **{c1_name: [99, 100, 101],
                        c2_name: [201, 200, 199]})
    c1 = getattr(p, c1_name)
    c2 = getattr(p, c2_name)

    slicer_single_bunch.slice(p)

    # Test copy
    slicer_single_bunch_copy = slicer_single_bunch.copy()

    # Test round-trip with _to/_from_npbuffer
    slicer_single_bunch_buffer = UniformBinSlicer._from_npbuffer(
                                        slicer_single_bunch._to_npbuffer())

    # Test sum
    pp1 = p.filter(p.zeta < 0.98)
    pp2 = p.filter(p.zeta >= 0.98)

    slicer_single_bunch_1.slice(pp1)
    slicer_single_bunch_2.slice(pp2)
    slicer_single_bunch_sum = sum([slicer_single_bunch_1,
                                   slicer_single_bunch_2])

    for sl in [slicer_single_bunch, slicer_single_bunch_copy,
               slicer_single_bunch_buffer, slicer_single_bunch_sum]:

        assert np.allclose(sl.zeta_centers, np.array([-2/3, 0, 2/3]), rtol=0,
                           atol=1e-12)
        assert np.allclose(sl.num_particles, [0, 0, p.weight.sum()], rtol=0,
                           atol=1e-12)
        assert np.allclose(sl.sum(c1_name), [0, 0, (c1 * p.weight).sum()],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum(c2_name), [0, 0, (c2 * p.weight).sum()],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum('zeta'), [0, 0, (p.zeta * p.weight).sum()],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum(c1_name + c1_name),
                           [0, 0, (c1**2 * p.weight).sum()],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum(c2_name + c2_name),
                           [0, 0, (c2**2 * p.weight).sum()],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum(c1_name + c2_name),
                           [0, 0, (c1 * c2 * p.weight).sum()],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum('zetazeta'),
                           [0, 0, (p.zeta**2 * p.weight).sum()],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum(c1_name + 'zeta'),
                           [0, 0, (c1 * p.zeta * p.weight).sum()],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.mean(c1_name),
                           [0, 0, (c1 * p.weight).sum() / p.weight.sum()],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.mean(c2_name),
                           [0, 0, (c2 * p.weight).sum() / p.weight.sum()],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.mean(c1_name + c1_name),
                           [0, 0, (c1**2 * p.weight).sum() / p.weight.sum()],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.mean(c2_name + c2_name),
                           [0, 0, (c2**2 * p.weight).sum() / p.weight.sum()],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.mean(c1_name + c2_name),
                           [0, 0, (c1 * c2 * p.weight).sum() / p.weight.sum()],
                           rtol=0, atol=1e-12)
        assert np.allclose(
            sl.mean(c1_name + 'zeta'),
            [0, 0, (c1 * p.zeta * p.weight).sum() / p.weight.sum()],
            rtol=0, atol=1e-12)
        assert np.allclose(
            sl.cov(c1_name, c2_name),
            sl.mean(c1_name + c2_name) - sl.mean(c1_name) * sl.mean(c2_name),
            rtol=0, atol=1e-12)
        assert np.allclose(
            sl.var(c1_name), sl.cov(c1_name, c1_name),
            rtol=0, atol=1e-12)
        assert np.allclose(
            sl.var(c1_name), sl.mean(c1_name + c1_name) - sl.mean(c1_name)**2,
            rtol=0, atol=1e-12)
        assert np.allclose(
            sl.var('zeta'), sl.mean('zetazeta') - sl.mean('zeta')**2,
            rtol=0, atol=1e-12)

        assert np.all(sl.sum(c1_name + c2_name) ==
                      sl.sum(c1_name + '_' + c2_name))
        assert np.all(sl.sum(c1_name, c2_name) ==
                      sl.sum(c1_name + '_' + c2_name))
        assert np.all(sl.mean(c1_name + c2_name) ==
                      sl.mean(c1_name + '_' + c2_name))
        assert np.all(sl.mean(c1_name, c2_name) ==
                      sl.mean(c1_name + '_' + c2_name))
        assert np.all(sl.cov(c1_name + c2_name) ==
                      sl.cov(c1_name + '_' + c2_name))
        assert np.all(sl.cov(c1_name, c2_name) ==
                      sl.cov(c1_name + '_' + c2_name))

####################################
# Check moments (multi-bunch mode) #
####################################

slicer_multi_bunch = UniformBinSlicer(
    zeta_range=(-1, 1), num_slices=3,
    num_bunches=4,
    bunch_spacing_zeta=bunch_spacing_zeta)
slicer_multi_bunch_1 = slicer_multi_bunch.copy()
slicer_multi_bunch_2 = slicer_multi_bunch.copy()

slicer_multi_bunch_part = UniformBinSlicer(
    zeta_range=(-1, 1), num_slices=3,
    num_bunches=3,
    bunch_spacing_zeta=bunch_spacing_zeta)
slicer_multi_bunch_part_1 = slicer_multi_bunch_part.copy()
slicer_multi_bunch_part_2 = slicer_multi_bunch_part.copy()

p1 = xt.Particles(zeta=[0.97, 0.98, 0.99],
                  weight=[1, 2, 1],
                  x=[99, 100, 101],
                  y=[201, 200, 199])
p2 = xt.Particles(zeta=np.array([-0.01, 0, 0.01]) - 2 * bunch_spacing_zeta,
                  weight=[1, 2, 1],
                  x=[99, 100, 101],
                  y=[201, 200, 199])
p = xt.Particles.merge([p1, p2])

assert np.isclose(slicer_multi_bunch.bunch_spacing_zeta, 10, rtol=0, atol=1e-12)
assert np.isclose(slicer_multi_bunch_part.bunch_spacing_zeta, 10,
                  rtol=0, atol=1e-12)

assert slicer_multi_bunch.num_bunches == 4
assert slicer_multi_bunch_part.num_bunches == 3

slicer_multi_bunch.slice(p)
slicer_multi_bunch_part.slice(p)

# Test copy
slicer_multi_bunch_copy = slicer_multi_bunch.copy()
slicer_multi_bunch_part_copy = slicer_multi_bunch_part.copy()

# Test round-trip with _to/_from_npbuffer
slicer_multi_bunch_buffer = UniformBinSlicer._from_npbuffer(
                                    slicer_multi_bunch._to_npbuffer())
slicer_multi_bunch_part_buffer = UniformBinSlicer._from_npbuffer(
                                    slicer_multi_bunch_part._to_npbuffer())

# Test sum
pp1 = p.filter(p.zeta < 0.5)
pp2 = p.filter(p.zeta >= 0.5)

slicer_multi_bunch_1.slice(pp1)
slicer_multi_bunch_2.slice(pp2)
slicer_multi_bunch_sum = sum([slicer_multi_bunch_1, slicer_multi_bunch_2])

slicer_multi_bunch_part_1.slice(pp1)
slicer_multi_bunch_part_2.slice(pp2)
slicer_multi_bunch_part_sum = sum([slicer_multi_bunch_part_1,
                                   slicer_multi_bunch_part_2])

for sl in [slicer_multi_bunch, slicer_multi_bunch_copy,
           slicer_multi_bunch_buffer,
           slicer_multi_bunch_sum]:
    assert np.allclose(sl.zeta_centers,
                       np.array([[-2/3, 0, 2/3],
                                 [-10-2/3, -10, -10+2/3],
                                 [-20-2/3, -20, -20+2/3],
                                 [-30-2/3, -30, -30+2/3]]),
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.num_particles,
                       [[0, 0, p1.weight.sum()],
                        [0, 0, 0],
                        [0, p2.weight.sum(), 0],
                        [0, 0, 0]],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('x'),
                       [[0, 0, (p1.x * p1.weight).sum()],
                        [0, 0, 0],
                        [0, (p2.x * p2.weight).sum(), 0],
                        [0, 0, 0]],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('y'),
                       [[0, 0, (p1.y * p1.weight).sum()],
                        [0, 0, 0],
                        [0, (p2.y * p2.weight).sum(), 0],
                        [0, 0, 0]],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('zeta'),
                       [[0, 0, (p1.zeta * p1.weight).sum()],
                        [0, 0, 0],
                        [0, (p2.zeta * p2.weight).sum(), 0],
                        [0, 0, 0]],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('xx'),
                       [[0, 0, (p1.x**2 * p1.weight).sum()],
                        [0, 0, 0],
                        [0, (p2.x**2 * p2.weight).sum(), 0],
                        [0, 0, 0]],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('yy'),
                       [[0, 0, (p1.y**2 * p1.weight).sum()],
                        [0, 0, 0],
                        [0, (p2.y**2 * p2.weight).sum(), 0],
                        [0, 0, 0]],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('zetazeta'),
                       [[0, 0, (p1.zeta**2 * p1.weight).sum()],
                        [0, 0, 0],
                        [0, (p2.zeta**2 * p2.weight).sum(), 0],
                        [0, 0, 0]],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('xy'),
                       [[0, 0, (p1.x * p1.y * p1.weight).sum()],
                        [0, 0, 0],
                        [0, (p2.x * p2.y * p2.weight).sum(), 0],
                        [0, 0, 0]],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('xzeta'),
                       [[0, 0, (p1.x * p1.zeta * p1.weight).sum()],
                        [0, 0, 0],
                        [0, (p2.x * p2.zeta * p2.weight).sum(), 0],
                        [0, 0, 0]],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.mean('x'),
                       [[0, 0, (p1.x * p1.weight).sum() / p1.weight.sum()],
                        [0, 0, 0],
                        [0, (p2.x * p2.weight).sum() / p2.weight.sum(), 0],
                        [0, 0, 0]],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.mean('y'),
                       [[0, 0, (p1.y * p1.weight).sum() / p1.weight.sum()],
                        [0, 0, 0],
                        [0, (p2.y * p2.weight).sum() / p2.weight.sum(), 0],
                        [0, 0, 0]],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.mean('xx'),
                       [[0, 0, (p1.x**2 * p1.weight).sum() / p1.weight.sum()],
                        [0, 0, 0],
                        [0, (p2.x**2 * p2.weight).sum() / p2.weight.sum(), 0],
                        [0, 0, 0]],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.mean('yy'),
                       [[0, 0, (p1.y**2 * p1.weight).sum() / p1.weight.sum()],
                        [0, 0, 0],
                        [0, (p2.y**2 * p2.weight).sum() / p2.weight.sum(), 0],
                        [0, 0, 0]],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.mean('xy'),
                       [[0, 0, (p1.x * p1.y * p1.weight).sum()/p1.weight.sum()],
                        [0, 0, 0],
                        [0, (p2.x * p2.y * p2.weight).sum()/p2.weight.sum(), 0],
                        [0, 0, 0]],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.mean('xzeta'),
                       [[0, 0, (p1.x*p1.zeta*p1.weight).sum()/p1.weight.sum()],
                        [0, 0, 0],
                        [0, (p2.x*p2.zeta*p2.weight).sum()/p2.weight.sum(), 0],
                        [0, 0, 0]],
                       rtol=0, atol=1e-12)
    assert np.allclose(
        sl.cov('x', 'y'),
        sl.mean('xy') - sl.mean('x') * sl.mean('y'),
        rtol=0, atol=1e-12)
    assert np.allclose(
        sl.var('x'), sl.cov('x', 'x'),
        rtol=0, atol=1e-12)
    assert np.allclose(
        sl.var('x'), sl.mean('xx') - sl.mean('x')**2,
        rtol=0, atol=1e-12)
    assert np.allclose(
        sl.var('zeta'), sl.mean('zetazeta') - sl.mean('zeta')**2,
        rtol=0, atol=1e-12)
    assert np.allclose(sl.std('x'), np.sqrt(sl.var('x')),
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.std('y'), np.sqrt(sl.var('y')),
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.std('zeta'), np.sqrt(sl.var('zeta')),
                       rtol=0, atol=1e-12)

    assert np.all(sl.sum('xy') == sl.sum('x_y'))
    assert np.all(sl.sum('x', 'y') == sl.sum('x_y'))
    assert np.all(sl.mean('xy') == sl.mean('x_y'))
    assert np.all(sl.mean('x', 'y') == sl.mean('x_y'))
    assert np.all(sl.cov('xy') == sl.cov('x_y'))
    assert np.all(sl.cov('x', 'y') == sl.cov('x_y'))

# Check slicer_part
for sl in [slicer_multi_bunch_part, slicer_multi_bunch_part_copy,
           slicer_multi_bunch_part_buffer,
           slicer_multi_bunch_part_sum]:
    assert np.allclose(sl.zeta_centers, slicer_multi_bunch.zeta_centers[:-1],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.num_particles, slicer_multi_bunch.num_particles[:-1],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('x'), slicer_multi_bunch.sum('x')[:-1],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('y'), slicer_multi_bunch.sum('y')[:-1],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('zeta'), slicer_multi_bunch.sum('zeta')[:-1],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('xx'), slicer_multi_bunch.sum('xx')[:-1],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('yy'), slicer_multi_bunch.sum('yy')[:-1],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('zetazeta'),
                       slicer_multi_bunch.sum('zetazeta')[:-1],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('xy'), slicer_multi_bunch.sum('xy')[:-1],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('xzeta'), slicer_multi_bunch.sum('xzeta')[:-1],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.mean('x'), slicer_multi_bunch.mean('x')[:-1],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.mean('y'), slicer_multi_bunch.mean('y')[:-1],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.mean('xx'), slicer_multi_bunch.mean('xx')[:-1],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.mean('yy'), slicer_multi_bunch.mean('yy')[:-1],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.mean('xy'), slicer_multi_bunch.mean('xy')[:-1],
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.mean('xzeta'), slicer_multi_bunch.mean('xzeta')[:-1],
                       rtol=0, atol=1e-12)
    assert np.allclose(
        sl.cov('x', 'y'),
        sl.mean('xy') - sl.mean('x') * sl.mean('y'),
        rtol=0, atol=1e-12)
    assert np.allclose(
        sl.var('x'), sl.cov('x', 'x'),
        rtol=0, atol=1e-12)
    assert np.allclose(
        sl.var('x'), sl.mean('xx') - sl.mean('x')**2,
        rtol=0, atol=1e-12)
    assert np.allclose(
        sl.var('zeta'), sl.mean('zetazeta') - sl.mean('zeta')**2,
        rtol=0, atol=1e-12)
    assert np.allclose(sl.std('x'), np.sqrt(sl.var('x')),
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.std('y'), np.sqrt(sl.var('y')),
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.std('zeta'), np.sqrt(sl.var('zeta')),
                       rtol=0, atol=1e-12)


# # Same parametrized

moms = ['x_px', 'x_y', 'x_py', 'x_delta',
        'px_y', 'px_py', 'px_delta',
        'y_py', 'y_delta',
        'py_delta']

for mm in moms:
    c1_name, c2_name = mm.split('_')

    p1 = xt.Particles(zeta=[0.97, 0.98, 0.99],
                      weight=[1, 2, 1],
                      **{c1_name: [99, 100, 101],
                         c2_name: [201, 200, 199]})
    p2 = xt.Particles(zeta=np.array([-0.01, 0, 0.01]) - 2 * bunch_spacing_zeta,
                      weight=[1, 2, 1],
                      **{c1_name: [99, 100, 101],
                         c2_name: [201, 200, 199]})

    p = xt.Particles.merge([p1, p2])

    slicer_multi_bunch.slice(p)
    slicer_multi_bunch_part.slice(p)

    pp1 = p.filter(p.zeta < 0.5)
    pp2 = p.filter(p.zeta >= 0.5)

    slicer_multi_bunch_1.slice(pp1)
    slicer_multi_bunch_2.slice(pp2)
    slicer_multi_bunch_sum = sum([slicer_multi_bunch_1, slicer_multi_bunch_2])

    slicer_multi_bunch_part_1.slice(pp1)
    slicer_multi_bunch_part_2.slice(pp2)
    slicer_multi_bunch_part_sum = sum([slicer_multi_bunch_part_1,
                                       slicer_multi_bunch_part_2])

    # Test copy
    slicer_multi_bunch_copy = slicer_multi_bunch.copy()
    slicer_multi_bunch_part_copy = slicer_multi_bunch_part.copy()

    # Test round-trip with _to/_from_npbuffer
    slicer_multi_bunch_buffer = UniformBinSlicer._from_npbuffer(
                                        slicer_multi_bunch._to_npbuffer())
    slicer_multi_bunch_part_buffer = UniformBinSlicer._from_npbuffer(
                                        slicer_multi_bunch_part._to_npbuffer())

    c1_p1 = getattr(p1, c1_name)
    c2_p1 = getattr(p1, c2_name)
    c1_p2 = getattr(p2, c1_name)
    c2_p2 = getattr(p2, c2_name)
    for sl in [slicer_multi_bunch, slicer_multi_bunch_copy,
               slicer_multi_bunch_buffer, slicer_multi_bunch_sum]:
        assert np.allclose(sl.zeta_centers,
                           np.array([[-2/3, 0, 2/3],
                                     [-10-2/3, -10, -10+2/3],
                                     [-20-2/3, -20, -20+2/3],
                                     [-30-2/3, -30, -30+2/3]]),
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.num_particles, [[0, 0, p1.weight.sum()],
                                              [0, 0, 0],
                                              [0, p2.weight.sum(), 0],
                                              [0, 0, 0]],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum(c1_name),
                           [[0, 0, (c1_p1 * p1.weight).sum()],
                            [0, 0, 0],
                            [0, (c1_p2 * p2.weight).sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum(c2_name),
                           [[0, 0, (c2_p1 * p1.weight).sum()],
                            [0, 0, 0],
                            [0, (c2_p2 * p2.weight).sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum('zeta'),
                           [[0, 0, (p1.zeta * p1.weight).sum()],
                            [0, 0, 0],
                            [0, (p2.zeta * p2.weight).sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum(c1_name + c1_name),
                           [[0, 0, (c1_p1**2 * p1.weight).sum()],
                            [0, 0, 0],
                            [0, (c1_p2**2 * p2.weight).sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum(c2_name + c2_name),
                           [[0, 0, (c2_p1**2 * p1.weight).sum()],
                            [0, 0, 0],
                            [0, (c2_p2**2 * p2.weight).sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum(c1_name + c2_name),
                           [[0, 0, (c1_p1 * c2_p1 * p1.weight).sum()],
                            [0, 0, 0],
                            [0, (c1_p2 * c2_p2 * p2.weight).sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum('zetazeta'),
                           [[0, 0, (p1.zeta**2 * p1.weight).sum()],
                            [0, 0, 0],
                            [0, (p2.zeta**2 * p2.weight).sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum(c1_name + 'zeta'),
                           [[0, 0, (c1_p1 * p1.zeta * p1.weight).sum()],
                            [0, 0, 0],
                            [0, (c1_p2 * p2.zeta * p2.weight).sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.mean(c1_name),
                           [[0, 0, (c1_p1 * p1.weight).sum() / p1.weight.sum()],
                            [0, 0, 0],
                            [0, (c1_p2 * p2.weight).sum() / p2.weight.sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.mean(c2_name),
                           [[0, 0, (c2_p1 * p1.weight).sum() / p1.weight.sum()],
                            [0, 0, 0],
                            [0, (c2_p2 * p2.weight).sum() / p2.weight.sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.mean(c1_name + c1_name),
                           [[0, 0, (c1_p1**2*p1.weight).sum()/p1.weight.sum()],
                            [0, 0, 0],
                            [0, (c1_p2**2*p2.weight).sum()/p2.weight.sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.mean(c2_name + c2_name),
                           [[0, 0, (c2_p1**2*p1.weight).sum()/p1.weight.sum()],
                            [0, 0, 0],
                            [0, (c2_p2**2*p2.weight).sum()/p2.weight.sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        assert np.allclose(
            sl.mean(c1_name + c2_name),
            [[0, 0, (c1_p1 * c2_p1 * p1.weight).sum() / p1.weight.sum()],
             [0, 0, 0],
             [0, (c1_p2 * c2_p2 * p2.weight).sum() / p2.weight.sum(), 0],
             [0, 0, 0]],
            rtol=0, atol=1e-12)
        assert np.allclose(
            sl.mean(c1_name + 'zeta'),
            [[0, 0, (c1_p1 * p1.zeta * p1.weight).sum() / p1.weight.sum()],
             [0, 0, 0],
             [0, (c1_p2 * p2.zeta * p2.weight).sum() / p2.weight.sum(), 0],
             [0, 0, 0]],
            rtol=0, atol=1e-12)
        assert np.allclose(
            sl.cov(c1_name, c2_name),
            sl.mean(c1_name + c2_name) - sl.mean(c1_name) * sl.mean(c2_name),
            rtol=0, atol=1e-12)
        assert np.allclose(
            sl.var(c1_name), sl.cov(c1_name, c1_name),
            rtol=0, atol=1e-12)
        assert np.allclose(
            sl.var(c1_name), sl.mean(c1_name + c1_name) - sl.mean(c1_name)**2,
            rtol=0, atol=1e-12)
        assert np.allclose(
            sl.var('zeta'), sl.mean('zetazeta') - sl.mean('zeta')**2,
            rtol=0, atol=1e-12)
        assert np.allclose(sl.std(c1_name), np.sqrt(sl.var(c1_name)),
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.std(c2_name), np.sqrt(sl.var(c2_name)),
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.std('zeta'), np.sqrt(sl.var('zeta')),
                           rtol=0, atol=1e-12)

        assert np.all(sl.sum(c1_name + c2_name) ==
                      sl.sum(c1_name + '_' + c2_name))
        assert np.all(sl.sum(c1_name, c2_name) ==
                      sl.sum(c1_name + '_' + c2_name))
        assert np.all(sl.mean(c1_name + c2_name) ==
                      sl.mean(c1_name + '_' + c2_name))
        assert np.all(sl.mean(c1_name, c2_name) ==
                      sl.mean(c1_name + '_' + c2_name))
        assert np.all(sl.cov(c1_name + c2_name) ==
                      sl.cov(c1_name + '_' + c2_name))
        assert np.all(sl.cov(c1_name, c2_name) ==
                      sl.cov(c1_name + '_' + c2_name))

    # Check slicer_part
    for sl in [slicer_multi_bunch_part, slicer_multi_bunch_part_copy,
               slicer_multi_bunch_part_buffer, slicer_multi_bunch_part_sum]:
        assert np.allclose(sl.zeta_centers,
                           slicer_multi_bunch.zeta_centers[:-1],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.num_particles,
                           slicer_multi_bunch.num_particles[:-1],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum(c1_name),
                           slicer_multi_bunch.sum(c1_name)[:-1],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum(c2_name),
                           slicer_multi_bunch.sum(c2_name)[:-1],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum('zeta'),
                           slicer_multi_bunch.sum('zeta')[:-1],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum(c1_name + c1_name),
                           slicer_multi_bunch.sum(c1_name + c1_name)[:-1],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum(c2_name + c2_name),
                           slicer_multi_bunch.sum(c2_name + c2_name)[:-1],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum(c1_name + c2_name),
                           slicer_multi_bunch.sum(c1_name + c2_name)[:-1],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum('zetazeta'),
                           slicer_multi_bunch.sum('zetazeta')[:-1],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.sum(c1_name + 'zeta'),
                           slicer_multi_bunch.sum(c1_name + 'zeta')[:-1],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.mean(c1_name),
                           slicer_multi_bunch.mean(c1_name)[:-1],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.mean(c2_name),
                           slicer_multi_bunch.mean(c2_name)[:-1],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.mean(c1_name + c1_name),
                           slicer_multi_bunch.mean(c1_name + c1_name)[:-1],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.mean(c2_name + c2_name),
                           slicer_multi_bunch.mean(c2_name + c2_name)[:-1],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.mean(c1_name + c2_name),
                           slicer_multi_bunch.mean(c1_name + c2_name)[:-1],
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.mean(c1_name + 'zeta'),
                           slicer_multi_bunch.mean(c1_name + 'zeta')[:-1],
                           rtol=0, atol=1e-12)
        assert np.allclose(
            sl.cov(c1_name, c2_name),
            sl.mean(c1_name + c2_name) - sl.mean(c1_name) * sl.mean(c2_name),
            rtol=0, atol=1e-12)
        assert np.allclose(
            sl.var(c1_name), sl.cov(c1_name, c1_name),
            rtol=0, atol=1e-12)
        assert np.allclose(
            sl.var(c1_name), sl.mean(c1_name + c1_name) - sl.mean(c1_name)**2,
            rtol=0, atol=1e-12)
        assert np.allclose(
            sl.var('zeta'), sl.mean('zetazeta') - sl.mean('zeta')**2,
            rtol=0, atol=1e-12)
        assert np.allclose(sl.std(c1_name), np.sqrt(sl.var(c1_name)),
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.std(c2_name), np.sqrt(sl.var(c2_name)),
                           rtol=0, atol=1e-12)
        assert np.allclose(sl.std('zeta'), np.sqrt(sl.var('zeta')),
                           rtol=0, atol=1e-12)

# Try selected moments
slicer_multi_bunch_mom = UniformBinSlicer(
    zeta_range=(-1, 1), num_slices=3,
    num_bunches=4, bunch_spacing_zeta=bunch_spacing_zeta,
    moments=['delta', 'xy', 'px_px'])
slicer_multi_bunch_mom_1 = slicer_multi_bunch_mom.copy()
slicer_multi_bunch_mom_2 = slicer_multi_bunch_mom.copy()

assert np.all(np.array(slicer_multi_bunch_mom.moments) == np.array(
    ['x', 'px', 'y', 'delta', 'x_y', 'px_px']))

p1 = xt.Particles(zeta=[0.97, 0.98, 0.99],
                  weight=[1, 2, 1],
                  x=[99, 100, 101],
                  y=[201, 200, 199])
p2 = xt.Particles(zeta=np.array([-0.01, 0, 0.01]) + 2 * bunch_spacing_zeta,
                  weight=[1, 2, 1],
                  x=[99, 100, 101],
                  y=[201, 200, 199])
p = xt.Particles.merge([p1, p2])

slicer_multi_bunch_mom.slice(p)
slicer_multi_bunch.slice(p)

# Test copy
slicer_multi_bunch_mom_copy = slicer_multi_bunch_mom.copy()

# Test round-trip with _to/_from_npbuffer
slicer_multi_bunch_mom_buffer = UniformBinSlicer._from_npbuffer(
                                    slicer_multi_bunch_mom._to_npbuffer())

# Test sum
pp1 = p.filter(p.zeta < 10)
pp2 = p.filter(p.zeta >= 10)
slicer_multi_bunch_mom_1.slice(pp1)
slicer_multi_bunch_mom_2.slice(pp2)
slicer_multi_bunch_mom_sum = sum([slicer_multi_bunch_mom_1,
                                  slicer_multi_bunch_mom_2])

for sl in [slicer_multi_bunch_mom, slicer_multi_bunch_mom_copy,
           slicer_multi_bunch_mom_buffer, slicer_multi_bunch_mom_sum]:
    assert np.allclose(sl.num_particles,
                       slicer_multi_bunch.num_particles,
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.zeta_centers,
                       slicer_multi_bunch.zeta_centers,
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('x'),
                       slicer_multi_bunch.sum('x'),
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('y'),
                       slicer_multi_bunch.sum('y'),
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('px'),
                       slicer_multi_bunch.sum('px'),
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.sum('delta'),
                       slicer_multi_bunch.sum('delta'),
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.cov('x_y'),
                       slicer_multi_bunch.cov('x_y'),
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.cov('px_px'),
                       slicer_multi_bunch.cov('px_px'),
                       rtol=0, atol=1e-12)
    assert np.allclose(sl.var('px'),
                       slicer_multi_bunch.var('px'),
                       rtol=0, atol=1e-12)

p = xt.Particles(zeta=np.random.uniform(-1, 1, int(1e6)),
                 x=np.random.normal(0, 1, int(1e6)))
slicer_time = UniformBinSlicer(zeta_range=(-1, 1), num_slices=100,
                               moments=['x', 'xx'])
