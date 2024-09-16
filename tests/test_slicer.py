import numpy as np
import xfields as xf
import xtrack as xt
import xobjects as xo

from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_slicer_zeta(test_context):
    zeta_range = (-1.0, 1.0)
    num_slices = 10
    dzeta = (zeta_range[1]-zeta_range[0])/num_slices
    zeta_slice_edges = np.linspace(zeta_range[0],
                                   zeta_range[1],
                                   num_slices+1)
    zeta_centers = zeta_slice_edges[:-1]+dzeta/2
    slicer_0 = xf.UniformBinSlicer(_context=test_context, zeta_range=zeta_range,
                                   num_slices=num_slices)
    slicer_1 = xf.UniformBinSlicer(_context=test_context, zeta_range=zeta_range,
                                   dzeta=dzeta)
    slicer_2 = xf.UniformBinSlicer(_context=test_context,
                                   zeta_slice_edges=zeta_slice_edges)
    xo.assert_allclose(slicer_0.zeta_centers,
                       zeta_centers)
    xo.assert_allclose(slicer_1.zeta_centers,
                       zeta_centers)
    xo.assert_allclose(slicer_2.zeta_centers,
                       zeta_centers)


@for_all_test_contexts
def test_slice_attribution_single_bunch(test_context):
    slicer = xf.UniformBinSlicer(zeta_range=(-1, 1), num_slices=3,
                                 _context=test_context)
    assert slicer.num_bunches == 1

    p0 = xt.Particles(
        zeta=[-2, -1.51, -1.49, -1, -0.51, -0.49, 0, 0.49, 0.51, 1, 1.49, 1.51,
              2,
              2.51],
        weight=[10, 10, 10, 10, 10, 20, 20, 20, 30, 30, 30, 40, 40, 40],
        x=[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.,
           12., 13.],
        y=[13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2.,
           1., 0.],
        _context=test_context
    )

    p0.state[-1] = 0

    p = p0.copy()

    i_slice_expected = [-1, -1, -1, 0, 0, 0, 1, 2, 2, -1, -1, -1, -1, -999]
    i_bunch_expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -9999]
    i_slice_particles = p.particle_id * 0 - 999
    i_slot_particles = p.particle_id * 0 - 9999
    slicer.slice(particles=p, i_slot_particles=i_slot_particles,
                 i_slice_particles=i_slice_particles)

    xo.assert_allclose(np.array(i_slice_expected), i_slice_particles, atol=0, rtol=0)
    xo.assert_allclose(i_slot_particles, i_bunch_expected,  atol=0, rtol=0)

    expected_num_particles = np.array([40, 20, 50])
    xo.assert_allclose(slicer.num_particles, expected_num_particles,
                       atol=1e-12, rtol=0)


@for_all_test_contexts
def test_slice_attribution_multi_bunch(test_context):
    p0 = xt.Particles(
        zeta=[-2, -1.51, -1.49, -1, -0.51, -0.49, 0, 0.49, 0.51, 1, 1.49, 1.51,
              2,
              2.51],
        weight=[10, 10, 10, 10, 10, 20, 20, 20, 30, 30, 30, 40, 40, 40],
        x=[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.,
           12., 13.],
        y=[13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2.,
           1., 0.],
        _context=test_context
    )

    p0.state[-1] = 0

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

    i_slot_particles = p.particle_id * 0 - 999
    i_slice_particles = p.particle_id * 0 - 999

    slicer = xf.UniformBinSlicer(zeta_range=(-1, 1), num_slices=3,
                                 num_bunches=4,
                                 bunch_spacing_zeta=bunch_spacing_zeta,
                                 _context=test_context)
    slicer.slice(particles=p,
                 i_slice_particles=i_slice_particles,
                 i_slot_particles=i_slot_particles)

    # when we merge the particles the ones with state zero end up at the end so
    # the -999 are now at the end of the array
    i_slice_expected = [-1, -1, -1, 0, 0, 0, 1, 2, 2, -1, -1, -1, -1, -999]
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

    xo.assert_allclose(i_slot_particles, i_bunch_expected_mb, rtol=0, atol=0)
    xo.assert_allclose(i_slice_particles, i_slice_expected_mb, rtol=0, atol=0)
    xo.assert_allclose(slicer.num_particles, expected_num_particles_mb,
                       atol=1e-12, rtol=0)


@for_all_test_contexts
def test_slicer_moments_single_bunch(test_context):
    slicer_single_bunch = xf.UniformBinSlicer(zeta_range=(-1, 1), num_slices=3,
                                              _context=test_context)
    slicer_single_bunch_1 = slicer_single_bunch.copy()
    slicer_single_bunch_2 = slicer_single_bunch.copy()

    p = xt.Particles(zeta=[0.97, 0.98, 0.99],
                     weight=[1, 2, 1],
                     x=[99, 100, 101],
                     y=[201, 200, 199],
                     _context=test_context)
    slicer_single_bunch.slice(p)

    # Test copy
    slicer_single_bunch_copy = slicer_single_bunch.copy()

    # try round-trip with _to/_from_npbuffer
    slicer_single_bunch_buffer = xf.UniformBinSlicer._from_npbuffer(
        slicer_single_bunch.copy(_context=xo.context_default)._to_npbuffer())

    # Test sum
    pp1 = p.filter(p.zeta < 0.98)
    pp2 = p.filter(p.zeta >= 0.98)

    slicer_single_bunch_1.slice(pp1)
    slicer_single_bunch_2.slice(pp2)
    slicer_single_bunch_sum = sum(
        [slicer_single_bunch_1, slicer_single_bunch_2])

    for sl in [slicer_single_bunch, slicer_single_bunch_copy,
               slicer_single_bunch_buffer, slicer_single_bunch_sum]:

        sl.move(_context=xo.context_default)
        p.move(_context=xo.context_default)

        assert sl.bunch_spacing_zeta == 0

        xo.assert_allclose(sl.zeta_centers, np.array([-2 / 3, 0, 2 / 3]),
                           rtol=0,
                           atol=1e-12)
        xo.assert_allclose(sl.num_particles, [0, 0, p.weight.sum()], rtol=0,
                           atol=1e-12)
        xo.assert_allclose(sl.sum('x'), [0, 0, (p.x * p.weight).sum()], rtol=0,
                           atol=1e-12)
        xo.assert_allclose(sl.sum('y'), [0, 0, (p.y * p.weight).sum()], rtol=0,
                           atol=1e-12)
        xo.assert_allclose(sl.sum('zeta'), [0, 0, (p.zeta * p.weight).sum()],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.sum('xx'), [0, 0, (p.x ** 2 * p.weight).sum()],
                           rtol=0,
                           atol=1e-12)
        xo.assert_allclose(sl.sum('yy'), [0, 0, (p.y ** 2 * p.weight).sum()],
                           rtol=0,
                           atol=1e-12)
        xo.assert_allclose(sl.sum('zetazeta'),
                           [0, 0, (p.zeta ** 2 * p.weight).sum()],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.sum('xy'), [0, 0, (p.x * p.y * p.weight).sum()],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.sum('xzeta'),
                           [0, 0, (p.x * p.zeta * p.weight).sum()],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.mean('x'),
                           [0, 0, (p.x * p.weight).sum() / p.weight.sum()],
                           rtol=0,
                           atol=1e-12)
        xo.assert_allclose(sl.mean('y'),
                           [0, 0, (p.y * p.weight).sum() / p.weight.sum()],
                           rtol=0,
                           atol=1e-12)
        xo.assert_allclose(sl.mean('xx'),
                           [0, 0, (p.x ** 2 * p.weight).sum() / p.weight.sum()],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.mean('yy'),
                           [0, 0, (p.y ** 2 * p.weight).sum() / p.weight.sum()],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.mean('zetazeta'),
                           [0, 0,
                            (p.zeta ** 2 * p.weight).sum() / p.weight.sum()],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.mean('xy'),
                           [0, 0,
                            (p.x * p.y * p.weight).sum() / p.weight.sum()],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.mean('xzeta'),
                           [0, 0,
                            (p.x * p.zeta * p.weight).sum() / p.weight.sum()],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.cov('x', 'y'),
                           sl.mean('xy') - sl.mean('x') * sl.mean('y'),
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.var('x'), sl.cov('x', 'x'),
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.var('x'), sl.mean('xx') - sl.mean('x') ** 2,
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.var('zeta'),
                           sl.mean('zetazeta') - sl.mean('zeta') ** 2,
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.std('x'), np.sqrt(sl.var('x')),
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.std('y'), np.sqrt(sl.var('y')),
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.std('zeta'), np.sqrt(sl.var('zeta')),
                           rtol=0, atol=1e-12)

        assert np.all(sl.sum('xy') == sl.sum('x_y'))
        assert np.all(sl.sum('x', 'y') == sl.sum('x_y'))
        assert np.all(sl.mean('xy') == sl.mean('x_y'))
        assert np.all(sl.mean('x', 'y') == sl.mean('x_y'))
        assert np.all(sl.cov('xy') == sl.cov('x_y'))
        assert np.all(sl.cov('x', 'y') == sl.cov('x_y'))

        sl.move(_context=test_context)
        p.move(_context=test_context)

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
                            c2_name: [201, 200, 199]},
                        _context=test_context)


        slicer_single_bunch.slice(p)

        # Test copy
        slicer_single_bunch_copy = slicer_single_bunch.copy()

        # Test round-trip with _to/_from_npbuffer
        slicer_single_bunch_buffer = xf.UniformBinSlicer._from_npbuffer(
            slicer_single_bunch.copy(_context=xo.context_default)._to_npbuffer())

        # Test sum
        pp1 = p.filter(p.zeta < 0.98)
        pp2 = p.filter(p.zeta >= 0.98)

        slicer_single_bunch_1.slice(pp1)
        slicer_single_bunch_2.slice(pp2)
        slicer_single_bunch_sum = sum([slicer_single_bunch_1,
                                       slicer_single_bunch_2])

        for sl in [slicer_single_bunch, slicer_single_bunch_copy,
                   slicer_single_bunch_buffer, slicer_single_bunch_sum]:

            sl.move(_context=xo.context_default)
            p.move(_context=xo.context_default)

            c1 = getattr(p, c1_name)
            c2 = getattr(p, c2_name)

            xo.assert_allclose(sl.zeta_centers, np.array([-2 / 3, 0, 2 / 3]),
                               rtol=0,
                               atol=1e-12)
            xo.assert_allclose(sl.num_particles, [0, 0, p.weight.sum()], rtol=0,
                               atol=1e-12)
            xo.assert_allclose(sl.sum(c1_name), [0, 0, (c1 * p.weight).sum()],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum(c2_name), [0, 0, (c2 * p.weight).sum()],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum('zeta'),
                               [0, 0, (p.zeta * p.weight).sum()],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum(c1_name + c1_name),
                               [0, 0, (c1 ** 2 * p.weight).sum()],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum(c2_name + c2_name),
                               [0, 0, (c2 ** 2 * p.weight).sum()],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum(c1_name + c2_name),
                               [0, 0, (c1 * c2 * p.weight).sum()],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum('zetazeta'),
                               [0, 0, (p.zeta ** 2 * p.weight).sum()],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum(c1_name + 'zeta'),
                               [0, 0, (c1 * p.zeta * p.weight).sum()],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.mean(c1_name),
                               [0, 0, (c1 * p.weight).sum() / p.weight.sum()],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.mean(c2_name),
                               [0, 0, (c2 * p.weight).sum() / p.weight.sum()],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.mean(c1_name + c1_name),
                               [0, 0,
                                (c1 ** 2 * p.weight).sum() / p.weight.sum()],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.mean(c2_name + c2_name),
                               [0, 0,
                                (c2 ** 2 * p.weight).sum() / p.weight.sum()],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.mean(c1_name + c2_name),
                               [0, 0,
                                (c1 * c2 * p.weight).sum() / p.weight.sum()],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(
                sl.mean(c1_name + 'zeta'),
                [0, 0, (c1 * p.zeta * p.weight).sum() / p.weight.sum()],
                rtol=0, atol=1e-12)
            xo.assert_allclose(
                sl.cov(c1_name, c2_name),
                sl.mean(c1_name + c2_name) - sl.mean(c1_name) * sl.mean(
                    c2_name),
                rtol=0, atol=1e-12)
            xo.assert_allclose(
                sl.var(c1_name), sl.cov(c1_name, c1_name),
                rtol=0, atol=1e-12)
            xo.assert_allclose(
                sl.var(c1_name),
                sl.mean(c1_name + c1_name) - sl.mean(c1_name) ** 2,
                rtol=0, atol=1e-12)
            xo.assert_allclose(
                sl.var('zeta'), sl.mean('zetazeta') - sl.mean('zeta') ** 2,
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

            sl.move(_context=test_context)
            p.move(_context=test_context)


@for_all_test_contexts
def test_slicer_moments_multi_bunch(test_context):
    bunch_spacing_zeta = 10.
    slicer_multi_bunch = xf.UniformBinSlicer(
        zeta_range=(-1, 1), num_slices=3,
        num_bunches=4,
        bunch_spacing_zeta=bunch_spacing_zeta,
        _context=test_context)
    slicer_multi_bunch_1 = slicer_multi_bunch.copy()
    slicer_multi_bunch_2 = slicer_multi_bunch.copy()

    slicer_multi_bunch_part = xf.UniformBinSlicer(
        zeta_range=(-1, 1), num_slices=3,
        num_bunches=3,
        bunch_spacing_zeta=bunch_spacing_zeta,
        _context=test_context)
    slicer_multi_bunch_part_1 = slicer_multi_bunch_part.copy()
    slicer_multi_bunch_part_2 = slicer_multi_bunch_part.copy()

    p1 = xt.Particles(zeta=[0.97, 0.98, 0.99],
                      weight=[1, 2, 1],
                      x=[99, 100, 101],
                      y=[201, 200, 199],
                      _context=test_context)
    p2 = xt.Particles(zeta=np.array([-0.01, 0, 0.01]) - 2 * bunch_spacing_zeta,
                      weight=[1, 2, 1],
                      x=[99, 100, 101],
                      y=[201, 200, 199],
                      _context=test_context)
    p = xt.Particles.merge([p1, p2], _context=test_context)

    assert np.isclose(slicer_multi_bunch.bunch_spacing_zeta, 10, rtol=0,
                      atol=1e-12)
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
    slicer_multi_bunch_buffer = xf.UniformBinSlicer._from_npbuffer(
        slicer_multi_bunch.copy(_context=xo.context_default)._to_npbuffer())
    slicer_multi_bunch_part_buffer = xf.UniformBinSlicer._from_npbuffer(
        slicer_multi_bunch_part.copy(_context=xo.context_default)._to_npbuffer())

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

        sl.move(_context=xo.context_default)
        p.move(_context=xo.context_default)
        p1.move(_context=xo.context_default)
        p2.move(_context=xo.context_default)

        xo.assert_allclose(sl.zeta_centers,
                           np.array([[-2 / 3, 0, 2 / 3],
                                     [-10 - 2 / 3, -10, -10 + 2 / 3],
                                     [-20 - 2 / 3, -20, -20 + 2 / 3],
                                     [-30 - 2 / 3, -30, -30 + 2 / 3]]),
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.num_particles,
                           [[0, 0, p1.weight.sum()],
                            [0, 0, 0],
                            [0, p2.weight.sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.sum('x'),
                           [[0, 0, (p1.x * p1.weight).sum()],
                            [0, 0, 0],
                            [0, (p2.x * p2.weight).sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.sum('y'),
                           [[0, 0, (p1.y * p1.weight).sum()],
                            [0, 0, 0],
                            [0, (p2.y * p2.weight).sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.sum('zeta'),
                           [[0, 0, (p1.zeta * p1.weight).sum()],
                            [0, 0, 0],
                            [0, (p2.zeta * p2.weight).sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.sum('xx'),
                           [[0, 0, (p1.x ** 2 * p1.weight).sum()],
                            [0, 0, 0],
                            [0, (p2.x ** 2 * p2.weight).sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.sum('yy'),
                           [[0, 0, (p1.y ** 2 * p1.weight).sum()],
                            [0, 0, 0],
                            [0, (p2.y ** 2 * p2.weight).sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.sum('zetazeta'),
                           [[0, 0, (p1.zeta ** 2 * p1.weight).sum()],
                            [0, 0, 0],
                            [0, (p2.zeta ** 2 * p2.weight).sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.sum('xy'),
                           [[0, 0, (p1.x * p1.y * p1.weight).sum()],
                            [0, 0, 0],
                            [0, (p2.x * p2.y * p2.weight).sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.sum('xzeta'),
                           [[0, 0, (p1.x * p1.zeta * p1.weight).sum()],
                            [0, 0, 0],
                            [0, (p2.x * p2.zeta * p2.weight).sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.mean('x'),
                           [[0, 0, (p1.x * p1.weight).sum() / p1.weight.sum()],
                            [0, 0, 0],
                            [0, (p2.x * p2.weight).sum() / p2.weight.sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.mean('y'),
                           [[0, 0, (p1.y * p1.weight).sum() / p1.weight.sum()],
                            [0, 0, 0],
                            [0, (p2.y * p2.weight).sum() / p2.weight.sum(), 0],
                            [0, 0, 0]],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(
            sl.mean('xx'),
            [[0, 0,
             (p1.x ** 2 * p1.weight).sum() / p1.weight.sum()],
             [0, 0, 0],
             [0, (p2.x ** 2 * p2.weight).sum() / p2.weight.sum(), 0],
             [0, 0, 0]],
            rtol=0, atol=1e-12)
        xo.assert_allclose(
            sl.mean('yy'),
            [[0, 0, (p1.y ** 2 * p1.weight).sum() / p1.weight.sum()],
             [0, 0, 0],
             [0, (p2.y ** 2 * p2.weight).sum() / p2.weight.sum(), 0],
             [0, 0, 0]],
            rtol=0, atol=1e-12)
        xo.assert_allclose(
            sl.mean('xy'),
            [[0, 0, (p1.x * p1.y * p1.weight).sum() / p1.weight.sum()],
             [0, 0, 0],
             [0, (p2.x * p2.y * p2.weight).sum() / p2.weight.sum(), 0],
             [0, 0, 0]],
            rtol=0, atol=1e-12)
        xo.assert_allclose(
            sl.mean('xzeta'),
            [[0, 0, (p1.x * p1.zeta * p1.weight).sum() / p1.weight.sum()],
             [0, 0, 0],
             [0, (p2.x * p2.zeta * p2.weight).sum() / p2.weight.sum(), 0],
             [0, 0, 0]],
            rtol=0, atol=1e-12)
        xo.assert_allclose(
            sl.cov('x', 'y'),
            sl.mean('xy') - sl.mean('x') * sl.mean('y'),
            rtol=0, atol=1e-12)
        xo.assert_allclose(
            sl.var('x'), sl.cov('x', 'x'),
            rtol=0, atol=1e-12)
        xo.assert_allclose(
            sl.var('x'), sl.mean('xx') - sl.mean('x') ** 2,
            rtol=0, atol=1e-12)
        xo.assert_allclose(
            sl.var('zeta'), sl.mean('zetazeta') - sl.mean('zeta') ** 2,
            rtol=0, atol=1e-12)
        xo.assert_allclose(sl.std('x'), np.sqrt(sl.var('x')),
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.std('y'), np.sqrt(sl.var('y')),
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.std('zeta'), np.sqrt(sl.var('zeta')),
                           rtol=0, atol=1e-12)

        assert np.all(sl.sum('xy') == sl.sum('x_y'))
        assert np.all(sl.sum('x', 'y') == sl.sum('x_y'))
        assert np.all(sl.mean('xy') == sl.mean('x_y'))
        assert np.all(sl.mean('x', 'y') == sl.mean('x_y'))
        assert np.all(sl.cov('xy') == sl.cov('x_y'))
        assert np.all(sl.cov('x', 'y') == sl.cov('x_y'))

        sl.move(_context=test_context)
        p.move(_context=test_context)
        p1.move(_context=test_context)
        p2.move(_context=test_context)

    # Check slicer_part
    for sl in [slicer_multi_bunch_part, slicer_multi_bunch_part_copy,
               slicer_multi_bunch_part_buffer,
               slicer_multi_bunch_part_sum]:

        sl.move(_context=xo.context_default)
        p.move(_context=xo.context_default)
        p1.move(_context=xo.context_default)
        p2.move(_context=xo.context_default)
        slicer_multi_bunch.move(_context=xo.context_default)

        xo.assert_allclose(sl.zeta_centers,
                           slicer_multi_bunch.zeta_centers[:-1],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.num_particles,
                           slicer_multi_bunch.num_particles[:-1],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.sum('x'), slicer_multi_bunch.sum('x')[:-1],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.sum('y'), slicer_multi_bunch.sum('y')[:-1],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.sum('zeta'), slicer_multi_bunch.sum('zeta')[:-1],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.sum('xx'), slicer_multi_bunch.sum('xx')[:-1],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.sum('yy'), slicer_multi_bunch.sum('yy')[:-1],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.sum('zetazeta'),
                           slicer_multi_bunch.sum('zetazeta')[:-1],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.sum('xy'), slicer_multi_bunch.sum('xy')[:-1],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.sum('xzeta'),
                           slicer_multi_bunch.sum('xzeta')[:-1],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.mean('x'), slicer_multi_bunch.mean('x')[:-1],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.mean('y'), slicer_multi_bunch.mean('y')[:-1],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.mean('xx'), slicer_multi_bunch.mean('xx')[:-1],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.mean('yy'), slicer_multi_bunch.mean('yy')[:-1],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.mean('xy'), slicer_multi_bunch.mean('xy')[:-1],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.mean('xzeta'),
                           slicer_multi_bunch.mean('xzeta')[:-1],
                           rtol=0, atol=1e-12)
        xo.assert_allclose(
            sl.cov('x', 'y'),
            sl.mean('xy') - sl.mean('x') * sl.mean('y'),
            rtol=0, atol=1e-12)
        xo.assert_allclose(
            sl.var('x'), sl.cov('x', 'x'),
            rtol=0, atol=1e-12)
        xo.assert_allclose(
            sl.var('x'), sl.mean('xx') - sl.mean('x') ** 2,
            rtol=0, atol=1e-12)
        xo.assert_allclose(
            sl.var('zeta'), sl.mean('zetazeta') - sl.mean('zeta') ** 2,
            rtol=0, atol=1e-12)
        xo.assert_allclose(sl.std('x'), np.sqrt(sl.var('x')),
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.std('y'), np.sqrt(sl.var('y')),
                           rtol=0, atol=1e-12)
        xo.assert_allclose(sl.std('zeta'), np.sqrt(sl.var('zeta')),
                           rtol=0, atol=1e-12)

        sl.move(_context=test_context)
        p.move(_context=test_context)
        p1.move(_context=test_context)
        p2.move(_context=test_context)
        slicer_multi_bunch.move(_context=test_context)

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
        p2 = xt.Particles(
            zeta=np.array([-0.01, 0, 0.01]) - 2 * bunch_spacing_zeta,
            weight=[1, 2, 1],
            **{c1_name: [99, 100, 101],
               c2_name: [201, 200, 199]})

        p = xt.Particles.merge([p1, p2])

        p.move(_context=test_context)

        slicer_multi_bunch.slice(p)
        slicer_multi_bunch_part.slice(p)

        pp1 = p.filter(p.zeta < 0.5)
        pp2 = p.filter(p.zeta >= 0.5)

        slicer_multi_bunch_1.slice(pp1)
        slicer_multi_bunch_2.slice(pp2)
        slicer_multi_bunch_sum = sum(
            [slicer_multi_bunch_1, slicer_multi_bunch_2])

        slicer_multi_bunch_part_1.slice(pp1)
        slicer_multi_bunch_part_2.slice(pp2)
        slicer_multi_bunch_part_sum = sum([slicer_multi_bunch_part_1,
                                           slicer_multi_bunch_part_2])

        # Test copy
        slicer_multi_bunch_copy = slicer_multi_bunch.copy()
        slicer_multi_bunch_part_copy = slicer_multi_bunch_part.copy()

        # Test round-trip with _to/_from_npbuffer
        slicer_multi_bunch_buffer = xf.UniformBinSlicer._from_npbuffer(
            slicer_multi_bunch.copy(_context=xo.context_default)._to_npbuffer())
        slicer_multi_bunch_part_buffer = xf.UniformBinSlicer._from_npbuffer(
            slicer_multi_bunch_part.copy(_context=xo.context_default)._to_npbuffer())

        c1_p1 = getattr(p1, c1_name)
        c2_p1 = getattr(p1, c2_name)
        c1_p2 = getattr(p2, c1_name)
        c2_p2 = getattr(p2, c2_name)
        for sl in [slicer_multi_bunch, slicer_multi_bunch_copy,
                   slicer_multi_bunch_buffer, slicer_multi_bunch_sum]:
            sl.move(_context=xo.context_default)
            xo.assert_allclose(sl.zeta_centers,
                               np.array([[-2 / 3, 0, 2 / 3],
                                         [-10 - 2 / 3, -10, -10 + 2 / 3],
                                         [-20 - 2 / 3, -20, -20 + 2 / 3],
                                         [-30 - 2 / 3, -30, -30 + 2 / 3]]),
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.num_particles, [[0, 0, p1.weight.sum()],
                                                  [0, 0, 0],
                                                  [0, p2.weight.sum(), 0],
                                                  [0, 0, 0]],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum(c1_name),
                               [[0, 0, (c1_p1 * p1.weight).sum()],
                                [0, 0, 0],
                                [0, (c1_p2 * p2.weight).sum(), 0],
                                [0, 0, 0]],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum(c2_name),
                               [[0, 0, (c2_p1 * p1.weight).sum()],
                                [0, 0, 0],
                                [0, (c2_p2 * p2.weight).sum(), 0],
                                [0, 0, 0]],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum('zeta'),
                               [[0, 0, (p1.zeta * p1.weight).sum()],
                                [0, 0, 0],
                                [0, (p2.zeta * p2.weight).sum(), 0],
                                [0, 0, 0]],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum(c1_name + c1_name),
                               [[0, 0, (c1_p1 ** 2 * p1.weight).sum()],
                                [0, 0, 0],
                                [0, (c1_p2 ** 2 * p2.weight).sum(), 0],
                                [0, 0, 0]],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum(c2_name + c2_name),
                               [[0, 0, (c2_p1 ** 2 * p1.weight).sum()],
                                [0, 0, 0],
                                [0, (c2_p2 ** 2 * p2.weight).sum(), 0],
                                [0, 0, 0]],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum(c1_name + c2_name),
                               [[0, 0, (c1_p1 * c2_p1 * p1.weight).sum()],
                                [0, 0, 0],
                                [0, (c1_p2 * c2_p2 * p2.weight).sum(), 0],
                                [0, 0, 0]],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum('zetazeta'),
                               [[0, 0, (p1.zeta ** 2 * p1.weight).sum()],
                                [0, 0, 0],
                                [0, (p2.zeta ** 2 * p2.weight).sum(), 0],
                                [0, 0, 0]],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum(c1_name + 'zeta'),
                               [[0, 0, (c1_p1 * p1.zeta * p1.weight).sum()],
                                [0, 0, 0],
                                [0, (c1_p2 * p2.zeta * p2.weight).sum(), 0],
                                [0, 0, 0]],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(
                sl.mean(c1_name),
                [[0, 0, (c1_p1 * p1.weight).sum() / p1.weight.sum()],
                 [0, 0, 0],
                 [0, (c1_p2 * p2.weight).sum() / p2.weight.sum(), 0],
                 [0, 0, 0]],
                rtol=0, atol=1e-12)
            xo.assert_allclose(
                sl.mean(c2_name),
                [[0, 0,
                 (c2_p1 * p1.weight).sum() / p1.weight.sum()],
                 [0, 0, 0],
                 [0, (c2_p2 * p2.weight).sum() / p2.weight.sum(), 0],
                 [0, 0, 0]],
                rtol=0, atol=1e-12)
            xo.assert_allclose(
                sl.mean(c1_name + c1_name),
                [[0, 0, (c1_p1 ** 2 * p1.weight).sum() / p1.weight.sum()],
                 [0, 0, 0],
                 [0, (c1_p2 ** 2 * p2.weight).sum() / p2.weight.sum(), 0],
                 [0, 0, 0]],
                rtol=0, atol=1e-12)
            xo.assert_allclose(
                sl.mean(c2_name + c2_name),
                [[0, 0, (c2_p1 ** 2 * p1.weight).sum() / p1.weight.sum()],
                 [0, 0, 0],
                 [0, (c2_p2 ** 2 * p2.weight).sum() / p2.weight.sum(), 0],
                 [0, 0, 0]],
                rtol=0, atol=1e-12)
            xo.assert_allclose(
                sl.mean(c1_name + c2_name),
                [[0, 0, (c1_p1 * c2_p1 * p1.weight).sum() / p1.weight.sum()],
                 [0, 0, 0],
                 [0, (c1_p2 * c2_p2 * p2.weight).sum() / p2.weight.sum(), 0],
                 [0, 0, 0]],
                rtol=0, atol=1e-12)
            xo.assert_allclose(
                sl.mean(c1_name + 'zeta'),
                [[0, 0, (c1_p1 * p1.zeta * p1.weight).sum() / p1.weight.sum()],
                 [0, 0, 0],
                 [0, (c1_p2 * p2.zeta * p2.weight).sum() / p2.weight.sum(), 0],
                 [0, 0, 0]],
                rtol=0, atol=1e-12)
            xo.assert_allclose(
                sl.cov(c1_name, c2_name),
                sl.mean(c1_name + c2_name) - sl.mean(c1_name) * sl.mean(
                    c2_name),
                rtol=0, atol=1e-12)
            xo.assert_allclose(
                sl.var(c1_name), sl.cov(c1_name, c1_name),
                rtol=0, atol=1e-12)
            xo.assert_allclose(
                sl.var(c1_name),
                sl.mean(c1_name + c1_name) - sl.mean(c1_name) ** 2,
                rtol=0, atol=1e-12)
            xo.assert_allclose(
                sl.var('zeta'), sl.mean('zetazeta') - sl.mean('zeta') ** 2,
                rtol=0, atol=1e-12)
            xo.assert_allclose(sl.std(c1_name), np.sqrt(sl.var(c1_name)),
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.std(c2_name), np.sqrt(sl.var(c2_name)),
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.std('zeta'), np.sqrt(sl.var('zeta')),
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
            sl.move(_context=test_context)

        # Check slicer_part
        for sl in [slicer_multi_bunch_part, slicer_multi_bunch_part_copy,
                   slicer_multi_bunch_part_buffer, slicer_multi_bunch_part_sum]:

            sl.move(_context=xo.context_default)
            slicer_multi_bunch.move(_context=xo.context_default)

            xo.assert_allclose(sl.zeta_centers,
                               slicer_multi_bunch.zeta_centers[:-1],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.num_particles,
                               slicer_multi_bunch.num_particles[:-1],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum(c1_name),
                               slicer_multi_bunch.sum(c1_name)[:-1],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum(c2_name),
                               slicer_multi_bunch.sum(c2_name)[:-1],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum('zeta'),
                               slicer_multi_bunch.sum('zeta')[:-1],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum(c1_name + c1_name),
                               slicer_multi_bunch.sum(c1_name + c1_name)[:-1],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum(c2_name + c2_name),
                               slicer_multi_bunch.sum(c2_name + c2_name)[:-1],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum(c1_name + c2_name),
                               slicer_multi_bunch.sum(c1_name + c2_name)[:-1],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum('zetazeta'),
                               slicer_multi_bunch.sum('zetazeta')[:-1],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.sum(c1_name + 'zeta'),
                               slicer_multi_bunch.sum(c1_name + 'zeta')[:-1],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.mean(c1_name),
                               slicer_multi_bunch.mean(c1_name)[:-1],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.mean(c2_name),
                               slicer_multi_bunch.mean(c2_name)[:-1],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.mean(c1_name + c1_name),
                               slicer_multi_bunch.mean(c1_name + c1_name)[:-1],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.mean(c2_name + c2_name),
                               slicer_multi_bunch.mean(c2_name + c2_name)[:-1],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.mean(c1_name + c2_name),
                               slicer_multi_bunch.mean(c1_name + c2_name)[:-1],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.mean(c1_name + 'zeta'),
                               slicer_multi_bunch.mean(c1_name + 'zeta')[:-1],
                               rtol=0, atol=1e-12)
            xo.assert_allclose(
                sl.cov(c1_name, c2_name),
                sl.mean(c1_name + c2_name) - sl.mean(c1_name) * sl.mean(
                    c2_name),
                rtol=0, atol=1e-12)
            xo.assert_allclose(
                sl.var(c1_name), sl.cov(c1_name, c1_name),
                rtol=0, atol=1e-12)
            xo.assert_allclose(
                sl.var(c1_name),
                sl.mean(c1_name + c1_name) - sl.mean(c1_name) ** 2,
                rtol=0, atol=1e-12)
            xo.assert_allclose(
                sl.var('zeta'), sl.mean('zetazeta') - sl.mean('zeta') ** 2,
                rtol=0, atol=1e-12)
            xo.assert_allclose(sl.std(c1_name), np.sqrt(sl.var(c1_name)),
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.std(c2_name), np.sqrt(sl.var(c2_name)),
                               rtol=0, atol=1e-12)
            xo.assert_allclose(sl.std('zeta'), np.sqrt(sl.var('zeta')),
                               rtol=0, atol=1e-12)

            sl.move(_context=test_context)
            slicer_multi_bunch.move(_context=test_context)
