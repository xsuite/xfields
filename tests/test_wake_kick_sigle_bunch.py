import numpy as np
from scipy.constants import c, e
from xfields import ResonatorWake, Wakefield
from xobjects.test_helpers import for_all_test_contexts
import xtrack as xt

exclude_contexts = ['ContextPyopencl', 'ContextCupy']

@for_all_test_contexts(excluding=exclude_contexts)
def test_longitudinal_wake_kick(test_context):
    n_turns_wake = 1
    flatten = False

    p0 = 7000e9 * e / c
    h_RF = 600
    bunch_spacing_buckets = 10
    n_slices = 100
    n_macroparticles = 2  # per bunch
    circumference = 26658.883
    bucket_length = circumference/h_RF
    zeta_range = (-0.5*bucket_length, 0.5*bucket_length)
    dz = (zeta_range[1] - zeta_range[0])/n_slices
    zeta = np.linspace(zeta_range[0] + dz/2, zeta_range[1]-dz/2,
                       n_slices)

    i_source = -10
    i_test = 10

    flag_plot = False

    particles = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        p0=p0,
        x=np.zeros(n_macroparticles),
        px=np.zeros(n_macroparticles),
        y=np.zeros(n_macroparticles),
        py=np.zeros(n_macroparticles),
        zeta=np.array([zeta[i_test], zeta[i_source]]),
        delta=np.zeros(n_macroparticles),
        weight=np.ones(n_macroparticles),
        _context=test_context
    )

    delta_bef = particles.delta.copy()

    wfz = ResonatorWake(
        r_shunt=1e8,
        q_factor=1e7,
        frequency=1e3,
        source_moments=['num_particles'],
        kick='delta',
        scale_kick=None
    )

    zeta_range_xf = zeta_range

    wf = Wakefield(
        components=[wfz],
        zeta_range=zeta_range_xf,
        num_slices=n_slices,  # per bunch
        bunch_spacing_zeta=bunch_spacing_buckets*bucket_length,
        num_turns=n_turns_wake,
        circumference=circumference,
        log_moments=['px'],
        _flatten=flatten
    )

    line = xt.Line(elements=[wf],
                   element_names=['wf'])
    line.build_tracker()
    line.track(particles, num_turns=1)

    scale = -particles.q0**2 * e**2 / (particles.p0c * e)

    assert np.allclose((particles.delta - delta_bef)[0],
                       (wfz.function(-particles.zeta[1] +
                                     particles.zeta[0]) * scale[0]),
                       rtol=1e-4, atol=0)


@for_all_test_contexts(excluding=exclude_contexts)
def test_constant_wake_kick(test_context):
    n_turns_wake = 1
    flatten = False

    p0 = 7000e9 * e / c
    h_RF = 600
    bunch_spacing_buckets = 10
    n_slices = 100
    n_macroparticles = 2  # per bunch
    circumference = 26658.883
    bucket_length = circumference/h_RF
    zeta_range = (-0.5*bucket_length, 0.5*bucket_length)
    dz = (zeta_range[1] - zeta_range[0])/n_slices
    zeta = np.linspace(zeta_range[0] + dz/2, zeta_range[1]-dz/2,
                       n_slices)

    i_source = -10
    i_test = 10

    particles = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        p0=p0,
        x=np.zeros(n_macroparticles),
        px=np.zeros(n_macroparticles),
        y=np.zeros(n_macroparticles),
        py=np.zeros(n_macroparticles),
        zeta=np.array([zeta[i_test], zeta[i_source]]),
        delta=np.zeros(n_macroparticles),
        weight=np.ones(n_macroparticles),
        _context=test_context
    )

    px_bef = particles.px.copy()
    py_bef = particles.py.copy()

    wfx = ResonatorWake(
        r_shunt=2e8,
        q_factor=1e7,
        frequency=1e3,
        source_moments=['num_particles'],
        kick='px',
        scale_kick=None,
    )

    wfy = ResonatorWake(
        r_shunt=3e8,
        q_factor=1e7,
        frequency=1e3,
        source_moments=['num_particles'],
        kick='py',
        scale_kick=None,
    )

    zeta_range_xf = zeta_range

    wf = Wakefield(
        components=[wfx, wfy],
        zeta_range=zeta_range_xf,
        num_slices=n_slices,  # per bunch
        bunch_spacing_zeta=bunch_spacing_buckets*bucket_length,
        num_turns=n_turns_wake,
        circumference=circumference,
        log_moments=['px'],
        _flatten=flatten
    )

    line = xt.Line(elements=[wf],
                   element_names=['wf'])
    line.build_tracker()
    line.track(particles, num_turns=1)

    scale = -particles.q0**2 * e**2 / (particles.p0c * e)

    assert np.allclose((particles.px - px_bef)[0],
                       (wfx.function(-particles.zeta[1] +
                                     particles.zeta[0]) * scale[0]),
                       rtol=1e-4, atol=0)
    assert np.allclose((particles.py - py_bef)[0],
                       (wfy.function(-particles.zeta[1] +
                                     particles.zeta[0]) * scale[0]),
                       rtol=1e-4, atol=0)


@for_all_test_contexts(excluding=exclude_contexts)
def test_direct_dipolar_wake_kick(test_context):
    n_turns_wake = 1
    flatten = False

    p0 = 7000e9 * e / c
    h_RF = 600
    bunch_spacing_buckets = 10
    n_slices = 100
    n_macroparticles = n_slices  # per bunch
    circumference = 26658.883
    bucket_length = circumference/h_RF
    zeta_range = (-0.5*bucket_length, 0.5*bucket_length)
    dz = (zeta_range[1] - zeta_range[0])/n_slices
    zeta = np.linspace(zeta_range[0] + dz/2, zeta_range[1]-dz/2,
                       n_macroparticles)
    i_source = -10

    source_moment_x = 'x'
    source_moment_y = 'y'

    scale_kick_x = None
    scale_kick_y = None

    displace_x = 2
    displace_y = 3

    particles = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        p0=p0,
        x=np.zeros(n_macroparticles),
        px=np.zeros(n_macroparticles),
        y=np.zeros(n_macroparticles),
        py=np.zeros(n_macroparticles),
        zeta=zeta,
        delta=np.zeros(n_macroparticles),
        weight=np.ones(n_macroparticles),
        _context=test_context
    )

    particles.x[i_source] += displace_x
    particles.y[i_source] += displace_y

    px_bef = particles.px.copy()
    py_bef = particles.py.copy()

    wfx = ResonatorWake(
        r_shunt=2e8,
        q_factor=1e7,
        frequency=1e3,
        source_moments=['num_particles', source_moment_x],
        kick='px',
        scale_kick=scale_kick_x,
    )

    wfy = ResonatorWake(
        r_shunt=3e8,
        q_factor=1e7,
        frequency=1e3,
        source_moments=['num_particles', source_moment_y],
        kick='py',
        scale_kick=scale_kick_y,
    )

    zeta_range_xf = zeta_range

    wf = Wakefield(
        components=[wfx, wfy],
        zeta_range=zeta_range_xf,
        num_slices=n_slices,  # per bunch
        bunch_spacing_zeta=bunch_spacing_buckets*bucket_length,
        num_turns=n_turns_wake,
        circumference=circumference,
        log_moments=['px'],
        _flatten=flatten
    )

    line = xt.Line(elements=[wf],
                   element_names=['wf'])
    line.build_tracker()
    line.track(particles, num_turns=1)

    scale = -particles.q0**2 * e**2 / (particles.p0c * e)

    mask = particles.zeta < particles.zeta[i_source]

    assert np.allclose((particles.px - px_bef)[mask]/displace_x,
                       (wfx.function(-particles.zeta[i_source] +
                                     particles.zeta) * scale[0])[mask],
                       rtol=1e-4, atol=0)
    assert np.allclose((particles.py - py_bef)[mask]/displace_y,
                       (wfy.function(-particles.zeta[i_source] +
                                     particles.zeta) * scale[0])[mask],
                       rtol=1e-4, atol=0)


@for_all_test_contexts(excluding=exclude_contexts)
def test_cross_dipolar_wake_kick(test_context):
    n_turns_wake = 1
    flatten = False

    p0 = 7000e9 * e / c
    h_RF = 600
    bunch_spacing_buckets = 10
    n_slices = 100
    n_macroparticles = n_slices  # per bunch
    circumference = 26658.883
    bucket_length = circumference/h_RF
    zeta_range = (-0.5*bucket_length, 0.5*bucket_length)
    dz = (zeta_range[1] - zeta_range[0])/n_slices
    zeta = np.linspace(zeta_range[0] + dz/2, zeta_range[1]-dz/2,
                       n_macroparticles)
    i_source = -10

    source_moment_x = 'y'
    source_moment_y = 'x'

    scale_kick_x = None
    scale_kick_y = None

    displace_x = 2
    displace_y = 3

    particles = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        p0=p0,
        x=np.zeros(n_macroparticles),
        px=np.zeros(n_macroparticles),
        y=np.zeros(n_macroparticles),
        py=np.zeros(n_macroparticles),
        zeta=zeta,
        delta=np.zeros(n_macroparticles),
        weight=np.ones(n_macroparticles),
        _context=test_context
    )

    particles.x[i_source] += displace_x
    particles.y[i_source] += displace_y

    px_bef = particles.px.copy()
    py_bef = particles.py.copy()

    wfx = ResonatorWake(
        r_shunt=2e8,
        q_factor=1e7,
        frequency=1e3,
        source_moments=['num_particles', source_moment_x],
        kick='px',
        scale_kick=scale_kick_x,
    )

    wfy = ResonatorWake(
        r_shunt=3e8,
        q_factor=1e7,
        frequency=1e3,
        source_moments=['num_particles', source_moment_y],
        kick='py',
        scale_kick=scale_kick_y,
    )

    zeta_range_xf = zeta_range

    wf = Wakefield(
        components=[wfx, wfy],
        zeta_range=zeta_range_xf,
        num_slices=n_slices,  # per bunch
        bunch_spacing_zeta=bunch_spacing_buckets*bucket_length,
        num_turns=n_turns_wake,
        circumference=circumference,
        log_moments=['px'],
        _flatten=flatten
    )

    line = xt.Line(elements=[wf],
                   element_names=['wf'])
    line.build_tracker()
    line.track(particles, num_turns=1)

    scale = -particles.q0**2 * e**2 / (particles.p0c * e)

    mask = particles.zeta < particles.zeta[i_source]

    assert np.allclose((particles.px - px_bef)[mask]/displace_y,
                       (wfx.function(-particles.zeta[i_source] +
                                     particles.zeta) * scale[0])[mask],
                       rtol=1e-4, atol=0)
    assert np.allclose((particles.py - py_bef)[mask]/displace_x,
                       (wfy.function(-particles.zeta[i_source] +
                                     particles.zeta) * scale[0])[mask],
                       rtol=1e-4, atol=0)


@for_all_test_contexts(excluding=exclude_contexts)
def test_direct_quadrupolar_wake_kick(test_context):
    n_turns_wake = 1
    flatten = False

    p0 = 7000e9 * e / c
    h_RF = 600
    bunch_spacing_buckets = 10
    n_slices = 100
    n_macroparticles = n_slices  # per bunch
    circumference = 26658.883
    bucket_length = circumference/h_RF
    zeta_range = (-0.5*bucket_length, 0.5*bucket_length)
    dz = (zeta_range[1] - zeta_range[0])/n_slices
    zeta = np.linspace(zeta_range[0] + dz/2, zeta_range[1]-dz/2,
                       n_macroparticles)
    i_source = -10
    i_test = 10

    source_moment_x = 'x'
    source_moment_y = 'y'

    scale_kick_x = 'x'
    scale_kick_y = 'y'

    displace_x_test = 2
    displace_y_test = 3
    displace_x_source = 4
    displace_y_source = 5

    particles = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        p0=p0,
        x=np.zeros(n_macroparticles),
        px=np.zeros(n_macroparticles),
        y=np.zeros(n_macroparticles),
        py=np.zeros(n_macroparticles),
        zeta=zeta,
        delta=np.zeros(n_macroparticles),
        weight=np.ones(n_macroparticles),
        _context=test_context
    )

    particles.x[i_source] += displace_x_source
    particles.y[i_source] += displace_y_source

    particles.x[i_test] += displace_x_test
    particles.y[i_test] += displace_y_test

    wfx = ResonatorWake(
        r_shunt=2e8,
        q_factor=1e7,
        frequency=1e3,
        source_moments=['num_particles', source_moment_x],
        kick='px',
        scale_kick=scale_kick_x,
    )

    wfy = ResonatorWake(
        r_shunt=3e8,
        q_factor=1e7,
        frequency=1e3,
        source_moments=['num_particles', source_moment_y],
        kick='py',
        scale_kick=scale_kick_y,
    )

    zeta_range_xf = zeta_range

    wf = Wakefield(
        components=[wfx, wfy],
        zeta_range=zeta_range_xf,
        num_slices=n_slices,  # per bunch
        bunch_spacing_zeta=bunch_spacing_buckets*bucket_length,
        num_turns=n_turns_wake,
        circumference=circumference,
        log_moments=['px'],
        _flatten=flatten
    )

    line = xt.Line(elements=[wf],
                   element_names=['wf'])
    line.build_tracker()
    line.track(particles, num_turns=1)

    scale = -particles.q0**2 * e**2 / (particles.p0c * e)

    assert np.allclose(particles.px[i_test]/(displace_x_test*displace_x_source),
                       (wfx.function(-particles.zeta[i_source] +
                                     particles.zeta) * scale[0])[i_test],
                       rtol=1e-4, atol=0)
    assert np.allclose(particles.py[i_test]/(displace_y_test*displace_y_source),
                       (wfy.function(-particles.zeta[i_source] +
                                     particles.zeta) * scale[0])[i_test],
                       rtol=1e-4, atol=0)


@for_all_test_contexts(excluding=exclude_contexts)
def test_cross_quadrupolar_wake_kick(test_context):
    n_turns_wake = 1
    flatten = False

    p0 = 7000e9 * e / c
    h_RF = 600
    bunch_spacing_buckets = 10
    n_slices = 100
    n_macroparticles = n_slices  # per bunch
    circumference = 26658.883
    bucket_length = circumference/h_RF
    zeta_range = (-0.5*bucket_length, 0.5*bucket_length)
    dz = (zeta_range[1] - zeta_range[0])/n_slices
    zeta = np.linspace(zeta_range[0] + dz/2, zeta_range[1]-dz/2,
                       n_macroparticles)
    i_source = -10
    i_test = 10

    source_moment_x = 'x'
    source_moment_y = 'y'

    scale_kick_x = 'y'
    scale_kick_y = 'x'

    displace_x_test = 2
    displace_y_test = 3
    displace_x_source = 4
    displace_y_source = 5

    particles = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        p0=p0,
        x=np.zeros(n_macroparticles),
        px=np.zeros(n_macroparticles),
        y=np.zeros(n_macroparticles),
        py=np.zeros(n_macroparticles),
        zeta=zeta,
        delta=np.zeros(n_macroparticles),
        weight=np.ones(n_macroparticles),
        _context=test_context
    )

    particles.x[i_source] += displace_x_source
    particles.y[i_source] += displace_y_source

    particles.x[i_test] += displace_x_test
    particles.y[i_test] += displace_y_test

    wfx = ResonatorWake(
        r_shunt=2e8,
        q_factor=1e7,
        frequency=1e3,
        source_moments=['num_particles', source_moment_x],
        kick='px',
        scale_kick=scale_kick_x,
    )

    wfy = ResonatorWake(
        r_shunt=3e8,
        q_factor=1e7,
        frequency=1e3,
        source_moments=['num_particles', source_moment_y],
        kick='py',
        scale_kick=scale_kick_y,
    )

    zeta_range_xf = zeta_range

    wf = Wakefield(
        components=[wfx, wfy],
        zeta_range=zeta_range_xf,
        num_slices=n_slices,  # per bunch
        bunch_spacing_zeta=bunch_spacing_buckets*bucket_length,
        num_turns=n_turns_wake,
        circumference=circumference,
        log_moments=['px'],
        _flatten=flatten
    )

    line = xt.Line(elements=[wf],
                   element_names=['wf'])
    line.build_tracker()
    line.track(particles, num_turns=1)

    scale = -particles.q0**2 * e**2 / (particles.p0c * e)

    assert np.allclose(particles.px[i_test]/(displace_y_test*displace_x_source),
                       (wfx.function(-particles.zeta[i_source] +
                                     particles.zeta) * scale[0])[i_test],
                       rtol=1e-4, atol=0)
    assert np.allclose(particles.py[i_test]/(displace_x_test*displace_y_source),
                       (wfy.function(-particles.zeta[i_source] +
                                     particles.zeta) * scale[0])[i_test],
                       rtol=1e-4, atol=0)


@for_all_test_contexts(excluding=exclude_contexts)
def test_direct_dipolar_wake_kick_multiturn(test_context):
    n_turns_wake = 2
    flatten = False

    p0 = 7000e9 * e / c
    h_RF = 600
    bunch_spacing_buckets = 10
    n_slices = 100
    n_macroparticles = n_slices  # per bunch
    circumference = 26658.883
    bucket_length = circumference/h_RF
    zeta_range = (-0.5*bucket_length, 0.5*bucket_length)
    dz = (zeta_range[1] - zeta_range[0])/n_slices
    zeta = np.linspace(zeta_range[0] + dz/2, zeta_range[1]-dz/2,
                       n_macroparticles)
    i_source = -10

    source_moment_x = 'x'
    source_moment_y = 'y'

    scale_kick_x = None
    scale_kick_y = None

    displace_x_source = 4
    displace_y_source = 5

    particles = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        p0=p0,
        x=np.zeros(n_macroparticles),
        px=np.zeros(n_macroparticles),
        y=np.zeros(n_macroparticles),
        py=np.zeros(n_macroparticles),
        zeta=zeta,
        delta=np.zeros(n_macroparticles),
        weight=np.ones(n_macroparticles),
        _context=test_context
    )

    particles.x[i_source] += displace_x_source
    particles.y[i_source] += displace_y_source

    px_bef = particles.px.copy()
    py_bef = particles.py.copy()

    wfx = ResonatorWake(
        r_shunt=2e8,
        q_factor=1e7,
        frequency=1e4,
        source_moments=['num_particles', source_moment_x],
        kick='px',
        scale_kick=scale_kick_x,
    )

    wfy = ResonatorWake(
        r_shunt=3e8,
        q_factor=1e7,
        frequency=1e4,
        source_moments=['num_particles', source_moment_y],
        kick='py',
        scale_kick=scale_kick_y
    )

    zeta_range_xf = zeta_range

    wf = Wakefield(
        components=[wfx, wfy],
        zeta_range=zeta_range_xf,
        num_slices=n_slices,  # per bunch
        bunch_spacing_zeta=bunch_spacing_buckets * bucket_length,
        num_turns=n_turns_wake,
        circumference=circumference,
        log_moments=['px'],
        _flatten=flatten
    )

    line = xt.Line(elements=[wf],
                   element_names=['wf'])
    line.build_tracker()
    line.track(particles, num_turns=n_turns_wake)

    scale = -particles.q0 ** 2 * e ** 2 / (particles.p0c * e)

    assert np.allclose((particles.px - px_bef)/displace_x_source,
                       wfx.function(-particles.zeta[i_source] +
                                    particles.zeta - circumference) * scale +
                       wfx.function(-particles.zeta[i_source] +
                                    particles.zeta) * scale +
                       wfx.function(-particles.zeta[i_source] +
                                    particles.zeta) * scale,
                       rtol=1e-4, atol=0)
    assert np.allclose((particles.py - py_bef)/displace_y_source,
                       wfy.function(-particles.zeta[i_source] +
                                    particles.zeta - circumference) * scale +
                       wfy.function(-particles.zeta[i_source] +
                                    particles.zeta) * scale +
                       wfy.function(-particles.zeta[i_source] +
                                    particles.zeta) * scale,
                       rtol=1e-4, atol=0)

