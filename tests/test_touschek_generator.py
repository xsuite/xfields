# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #
from pathlib import Path
import warnings

import numpy as np
import pytest

import xtrack as xt
import xfields as xf


FOUR_CELL_RING = (
    Path(__file__).parent.parent / 'test_data' / 'four_cell_ring' / 'line.json')


def _ring():
    """Four-cell electron ring of the Touschek examples, without markers."""
    return xt.load(FOUR_CELL_RING)


LMA_SCALE = 0.85
BEAM = dict(nemitt_x=1e-5, nemitt_y=1e-7, sigma_z=4e-3, sigma_delta=1e-3,
            bunch_intensity=4e9)


def _ring_with_markers():
    """The ring with a marker at the centre of each magnet, ring closed."""
    line = _ring()
    tab = line.get_table()
    magnets = tab.rows[(tab.element_type == 'Bend')
                       | (tab.element_type == 'Quadrupole')].name
    line.xfields.touschek.install_markers(at=list(magnets))
    return line


def _lma(line):
    """A fixed momentum acceptance at the markers, different at each one."""
    names = line.xfields.touschek.marker_names
    tab = line.get_table()
    delta = 0.010 + 0.0005 * np.arange(len(names))
    return xt.Table({
        'name': np.array(names),
        's': np.array([tab['s', nn] for nn in names]),
        'delta_neg': -delta,
        'delta_pos': delta,
    })


@pytest.fixture(scope='module')
def ring():
    """
    The ring with markers and its 4d twiss. Building a tracker compiles its
    kernels, for each new line: the other tests configure fresh rings with
    the same layout using this twiss, and track in this line.
    """
    line = _ring_with_markers()
    return line, line.twiss(method='4d')


def _configure(line, **kwargs):
    config = dict(
        local_momentum_acceptance=_lma(line),
        local_momentum_acceptance_scale=LMA_SCALE,
        n_scattering_events=20_000,
        weight_retention_fraction=0.99,
        seed=1997,
        method='4d',
        verbose=False,
        **BEAM)
    config.update(kwargs)
    line.xfields.touschek.configure(**config)


def _place_markers(line, positions):
    """Install markers by hand, as in the examples: env.place + insert."""
    env = line.env
    places = []
    for ii, at in enumerate(positions):
        env.elements[f'ts.{ii}'] = xf.TouschekMarker()
        places.append(env.place(f'ts.{ii}', at=at))
    line.insert(places)
    return [f'ts.{ii}' for ii in range(len(positions))]


def test_facade_returns_a_generator_bound_to_the_line():
    line = _ring()
    tsc = line.xfields.touschek
    assert isinstance(tsc, xf.TouschekGenerator)
    assert tsc.line is line


def test_generator_needs_a_line():
    with pytest.raises(TypeError, match='xtrack.Line'):
        xf.TouschekGenerator(None)


def test_marker_names_are_in_line_order():
    line = _ring()
    assert line.xfields.touschek.marker_names == []

    length = line.get_length()
    names = _place_markers(line, [length, 3.0, 0.5 * length])
    assert line.xfields.touschek.marker_names == [
        names[1], names[2], names[0]]


def test_repeated_marker_is_refused():
    line = _ring()
    env = line.env
    env.elements['ts'] = xf.TouschekMarker()
    line.insert([env.place('ts', at=1.0), env.place('ts', at=2.0)])
    with pytest.raises(ValueError, match='more than once'):
        line.xfields.touschek.marker_names


# The same positions given to env.place directly and to install_markers
POSITIONS = {
    'number':         lambda env: dict(at=3.0),
    'element':        lambda env: dict(at='mb1.1'),
    'start':          lambda env: dict(at='mqf.2@start'),
    'end':            lambda env: dict(at='mqd.1@end'),
    'from':           lambda env: dict(at=0.5, from_='mb2.1'),
    'from_anchor':    lambda env: dict(at=-0.1, from_='mb2.1',
                                       from_anchor='end'),
    'from_at_anchor': lambda env: dict(at=0.2, from_='mb1.2@start'),
    'expression':     lambda env: dict(at='2 * s_ts'),
    'reference':      lambda env: dict(at=3 * env.ref['s_ts']),
}


@pytest.mark.parametrize('position', POSITIONS.values(), ids=POSITIONS.keys())
def test_install_markers_places_like_env_place(position):
    manual = _ring()
    env = manual.env
    env['s_ts'] = 2.6
    env.elements['ts'] = xf.TouschekMarker()
    manual.insert([env.place('ts', **position(env))])

    line = _ring()
    line.env['s_ts'] = 2.6
    names = line.xfields.touschek.install_markers(
        close_ring=False, **position(line.env))

    assert names == ['tscatter.0']
    assert isinstance(line['tscatter.0'], xf.TouschekMarker)
    tab, tab_manual = line.get_table(), manual.get_table()
    assert tab['s', 'tscatter.0'] == tab_manual['s', 'ts']
    # The rest of the line is cut in the same way
    np.testing.assert_array_equal(tab.s, tab_manual.s)
    np.testing.assert_array_equal(tab.element_type, tab_manual.element_type)


def test_install_markers_closes_the_ring():
    line = _ring()
    tsc = line.xfields.touschek

    names = tsc.install_markers(at=[1.0, 5.0])
    assert names == ['tscatter.0', 'tscatter.1', 'tscatter.2']
    assert tsc.marker_names == names
    assert line.get_table()['s', 'tscatter.2'] == line.get_length()

    # The ring is already closed: no other marker at the end
    assert tsc.install_markers(at=8.0) == ['tscatter.3']
    assert tsc.marker_names[-1] == 'tscatter.2'


def test_install_markers_with_a_marker_at_the_end():
    line = _ring()
    names = line.xfields.touschek.install_markers(at=[1.0, line.get_length()])
    assert names == ['tscatter.0', 'tscatter.1']


def test_install_markers_without_closing_the_ring():
    line = _ring()
    names = line.xfields.touschek.install_markers(
        at=[1.0, 5.0], close_ring=False)
    assert names == ['tscatter.0', 'tscatter.1']
    assert line.get_table()['s', 'tscatter.1'] < line.get_length()


def test_install_markers_names():
    line = _ring()
    tsc = line.xfields.touschek
    line.env.elements['tscatter.1'] = xt.Marker()

    assert tsc.install_markers(
        at=[1.0, 2.0], close_ring=False) == ['tscatter.0', 'tscatter.2']
    assert tsc.install_markers(
        at=3.0, names='my_ts', close_ring=False) == ['my_ts']
    assert tsc.install_markers(
        at=4.0, prefix='ts', close_ring=False) == ['ts.0']
    with pytest.raises(ValueError, match='already used'):
        tsc.install_markers(at=5.0, names='my_ts')
    with pytest.raises(ValueError, match='repeated'):
        tsc.install_markers(at=[5.0, 6.0], names=['a', 'a'])


def test_install_markers_checks_its_arguments():
    line = _ring()
    tsc = line.xfields.touschek
    with pytest.raises(ValueError, match='`from_` has 1 entries'):
        tsc.install_markers(at=[1.0, 2.0], from_=['mb1.1'])
    with pytest.raises(ValueError, match='`names` has 1 entries'):
        tsc.install_markers(at=[1.0, 2.0], names=['a'])
    with pytest.raises(ValueError, match='at least one'):
        tsc.install_markers(at=[])
    assert tsc.marker_names == []


def test_configure_stores_the_configuration_on_the_markers(ring):
    _, tw = ring
    line = _ring_with_markers()
    lma = _lma(line)
    _configure(line, twiss=tw, local_momentum_acceptance=lma, rng='xtrack')

    ref = line.particle_ref
    gemitt_x = BEAM['nemitt_x'] / (ref.beta0[0] * ref.gamma0[0])
    for ii, nn in enumerate(line.xfields.touschek.marker_names):
        marker = line[nn]
        assert marker.p0c == ref.p0c[0]
        for name, col in [('betx', 'betx'), ('alfy', 'alfy'), ('dx', 'dx'),
                          ('dpx', 'dpx'), ('x_co', 'x'), ('delta_co', 'delta')]:
            assert getattr(marker, name) == pytest.approx(
                tw[col, nn], rel=1e-12, abs=1e-15)
        assert marker.delta_neg == pytest.approx(LMA_SCALE * lma.delta_neg[ii])
        assert marker.delta_pos == pytest.approx(LMA_SCALE * lma.delta_pos[ii])
        assert marker.gemitt_x == pytest.approx(gemitt_x)
        assert marker.bunch_intensity == BEAM['bunch_intensity']
        assert marker.sigma_z == BEAM['sigma_z']
        assert marker.n_scattering_events == 20_000
        assert marker.weight_retention_fraction == 0.99
        assert marker.rng == 'xtrack'
        assert marker.seed == 1997
        assert marker.nz <= 3
        assert marker.piwinski_rate > 0
        assert marker.integrated_piwinski_rate > 0


def test_configure_does_not_modify_the_momentum_acceptance(ring):
    line = _ring_with_markers()
    lma = _lma(line)
    delta_neg, delta_pos = lma.delta_neg.copy(), lma.delta_pos.copy()
    _configure(line, twiss=ring[1], local_momentum_acceptance=lma)
    np.testing.assert_array_equal(lma.delta_neg, delta_neg)
    np.testing.assert_array_equal(lma.delta_pos, delta_pos)


def test_configure_stores_a_drawn_seed(ring):
    line = _ring_with_markers()
    _configure(line, twiss=ring[1], seed=None)
    seeds = {line[nn].seed for nn in line.xfields.touschek.marker_names}
    assert len(seeds) == 1
    assert seeds.pop() > 0

    with pytest.raises(ValueError, match='non-negative'):
        _configure(line, seed=-1)
    with pytest.raises(TypeError, match='integer'):
        _configure(line, seed=1.5)


def test_configure_a_subset_of_the_markers(ring):
    line, tw = _ring_with_markers(), ring[1]
    _configure(line, twiss=tw)
    names = line.xfields.touschek.marker_names
    rates = {nn: line[nn].integrated_piwinski_rate for nn in names}

    # One section integrated on its own gives the same rate
    for nn in names:
        line[nn].integrated_piwinski_rate = 0
        _configure(line, twiss=tw, at=nn)
        assert line[nn].integrated_piwinski_rate == pytest.approx(
            rates[nn], rel=1e-12)

    # Only the selected markers are configured: the rate goes as N^2
    _configure(line, twiss=tw, at=names[2:4],
               bunch_intensity=2 * BEAM['bunch_intensity'])
    for nn in names:
        factor = 4 if nn in names[2:4] else 1
        assert line[nn].integrated_piwinski_rate == pytest.approx(
            factor * rates[nn], rel=1e-12)


def test_configure_computes_the_twiss(ring):
    line, tw = ring
    _configure(line)
    computed = {nn: line[nn].to_dict()
                for nn in line.xfields.touschek.marker_names}
    _configure(line, twiss=tw)
    for nn, dct in computed.items():
        assert line[nn].to_dict() == dct


def test_configure_requires_the_last_marker_at_the_end_of_the_line():
    line = _ring()
    line.xfields.touschek.install_markers(at=[1.0, 5.0], close_ring=False)
    with pytest.raises(ValueError, match='must sit at the end of the line'):
        _configure(line, local_momentum_acceptance=xt.Table({
            'name': np.array(['a']), 's': np.array([0.0]),
            'delta_neg': np.array([-0.01]), 'delta_pos': np.array([0.01])}))


def test_configure_checks_its_arguments(ring):
    line = ring[0]
    with pytest.raises(ValueError, match='`sigma_z` is required'):
        _configure(line, sigma_z=None)
    with pytest.raises(ValueError, match='not both'):
        _configure(line, gemitt_x=1e-9, gemitt_y=1e-11)
    with pytest.raises(TypeError, match=r'xt\.Table'):
        _configure(line, local_momentum_acceptance={'s': [0.0]})
    with pytest.raises(ValueError, match="'mb1.1' is not present"):
        _configure(line, at='mb1.1')
    with pytest.raises(TypeError, match='not a TouschekMarker'):
        _configure(line, at='mqf.1_entry')
    with pytest.raises(ValueError, match='rng'):
        _configure(line, rng='other')
    for at in (42, [42]):
        with pytest.raises(TypeError, match='selected by name'):
            _configure(line, at=at)

    no_markers = _ring()
    with pytest.raises(ValueError, match='install_markers'):
        _configure(no_markers, local_momentum_acceptance=_lma(line))


def test_stream_seeds_are_keyed_by_marker_and_replica():
    from xfields.touschek.generator import _stream_seed

    names = [f'tscatter.{ii}' for ii in range(500)]
    seeds = {(nn, rr): _stream_seed(1997, nn, rr)
             for nn in names for rr in range(4)}
    assert len(set(seeds.values())) == len(seeds)
    assert all(0 <= ss < 2**31 for ss in seeds.values())
    # Reproducible, and changed by the configured seed
    assert _stream_seed(1997, 'tscatter.3', 1) == seeds['tscatter.3', 1]
    assert _stream_seed(1998, 'tscatter.3', 1) != seeds['tscatter.3', 1]


COORDINATES = ('x', 'px', 'y', 'py', 'zeta', 'delta', 'weight', 's',
               'at_element', 'particle_id')


def _assert_same_particles(part, other):
    alive, alive_other = part.state > 0, other.state > 0
    assert np.sum(alive) == np.sum(alive_other) > 0
    for nn in COORDINATES:
        np.testing.assert_array_equal(
            getattr(part, nn)[alive], getattr(other, nn)[alive_other],
            err_msg=f'{nn} differs')


def test_generate_at_one_or_several_markers(ring):
    line = _ring_with_markers()
    _configure(line, twiss=ring[1])
    tsc = line.xfields.touschek
    names = tsc.marker_names

    assert isinstance(tsc.generate(at=names[1]), xt.Particles)
    assert list(tsc.generate(at=[names[3], names[1]])) == [names[1], names[3]]
    assert list(tsc.generate()) == names


def test_generated_particles(ring):
    line = _ring_with_markers()
    lma = _lma(line)
    _configure(line, twiss=ring[1], local_momentum_acceptance=lma)
    tsc = line.xfields.touschek
    ref = line.particle_ref

    tab = line.get_table()
    for ii, nn in enumerate(tsc.marker_names):
        part = tsc.generate(at=nn)
        alive = part.state > 0
        assert np.all(part.s[alive] == tab['s', nn])
        assert np.all(part.at_element[alive]
                      == line.element_names.index(nn))
        assert part.q0 == ref.q0 and part.mass0 == ref.mass0
        assert np.all(part.pdg_id[alive] == ref.pdg_id[0])
        delta = part.delta[alive]
        assert np.all((delta < LMA_SCALE * lma.delta_neg[ii])
                      | (delta > LMA_SCALE * lma.delta_pos[ii]))
        # The weights carry the retained rate of the section
        assert np.sum(part.weight[alive]) == pytest.approx(
            0.99 * line[nn].integrated_piwinski_rate, rel=2e-3)


def test_generation_does_not_depend_on_the_other_markers(ring):
    line = _ring_with_markers()
    _configure(line, twiss=ring[1])
    tsc = line.xfields.touschek
    names = tsc.marker_names

    single_job = tsc.generate()
    for nn in reversed(names):
        _assert_same_particles(tsc.generate(at=nn), single_job[nn])
    for nn, part in tsc.generate(at=names[::-2]).items():
        _assert_same_particles(part, single_job[nn])


def test_replicas(ring):
    line = _ring_with_markers()
    _configure(line, twiss=ring[1])
    tsc = line.xfields.touschek
    nn = tsc.marker_names[2]

    one = tsc.generate(at=nn)
    first = tsc.generate(at=nn, replica=0, n_replicas=3)
    replicas = [first] + [tsc.generate(at=nn, replica=rr, n_replicas=3)
                          for rr in (1, 2)]

    # The same stream, weights divided by the number of replicas
    alive = one.state > 0
    np.testing.assert_array_equal(first.x[first.state > 0], one.x[alive])
    np.testing.assert_array_equal(
        first.weight[first.state > 0], one.weight[alive] * (1 / 3))
    # Independent samples, which together carry the rate once
    assert not np.array_equal(replicas[1].x[:10], replicas[2].x[:10])
    total = sum(np.sum(pp.weight[pp.state > 0]) for pp in replicas)
    assert total == pytest.approx(
        0.99 * line[nn].integrated_piwinski_rate, rel=2e-3)


@pytest.mark.parametrize('rng', ['elegant', 'xtrack'])
def test_generation_from_a_reloaded_line(ring, tmp_path, rng):
    line = _ring_with_markers()
    _configure(line, twiss=ring[1], rng=rng)
    names = line.xfields.touschek.marker_names
    expected = line.xfields.touschek.generate(at=names[1:4])

    line.to_json(tmp_path / 'line.json')
    line.env['ring'] = line
    line.env.to_json(tmp_path / 'env.json')
    for loaded in (xt.load(tmp_path / 'line.json'),
                   xt.load(tmp_path / 'env.json').lines['ring']):
        for nn in names:
            assert loaded[nn].to_dict() == line[nn].to_dict()
        reloaded = loaded.xfields.touschek.generate(at=names[1:4])
        for nn, part in expected.items():
            _assert_same_particles(reloaded[nn], part)


@pytest.mark.parametrize('rng', ['elegant', 'xtrack'])
def test_continuous_stream_runs_through_the_markers(ring, rng):
    """
    With rng_stream='continuous', one stream seeded with the configured seed
    runs through the markers in line order, as in ELEGANT (whose sequence the
    ELEGANT reference test checks): the particles of a marker depend on the
    markers generated before it.
    """
    line = _ring_with_markers()
    _configure(line, twiss=ring[1], rng=rng)
    tsc = line.xfields.touschek
    first, second = tsc.marker_names[1:3]

    both = tsc.generate(at=[second, first], rng_stream='continuous')
    _assert_same_particles(
        both[first], tsc.generate(at=first, rng_stream='continuous'))
    alone = tsc.generate(at=second, rng_stream='continuous')
    assert not np.array_equal(both[second].x[:10], alone.x[:10])

    # Not the streams of the markers
    assert not np.array_equal(both[first].x[:10],
                              tsc.generate(at=first).x[:10])


def test_generate_checks_its_arguments(ring):
    line = _ring_with_markers()
    tsc = line.xfields.touschek
    nn = tsc.marker_names[1]
    with pytest.raises(ValueError, match='not configured'):
        tsc.generate(at=nn)

    _configure(line, twiss=ring[1])
    with pytest.raises(ValueError, match='replica'):
        tsc.generate(at=nn, replica=2, n_replicas=2)
    with pytest.raises(ValueError, match='n_replicas'):
        tsc.generate(at=nn, n_replicas=0)
    with pytest.raises(ValueError, match='rng_stream'):
        tsc.generate(at=nn, rng_stream='other')
    with pytest.raises(ValueError, match='n_replicas=1'):
        tsc.generate(rng_stream='continuous', replica=1, n_replicas=2)

    line.set_particle_ref('electron', p0c=2e9)
    with pytest.raises(ValueError, match='configure the markers again'):
        tsc.generate(at=nn)


def _fake_track(part, seed):
    """
    Stand-in for tracking: half of the particles are lost, on an aperture
    (state 0) or in a collimator (xcoll LOST_ON_MATERIAL = -330), and the
    first lost one produces a lost secondary, which carries its weight.
    """
    rng = np.random.default_rng(seed)
    alive = np.flatnonzero(part.state > 0)
    lost = alive[rng.random(len(alive)) < 0.5]
    part.state[lost] = rng.choice([0, -330], size=len(lost))
    part.at_turn[alive] = 10

    free = np.flatnonzero(part.state == xt.particles.LAST_INVALID_STATE)[0]
    part.state[free] = 0
    part.particle_id[free] = np.max(part.particle_id) + 1
    part.parent_particle_id[free] = part.particle_id[lost[0]]
    part.weight[free] = part.weight[lost[0]]


def _lost_weight(part):
    primary = part.parent_particle_id == part.particle_id
    lost = (part.state <= 0) & (part.state > xt.particles.LAST_INVALID_STATE)
    return np.sum(part.weight[lost & primary])


def test_analyse_without_particles(ring):
    line = _ring_with_markers()
    _configure(line, twiss=ring[1])
    tsc = line.xfields.touschek
    result = tsc.analyse()

    names = tsc.marker_names
    rates = [line[nn].integrated_piwinski_rate for nn in names]
    assert result.element_names == names
    assert result.rate_scattering == pytest.approx(np.sum(rates), rel=1e-12)
    assert result.lifetime_scattering == pytest.approx(
        BEAM['bunch_intensity'] / np.sum(rates), rel=1e-12)
    assert result.tracked is False
    assert result.rate_generated is None
    assert result.rate_tracking is None
    assert result.lifetime_tracking is None
    assert set(result.local_rates._col_names) == {
        'name', 's', 'delta_neg', 'delta_pos', 'piwinski_rate',
        'integrated_piwinski_rate'}
    np.testing.assert_array_equal(
        result.local_rates.integrated_piwinski_rate, rates)
    np.testing.assert_array_equal(
        result.local_momentum_acceptance.delta_neg,
        [line[nn].delta_neg for nn in names])


def test_analyse_merged_particles_and_samples_by_marker(ring):
    line = _ring_with_markers()
    _configure(line, twiss=ring[1])
    tsc = line.xfields.touschek
    samples = tsc.generate()

    # Not tracked yet: only the generated weight
    untracked = tsc.analyse(xt.Particles.merge(list(samples.values())))
    assert untracked.tracked is False
    assert untracked.rate_tracking is None
    assert untracked.rate_generated == pytest.approx(
        0.99 * untracked.rate_scattering, rel=2e-3)

    for ii, part in enumerate(samples.values()):
        _fake_track(part, seed=ii)
    expected = sum(_lost_weight(pp) for pp in samples.values())

    by_marker = tsc.analyse(samples)
    shuffled = list(samples.values())[::-1]
    merged = tsc.analyse(xt.Particles.merge(shuffled))
    singles = [tsc.analyse(pp, at=nn) for nn, pp in samples.items()]

    for result in (by_marker, merged):
        assert result.tracked is True
        assert result.rate_tracking == pytest.approx(expected, rel=1e-12)
        assert result.lifetime_tracking == pytest.approx(
            BEAM['bunch_intensity'] / expected, rel=1e-12)
        assert result.rate_generated == pytest.approx(
            untracked.rate_generated, rel=1e-12)
        assert result.rate_tracking_error is None
    assert sum(rr.rate_tracking for rr in singles) == pytest.approx(
        expected, rel=1e-12)

    # Per-marker counts: from the samples of each marker only
    assert 'sum_lost_weight' not in merged.local_rates._col_names
    for single, nn in zip(singles, samples):
        for col in ('num_particles', 'sum_weight', 'num_lost_particles',
                    'sum_lost_weight'):
            assert single.local_rates[col, nn] == by_marker.local_rates[col, nn]
        assert by_marker.local_rates['sum_lost_weight', nn] == pytest.approx(
            _lost_weight(samples[nn]), rel=1e-12)


def test_analyse_replicas(ring):
    line = _ring_with_markers()
    _configure(line, twiss=ring[1])
    tsc = line.xfields.touschek
    n_replicas = 3
    replicas = {nn: [] for nn in tsc.marker_names}
    for rr in range(n_replicas):
        for nn, part in tsc.generate(replica=rr, n_replicas=n_replicas).items():
            _fake_track(part, seed=rr)
            replicas[nn].append(part)

    result = tsc.analyse(replicas)
    merged = tsc.analyse(xt.Particles.merge(
        [pp for vv in replicas.values() for pp in vv]))

    assert result.rate_tracking == pytest.approx(merged.rate_tracking, rel=1e-12)
    assert result.rate_generated == pytest.approx(
        0.99 * result.rate_scattering, rel=2e-3)
    np.testing.assert_array_equal(result.local_rates.n_replicas, n_replicas)

    # Standard error of the mean of the estimates of the replicas
    variance = 0
    for nn, parts in replicas.items():
        estimates = [n_replicas * _lost_weight(pp) for pp in parts]
        error = np.std(estimates, ddof=1) / np.sqrt(n_replicas)
        assert result.local_rates['sum_lost_weight_error', nn] == (
            pytest.approx(error, rel=1e-9))
        variance += error**2
    assert result.rate_tracking_error == pytest.approx(
        np.sqrt(variance), rel=1e-9)
    assert result.lifetime_tracking_error == pytest.approx(
        result.lifetime_tracking * result.rate_tracking_error
        / result.rate_tracking, rel=1e-12)


def test_analyse_warns_about_missing_samples(ring):
    line = _ring_with_markers()
    _configure(line, twiss=ring[1])
    tsc = line.xfields.touschek
    samples = tsc.generate()
    names = list(samples)
    some = [samples[nn] for nn in names[2:]]

    with pytest.warns(UserWarning, match='missing or repeated'):
        tsc.analyse(xt.Particles.merge(some))
    with pytest.warns(UserWarning, match='missing or repeated'):
        tsc.analyse({nn: [samples[nn]] * 2 for nn in names})

    # Analyses of a part of the markers are not checked
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        tsc.analyse(xt.Particles.merge(some), at=names[2:])
        tsc.analyse({nn: samples[nn] for nn in names[2:]})


def test_analyse_checks_its_arguments(ring):
    line = _ring_with_markers()
    tsc = line.xfields.touschek
    nn = tsc.marker_names[1]
    with pytest.raises(ValueError, match='not configured'):
        tsc.analyse()

    _configure(line, twiss=ring[1])
    part = tsc.generate(at=nn)
    with pytest.raises(ValueError, match='keys select the markers'):
        tsc.analyse({nn: part}, at=nn)
    with pytest.raises(TypeError, match='xtrack.Particles'):
        tsc.analyse({nn: [part, 'other']})
    with pytest.raises(ValueError, match='No particles'):
        tsc.analyse({nn: []})


def test_run_tracks_each_sample_from_its_marker(ring, monkeypatch):
    line = _ring_with_markers()
    _configure(line, twiss=ring[1])
    tsc = line.xfields.touschek
    names = tsc.marker_names

    calls = []

    def fake_track(particles, **kwargs):
        calls.append(kwargs)
        _fake_track(particles, seed=len(calls))

    monkeypatch.setattr(line, 'track', fake_track)
    result = tsc.run(n_turns=7, at=names[1:4], n_replicas=2,
                     keep_particles=True)

    assert [cc['ele_start'] for cc in calls] == 2 * names[1:4]
    assert all(cc['ele_stop'] == cc['ele_start'] and cc['num_turns'] == 7
               for cc in calls)

    # The explicit steps give the same result
    samples = {nn: [] for nn in names[1:4]}
    for rr in range(2):
        for nn in names[1:4]:
            part = tsc.generate(at=nn, replica=rr, n_replicas=2)
            _fake_track(part, seed=3 * rr + names[1:4].index(nn) + 1)
            samples[nn].append(part)
    explicit = tsc.analyse(samples)
    assert result.element_names == explicit.element_names
    assert result.rate_tracking == pytest.approx(explicit.rate_tracking,
                                                 rel=1e-12)
    assert result.rate_tracking_error == pytest.approx(
        explicit.rate_tracking_error, rel=1e-12)

    assert list(result.particles_by_element) == names[1:4]
    assert np.sum(result.lost_particles.weight[
        result.lost_particles.parent_particle_id
        == result.lost_particles.particle_id]) == pytest.approx(
            result.rate_tracking, rel=1e-12)
    assert np.all(result.lost_particles.state <= 0)


def test_run_tracks_the_particles(ring):
    line, tw = ring
    _configure(line, twiss=tw)
    result = line.xfields.touschek.run(n_turns=3, keep_particles=True)

    assert result.tracked is True
    # No apertures in this ring: nothing is lost
    assert result.rate_tracking == 0
    assert result.lifetime_tracking == np.inf
    assert result.lost_particles is None
    particles = result.particles
    assert np.all(particles.at_turn[particles.state > 0] == 3)
    for nn, part in result.particles_by_element.items():
        alive = part.state > 0
        assert np.all(part.at_element[alive] == line.element_names.index(nn))


def test_run_checks_its_arguments(ring):
    with pytest.raises(ValueError, match='n_turns'):
        ring[0].xfields.touschek.run(n_turns=0)


def test_generate_explains_an_empty_sample(ring):
    line = _ring_with_markers()
    _configure(line, twiss=ring[1], n_scattering_events=2000)
    tsc = line.xfields.touschek
    # With this seed, one event dominates the small sample of this marker
    with pytest.raises(ValueError, match="'tscatter.2'.*single scattering"):
        tsc.generate(at='tscatter.2')

    _configure(line, twiss=ring[1], n_scattering_events=2000,
               weight_retention_fraction=1.0)
    part = tsc.generate(at='tscatter.2')
    assert np.sum(part.state > 0) == 2000
