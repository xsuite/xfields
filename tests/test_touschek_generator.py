# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #
from pathlib import Path

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
