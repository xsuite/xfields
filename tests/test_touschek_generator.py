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
