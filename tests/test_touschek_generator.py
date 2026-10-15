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
