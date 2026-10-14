# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #
import csv
import json
from pathlib import Path

import numpy as np
import pytest

import xfields as xf
import xtrack as xt


REFERENCE_DIR = (
    Path(__file__).parent.parent
    / 'test_data' / 'touschek' / 'elegant_reference'
)


def _build_reference_study(reference_rows):
    env = xt.Environment()
    line = env.new_line(components=[
        env.new('qf1', xt.Quadrupole, length=0.3, k1=0.1),
        env.new('d11', xt.Drift, length=1.0),
        env.new('b1', xt.Bend, length=3.0, angle=np.pi / 2),
        env.new('d21', xt.Drift, length=1.0),
        env.new('qd1', xt.Quadrupole, length=0.3, k1=-0.7),
        env.new('d31', xt.Drift, length=1.0),
        env.new('b2', xt.Bend, length=3.0, angle=np.pi / 2),
        env.new('d41', xt.Drift, length=1.0),
        env.new('qf2', xt.Quadrupole, length=0.3, k1=0.1),
        env.new('d12', xt.Drift, length=1.0),
        env.new('b3', xt.Bend, length=3.0, angle=np.pi / 2),
        env.new('d22', xt.Drift, length=1.0),
        env.new('qd2', xt.Quadrupole, length=0.3, k1=-0.7),
        env.new('d32', xt.Drift, length=1.0),
        env.new('b4', xt.Bend, length=3.0, angle=np.pi / 2),
        env.new('d42', xt.Drift, length=1.0),
    ])
    line.set_particle_ref('electron', p0c=1e9)
    line.configure_bend_model(core='full', edge=None)

    placements = []
    for index, row in enumerate(reference_rows):
        env.elements[f'TS{index}'] = xf.TouschekScattering()
        placements.append(env.place(f'TS{index}', at=float(row['s_m'])))
    line.insert(placements)

    twiss = line.twiss(method='4d')
    table = line.get_table()
    names = table.rows.match(element_type='TouschekScattering').name
    xfields_s = np.array([table['s', name] for name in names])
    elegant_s = np.array([float(row['s_m']) for row in reference_rows])
    np.testing.assert_allclose(xfields_s, elegant_s, atol=1e-12)

    lma = xt.Table({
        'name': names,
        's': xfields_s,
        'delta_neg': np.full(len(names), -0.012),
        'delta_pos': np.full(len(names), 0.012),
    })
    touschek = line.xfields.touschek_configure(
        twiss=twiss,
        local_momentum_acceptance=lma,
        local_momentum_acceptance_scale=0.85,
        nemitt_x=1e-5,
        nemitt_y=1e-7,
        sigma_z=4e-3,
        sigma_delta=1e-3,
        bunch_intensity=4e9,
        n_scattering_events=100_000,
        nx=3,
        ny=3,
        nz=3,
        weight_retention_fraction=0.99,
        seed=1997,
        method='4d',
    )
    return line, names, touschek


@pytest.fixture(scope='module')
def elegant_comparison():
    with (REFERENCE_DIR / 'reference.csv').open(newline='') as stream:
        reference_rows = list(csv.DictReader(stream))
    with (REFERENCE_DIR / 'reference_summary.json').open() as stream:
        reference_summary = json.load(stream)
    line, names, touschek = _build_reference_study(reference_rows)
    return line, names, touschek, reference_rows, reference_summary


def test_local_piwinski_rates_against_elegant(elegant_comparison):
    line, names, _, reference_rows, _ = elegant_comparison
    xfields_rates = np.array([line[name].piwinski_rate for name in names])
    elegant_rates = np.array([
        float(row['piwinski_local_rate_hz']) for row in reference_rows
    ])

    # ELEGANT records zero at the first, zero-length TSCATTER section. The
    # remaining local values agree to a few ppm for this fixed-aperture case.
    np.testing.assert_allclose(
        xfields_rates[1:], elegant_rates[1:], rtol=1e-5)


def test_total_rate_and_lifetime_against_elegant(elegant_comparison):
    line, names, _, _, reference_summary = elegant_comparison
    total_rate = sum(line[name].integrated_piwinski_rate for name in names)
    lifetime = 4e9 / total_rate

    assert total_rate == pytest.approx(
        reference_summary['total_piwinski_rate_hz'], rel=1e-5)
    assert lifetime == pytest.approx(
        reference_summary['touschek_lifetime_s'], rel=1e-5)


@pytest.fixture(scope='module')
def scattered_distribution(elegant_comparison):
    _, names, touschek, _, _ = elegant_comparison
    # TS0 represents a zero-length section and has zero particle weight.
    result = touschek.run(track=False, keep_particles=True)
    particles = [result.particles_by_element[name] for name in names[1:]]
    with np.load(
            REFERENCE_DIR / 'scattered_distribution_reference.npz') as data:
        reference = {name: data[name].copy() for name in data.files}
    return particles, reference


def _xfields_scatter_summary(particles, quantity, bin_edges):
    attribute = {'xp': 'kin_xp', 'yp': 'kin_yp'}.get(quantity, quantity)
    values = []
    weights = []
    for part in particles:
        alive = part.state == 1
        values.append(np.asarray(getattr(part, attribute)[alive]))
        weights.append(np.asarray(part.weight[alive]))
    values = np.concatenate(values)
    weights = np.concatenate(weights)
    histogram = np.histogram(values, bins=bin_edges, weights=weights)[0]
    histogram /= histogram.sum()
    mean = np.average(values, weights=weights)
    standard_deviation = np.sqrt(
        np.average((values - mean)**2, weights=weights))
    return histogram, np.array([
        weights.sum(), mean, standard_deviation])


def _aggregate_elegant_summary(reference, quantity):
    element_moments = reference[f'{quantity}_moments']
    weights = element_moments[:, 0]
    mean = np.average(element_moments[:, 1], weights=weights)
    variance = np.average(
        element_moments[:, 2]**2 + (element_moments[:, 1] - mean)**2,
        weights=weights)
    histogram = np.average(
        reference[f'{quantity}_histogram'], axis=0, weights=weights)
    return histogram, np.array([
        weights.sum(), mean, np.sqrt(variance)])


def test_scattered_delta_distribution_against_elegant(
        scattered_distribution):
    particles, reference = scattered_distribution
    xfields_histogram, _ = _xfields_scatter_summary(
        particles, 'delta', reference['delta_bin_edges'])
    elegant_histogram, _ = _aggregate_elegant_summary(reference, 'delta')

    # Weighted samples contain only a few thousand retained particles. The L1
    # distance is stable while avoiding fragile, sparsely populated bin tests.
    assert np.sum(np.abs(xfields_histogram - elegant_histogram)) < 0.02


@pytest.mark.parametrize(
    'quantity, mean_sigma_tol, std_rtol',
    [
        ('x', 0.02, 0.02),
        ('xp', 0.02, 0.02),
        ('y', 0.02, 0.02),
        ('yp', 0.02, 0.02),
        ('delta', 0.02, 0.02),
    ])
def test_scattered_weighted_moments_against_elegant(
        scattered_distribution, quantity, mean_sigma_tol, std_rtol):
    particles, reference = scattered_distribution
    _, xfields_moments = _xfields_scatter_summary(
        particles, quantity, reference[f'{quantity}_bin_edges'])
    _, elegant_moments = _aggregate_elegant_summary(reference, quantity)

    assert xfields_moments[0] == pytest.approx(
        elegant_moments[0], rel=2e-3)
    normalized_mean_difference = (
        xfields_moments[1] - elegant_moments[1]) / elegant_moments[2]
    assert abs(normalized_mean_difference) < mean_sigma_tol
    assert xfields_moments[2] == pytest.approx(
        elegant_moments[2], rel=std_rtol)
