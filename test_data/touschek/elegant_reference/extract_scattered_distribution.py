#!/usr/bin/env python3
"""Freeze compact summaries of ELEGANT's scattered-particle SDDS output."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np


QUANTITIES = {
    'x': np.linspace(-7e-3, 7e-3, 57),
    'xp': np.linspace(-6e-4, 6e-4, 61),
    'y': np.linspace(-8e-5, 8e-5, 65),
    'yp': np.linspace(-8e-5, 8e-5, 65),
    'delta': np.linspace(-0.15, 0.15, 61),
}


def _stream(executable, source, page, option):
    return subprocess.run(
        [str(executable), str(source), f'-page={page}', option],
        check=True, text=True, capture_output=True).stdout


def _moments(values, weights):
    weight_sum = weights.sum()
    mean = np.sum(weights * values) / weight_sum
    variance = np.sum(weights * (values - mean)**2) / weight_sum
    return weight_sum, mean, np.sqrt(variance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdds2stream', required=True, type=Path)
    parser.add_argument('--input', default='toy_ring.scatter.sdds', type=Path)
    args = parser.parse_args()
    case_dir = Path(__file__).resolve().parent
    source = args.input if args.input.is_absolute() else case_dir / args.input
    with (case_dir / 'reference.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    histograms = {name: [] for name in QUANTITIES}
    moments = {name: [] for name in QUANTITIES}
    overflows = {name: [] for name in QUANTITIES}
    counts = []
    # TS0 has a zero-length preceding section and zero physical weight.
    for page in range(2, len(rows) + 1):
        columns = np.loadtxt(_stream(
            args.sdds2stream, source, page,
            '-columns=x,xp,y,yp,p,TRate').splitlines())
        particles, p_central = map(float, _stream(
            args.sdds2stream, source, page,
            '-parameters=Particles,pCentral').splitlines())
        assert columns.shape == (int(particles), 6)
        values = dict(
            x=columns[:, 0], xp=columns[:, 1], y=columns[:, 2],
            yp=columns[:, 3], delta=columns[:, 4] / p_central - 1)
        weights = columns[:, 5]
        counts.append(len(weights))
        for name, edges in QUANTITIES.items():
            histogram = np.histogram(
                values[name], bins=edges, weights=weights)[0]
            in_range_weight = histogram.sum()
            histograms[name].append(histogram / in_range_weight)
            moments[name].append(_moments(values[name], weights))
            overflows[name].append(1 - in_range_weight / weights.sum())
    arrays = dict(
        element_names=np.array([row['element_name'] for row in rows[1:]]),
        s_m=np.array([float(row['s_m']) for row in rows[1:]]),
        selected_counts=np.array(counts),
        moment_names=np.array([
            'weight_sum_hz', 'mean', 'standard_deviation']))
    for name, edges in QUANTITIES.items():
        arrays[f'{name}_bin_edges'] = edges
        arrays[f'{name}_histogram'] = np.asarray(histograms[name])
        arrays[f'{name}_moments'] = np.asarray(moments[name])
    np.savez_compressed(
        case_dir / 'scattered_distribution_reference.npz', **arrays)
    metadata = {
        'elegant_version': '2026.3.0',
        'elegant_source_tag': 'elegant-2026.3.0',
        'generation_command': './run_reference.sh',
        'extraction_command': (
            'python extract_scattered_distribution.py '
            '--sdds2stream "$SDDS_BIN_DIR/sdds2stream"'),
        'input_files': [
            'toy_ring.ele', 'toy_ring.lte', 'momentum_aperture.sdds'],
        'raw_source': 'toy_ring.scatter.sdds',
        'random_seed': 1997,
        'n_scattering_events_per_element': 100000,
        'weight_retention_fraction': 0.99,
        'bunch_intensity': 4000000000,
        'beam_energy_ev': 1000000000,
        'nemitt_x_m': 1e-5,
        'nemitt_y_m': 1e-7,
        'sigma_z_m': 4e-3,
        'sigma_delta': 1e-3,
        'momentum_acceptance': [-0.012, 0.012],
        'momentum_acceptance_scale': 0.85,
        'weight_column': 'TRate',
        'weight_meaning': (
            'Per-row section scattering rate in 1/s, used as event weight.'),
        'included_elements': [row['element_name'] for row in rows[1:]],
        'excluded_elements': {
            rows[0]['element_name']: 'zero physical section weight'},
        'selected_event_count': int(sum(counts)),
        'overflow_weight_fraction': overflows,
        'scattering_angle_available': False,
        'notes': (
            'Post-scattering, pre-tracking coordinates (do_track=0). '
            'delta=p/pCentral-1. ELEGANT does not output the centre-of-mass '
            'scattering angle or a scatter-element column; SDDS page order '
            'maps to element names and s positions in reference.csv.'),
    }
    (case_dir / 'scattered_distribution_metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
