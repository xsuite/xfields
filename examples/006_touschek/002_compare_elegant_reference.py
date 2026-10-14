# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #
"""Compare xfields with the frozen ELEGANT Touschek reference.

This script only reads committed CSV/JSON reference data. It does not run or
require ELEGANT or SDDS.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import xfields as xf
import xtrack as xt


REFERENCE_DIR = (
    Path(__file__).resolve().parents[2]
    / 'test_data' / 'touschek' / 'elegant_reference'
)

LOCAL_RATE_RTOL = 1e-5
TOTAL_RATE_RTOL = 1e-5
LIFETIME_RTOL = 1e-5

NEMITT_X = 1e-5
NEMITT_Y = 1e-7
SIGMA_Z = 4e-3
SIGMA_DELTA = 1e-3
BUNCH_INTENSITY = 4e9


######################################################
# Load the frozen ELEGANT scalar/table outputs
######################################################
with (REFERENCE_DIR / 'reference.csv').open(newline='') as stream:
    elegant_rows = list(csv.DictReader(stream))

with (REFERENCE_DIR / 'reference_summary.json').open() as stream:
    elegant_summary = json.load(stream)

elegant_s = np.array([float(row['s_m']) for row in elegant_rows])
elegant_local_rates = np.array([
    float(row['piwinski_local_rate_hz']) for row in elegant_rows
])
elegant_total_rate = elegant_summary['total_piwinski_rate_hz']
elegant_lifetime = elegant_summary['touschek_lifetime_s']


######################################################
# Build the same xfields toy ring used by the test
######################################################
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
for index, row in enumerate(elegant_rows):
    env.elements[f'TS{index}'] = xf.TouschekScattering()
    placements.append(env.place(f'TS{index}', at=float(row['s_m'])))
line.insert(placements)

twiss = line.twiss(method='4d')
table = line.get_table()
touschek_names = table.rows.match(element_type='TouschekScattering').name
xfields_s = np.array([table['s', name] for name in touschek_names])
np.testing.assert_allclose(xfields_s, elegant_s, atol=1e-12)

# Use the same fixed +/-1.2% aperture as the frozen ELEGANT case. This avoids
# dynamic-aperture tracking and isolates the Touschek rate comparison.
lma = xt.Table({
    'name': touschek_names,
    's': xfields_s,
    'delta_neg': np.full(len(touschek_names), -0.012),
    'delta_pos': np.full(len(touschek_names), 0.012),
})

touschek = line.xfields.touschek_configure(
    twiss=twiss,
    local_momentum_acceptance=lma,
    local_momentum_acceptance_scale=0.85,
    nemitt_x=NEMITT_X,
    nemitt_y=NEMITT_Y,
    sigma_z=SIGMA_Z,
    sigma_delta=SIGMA_DELTA,
    bunch_intensity=BUNCH_INTENSITY,
    n_scattering_events=100_000,
    nx=3,
    ny=3,
    nz=3,
    weight_retention_fraction=0.99,
    seed=1997,
    method='4d',
)

xfields_local_rates = np.array([
    line[name].piwinski_rate for name in touschek_names
])
xfields_total_rate = sum(
    line[name].integrated_piwinski_rate for name in touschek_names
)
xfields_lifetime = BUNCH_INTENSITY / xfields_total_rate


######################################################
# Apply the same comparisons and tolerances as pytest
######################################################
# ELEGANT reports zero at the first zero-length TSCATTER section, so the local
# comparison starts at TS1. Both codes' TS0 values remain visible in the plot.
np.testing.assert_allclose(
    xfields_local_rates[1:],
    elegant_local_rates[1:],
    rtol=LOCAL_RATE_RTOL,
)
np.testing.assert_allclose(
    xfields_total_rate,
    elegant_total_rate,
    rtol=TOTAL_RATE_RTOL,
)
np.testing.assert_allclose(
    xfields_lifetime,
    elegant_lifetime,
    rtol=LIFETIME_RTOL,
)

print('Frozen ELEGANT comparison passed')
print(f'  total rate: xfields={xfields_total_rate:.6f} Hz, '
      f'ELEGANT={elegant_total_rate:.6f} Hz')
print(f'  lifetime:   xfields={xfields_lifetime:.6f} s, '
      f'ELEGANT={elegant_lifetime:.6f} s')


######################################################
# Plot the quantities asserted above
######################################################
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.plot(
    elegant_s,
    elegant_local_rates * 1e-3,
    'o-',
    label='ELEGANT (frozen)',
)
ax.plot(
    xfields_s,
    xfields_local_rates * 1e-3,
    's--',
    label='xfields',
)
ax.set_xlabel('s [m]')
ax.set_ylabel('Local Piwinski rate [kHz]')
ax.set_title(f'Local rates (asserted at TS1-TS8, rtol={LOCAL_RATE_RTOL:.0%})')
ax.grid(alpha=0.3)
ax.legend()

labels = ['ELEGANT\n(frozen)', 'xfields']
colors = ['tab:blue', 'tab:orange']

ax = axes[1]
ax.bar(
    labels,
    np.array([elegant_total_rate, xfields_total_rate]) * 1e-3,
    color=colors,
)
ax.set_ylabel('Total Piwinski rate [kHz]')
ax.set_title(f'Total rate (rtol={TOTAL_RATE_RTOL:.0%})')
ax.grid(axis='y', alpha=0.3)

ax = axes[2]
ax.bar(
    labels,
    np.array([elegant_lifetime, xfields_lifetime]) / 60,
    color=colors,
)
ax.set_ylabel('Touschek lifetime [min]')
ax.set_title(f'Lifetime (rtol={LIFETIME_RTOL:.0%})')
ax.grid(axis='y', alpha=0.3)

fig.suptitle('xfields comparison with frozen ELEGANT Touschek reference')
fig.tight_layout()

######################################################
# Compare and plot frozen scattered-particle summaries
######################################################
with np.load(
        REFERENCE_DIR / 'scattered_distribution_reference.npz') as data:
    scatter_reference = {name: data[name].copy() for name in data.files}

# TS0 has zero section weight. Generate one fixed-seed sample at TS1--TS8.
scatter_result = touschek.run(track=False, keep_particles=True)
particles = [
    scatter_result.particles_by_element[name] for name in touschek_names[1:]
]


def xfields_summary(quantity):
    attribute = {'xp': 'kin_xp', 'yp': 'kin_yp'}.get(quantity, quantity)
    values = []
    weights = []
    for part in particles:
        alive = part.state == 1
        values.append(np.asarray(getattr(part, attribute)[alive]))
        weights.append(np.asarray(part.weight[alive]))
    values = np.concatenate(values)
    weights = np.concatenate(weights)
    edges = scatter_reference[f'{quantity}_bin_edges']
    histogram = np.histogram(values, edges, weights=weights)[0]
    histogram /= histogram.sum()
    mean = np.average(values, weights=weights)
    std = np.sqrt(np.average((values - mean)**2, weights=weights))
    return histogram, np.array([weights.sum(), mean, std])


def elegant_summary(quantity):
    element_moments = scatter_reference[f'{quantity}_moments']
    weights = element_moments[:, 0]
    mean = np.average(element_moments[:, 1], weights=weights)
    std = np.sqrt(np.average(
        element_moments[:, 2]**2 + (element_moments[:, 1] - mean)**2,
        weights=weights))
    histogram = np.average(
        scatter_reference[f'{quantity}_histogram'], axis=0, weights=weights)
    return histogram, np.array([weights.sum(), mean, std])


quantities = ['x', 'xp', 'y', 'yp', 'delta']
xfields_scatter = {name: xfields_summary(name) for name in quantities}
elegant_scatter = {name: elegant_summary(name) for name in quantities}

assert np.sum(np.abs(
    xfields_scatter['delta'][0] - elegant_scatter['delta'][0])) < 0.02
np.testing.assert_allclose(
    xfields_scatter['delta'][1][2], elegant_scatter['delta'][1][2], rtol=0.02)

fig_scatter, scatter_axes = plt.subplots(2, 2, figsize=(11, 8))
scatter_axes = scatter_axes.ravel()
edges = scatter_reference['delta_bin_edges']
centers = (edges[:-1] + edges[1:]) / 2
scatter_axes[0].step(
    centers, elegant_scatter['delta'][0], where='mid', label='ELEGANT')
scatter_axes[0].step(
    centers, xfields_scatter['delta'][0], where='mid', label='xfields')
scatter_axes[0].set_xlabel('Momentum deviation delta')
scatter_axes[0].set_ylabel('Normalized weighted probability')
scatter_axes[0].set_title('Scattered momentum distribution')
scatter_axes[0].grid(alpha=0.3)
scatter_axes[0].legend()

width = 0.35
positions = np.arange(len(quantities))
elegant_std = [elegant_scatter[name][1][2] for name in quantities]
xfields_std = [xfields_scatter[name][1][2] for name in quantities]
scatter_axes[1].bar(positions - width / 2, elegant_std, width, label='ELEGANT')
scatter_axes[1].bar(positions + width / 2, xfields_std, width, label='xfields')
scatter_axes[1].set_xticks(positions, quantities)
scatter_axes[1].set_yscale('log')
scatter_axes[1].set_ylabel('Weighted standard deviation')
scatter_axes[1].set_title('Asserted scattered-coordinate widths')
scatter_axes[1].grid(axis='y', alpha=0.3)
scatter_axes[1].legend()

normalized_mean_difference = [
    (xfields_scatter[name][1][1] - elegant_scatter[name][1][1])
    / elegant_scatter[name][1][2]
    for name in quantities
]
scatter_axes[2].bar(positions, normalized_mean_difference, width)
scatter_axes[2].set_xticks(positions, quantities)
scatter_axes[2].axhline(0.0, color='black', linewidth=0.8)
scatter_axes[2].set_ylabel('(xfields - ELEGANT) / ELEGANT std')
scatter_axes[2].set_title('Normalized weighted-mean difference')
scatter_axes[2].grid(axis='y', alpha=0.3)

elegant_weight = elegant_scatter['delta'][1][0]
xfields_weight = xfields_scatter['delta'][1][0]
scatter_axes[3].bar(
    ['ELEGANT\n(frozen)', 'xfields'],
    [elegant_weight * 1e-3, xfields_weight * 1e-3],
    color=colors)
scatter_axes[3].set_ylabel('Retained section weight [kHz]')
scatter_axes[3].set_title('Asserted aggregate particle-weight sum')
scatter_axes[3].grid(axis='y', alpha=0.3)

fig_scatter.suptitle('Post-scattering, pre-tracking distributions')
fig_scatter.tight_layout()

plt.show()
