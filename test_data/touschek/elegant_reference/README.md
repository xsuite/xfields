# ELEGANT Touschek reference for the xfields toy ring

This directory contains a frozen ELEGANT reference for the xfields Touschek
implementation. ELEGANT is used only to regenerate these files manually. CI
and pytest read the committed CSV/JSON data and never invoke ELEGANT or SDDS.

## Platform and versions

- Host: Ubuntu 25.04 (Plucky Puffin), Linux 6.14.0-37-generic
- Architecture: x86_64
- ELEGANT: 2026.3.0, tag `elegant-2026.3.0`, commit
  `a588d0781f99e3c598abede89c2545db65a9154a`
- SDDS: 5.11, tag `SDDS-5.11`, commit
  `a6c48a75c6b315fed486684b6ae16320223ac8cb`
- Compiler: conda-forge GCC/G++ 14.3.0
- Build tool: GNU Make 4.4.1

The official APS-maintainer source repositories were built in a contained
directory under `/tmp`; no root access or system package installation was
used. Binaries are intentionally not stored in this test-data directory.

## Reproducible installation

These are the complete commands needed on this host. Replace the conda
environment name if necessary, and use its absolute prefix in `LIB_DIRS`.

```bash
conda install -n py313 -y make gsl zlib xz fftw liblapack

mkdir -p /tmp/elegant-reference-build
git clone --branch elegant-2026.3.0 --depth 1 \
  https://github.com/rtsoliday/elegant.git \
  /tmp/elegant-reference-build/elegant
git clone --branch SDDS-5.11 --depth 1 \
  https://github.com/rtsoliday/SDDS.git \
  /tmp/elegant-reference-build/SDDS

cd /tmp/elegant-reference-build/elegant
make CUDA_AUTO=0 \
  LIB_DIRS='/home/giadarol/miniforge3/envs/py313/lib /usr/lib/x86_64-linux-gnu /usr/lib /lib' \
  -j4

cd /tmp/elegant-reference-build/SDDS
make LIB_DIRS='/home/giadarol/miniforge3/envs/py313/lib /usr/lib/x86_64-linux-gnu /usr/lib /lib' \
  -C 2d_interpolate/nn
make LIB_DIRS='/home/giadarol/miniforge3/envs/py313/lib /usr/lib/x86_64-linux-gnu /usr/lib /lib' \
  -C 2d_interpolate/csa

cd /tmp/elegant-reference-build/SDDS/SDDSaps
make LIB_DIRS='/home/giadarol/miniforge3/envs/py313/lib /usr/lib/x86_64-linux-gnu /usr/lib /lib' \
  O.Linux-x86_64/sdds2stream \
  O.Linux-x86_64/sddsprintout \
  O.Linux-x86_64/sddsquery \
  O.Linux-x86_64/sddsconvert \
  O.Linux-x86_64/sddsprocess
```

The top-level SDDS `make` target also attempts unrelated optional converters;
on this host it stops at the optional HDF5 converter because HDF5 development
headers are absent. The targeted commands above build every tool required for
this reference and avoid adding unused dependencies.

Verification commands:

```bash
/tmp/elegant-reference-build/elegant/bin/Linux-x86_64/elegant
/tmp/elegant-reference-build/elegant/bin/Linux-x86_64/elegant -version

for tool in sdds2stream sddsprintout sddsquery sddsconvert sddsprocess; do
  test -x "/tmp/elegant-reference-build/SDDS/bin/Linux-x86_64/${tool}"
done
```

ELEGANT reports `This is elegant 2026.3.0`. The source build requires
`RPN_DEFNS` to point to the pinned SDDS `defns.rpn` file.

## Reference case

`toy_ring.lte` is the ELEGANT analogue of
`examples/006_touschek/000_touschek_toy_ring.py` and
`tests/test_touschek.py`: a 21.2 m, 1 GeV electron ring with four 3 m, 90
degree sector bends, four 0.3 m quadrupoles, and nine `TSCATTER` markers.
`momentum_aperture.sdds` supplies a fixed symmetric +/-1.2% local momentum
acceptance at the markers. ELEGANT scales this by 0.85.

Fixed settings in `toy_ring.ele`:

- random seed: 1997
- bunch population: 4e9 electrons (`6.408706536e-10 C`)
- revolution frequency: `14141153.679245283 Hz` (`c / 21.2 m`)
- normalized emittances: `1e-5` and `1e-7 m rad`
- RMS bunch length and momentum spread: `4e-3 m` and `1e-3`
- accepted scattering events: 100,000 per marker
- Gaussian cutoffs: ELEGANT defaults of 3 sigma in x, y, and z
- ignored weight portion: 0.01 (99% retained)
- particle tracking: disabled; this reference concerns rates, not individual
  downstream loss coordinates

Regenerate from this directory with:

```bash
export RPN_DEFNS=/tmp/elegant-reference-build/SDDS/defns.rpn
export ELEGANT_BIN=/tmp/elegant-reference-build/elegant/bin/Linux-x86_64/elegant
export SDDS_BIN_DIR=/tmp/elegant-reference-build/SDDS/bin/Linux-x86_64
export PYTHON_BIN=/home/giadarol/miniforge3/envs/py313/bin/python
./run_reference.sh
sha256sum -c SHA256SUMS
```

## Files

- `toy_ring.ele`: ELEGANT command file, including seed and Touschek settings.
- `toy_ring.lte`: lattice and `TSCATTER` marker definitions.
- `momentum_aperture.sdds`: human-readable ASCII SDDS aperture input.
- `run_reference.sh`: the only regeneration and extraction command.
- `toy_ring.twi`: frozen ELEGANT optics in binary SDDS form.
- `toy_ring.scatter.sdds`: selected scattered particles and section weights.
  Each page is a TSCATTER element and contains post-scattering, pre-tracking
  `x`, `xp`, `y`, `yp`, `p`, `LRate`, and `TRate` values. ELEGANT
  does not expose the centre-of-mass scattering angle in this file.
- `extract_scattered_distribution.py`: converts the raw scatter pages into
  deterministic weighted histograms and moments.
- `scattered_distribution_reference.npz`: compact fixed-bin histograms and
  weighted moments for `x`, `xp`, `y`, `yp`, and `delta=p/pCentral-1`.
- `scattered_distribution_metadata.json`: generation settings and weighting.
- `toy_ring.distribution.sdds`: per-marker histograms and Piwinski/Monte Carlo
  rate parameters in binary SDDS form.
- `reference.csv`: precision-preserving per-marker scalar rates and counts.
- `reference_summary.json`: total Piwinski rate, corresponding lifetime, and
  total selected count. The frozen values are 644171.480142571 Hz,
  6209.52669173541 s, and 2458, respectively.
- `twiss_table.txt`, `touschek_rates.txt`, `scatter_summary.txt`: readable SDDS
  extracts for review.
- `*.layout.txt`: SDDS schemas produced by `sddsquery`.
- `elegant.stdout.txt` and `elegant.stderr.txt`: complete run logs.
- `SHA256SUMS`: checksums for physics inputs and frozen numeric outputs.

## Comparison policy and tolerances

Do not compare individual scattered particles, since selection and particle
ordering are Monte Carlo implementation details. Distribution comparisons use
ELEGANT's `TRate` as the event weight, which is the section-normalized row
rate and corresponds directly to the xfields particle weight.

The pytest comparison uses:

- local nonzero Piwinski rates: relative tolerance `1e-5`;
- total integrated Piwinski rate: relative tolerance `1e-5`;
- Touschek lifetime inferred as `bunch_intensity / total_rate`: relative
  tolerance `1e-5`;
- aggregate weighted `delta` histogram: L1 distance below `0.02`;
- aggregate retained-weight sum: relative tolerance `2e-3`;
- weighted means: normalized mean-difference tolerances below `0.02` ELEGANT
  standard deviations;
- weighted standard deviations: relative tolerance `0.02`.

The xfields comparison places the `TouschekScattering` markers at the exact
`s_m` positions stored in `reference.csv`. With matching marker locations, the
measured differences for this frozen case are a few ppm locally and for the
total rate. ELEGANT records a zero local Piwinski rate at the first zero-length
`TSCATTER` section, so that entry is excluded from the local-rate comparison.
The same zero-weight TS0 page is excluded from the aggregate distribution.
TS1--TS8 contain 2,155 retained ELEGANT particles. Fixed bin edges and weighted
moments are frozen instead of raw particle equality. The tolerances reflect the
few-thousand-particle, strongly weighted Monte Carlo sample; they are broadest
for transverse summaries and tighter for the momentum-deviation shape and total
weight.
