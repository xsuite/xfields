# Touschek API: markers and generator

This note summarizes the restructuring of the Touschek API into passive
markers and a generator bound to the line. The same design is used for the
beam-gas scattering of xcoll (`BeamGasMarker`, `line.xcoll.beamgas`). It
replaces the `xf.TouschekStudy` workflow, introduced after the original
Touschek merge (`3f1496b3`, PR #207), which is removed without a deprecation
period.

## Design

- `xf.TouschekMarker` (formerly `xf.TouschekScattering`) is a passive marker,
  transparent for tracking. Each marker represents the section of the line
  from the previous marker (from `s = 0` for the first one) to itself, and
  stores, in its fields, the whole configuration of the Monte Carlo of its
  section: local optics and closed orbit, local momentum acceptance, beam
  parameters, Monte Carlo settings, random-number seed, and the scattering
  rate of the section.
- `xf.TouschekGenerator`, available as `line.xfields.touschek`, finds the
  markers of the line, configures them, generates the scattered particles and
  analyses them. It holds no state other than the line.

Since the configuration is stored in the markers, the configured line can be
saved to JSON and loaded again, e.g. in separate jobs, and generates the same
particles.

## Standard workflow

```python
tsc = line.xfields.touschek
tsc.install_markers(at=[...])        # or by hand: env.place + line.insert
tsc.configure(local_momentum_acceptance=lma, nemitt_x=..., nemitt_y=...,
              sigma_z=..., sigma_delta=..., bunch_intensity=...,
              n_scattering_events=..., seed=..., method='4d')
# The line can be saved here, with its configuration

tab = line.get_table()
touschek_elements = tab.rows.match(element_type='TouschekMarker').name

particles_list = []
for name in touschek_elements:
    part = tsc.generate(at=name)
    line.track(part, ele_start=name, ele_stop=name, num_turns=n_turns)
    particles_list.append(part)
particles = xt.Particles.merge(particles_list)

# Optional: xt.LossLocationRefinement(...).refine_loss_location(particles)
result = tsc.analyse(particles)
```

`tsc.run(n_turns)` does the same steps in one call. The examples are
`000_touschek_toy_ring.py` (the workflow above),
`002_compare_elegant_reference.py` (the ELEGANT benchmark) and
`003_touschek_jobs.py` (independent jobs, merged afterwards).

## Installation of the markers

`install_markers(at, from_=..., anchor=..., from_anchor=..., names=...)`
creates the markers and places them with `env.place` and `line.insert`: `at`
can be an `s` position, an expression of the environment variables, an xdeps
reference, an element name (its centre), or `'name@start'`/`'name@end'`.
Scalars apply to all the markers, lists give one value per marker. The
default names are `tscatter.<i>`.

The last marker must sit at the end of the line, otherwise the part of the
line after it belongs to no section (as for the TSCATTER elements of ELEGANT):
`install_markers` adds such a marker if needed (`close_ring=True`), and
`configure` refuses a line without it.

## Independent jobs and merging

- Each marker has its own random-number stream, derived with
  `numpy.random.SeedSequence` from the configured seed, the name of the marker
  and the replica index. The particles of a marker do not depend on the other
  markers generated, nor on the order: one job generating all the markers and
  one job per marker give the same particles.
- `generate(at=name, replica=r, n_replicas=R)` generates an independent sample
  of a marker and divides its weights by `R`. The weights are rates [1/s per
  bunch], so merged with `xt.Particles.merge`, the samples of all the markers
  and replicas give the loss rate of the ring.
- `generate(rng_stream='continuous')` runs one stream, seeded with the seed,
  through the markers in line order, as in ELEGANT, whose sequence it
  reproduces. The ELEGANT benchmark relies on it.

## Analysis

`analyse(particles)` accepts:

- nothing: the scattering rates configured on the markers;
- an `xt.Particles` object, i.e. the sample of a marker (`at=name`) or the
  particles of several markers merged, possibly from different jobs: the
  scattering, generated and loss rates, and the lifetimes;
- a mapping `{name: particles or [replicas]}`: in addition, the per-marker
  counts and rates, and the Monte Carlo error of the loss rate from the spread
  between the replicas.

The particles do not record the marker at which they were generated: there is
no free per-particle field in `xt.Particles`, `xt.Particles.merge` renumbers
`particle_id`, and xcoll's Geant4 and FLUKA couplings require small particle
ids. So the per-marker results need the samples given separately.

When all the markers are analysed, the total weight of the scattered particles
is compared with the retained fraction of the scattering rate, to detect
missing or repeated samples.

## Changes from the TouschekStudy API

- `TouschekScattering` -> `TouschekMarker`; the fields `n_simulated` and
  `ignored_portion` -> `n_scattering_events` and `weight_retention_fraction`.
  The closed orbit, the local Piwinski rate and the seed are now fields, so
  they are saved with the line, as is the random-number generator (which was
  reset to `'elegant'` on loading). Lines saved with `TouschekScattering`
  elements must be created again.
- `line.xfields.touschek_configure(...)` / `xf.TouschekStudy(...)` and
  `initialise_touschek(element)` -> `line.xfields.touschek.configure(...)`,
  with `at=` to (re)configure a part of the markers. The local momentum
  acceptance table is no longer scaled in place, and a seed is drawn and
  stored if none is given.
- `TouschekStudy.run`, `generate_particles` and `local_rates` ->
  `line.xfields.touschek.run`, `generate` and `analyse`.
- `TouschekMarker.scatter()` is removed: `generate` sets the position, the
  line index and the species of the particles from the line.
- The deprecated aliases `n_simulated` and `ignored_portion` are removed.

## Rate and weight handling

The study uses the Piwinski rate to normalize the weighted Monte Carlo
particles. `weight_retention_fraction=0.99` means that the returned particle
set represents approximately 99 percent of the generated scattering weight;
the discarded low-weight tail is not renormalized onto the retained particles.
`weight_retention_fraction=1.0` keeps all generated particles.

The section integration uses `twiss.line_length`, and the longitudinal
sampling cutoff `nz` is reduced, marker by marker, so that the initial
particles are never generated outside the local momentum acceptance before
the Touschek kick.

`xtrack.Line.get_local_momentum_acceptance()` returns the columns `name`, `s`,
`delta_neg` and `delta_pos`, used by `configure`.
