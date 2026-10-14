# Touschek Refactor Summary

Reference commit:

```text
3f1496b36a37069222b4082207bac84ff8a9e883
Merge pull request #207 from gbrogginess/touschek
```

This note summarizes the changes made after the original Touschek merge, with
the intent of making the feature easier to use, easier to document, and more
consistent with Xsuite naming conventions.

## Main API Changes

The original high-level workflow has been reshaped around a single public study
object:

```python
touschek = line.xfields.touschek_configure(...)
result = touschek.run(track=True, n_turns=...)
```

`line.xfields.touschek_configure(...)` now returns a `xf.TouschekStudy`.
The study object owns the configuration and exposes:

- `local_rates()`
- `generate_particles()`
- `run(...)`

The previous public `TouschekManager` / `TouschekCalculator` split was removed.
The implementation now uses `xf.TouschekStudy` as the public object.

## Naming Changes

Several parameters were renamed to make the API more explicit:

```text
ignored_portion          -> weight_retention_fraction
n_simulated              -> n_scattering_events
bunch_population         -> bunch_intensity
deltan / deltap          -> delta_neg / delta_pos
deltaN / deltaP          -> delta_neg / delta_pos
```

Compatibility aliases are still kept for:

- `ignored_portion`
- `n_simulated`

No backward-compatibility alias was kept for:

- `bunch_population`
- `deltan` / `deltap`
- `deltaN` / `deltaP`

Those names belonged to the new Touschek functionality and were removed before
the API reached users.

## Local Momentum Acceptance

`xtrack.Line.get_local_momentum_acceptance()` now returns clearer table columns:

```text
name
s
delta_neg
delta_pos
```

instead of:

```text
deltan
deltap
```

The Touschek study now requires an LMA table with `delta_neg` and `delta_pos`.
The same names are used in the `TouschekScattering` xobject fields and in the C
kernel accessors.

## Touschek Facade

A line-level xfields facade was added, following the style of other line
facades:

```python
line.xfields.touschek_configure(...)
```

The facade workflow is to configure once and then call methods on the returned
`TouschekStudy`.

## Examples

The original example was kept:

```text
examples/006_touschek/000_touschek_toy_ring.py
```

A new facade-based example was added:

```text
examples/006_touschek/001_touschek_toy_ring_facade.py
```

The examples now use the clearer API names:

```python
bunch_intensity=...
n_scattering_events=...
weight_retention_fraction=...
```

and plot/use LMA columns as:

```python
lma.delta_neg
lma.delta_pos
```

## Rate And Weight Handling

The study still uses the Piwinski rate to normalize the weighted Monte Carlo
particles.

`weight_retention_fraction` is now phrased from the user perspective:

```python
weight_retention_fraction=0.99
```

means that the returned particle set represents approximately 99 percent of the
generated scattering weight. The discarded low-weight tail is not renormalized
onto the retained particles.

Set:

```python
weight_retention_fraction=1.0
```

to keep all generated particles.

## Physics And Implementation Fixes

The Touschek study now uses:

```python
twiss.line_length
```

instead of reconstructing the line length as:

```python
C_LIGHT_VACUUM * twiss.t_rev0
```

The Piwinski-integral helper is now a static method of `TouschekStudy`, and the
local-rate construction is owned directly by `TouschekStudy.local_rates()`.

The longitudinal sampling cutoff logic and comments were clarified. In
particular, the effective `nz` can be reduced element by element so that
initial particles are not generated outside the local momentum aperture before
the Touschek kick.

## Tests

The Touschek tests were updated for the new API and naming. The end-to-end
tracking test now uses the standard Xsuite test-context parametrization:

```python
@for_all_test_contexts(excluding=("ContextCupy", "ContextPyopencl"))
```

The local OpenMP build was not available on this machine, so the validated local
run used the serial CPU context.

Validated locally:

```text
xtrack/tests/test_local_momentum_acceptance.py: 12 passed
xfields/tests/test_touschek.py: 41 passed
```

## Related Commits

Relevant `xfields` commits after the reference merge:

```text
c0516a3 Simplify Touschek study API
8409cfc Parametrize Touschek tracking test contexts
b9e4f8f Make Touschek study own rate helpers
b06cd77 Rename Touschek weight retention setting
a315a38 Rename Touschek scattering event count
8bda6dc Use bunch intensity in Touschek API
80ea9c2 Use explicit Touschek momentum acceptance names
eacbc28 Simplify Touschek docstring notation
```

Related `xtrack` commits:

```text
74c2b4e14 Add xfields line facade
4f4cf1175 Rename local momentum acceptance columns
```

Related `xsuite` documentation commits:

```text
9c6f0cf Clarify Touschek physics manual
f32d8f5 Rename Touschek retention parameter in manual
```
