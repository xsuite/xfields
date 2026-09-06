# Multibunch beam-beam design rationalization

This note records a proposed cleanup of the multibunch beam-beam work before
the API becomes established. The goal is to retain the new physics and solver
capabilities while integrating them into the existing Xfields and Xtrack
abstractions.

## Motivation

The current implementation introduces two concepts parallel to existing ones:

1. `BeamBeamBiGaussianRigidBunch2D` originally duplicated the physical kick
   calculation in `BeamBeamBiGaussian2D`.
2. `env.xfields.install_multibunch_beambeam(...)` duplicates much of the
   existing `install_beambeam_interactions(...)` and
   `configure_beambeam_interactions(...)` workflow.

The physical kick and the installation/configuration workflow should not have
parallel implementations. The element storage, however, has a real structural
difference: scalar BB2D has fixed-size data, while the rigid-bunch element owns
arrays indexed by RF slot. Combining those layouts would enlarge every scalar
BB2D element, introduce a per-particle mode branch and complicate the API.

The multibunch-specific behavior is narrower than these duplicated surfaces:

- Match a tracked bunch to an opposing bunch using `zeta`, with periodic ring
  wrapping.
- Store and update per-bunch centroids, populations and covariances.
- In coherent mode, use the convolution of the two matched bunch covariances.
- Solve the two beams self-consistently and optionally update dynamic beta.

These capabilities should therefore keep a dedicated element class, while the
physical kick and installation machinery are shared wherever their behavior is
genuinely common.

## Rationalization 1: keep two element classes and share the kick

Keep both public element classes because they represent different storage and
tracking contracts:

- `BeamBeamBiGaussian2D` keeps its compact scalar layout, established
  constructor, weak-strong behavior and pipeline strong-strong updater.
- `BeamBeamBiGaussianRigidBunch2D` owns the slot-indexed bunch arrays and
  the rigid-bunch train behavior.

The classes must call one common BB2D kick helper that owns:

- covariance rotation;
- the bi-Gaussian field calculation;
- relativistic and charge scaling;
- strength scaling; and
- the final transverse momentum kick and optional post-subtraction.

Each wrapper remains responsible for selecting those inputs. Scalar BB2D reads
its scalar fields. The rigid-bunch wrapper matches the opposing bunch on the
periodic `zeta` grid, selects its centroid, population and covariance, and, in
coherent mode, adds the matched own-beam covariance before calling the helper.
The bunch-matching code is multibunch-specific and does not need to be added to
the scalar element.

For coherent tracking, the effective covariance is the sum of the matched own
and opposing bunch covariances. Using covariances instead of separate
`sigma_x`/`sigma_y` logic remains consistent with scalar BB2D and leaves room
for transverse coupling. The API and serialized layout include ``Sigma_11``,
``Sigma_13`` and ``Sigma_33``. Coupled rigid-bunch operation is not yet
validated, so ``Sigma_13`` is currently ignored by the kick and a warning is
emitted when the normalized correlation
``abs(Sigma_13) / sqrt(Sigma_11 * Sigma_33)`` exceeds ``1e-2``.

In the high-level ring workflow, rigid-bunch arrays have one entry per physical
RF slot. ``harmonic_number`` and ``bunch_spacing_buckets`` determine that size
at installation, so installation can actually place the elements before the
filling is configured. Empty slots carry zero population, and any subsequent
filling change updates the existing elements in place. The standalone Xfields
element can still infer its allocation from explicitly supplied bunch data.

Keeping the classes separate also makes the semantics visible in the API:
pipeline strong-strong remains a configuration of scalar BB2D, whereas
`mode='rigid_bunch'` constructs `BeamBeamBiGaussianRigidBunch2D`.
“Multibunch” describes the capability shared by both approaches; “rigid bunch”
identifies the physical approximation used by this workflow. “Fixed point” is
reserved for the solver algorithm, and “train” for the collection of bunches.
Permanent tests should
include one-bunch and per-selected-bunch comparisons against scalar BB2D so the
shared physical model remains locked down.

## Rationalization 2: extend the existing installation workflow

The public entry point should remain
`env.xfields.install_beambeam_interactions(...)`, followed by
`env.xfields.configure_beambeam_interactions(...)`. A parallel
`install_multibunch_beambeam(...)` entry point should not be needed.

The existing configuration machinery already owns:

- encounter generation and placement;
- clockwise/anticlockwise conventions;
- bunch-slot delays and encounter pairing;
- Twiss and survey geometry;
- beam covariance computation;
- opposing-beam coordinate transformations and separation;
- beam-beam strength knobs;
- storage of beam-beam configuration on the environment.

These operations should be factored into element-independent helpers that
produce an encounter description and its geometry. The particles and
rigid-bunch workflows should consume the same description.

The installation API can select a mode, for example:

```python
env.xfields.install_beambeam_interactions(
    clockwise_line='lhcb1',
    anticlockwise_line='lhcb2',
    ip_names=['ip1', 'ip2', 'ip5', 'ip8'],
    num_long_range_encounters_per_side=45,
    harmonic_number=35640,
    bunch_spacing_buckets=10,
    mode='rigid_bunch',
)

rigid_bunch_study = env.xfields.configure_beambeam_interactions(
    num_particles={'cw': bunch_intensity_cw, 'acw': bunch_intensity_acw},
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    filling_pattern_cw=filling_pattern_cw,
    filling_pattern_acw=filling_pattern_acw,
)
```

The default mode must preserve the existing sliced head-on and long-range
workflow. Rigid-bunch mode installs the extended BB2D element with one 2D lens
per head-on or long-range encounter and allocates its bunch arrays.
Configuration loads the populations, geometry and design covariances, then
returns a `BeamBeamRigidBunchStudy`. Filling patterns can be supplied during
configuration or changed later on the returned study.

Dense and sparse filling inputs share one contract across Xtrack, Xpart,
Xfields and Xwakes. A caller provides exactly one of `filling_pattern` or
`filled_slots`; `filling_scheme` remains a compatibility alias for the dense
form where it was previously exposed. APIs that know the machine layout use
`num_slots` (or their installed harmonic configuration) to retain trailing
empty slots. Bunch
intensities remain separate from occupancy.

Normalization is implemented by the internal, immutable
`xtrack._filling_pattern._FillingPattern` value object. It validates dense
binary patterns and sparse non-negative, unique slot indices, copies caller
inputs, and creates fresh arrays when either representation is exposed. Device
arrays needed by tracking remain private. Filling changes are made through the
owning object's configuration/apply method, where all inputs are normalized
before state is replaced, rather than by mutating a public slot array.

`BeamBeamRigidBunchStudy` remains useful, but should contain only the genuinely
stateful rigid-bunch operations:

- `apply_filling_pattern(...)`;
- `twiss(...)`;
- `solve(...)`;
- `second_order_maps(...)`;
- `load_solution(...)`;
- convergence and optional dynamic-beta updates.

Element placement, survey geometry, design covariance calculation, reversed
beam transformations and beam-beam knob creation should move to the shared
configuration layer. In particular, beam sizes should come from the standard
Twiss beam-covariance API rather than a separate `sqrt(beta * nemitt / gamma)`
calculation.

`configure_orbit_dependent_parameters_for_bb(...)` cannot be applied unchanged
to the coherent rigid-bunch solve: it subtracts the reference dipole kick,
whereas the solver intentionally retains that kick to find the distorted
closed orbit. Its lower-level coordinate conventions may still be shared.

## Rationalization 3: keep rigid-bunch optics in Xfields

The per-bunch optics implementation was initially exposed as the generic
Xtrack APIs `Line.twiss_multibunch(...)`, `twiss_line_multibunch(...)` and
`MultiBunchTwiss`. In practice it assumes a fixed longitudinal coordinate and
a bunch-dependent force selected by `BeamBeamBiGaussianRigidBunch2D`. It does
not describe pipeline strong-strong operation, wakefields or arbitrary
multibunch tracking. The generic name therefore promises a wider contract than
the implementation provides.

Rigid-bunch optics belongs to the beam-beam study that already knows both
lines, their filling patterns and their physical RF slots. The public API is:

```python
twiss = rigid_bunch_study.twiss(mode='fast')
solution = rigid_bunch_study.solve(...)

twiss.cw.qx
twiss.acw.qx
rigid_bunch_study.load_solution(solution)
```

`rigid_bunch_study.twiss(...)` observes the opposing-beam state currently
loaded in the elements; `rigid_bunch_study.solve(...)` repeatedly calls that
operation while feeding the two beams back into each other. Bunch positions
are derived from the study, so
users cannot accidentally provide a `zeta_bunches` array inconsistent with the
configured filling. `RigidBunchTwiss` contains the two beam results as `cw` and
`acw`; each is a `BunchTwiss` collection with one Xtrack `TwissTable` per filled
bunch. The containers live in Xfields together with
`BeamBeamRigidBunchStudy` and its solver.

``solve(...)`` raises ``RuntimeError`` by default if ``max_iterations`` is
reached before ``tol_sigma``. A deliberate fixed-iteration study can pass
``require_convergence=False`` and inspect ``converged``,
``num_iterations`` and ``max_orbit_change`` on the returned result before using
the last iterate.

For the same ownership reason, the rigid-bunch installer, study, examples,
tests and LHC regression data live in Xfields. Xtrack retains only a generic
lazy environment façade: `env.xfields` constructs
`xfields.XfieldsEnvironmentAPI`, just as the line-level Xfields and Xcoll
façades load their owning packages. This small integration hook prevents
Xtrack from acquiring beam-beam-specific implementation code.

## Bunch-pattern API consistency

The rigid-bunch beam-beam API should use the same bunch-pattern concepts as
Xpart, Xwakes and `BeamStatsMonitor`. Their common public model is:

- `filling_pattern` is a slot-indexed boolean/integer occupancy pattern;
- `filled_slots` contains the corresponding physical slot numbers;
- `bunch_spacing_zeta` is the positive physical distance between adjacent
  slots of that pattern; and
- physical slot `i` is centred at `zeta = -i * bunch_spacing_zeta`.

Occupancy and intensity must remain separate. In particular, a floating-point
array containing the population of every slot should not be called a filling
pattern. Rigid-bunch configuration therefore accepts `num_particles` as either
a common scalar or a `cw` / `acw` mapping. Each value can be uniform or
slot-indexed when bunch populations are not uniform. The separate
`apply_filling_pattern(...)` operation selects occupied slots. The study
exposes the derived physical slot identifiers as `filled_slots_cw` and
`filled_slots_acw`, rather than the ambiguous `bunches_cw` and `bunches_acw`.

The high-level installer allocates own- and opposing-beam arrays for every RF
slot. This is not a user-selected reserve capacity: the ring topology fixes it
uniquely as ``harmonic_number // bunch_spacing_buckets``. The filling patterns
select populated physical slots, while empty slots remain present with zero
population. This keeps the public filling semantics aligned with
`BeamStatsMonitor` and Xwakes, and lets `apply_filling_pattern(...)` update
elements in place without a hidden reallocation lifecycle.

Keeping `harmonic_number` and `bunch_spacing_buckets` in the high-level
beam-beam installation API is useful because encounter pairing needs the
integer ring topology. The normalized study should additionally expose
`bunch_spacing_zeta`. Any different phase or sign convention needed inside a
beam-beam kernel should be converted at the implementation boundary and should
not change the public bunch-position convention.

There is an existing selection-semantic difference which this refactoring must
not spread further:

- `BeamStatsMonitor.selected_slots` contains physical slot numbers;
- Xpart and Xwakes `bunch_selection` contains ordinal indices into the compact
  list of filled bunches.

For example, with `filling_pattern=[1, 0, 1, 1]`,
`BeamStatsMonitor(selected_slots=[0, 2])` selects physical slots 0 and 2,
whereas the Xpart/Xwakes `bunch_selection=[0, 2]` selects physical slots 0 and
3. The new beam-beam API should use physical slot numbers whenever it exposes
a selection and should call that argument `selected_slots`.

### Cross-package naming decision

`filling_pattern` is the canonical public term in Xpart, Xfields, Xwakes and
Xtrack. Existing `filling_scheme` keyword arguments remain supported as
compatibility aliases, while new documentation and examples use
`filling_pattern`. Supplying both names is an error. Internal serialized and
Xobject fields may retain their established names to avoid a data-format
migration.

This naming migration does not change the meaning of the established
Xpart/Xwakes `bunch_selection` argument. A possible follow-up is to add
`selected_slots` to those packages, make it mutually exclusive with
`bunch_selection`, and perform the physical-slot to compact-index conversion
internally.

Add a small cross-package contract test, without Xmask, using a sparse pattern
such as:

```python
filling_pattern = [1, 0, 1, 1]
bunch_spacing_zeta = 5
selected_slots = [0, 3]
```

It should establish that `filled_slots == [0, 2, 3]` and that the selected
physical bunch centres are `[0, -15]`. This catches confusion between physical
slots and compact bunch ordinals, as well as spacing and `zeta`-sign mistakes.

## Implementation plan

The work should be staged so that each intermediate commit remains usable and
the two repositories can be migrated without requiring an atomic Xfields and
Xtrack update.

### Phase 1: freeze current behavior

Add the fast characterization tests before changing either implementation:

- scalar BB2D behavior and serialization;
- current multibunch matching and coherent convolution;
- one-bunch scalar/multibunch equivalence;
- sparse fillings, unequal intensities and periodic matching;
- a small deterministic Xtrack installer/study test;
- multibunch Twiss mode comparisons; and
- the cross-package bunch-pattern contract described above.

These tests are the normal development loop. Pytrain and Xmask remain the
realistic final acceptance tests.

### Phase 2: share Xfields physics without merging element layouts

In Xfields, extract the scalar BB2D field and kick calculation into a shared C
helper and call it from both element kernels. Preserve the scalar BB2D Xobject
layout, constructor defaults and pipeline behavior. Keep rigid-bunch matching,
slot-indexed arrays and update methods on
`BeamBeamBiGaussianRigidBunch2D`.

The standalone element allocates from the supplied own and opposing bunch data;
the high-level ring installer supplies all RF slots. No public reserve-capacity
argument is needed.

### Phase 3: align the Xtrack train API

Keep the train on `BeamBeamBiGaussianRigidBunch2D` and apply the bunch-pattern
API decisions:

- separate `filling_pattern_cw` / `filling_pattern_acw` from the corresponding
  `bunch_intensity_particles_*` inputs;
- expose `filled_slots_cw`, `filled_slots_acw` and `bunch_spacing_zeta`; and
- translate the common public negative-`zeta` slot convention at the kernel
  boundary if the internal matching implementation needs another convention;
- update full-slot element data in place when the filling changes.

Do not otherwise change the solver physics or iteration algorithm during this
migration.

### Phase 4: consolidate installation and configuration

Refactor the existing workflow incrementally:

1. Extract encounter generation from the current installer.
2. Extract Twiss/survey geometry and coordinate transformations.
3. Make the particles and rigid-bunch paths consume those helpers.
4. Add `mode='rigid_bunch'` to `install_beambeam_interactions(...)`.
5. Add the rigid-bunch filling, intensity and emittance inputs to
   `configure_beambeam_interactions(...)`.
6. Return `BeamBeamRigidBunchStudy` for the genuinely stateful operations.

During migration, keep `install_multibunch_beambeam(...)` as a temporary bridge
and compare its normalized output against the consolidated path. Remove it once
the examples and permanent behavioral tests use the standard workflow.

Step 1 is complete: Xfields now provides one logical encounter table containing
the IP, encounter type, signed long-range index, orientation-specific
displacement from the IP and CW/ACW bunch-pairing offsets. The particles
installer expands head-on slices from this table, while the rigid-bunch path
renders its own element names and consumes the same
placement and pairing data.

Step 2 is complete for the rigid-bunch path: Xfields now provides an
element-independent geometry helper that evaluates both Twiss tables, obtains
the standard transverse covariance with `twiss.get_beam_covariance()` and
computes local-survey separation without crossing the line seam. The
rigid-bunch configuration consumes this result.

The standard two-beam particles path now also consumes the shared result for
its explicitly oriented CW/ACW Twiss tables, closed-orbit coordinates,
transverse covariance components, local-survey separation and crossing slopes.
A curved toy-ring test compares every paired encounter against the former
MadPoint/survey calculation before checking the configured CW and
counter-rotating ACW elements. Crabbing, the final counter-rotating element
conversion, orbit-dependent kick subtraction and one-beam antisymmetry remain
mode-specific and unchanged.

Steps 4--6 are complete behind the explicit ``mode='rigid_bunch'`` selection.
In this mode installation places serializable elements with arrays covering
every RF slot. Configuration receives the two filling patterns, populations and
emittances, loads geometry and per-slot state, and returns
``BeamBeamRigidBunchStudy``. A bridge test compares the result with the
temporary all-in-one installer during migration. Permanent tests now protect
names, positions, geometry, element arrays, strength-knob response and a short
self-consistent solve. Calls without the mode continue to dispatch unchanged
to the particles workflow.

### Phase 5: migrate examples and remove the duplicate installer path

This phase is complete:

- migrate all examples to the consolidated install/configure workflow;
- remove `install_multibunch_beambeam(...)` and its environment façade;
- replace temporary bridge comparisons with permanent behavioral tests.

`BeamBeamBiGaussianRigidBunch2D` remains public, with its own serialized layout
and prebuilt-kernel entry, but its physical kick continues to use the common
BB2D helper.

### Phase 6: full validation

Run checks in increasing order of cost:

1. Focused Xfields element tests.
2. Focused Xfields installer, study and rigid-bunch-Twiss tests.
3. Existing Xfields and Xtrack beam-beam suites.
4. Serialization, xdeps knobs, prebuilt kernels and supported execution
   contexts.
5. The pytrain regression.
6. The slow Xmask tests.

## Test plan before refactoring

The xmask tests provide important end-to-end protection for the established
LHC beam-beam workflow, but they are too slow to be the main development loop.
Before changing the element or installer, add a compact test layer that runs in
seconds and isolates failures. Keep the xmask and pytrain tests as final
realistic acceptance tests.

### 1. Xfields element tests

Extend `xfields/tests/test_beambeam_rigid_bunch2d.py` with focused scalar and
rigid-bunch construction helpers.

Protect the current rigid-bunch behavior with focused tests for:

- one-bunch scalar/rigid-bunch equivalence for round and elliptical beams;
- several bunches with distinct centroids, intensities and sizes;
- coherent and incoherent covariance handling;
- an unmatched bunch receiving exactly zero kick;
- positive and negative offsets with periodic wrapping;
- unsorted updates preserving the association between `zeta`, centroids,
  populations and covariances;
- scalar size broadcasting and per-bunch sizes;
- exact allocation from the bunch data and explicit reconfiguration when the
  bunch count changes;
- `scale_strength` at `0`, an intermediate value and `1`.

Strengthen the scalar BB2D characterization tests to cover:

- modern and legacy constructors producing identical fields and kicks;
- `ref_shift_x/y` and `post_subtract_px/py`;
- nontrivial transverse covariance fields;
- copy and dictionary/JSON round trips;
- construction inside a line and xdeps control of `scale_strength`;
- the existing pipeline updater remaining functional.

Before migrating the particles installer/configurer, add a fast
characterization in `xfields/tests/test_beambeam_config_tools.py`. It must run
the public two-line install/configure workflow with sliced head-on and
long-range encounters and protect names, partner mapping, positions, delays,
Twiss covariances, nonzero orbit and separation geometry, configured 2D/3D
fields, orbit-dependent subtraction and the global strength knob. This test is
now in place; the slower Xmask LHC tests remain the acceptance layer.

Where the two paths execute the same kick formula, require agreement close to
machine precision. Use broader tolerances only when comparing genuinely
different numerical algorithms.

### 2. Xfields installation and configuration tests

Add `xfields/tests/test_rigid_bunch_beambeam.py` based on a small deterministic
two-beam environment: four or eight RF slots, two IP markers, at most one
long-range encounter per side and three populated bunches with different
intensities. Most tests should set `survey_separation=False`; use a separate
small curved lattice for the survey-sign checks.

Test the installation layout exactly:

- element names, order and number of head-on/LR encounters;
- longitudinal positions and CW/ACW suffixes;
- pairing offsets and their signs;
- `zeta_period` and matching tolerance;
- own and opposing array lengths equal to the number of RF slots;
- `beambeam_scale` references.

Test configuration and geometry:

- explicit IP offsets supplied as a mapping;
- inferred offsets supplied through an IP list;
- scalar and per-IP long-range encounter counts;
- assigned populations and covariances;
- CW/ACW transverse coordinate transformations;
- nonzero survey separation on the curved toy lattice.

Test the stateful study independently:

- `apply_filling_pattern(...)` updating full-slot arrays without replacing
  elements;
- a short symmetric two-beam solve;
- `load_solution(...)`;
- static and dynamic-beta updates;
- `second_order_maps(...)` preserving the exact beam-beam elements.

### 3. Rigid-bunch Twiss tests

Add `xfields/tests/test_rigid_bunch_twiss.py` using a `LineSegmentMap` and one
2D beam-beam lens through `BeamBeamRigidBunchStudy.twiss(...)`, without plotting or
external model data.

Cover:

- `full`, `fast` and `fast_orbit` on the same bunches;
- closed-orbit and fractional-tune agreement against `full`;
- beta, alpha, dispersion and phase from `fast` against `full`;
- `RigidBunchTwiss.cw` / `.acw` access and `BunchTwiss` integer, named-row and
  attribute access;
- bunch positions and labels derived from the study filling;
- unsupported modes, methods and kwargs;
- lost particles and closed-orbit failure handling.

The tests should explicitly document the current difference in `qx/qy`
semantics: `fast_orbit` exposes fractional tunes while `fast` exposes
accumulated tunes.

### 4. Permanent installer behavior tests

The scalar/rigid-bunch element comparisons compare each selected rigid-bunch
kick against an independently configured scalar BB2D.

For installation, exercise rigid-bunch mode through the standard
install/configure workflow, including a serialization round trip between the
two calls. Protect the normalized result through assertions covering:

- encounter names and positions;
- IP offsets and geometry;
- all active per-bunch element arrays;
- scale-knob expressions;
- per-bunch Twiss results after a small number of solver iterations.

These assertions replaced the temporary old-vs-new bridge comparison when the
duplicate installer was removed.

### 5. Test and refactoring sequence

Keep the preparatory tests and implementation changes in separate commits:

1. `Add BB2D and multibunch characterization tests`.
2. `Add toy multibunch installer and Twiss tests`.
3. `Share the BB2D kick implementation`.
4. `Infer multibunch element storage from bunch data`.
5. `Align rigid-bunch train bunch-pattern API`.
6. `Share beam-beam encounter and geometry configuration`.
7. `Add rigid-bunch mode to install/configure workflow`.
8. `Migrate examples and remove duplicate installer API`.
9. `Add final regression coverage`.

All nine original work packages are complete. Fast characterization protects the shared
encounter and geometry output, including exact comparison with the former
particles-mode survey calculation. Focused Xfields and Xtrack tests pass, as do
the LHC pytrain injection and collision regressions. The final Xmask beam-beam
test run also passes after removal of the duplicate installer path. Development
validation used the serial CPU context; OpenMP validation was intentionally left
out of scope for this work.

After moving rigid-bunch ownership into Xfields, the 16 focused Xfields
beam-beam/study/Twiss tests, the Xtrack lazy-façade test and both LHC pytrain
scenarios pass again. The recorded Xmask pass predates this package-ownership
move; Xmask has not been rerun after it.

## Pre-merge API and example review TODO

The following items were identified in the final API/example review and should
be resolved before the rigid-bunch interface is treated as established.

### Required fixes

- [x] Reconcile the covariance contract with the implementation. The
  rigid-bunch element stores slot-indexed ``Sigma_11``, ``Sigma_13`` and
  ``Sigma_33`` for both beams and adds the matched covariances component by
  component. Convenience RMS-size inputs are converted to diagonal covariance.
  ``Sigma_13`` is represented structurally but ignored by the kick; a warning
  identifies large coupling while the coupled case remains unvalidated.
- [x] Make non-convergence explicit. ``BeamBeamRigidBunchStudy.solve()`` raises
  by default after reaching ``max_iterations``. Deliberate fixed-iteration
  studies can request the last iterate with
  ``require_convergence=False``. Examples report ``converged``,
  ``num_iterations`` and ``max_orbit_change`` before using the result.
- [x] Preserve ``beambeam_scale`` during configuration. Geometry analysis
  temporarily disables the knob and restores the previous value or expression
  with exception-safe handling.
- [x] Fix ``examples/beambeam_multibunch/000_multibunch_2d.py``: the coherent
  calculation now explicitly enables ``coherent=True`` and provides the
  own-beam sizes and bunch grid.

### Public API decisions

- [x] Use one orientation vocabulary consistently. The machine-independent
  study and result APIs expose ``cw`` / ``acw`` state (for example
  ``filled_slots_cw`` and ``RigidBunchTwiss.cw``). The result container has no
  ``b1`` / ``b2`` aliases; machine-specific external formats may retain their
  native beam labels.
- [x] Use ``filling_pattern`` consistently across new public APIs,
  documentation and examples. Existing ``filling_scheme`` inputs remain
  supported as compatibility aliases in Xpart, Xfields, Xwakes and Xtrack;
  supplying both names is an error. Beam-beam retains its established
  ``filling_pattern_cw`` / ``filling_pattern_acw`` arguments.
- Avoid boolean orientation in public-looking helpers such as
  ``bb_name(base, mirror)`` and ``bunch_zeta(mirror)``. Use named CW/ACW
  accessors or an explicit orientation value, or make these helpers private.
- Clarify mode-specific configuration arguments. The shared
  ``configure_beambeam_interactions()`` signature contains both particles-only
  and rigid-bunch-only options; particle-only defaults such as
  ``crab_strong_beam=True`` are silently ignored in rigid-bunch mode. Use
  sentinel defaults and validation, or document the mode-dependent contract
  prominently.

### LHC example cleanup

- In ``000_lhc_multibunch_bb.py``, avoid configuring the complete filling and
  immediately replacing it with the bounded filling. Configure without a
  filling, derive the IP offsets, and apply the selected filling once.
- Make it obvious in the example output and introductory text that the default
  calculation uses a bounded bunch subset, controlled by ``LHC_ALL`` and
  ``LHC_WINDOW``, even though it uses the full thick lattice.
- Do not unconditionally write fixed pickle files into the source directory.
  Make result export explicit and direct outputs to a user-selected directory.
- Remove the unused ``line_b1`` / ``line_b2`` return values in the example, or
  simplify ``load_lhc()`` if callers generally use the lines through the
  environment.
- Keep the full-lattice LHC script as a realistic application example, but use
  the small deterministic example as the primary API introduction. Add a short
  README that distinguishes the quick API example, full thick-lattice study,
  second-order-map workflow, OpenMP example and pytrain comparison.

### Final validation gates

- Run the focused serial and OpenMP element, configuration, study and
  rigid-bunch-Twiss tests with the supported compiler setup.
- Run the current LHC injection and collision pytrain regressions.
- Rerun the Xmask beam-beam acceptance tests after the package-ownership move;
  the pass recorded above predates that change.
- Exercise at least one complete OpenMP rigid-bunch study, not only the element
  kernels.

## Non-goals

- This rationalization does not change the intended coherent rigid-bunch
  physics.
- It does not make the existing pipeline strong-strong updater multibunch
  aware.
- It does not make rigid-bunch Twiss a generic multibunch Xtrack facility.
