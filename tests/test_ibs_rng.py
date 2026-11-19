import numpy as np
import pytest
import xtrack as xt
from numpy.testing import assert_array_equal
from xobjects.test_helpers import fix_random_seed, for_all_test_contexts

from xfields.ibs import IBSAnalyticalKick, IBSKineticKick
from xfields.ibs._kicks import (
    DiffusionCoefficients,
    FrictionCoefficients,
    IBSKickCoefficients,
)


def _kick(kind):
    if kind == "analytical":
        kick = IBSAnalyticalKick("Nagaitsev", 20)
        kick.kick_coefficients = IBSKickCoefficients(1e-5, 2e-5, 1e-4)
    else:
        kick = IBSKineticKick(20)
        kick.diffusion_coefficients = DiffusionCoefficients(1.0, 2.0, 3.0)
        kick.friction_coefficients = FrictionCoefficients(0.2, 0.3, 0.4)
    kick._twiss = xt.TwissTable(
        dict(
            name=np.array(["ibs"]),
            dx=np.array([0.0]),
            dy=np.array([0.0]),
            dpx=np.array([0.0]),
            dpy=np.array([0.0]),
            alfx=np.array([0.0]),
            alfy=np.array([0.0]),
        )
    )
    kick._twiss["t_rev0"] = 1e-3
    kick._name = "ibs"
    kick.update_every = 100
    kick._scale_strength = 1
    return kick


def _particles(context):
    return xt.Particles(
        _context=context,
        p0c=1e9,
        at_turn=1,
        px=np.random.normal(0, 1e-4, 1001),
        py=np.random.normal(0, 1e-4, 1001),
        delta=np.random.normal(0, 1e-3, 1001),
        zeta=np.random.normal(0, 0.1, 1001),
    )


def _assert_same_bunch(first, second, context):
    for coordinate in ("px", "py", "delta"):
        assert_array_equal(
            context.nparray_from_context_array(getattr(first, coordinate)),
            context.nparray_from_context_array(getattr(second, coordinate)),
        )


def _assert_same_rng_state(first, second):
    assert first[0] == second[0]
    assert_array_equal(first[1], second[1])
    assert first[2:] == second[2:]


@for_all_test_contexts(excluding="ContextPyopencl")
@pytest.mark.parametrize("kind", ["analytical", "kinetic"])
@fix_random_seed(2026)
def test_numpy_seed_repeats_and_advances_kicks(test_context, kind):
    initial = _particles(test_context)
    bunches = []
    for seed in (42, 42, 43):
        particles = initial.copy()
        kick = _kick(kind)
        np.random.seed(seed)
        for _ in range(3):
            kick.track(particles)
        bunches.append(particles)
    _assert_same_bunch(bunches[0], bunches[1], test_context)
    to_numpy = test_context.nparray_from_context_array
    print(
        f"{kind}: max delta difference with repeated NumPy seed = "
        f"{np.max(np.abs(to_numpy(bunches[0].delta) - to_numpy(bunches[1].delta))):.9e}"
    )
    for coordinate in ("px", "py", "delta"):
        assert not np.array_equal(
            to_numpy(getattr(bunches[0], coordinate)),
            to_numpy(getattr(bunches[2], coordinate)),
        )

    # Successive calls without reseeding must consume fresh random draws.
    kick = _kick(kind)
    first, second = initial.copy(), initial.copy()
    kick.track(first)
    kick.track(second)
    assert not np.array_equal(to_numpy(first.delta), to_numpy(second.delta))


@for_all_test_contexts(excluding="ContextPyopencl")
@pytest.mark.parametrize("kind", ["analytical", "kinetic"])
@fix_random_seed(123)
def test_numpy_rng_checkpoint(test_context, kind):
    kick = _kick(kind)
    particles = _particles(test_context)
    kick.track(particles)
    saved_particles = particles.copy()
    saved_rng_state = np.random.get_state()
    for _ in range(3):
        kick.track(particles)
    final_rng_state = np.random.get_state()

    resumed = _kick(kind)
    np.random.set_state(saved_rng_state)
    for _ in range(3):
        resumed.track(saved_particles)
    _assert_same_bunch(particles, saved_particles, test_context)
    _assert_same_rng_state(final_rng_state, np.random.get_state())


@for_all_test_contexts(excluding="ContextPyopencl")
@pytest.mark.parametrize("kind", ["analytical", "kinetic"])
@fix_random_seed(2026)
def test_fix_random_seed_replays_kicks(test_context, kind):
    initial = _particles(test_context)
    results = []

    @fix_random_seed(42)
    def track():
        particles = initial.copy()
        kick = _kick(kind)
        for _ in range(3):
            kick.track(particles)
        results.append(particles)

    before = np.random.get_state()
    track()
    track()
    _assert_same_bunch(results[0], results[1], test_context)
    _assert_same_rng_state(before, np.random.get_state())


@for_all_test_contexts(excluding="ContextPyopencl")
@pytest.mark.parametrize("kind", ["analytical", "kinetic"])
@fix_random_seed(2026)
def test_disabled_kick_preserves_rng_state(test_context, kind):
    kick = _kick(kind)
    kick._scale_strength = 0
    particles = _particles(test_context)
    before = np.random.get_state()
    kick.track(particles)
    _assert_same_rng_state(before, np.random.get_state())
