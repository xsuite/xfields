from copy import deepcopy

import numpy as np
import pytest
import xtrack as xt
from numpy.testing import assert_array_equal
from xobjects.test_helpers import for_all_test_contexts

from xfields.ibs import IBSAnalyticalKick, IBSKineticKick
from xfields.ibs._kicks import (
    DiffusionCoefficients,
    FrictionCoefficients,
    IBSKickCoefficients,
)


def _kick(kind, rng=None):
    kwargs = {} if rng is None else {"rng": rng}
    if kind == "analytical":
        kick = IBSAnalyticalKick("Nagaitsev", 20, **kwargs)
        kick.kick_coefficients = IBSKickCoefficients(1e-5, 2e-5, 1e-4)
    else:
        kick = IBSKineticKick(20, **kwargs)
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
    rng = np.random.default_rng(2026)
    return xt.Particles(
        _context=context,
        p0c=1e9,
        at_turn=1,
        px=rng.normal(0, 1e-4, 1000),
        py=rng.normal(0, 1e-4, 1000),
        delta=rng.normal(0, 1e-3, 1000),
        zeta=rng.normal(0, 0.1, 1000),
    )


@for_all_test_contexts(excluding="ContextPyopencl")
@pytest.mark.parametrize("kind", ["analytical", "kinetic"])
def test_seeded_kicks_repeat_and_advance(test_context, kind):
    random = test_context.nplike_lib.random
    kicks = [_kick(kind, random.default_rng(seed)) for seed in (42, 42, 43)]
    initial = _particles(test_context)
    bunches = [initial.copy() for _ in kicks]
    for _ in range(3):
        for kick, particles in zip(kicks, bunches):
            kick.track(particles)
        for coord in ("px", "py", "delta"):
            arrays = [
                test_context.nparray_from_context_array(getattr(p, coord))
                for p in bunches
            ]
            assert_array_equal(arrays[0], arrays[1])
            assert not np.array_equal(arrays[0], arrays[2])

    # Reusing a generator must advance its stream, even for identical input.
    first, second = initial.copy(), initial.copy()
    kicks[0].track(first)
    kicks[0].track(second)
    assert not np.array_equal(
        test_context.nparray_from_context_array(first.delta),
        test_context.nparray_from_context_array(second.delta),
    )


@pytest.mark.parametrize("kind", ["analytical", "kinetic"])
def test_numpy_rng_checkpoint(kind):
    kick = _kick(kind, np.random.default_rng(123))
    particles = _particles(None)
    kick.track(particles)
    saved_particles = particles.copy()
    saved_rng_state = deepcopy(kick.rng.bit_generator.state)
    for _ in range(3):
        kick.track(particles)

    resumed = _kick(kind, np.random.default_rng())
    resumed.rng.bit_generator.state = saved_rng_state
    for _ in range(3):
        resumed.track(saved_particles)
    for coord in ("px", "py", "delta"):
        assert_array_equal(getattr(particles, coord), getattr(saved_particles, coord))


@pytest.mark.parametrize("kind", ["analytical", "kinetic"])
def test_replacing_rng_replays_kicks(kind):
    kick = _kick(kind)
    first = _particles(None)
    second = first.copy()
    kick.rng = np.random.default_rng(42)
    kick.track(first)
    kick.rng = np.random.default_rng(42)
    kick.track(second)
    print(
        f"{kind}: max delta difference with reset RNG = "
        f"{np.max(np.abs(first.delta - second.delta)):.9e}"
    )
    for coordinate in ("px", "py", "delta"):
        assert_array_equal(getattr(first, coordinate), getattr(second, coordinate))


@for_all_test_contexts(excluding="ContextPyopencl")
@pytest.mark.parametrize("kind", ["analytical", "kinetic"])
def test_default_rng_is_lazy_and_persistent(test_context, kind, monkeypatch):
    kick = _kick(kind)
    particles = _particles(test_context)
    kick._scale_strength = 0
    kick.track(particles)
    assert kick.rng is None
    kick._scale_strength = 1
    kick.track(particles)
    rng = kick.rng
    assert rng is not None

    def unexpected_new_generator():
        pytest.fail("Each kick element must reuse its generator")

    monkeypatch.setattr(
        test_context.nplike_lib.random, "default_rng", unexpected_new_generator
    )
    kick.track(particles)
    assert kick.rng is rng
