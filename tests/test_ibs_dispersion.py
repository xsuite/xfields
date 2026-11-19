import numpy as np
import pytest
import xtrack as xt
from numpy.testing import assert_allclose, assert_array_equal
from xobjects.test_helpers import fix_random_seed, for_all_test_contexts

from xfields.ibs import IBSAnalyticalKick, IBSKineticKick
from xfields.ibs._kicks import (
    DiffusionCoefficients,
    FrictionCoefficients,
    IBSKickCoefficients,
)


def _kick(kind, dispersion, diffusion=True, friction=True):
    if kind == "analytical":
        kick = IBSAnalyticalKick("Nagaitsev", 20)
        kick.kick_coefficients = IBSKickCoefficients(1e-5, 2e-5, 1e-4)
    else:
        kick = IBSKineticKick(20)
        kick.diffusion_coefficients = DiffusionCoefficients(
            *([1.0, 2.0, 3.0] if diffusion else [0.0, 0.0, 0.0])
        )
        kick.friction_coefficients = FrictionCoefficients(
            *([0.2, 0.3, 0.4] if friction else [0.0, 0.0, 0.0])
        )
    data = dict(name=np.array(["ibs"]), alfx=np.array([0.7]), alfy=np.array([-0.4]))
    data.update(
        {
            key: np.array([value])
            for key, value in zip(("dx", "dpx", "dy", "dpy"), dispersion)
        }
    )
    kick._twiss = xt.TwissTable(data)
    kick._twiss["t_rev0"] = 1e-3
    kick._name = "ibs"
    kick._scale_strength = 1
    kick.update_every = 100
    return kick


def _particles(context, dispersion):
    rng = np.random.default_rng(2026)
    delta = rng.normal(2e-4, 1e-3, 1000)
    data = {
        coord: rng.normal(1e-5, 1e-4, 1000) + value * delta
        for coord, value in zip(("x", "px", "y", "py"), dispersion)
    }
    return xt.Particles(
        _context=context,
        p0c=1e9,
        at_turn=1,
        delta=delta,
        zeta=rng.normal(0, 0.1, 1000),
        state=[1] * 999 + [0],
        **data,
    )


@for_all_test_contexts(excluding="ContextPyopencl")
@pytest.mark.parametrize(
    "kind,diffusion,friction",
    [
        ("analytical", True, False),
        ("kinetic", True, True),
        ("kinetic", True, False),
        ("kinetic", False, True),
    ],
)
@pytest.mark.parametrize(
    "dispersion",
    [
        (1.2, 0.3, -0.4, -0.2),
        (0.0, 0.3, 0.0, -0.2),
        (1.2, 0.0, -0.4, 0.0),
    ],
)
@fix_random_seed(42)
def test_kicks_in_betatron_coordinates(
    test_context, kind, diffusion, friction, dispersion, monkeypatch
):
    """Same betatron bunch and frozen rates must give the same kicks anywhere.

    Include a momentum offset, nonzero alpha, both dispersion planes, and a
    lost particle. Freeze the rates to isolate application from issue #1006.
    """
    zero = (0.0, 0.0, 0.0, 0.0)
    reference = _particles(test_context, zero)
    displaced = _particles(test_context, dispersion)
    before = displaced.copy()
    random = test_context.nplike_lib.random
    seeded_rng = random.default_rng
    for particles, optics in ((reference, zero), (displaced, dispersion)):
        # Seed the shared NumPy stream as well, for use with the RNG fix.
        np.random.seed(42)
        rng = seeded_rng(42)
        with monkeypatch.context() as patch:
            patch.setattr(random, "default_rng", lambda: rng)
            _kick(kind, optics, diffusion, friction).track(particles)

    to_numpy = test_context.nparray_from_context_array
    delta = to_numpy(displaced.delta)
    assert_allclose(delta, to_numpy(reference.delta), rtol=0, atol=1e-18)
    alive = to_numpy(displaced.state) > 0
    for coordinate, value in zip(("x", "px", "y", "py"), dispersion):
        actual = to_numpy(getattr(displaced, coordinate))
        expected = to_numpy(getattr(reference, coordinate))
        assert_allclose(
            (actual - value * delta)[alive], expected[alive], rtol=0, atol=2e-18
        )
        assert_array_equal(
            actual[~alive], to_numpy(getattr(before, coordinate))[~alive]
        )
    assert_array_equal(delta[~alive], to_numpy(before.delta)[~alive])


@fix_random_seed(42)
def test_longitudinal_diffusion_does_not_add_horizontal_action(monkeypatch):
    """Reproduce the spurious H_x * <Delta delta**2> / 2 contribution."""
    beta, alpha, dx, dpx = 12.0, 0.7, 1.2, 0.3
    particles = _particles(None, (0.0, 0.0, 0.0, 0.0))
    particles.x[:] = dx * particles.delta
    particles.px[:] = dpx * particles.delta
    before = particles.delta.copy()
    kick = _kick("analytical", (dx, dpx, 0.0, 0.0))
    kick.kick_coefficients = IBSKickCoefficients(0.0, 0.0, 1e-4)
    rng = np.random.default_rng(42)
    monkeypatch.setattr(np.random, "default_rng", lambda: rng)
    kick.track(particles)
    active = particles.state > 0
    xb = (particles.x - dx * particles.delta)[active]
    pb = (particles.px - dpx * particles.delta)[active]
    gamma = (1 + alpha**2) / beta
    action = np.mean((gamma * xb**2 + 2 * alpha * xb * pb + beta * pb**2) / 2)
    Hx = gamma * dx**2 + 2 * alpha * dx * dpx + beta * dpx**2
    old_extra_action = Hx * np.mean((particles.delta - before)[active] ** 2) / 2
    print(
        f"Uncorrected extra action: {old_extra_action:.9e} m; corrected: {action:.9e} m"
    )
    assert old_extra_action > 1e-9
    assert action < 1e-30
