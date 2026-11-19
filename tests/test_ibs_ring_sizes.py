import numpy as np
import pytest
import xtrack as xt
from numpy.testing import assert_allclose
from xobjects.test_helpers import for_all_test_contexts

from xfields.ibs import IBSKineticKick
from xfields.ibs._analytical import NagaitsevIBS


def _optics(constant_sizes=False):
    """Small ring-optics table with unequal segment lengths and beam sizes."""
    tw = xt.TwissTable(
        dict(
            name=np.array(["a", "b", "c", "_end_point"]),
            s=np.array([0.0, 10.0, 30.0, 60.0]),
            betx=np.array([10.0, 40.0, 20.0, 10.0]),
            bety=np.array([20.0, 40.0, 10.0, 20.0]),
            alfx=np.array([0.2, -0.3, 0.1, 0.2]),
            alfy=np.array([-0.1, 0.2, 0.3, -0.1]),
            dx=np.array([0.4, 0.0, 1.0, 0.4]),
            dy=np.zeros(4),
            dpx=np.array([0.02, 0.0, -0.01, 0.02]),
            dpy=np.zeros(4),
        )
    )
    if constant_sizes:
        for name in ("betx", "bety", "alfx", "alfy", "dx", "dpx"):
            tw[name][:] = tw[name][0]
    particle = xt.Particles(p0c=20e9, mass0=xt.PROTON_MASS_EV)
    tw["particle_on_co"] = particle
    tw["beta0"] = particle.beta0[0]
    tw["gamma0"] = particle.gamma0[0]
    tw["line_length"] = 60.0
    return tw


def _matched_particles(tw, name, context):
    # Exact unit covariance removes sampling differences between locations.
    normal = np.random.default_rng(2026).normal(size=(2000, 6))
    normal -= normal.mean(axis=0)
    normal, _ = np.linalg.qr(normal)
    x, px, y, py, delta, zeta = (normal * np.sqrt(len(normal))).T
    ex, ey, sigd, sigz = 2e-6, 1e-6, 1e-3, 0.1
    delta *= sigd
    bx, by = tw["betx", name], tw["bety", name]
    return xt.Particles(
        _context=context,
        p0c=20e9,
        weight=1e11 / len(normal),
        x=np.sqrt(ex * bx) * x + tw["dx", name] * delta,
        px=np.sqrt(ex / bx) * (px - tw["alfx", name] * x) + tw["dpx", name] * delta,
        y=np.sqrt(ey * by) * y + tw["dy", name] * delta,
        py=np.sqrt(ey / by) * (py - tw["alfy", name] * y) + tw["dpy", name] * delta,
        delta=delta,
        zeta=sigz * zeta,
    )


@for_all_test_contexts(excluding="ContextPyopencl")
@pytest.mark.parametrize("constant_sizes", [False, True])
def test_kinetic_coefficients_use_ring_beam_sizes(test_context, constant_sizes):
    tw = _optics(constant_sizes)
    rates = (
        NagaitsevIBS(tw)
        .growth_rates(
            gemitt_x=2e-6,
            gemitt_y=1e-6,
            sigma_delta=1e-3,
            bunch_length=0.1,
            total_beam_intensity=1e11,
        )
        .to_emittance_growth_rates()
        .as_tuple()
    )
    coefficients = []
    differences = []
    for name in ("a", "b", "c"):
        particles = _matched_particles(tw, name, test_context)
        kick = IBSKineticKick(num_slices=20)
        kick._twiss, kick._name = tw, name
        diffusion, friction = kick.compute_kinetic_coefficients(particles)
        diffusion, friction = (
            np.array(diffusion.as_tuple()),
            np.array(friction.as_tuple()),
        )
        coefficients.append(np.concatenate([diffusion, friction]))
        # Independent analytical implementation: D - F must recover the
        # Nagaitsev emittance-rate coefficients, including negative rates.
        differences.append(diffusion - friction)
    for name, values in zip(("b", "c"), coefficients[1:]):
        print(f"{name}/a coefficient ratios: {values / coefficients[0]}")
        assert_allclose(values, coefficients[0], rtol=2e-12, atol=0)
    for difference in differences:
        assert_allclose(difference, rates, rtol=2e-12, atol=1e-18)
