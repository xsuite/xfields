# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from __future__ import annotations  # important for sphinx to alias ArrayLike

import logging
from typing import TYPE_CHECKING

import xobjects as xo

if TYPE_CHECKING:
    import xtrack as xt
    from numpy.typing import ArrayLike

LOGGER = logging.getLogger(__name__)


def phi(beta: ArrayLike, alpha: ArrayLike, dx: ArrayLike, dpx: ArrayLike) -> ArrayLike:
    """
    Computes the ``Phi`` parameter of Eq (15) in
    :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`.

    Parameters
    ----------
    beta : ArrayLike
        Beta-function through the machine (chosen plane).
    alpha : ArrayLike
        Alpha-function through the machine (chosen plane).
    dxy : ArrayLike
        Dispersion function through the machine (chosen plane).
    dpxy : ArrayLike
        Dispersion of p[xy] function through the machine (chosen plane).

    Returns
    -------
    phi : ArrayLike
        The ``Phi`` values through the machine.
    """
    return dpx + alpha * dx / beta


# ----- Some helpers on xtrack.Particles objects ----- #


def _beam_intensity(particles: xt.Particles) -> float:
    """Get the beam intensity from the particles."""
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    return float(nplike.sum(particles.weight[particles.state > 0]))


def _bunch_length(particles: xt.Particles) -> float:
    """Get the bunch length from the particles."""
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    return float(nplike.std(particles.zeta[particles.state > 0]))


def _sigma_delta(particles: xt.Particles) -> float:
    """
    Get the standard deviation of the momentum spread
    from the particles.
    """
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    return float(nplike.std(particles.delta[particles.state > 0]))


def _sigma_x(particles: xt.Particles) -> float:
    """
    Get the horizontal coordinate standard deviation
    from the particles.
    """
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    return float(nplike.std(particles.x[particles.state > 0]))


def _sigma_y(particles: xt.Particles) -> float:
    """
    Get the vertical coordinate standard deviation
    from the particles.
    """
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    return float(nplike.std(particles.y[particles.state > 0]))


def _gemitt_x(particles: xt.Particles, betx: float, dx: float) -> float:
    """
    Horizontal geometric emittance at a location in the machine,
    for the beta and dispersion functions at this location.
    """
    # Context check is performed in the called functions
    sigma_x = _sigma_x(particles)
    sig_delta = _sigma_delta(particles)
    return float((sigma_x**2 - (dx * sig_delta) ** 2) / betx)


def _gemitt_y(particles: xt.Particles, bety: float, dy: float) -> float:
    """
    Vertical geometric emittance at a location in the machine,
    for the beta and dispersion functions at this location.
    """
    # Context check is performed in the called functions
    sigma_y = _sigma_y(particles)
    sig_delta = _sigma_delta(particles)
    return float((sigma_y**2 - (dy * sig_delta) ** 2) / bety)


def _current_turn(particles: xt.Particles) -> int:
    """
    Get the current tracking turn from one of
    the alive particles.
    """
    _assert_accepted_context(particles._context)
    return int(particles.at_turn[particles.state > 0][0])


def _sigma_px(particles: xt.Particles, dpx: float = 0) -> float:
    """
    Get the horizontal momentum standard deviation from
    the particles. The momentum dispersion can be provided
    to be taken out of the calculation (as we use the stdev
    of px, calling this function at a location with high dpx
    would skew the result).

    Parameters
    ----------
    particles : xt.Particles
        The particles object.
    dpx : float, optional
        Horizontal momentum dispersion function at the location
        where the sigma_px is computed. Defaults to 0.

    Returns
    -------
    sigma_px : float
        The standard deviation of the horizontal momentum.
    """
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    px: ArrayLike = particles.px[particles.state > 0]
    delta: ArrayLike = particles.delta[particles.state > 0]
    return float(nplike.std(px - dpx * delta))


def _sigma_py(particles: xt.Particles, dpy: float = 0) -> float:
    """
    Get the vertical momentum standard deviation from
    the particles. The momentum dispersion can be provided
    to be taken out of the calculation (as we use the stdev
    of py, calling this function at a location with high dpy
    would skew the result).

    Parameters
    ----------
    particles : xt.Particles
        The particles object.
    dpy : float, optional
        Vertical momentum dispersion function at the location
        where the sigma_py is computed. Defaults to 0.

    Returns
    -------
    sigma_py : float
        The standard deviation of the vertical momentum.
    """
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    py: ArrayLike = particles.py[particles.state > 0]
    delta: ArrayLike = particles.delta[particles.state > 0]
    return float(nplike.std(py - dpy * delta))


def _mean_px(particles: xt.Particles, dpx: float = 0) -> float:
    """
    Get the arithmetic mean of the horizontal momentum from
    the particles. The momentum dispersion can be provided to
    be taken out of the calculation (as we use the mean of
    px, calling this function at a location with high dpx
    would skew the result).

    Parameters
    ----------
    particles : xt.Particles
        The particles object.
    dpx : float, optional
        Horizontal momentum dispersion function at the location
        where the mean_px is computed. Defaults to 0.

    Returns
    -------
    mean_px : float
        The arithmetic mean of the horizontal momentum.
    """
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    px: ArrayLike = particles.px[particles.state > 0]
    delta: ArrayLike = particles.delta[particles.state > 0]
    return float(nplike.mean(px - dpx * delta))


def _mean_py(particles: xt.Particles, dpy: float = 0) -> float:
    """
    Get the arithmetic mean of the vertical momentum from
    the particles. The momentum dispersion can be provided to
    be taken out of the calculation (as we use the mean of
    py, calling this function at a location with high dpy
    would skew the result).

    Parameters
    ----------
    particles : xt.Particles
        The particles object.
    dpy : float, optional
        Vertical momentum dispersion function at the location
        where the mean_py is computed. Defaults to 0.

    Returns
    -------
    mean_py : float
        The arithmetic mean of the horizontal momentum.
    """
    _assert_accepted_context(particles._context)
    nplike = particles._context.nplike_lib
    py: ArrayLike = particles.py[particles.state > 0]
    delta: ArrayLike = particles.delta[particles.state > 0]
    return float(nplike.mean(py - dpy * delta))


# ----- Private helper to check the validity of the context ----- #


def _assert_accepted_context(ctx: xo.context.XContext):
    """
    Ensure the context is accepted for IBS computations. We do not
    support PyOpenCL because they have no booleans and lead to some
    wrong results when using boolean array masking, which we do to
    get the alive particles.
    """
    assert not isinstance(ctx, xo.ContextPyopencl), (
        "PyOpenCL context is not supported for IBS. " "Please use either the CPU or CuPy context."
    )
