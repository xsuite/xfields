# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from __future__ import annotations  # important for sphinx to alias ArrayLike

import logging

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


def _bunch_length(particles: xt.Particles) -> float:
    """Get the bunch length from the particles."""
    nplike = particles._context.nplike_lib
    return nplike.std(particles.zeta[particles.state > 0])


def _sigma_delta(particles: xt.Particles) -> float:
    """Get the standard deviation of the momentum spread from the particles."""
    nplike = particles._context.nplike_lib
    return nplike.std(particles.delta[particles.state > 0])


def _sigma_x(particles: xt.Particles) -> float:
    """Get the horizontal coordinate standard deviation from the particles."""
    nplike = particles._context.nplike_lib
    return nplike.std(particles.x[particles.state > 0])


def _sigma_y(particles: xt.Particles) -> float:
    """Get the vertical coordinate standard deviation from the particles."""
    nplike = particles._context.nplike_lib
    return nplike.std(particles.y[particles.state > 0])


def _geom_epsx(particles: xt.Particles, betx: float, dx: float) -> float:
    """
    Horizontal geometric emittance at a location in the machine, for the
    beta and dispersion functions at this location.
    """
    sigma_x = _sigma_x(particles)
    sig_delta = _sigma_delta(particles)
    return (sigma_x**2 - (dx * sig_delta) ** 2) / betx


def _geom_epsy(particles: xt.Particles, bety: float, dy: float) -> float:
    """
    Vertical geometric emittance at a location in the machine, for the
    beta and dispersion functions at this location.
    """
    sigma_y = _sigma_y(particles)
    sig_delta = _sigma_delta(particles)
    return (sigma_y**2 - (dy * sig_delta) ** 2) / bety
