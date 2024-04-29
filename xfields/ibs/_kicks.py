# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from __future__ import annotations  # important for sphinx to alias ArrayLike

from logging import getLogger
from typing import Tuple

import numpy as np
import xobjects as xo
import xtrack as xt
from numpy.typing import ArrayLike
from scipy.constants import c
from scipy.special import elliprd

from xfields.ibs._analytical import BjorkenMtingwaIBS, IBSGrowthRates
from xfields.ibs._formulary import (
    _assert_accepted_context,
    _beam_intensity,
    _bunch_length,
    _current_turn,
    _gemitt_x,
    _gemitt_y,
    _sigma_delta,
    _sigma_px,
    _sigma_py,
    _sigma_x,
    _sigma_y,
    phi,
)

LOGGER = getLogger(__name__)


# ----- Some classes to store results (as xo.HybridClass) ----- #


class DiffusionCoefficients(xo.HybridClass):
    """
    Holds the diffusion coefficients, named ``Dx``,
    ``Dy``, and ``Dz``, which are computed in the
    kinetic kick formalism.

    Attributes:
    -----------
    Dx : float
        Horizontal diffusion coefficient.
    Dy : float
        Vertical diffusion coefficient.
    Dz : float
        Longitudinal diffusion coefficient.
    """

    _xofields = {
        "Dx": xo.Float64,
        "Dy": xo.Float64,
        "Dz": xo.Float64,
    }

    def __init__(self, Dx: float, Dy: float, Dz: float) -> None:
        """Init by providing the diffusion coefficients."""
        self.xoinitialize(Dx=Dx, Dy=Dy, Dz=Dz)

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return the growth rates as a tuple."""
        return float(self.Dx), float(self.Dy), float(self.Dz)


class FrictionCoefficients(xo.HybridClass):
    """
    Holds the friction coefficients, named ``Fx``,
    ``Fy``, and ``Fz``, which are computed in the
    kinetic kick formalism.

    Attributes:
    -----------
    Fx : float
        Horizontal friction coefficient.
    Fy : float
        Vertical friction coefficient.
    Fz : float
        Longitudinal friction coefficient.
    """

    _xofields = {
        "Fx": xo.Float64,
        "Fy": xo.Float64,
        "Fz": xo.Float64,
    }

    def __init__(self, Fx: float, Fy: float, Fz: float) -> None:
        """Init by providing the friction coefficients."""
        self.xoinitialize(Fx=Fx, Fy=Fy, Fz=Fz)

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return the growth rates as a tuple."""
        return float(self.Fx), float(self.Fy), float(self.Fz)


class IBSKickCoefficients(xo.HybridClass):
    """
    Holds the kick coefficients, named ``Kx``,
    ``Ky``, and ``Kz``, which are used in order
    to determine the applied momenta kicks.

    Attributes:
    -----------
    Kx : float
        Horizontal kick coefficient.
    Ky : float
        Vertical kick coefficient.
    Kz : float
        Longitudinal kick coefficient.
    """

    _xofields = {
        "Kx": xo.Float64,
        "Ky": xo.Float64,
        "Kz": xo.Float64,
    }

    def __init__(self, Kx: float, Ky: float, Kz: float) -> None:
        """Init by providing the kick coefficients."""
        self.xoinitialize(Kx=Kx, Ky=Ky, Kz=Kz)

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return the growth rates as a tuple."""
        return float(self.Kx), float(self.Ky), float(self.Kz)
