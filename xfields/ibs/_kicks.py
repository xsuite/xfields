# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from __future__ import annotations  # important for sphinx to alias ArrayLike

from logging import getLogger

import numpy as np
import xobjects as xo
import xtrack as xt
from numpy.typing import ArrayLike

from xfields.ibs._analytical import AnalyticalIBS, BjorkenMtingwaIBS, IBSGrowthRates, NagaitsevIBS
from xfields.ibs._inputs import BeamParameters, OpticsParameters

LOGGER = getLogger(__name__)

# ----- Dataclasses-like as xo.HybridClass objects ----- #


class DiffusionCoefficients(xo.HybridClass):
    """Container dataclass for kinetic IBS diffusion coefficients.

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


class FrictionCoefficients(xo.HybridClass):
    """Container dataclass for kinetic IBS friction coefficients.

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


class IBSKickCoefficients(xo.HybridClass):
    """Container dataclass for IBS kick coefficients.

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


# ----- Useful Functions ----- #


# TODO: someday replace this with what Gianni has in xfields.longitudinal_profiles.qgaussian
def line_density(particles: xt.Particles, n_slices: int) -> ArrayLike:
    r"""
    Returns the longitudinal "line density" of the `Particles` object. It is used as a
    weighting factor for the application of IBS kicks: particles in the denser parts of
    the bunch will receive a larger kick, and vice versa.

    Parameters
    ----------
    particles : xtrack.Particles
        The xtrack.Particles object to compute the line density for.
    n_slices : int
        The number of slices to use for the computation of the bins.

    Returns:
        An array with the density values for each slice / bin of the xtrack.Particles object.
    """
    # ----------------------------------------------------------------------------------------------
    # Determine properties from longitudinal particles distribution: cuts, slice width, bunch length
    LOGGER.debug("Determining longitudinal particles distribution properties")
    zeta: np.ndarray = particles.zeta[particles.state > 0]  # careful to only consider active particles
    z_cut_head: float = np.max(zeta)  # z cut at front of bunch
    z_cut_tail: float = np.min(zeta)  # z cut at back of bunch
    slice_width: float = (z_cut_head - z_cut_tail) / n_slices  # slice width
    # ----------------------------------------------------------------------------------------------
    # Determine bin edges and bin centers for the distribution
    LOGGER.debug("Determining bin edges and bin centers for the distribution")
    bin_edges = np.linspace(
        z_cut_tail - 1e-7 * slice_width,
        z_cut_head + 1e-7 * slice_width,
        num=n_slices + 1,
        dtype=np.float64,
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    # ----------------------------------------------------------------------------------------------
    # Compute histogram on longitudinal distribution then compute and return line density
    counts_normed, bin_edges = np.histogram(zeta, bin_edges, density=True)  # density=True to normalize
    return np.interp(zeta, bin_centers, counts_normed)


# ----- Dataclasses-like as xo.HybridClass objects ----- #


# TODO: make the element aware of the tracking turn number so it can decide whether to recompute the growth rates or not
# Use the particle's .at_turn for this


class IBSSimpleKick(xt.BeamElement):
    """"""

    _xofields = {
        "beam_parameters": BeamParameters,
        "optics": OpticsParameters,
        "kick_coefficients": IBSKickCoefficients,
    }

    def __init__(self, beam_params: BeamParameters, optics_params: OpticsParameters, n_slices) -> None:
        # Check that we are above transition and raise and error if not (not applicable)
        if optics_params.slip_factor <= 0:  # below transition
            raise NotImplementedError(
                "SimpleKickIBS is not compatible with machines operating below transition. "
                "Please see the documentation and use the kinetic formalism with KineticKickIBS instead."
            )

        self.xoinitialize(beam_parameters=beam_params, optics=optics_params, kick_coefficients=None)

        # Analytical implementation for growth rates calculation, can be overridden by the user
        if np.count_nonzero(self.optics.dy) != 0:
            LOGGER.debug("Vertical dispersion detected, using Bjorken & Mtingwa formalism")
            self._analytical_ibs: AnalyticalIBS = BjorkenMtingwaIBS(beam_params, optics_params)
        else:
            LOGGER.debug("No vertical dispersion detected, using Nagaitsev formalism")
            self._analytical_ibs: AnalyticalIBS = NagaitsevIBS(beam_params, optics_params)
        LOGGER.debug("Override this manually by setting the self.analytical_ibs attribute")

    def to_dict(self):
        """Raises an error as the line should be saved without the IBS kick element."""
        raise NotImplementedError("IBS kick elements should not be saved as part of the line")

    @property
    def analytical_ibs(self) -> AnalyticalIBS:
        """The analytical IBS implementation used for growth rates calculation."""
        return self._analytical_ibs

    @analytical_ibs.setter
    def analytical_ibs(self, value: AnalyticalIBS) -> None:
        """The analytical_ibs has a setter so that .beam_params and .optics are updated when it is set."""
        # fmt: off
        LOGGER.debug("Overwriting analytical ibs implementation used for growth rates calculation")
        self._analytical_ibs: AnalyticalIBS = value
        LOGGER.debug("Re-pointing instance's beam & optics params to that of the new analytical")
        self.beam_parameters = self.analytical_ibs.beam_parameters
        self.optics = self.analytical_ibs.optics
        # fmt: on

    def compute_kick_coefficients(
        self, particles: "xpart.Particles", **kwargs  # noqa: F821
    ) -> IBSKickCoefficients:
        """"""
        pass

    def track(self, particles: xt.Particles) -> None:
        """"""
        pass


class IBSKineticKick(xt.BeamElement):
    """"""

    _xofields = {
        "beam_parameters": BeamParameters,
        "optics": OpticsParameters,
        "kick_coefficients": IBSKickCoefficients,
    }

    def __init__(self, beam_params: BeamParameters, optics_params: OpticsParameters) -> None:
        self.xoinitialize(beam_parameters=beam_params, optics=optics_params, kick_coefficients=None)

    def to_dict(self):
        """Raises an error as the line should be saved without the IBS kick element."""
        raise NotImplementedError("IBS kick elements should not be saved as part of the line")

    def compute_kick_coefficients(
        self, particles: "xpart.Particles", **kwargs  # noqa: F821
    ) -> IBSKickCoefficients:
        """"""
        pass

    def track(self, particles: xt.Particles) -> None:
        pass
