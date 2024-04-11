# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from __future__ import annotations  # important for sphinx to alias ArrayLike

from abc import ABC, abstractmethod
from logging import getLogger
from typing import Tuple

import numpy as np
import xobjects as xo
import xtrack as xt
from numpy.typing import ArrayLike
from scipy.constants import c
from scipy.special import elliprd

from xfields.ibs._analytical import AnalyticalIBS, BjorkenMtingwaIBS, IBSGrowthRates, NagaitsevIBS
from xfields.ibs._formulary import (
    _bunch_length,
    _geom_epsx,
    _geom_epsy,
    _percent_change,
    _sigma_delta,
    _sigma_x,
    _sigma_y,
    phi,
)
from xfields.ibs._inputs import BeamParameters, OpticsParameters

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
        return (self.Dx, self.Dy, self.Dz)


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
        return (self.Fx, self.Fy, self.Fz)


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
        return (self.Kx, self.Ky, self.Kz)


# ----- Useful Functions ----- #


# TODO: someday replace this with what Gianni has in xfields.longitudinal_profiles.qgaussian
def line_density(particles: xt.Particles, n_slices: int) -> ArrayLike:
    """
    Returns the longitudinal "line density" of the provided `xtrack.Particles`.
    It is used as a weighing factor for the application of IBS kicks, so that
    particles in the denser parts of the bunch will receive a larger kick, and
    vice versa.

    Parameters
    ----------
    particles : xtrack.Particles
        The `xtrack.Particles` object to compute the line density for.
    n_slices : int
        The number of slices to use for the computation of the bins.

    Returns
    -------
    ArrayLike
        An array with the weight value for each particle, to be used
        as a weight in the kicks application.
    """
    # ----------------------------------------------------------------------------------------------
    # Start with getting the nplike_lib from the particles' context, to compute on the context device
    nplike = particles._context.nplike_lib
    # ----------------------------------------------------------------------------------------------
    # Determine properties from longitudinal particles distribution: cuts, slice width, bunch length
    LOGGER.debug("Determining longitudinal particles distribution properties")
    zeta: ArrayLike = particles.zeta[
        particles.state > 0
    ]  # careful to only consider active particles    z_cut_head: float = np.max(zeta)  # z cut at front of bunch
    z_cut_head: float = nplike.max(zeta)  # z cut at front of bunch
    z_cut_tail: float = nplike.min(zeta)  # z cut at back of bunch
    slice_width: float = (z_cut_head - z_cut_tail) / n_slices  # slice width
    # ----------------------------------------------------------------------------------------------
    # Determine bin edges and bin centers for the distribution
    LOGGER.debug("Determining bin edges and bin centers for the distribution")
    bin_edges: ArrayLike = nplike.linspace(
        z_cut_tail - 1e-7 * slice_width,
        z_cut_head + 1e-7 * slice_width,
        num=n_slices + 1,
        dtype=np.float64,
    )
    bin_centers: ArrayLike = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    # ----------------------------------------------------------------------------------------------
    # Compute histogram on longitudinal distribution then compute and return line density
    counts_normed, bin_edges = nplike.histogram(zeta, bin_edges, density=True)  # density to normalize
    return nplike.interp(zeta, bin_centers, counts_normed)


# ----- Abstract Base Class to Inherit from ----- #

# TODO: make the element aware of the tracking turn number so it can decide whether to recompute the growth rates or not
# Use the particle's .at_turn for this


class IBSKick(xt.BeamElement, ABC):
    """
    Abstract base class for IBS kick elements, from which all
    formalism implementations should inherit.

    Attributes
    ----------
    beam_parameters : BeamParameters
        The necessary beam parameters to use for calculations.
    optics_parameters : OpticsParameters
        The necessary optics parameters to use for calculations.
    kick_coefficients : IBSKickCoefficients
        The computed kick coefficients. This self-updates when they
        are computed with the `.compute_kick_coefficients` method.
    auto_recompute_coefficients_percent : float, optional.
        If given, a check is performed after kicking the particles to
        determine if recomputing the kick coefficients is necessary, in
        which case it will be done before the next kick. **Please provide
        a value as a percentage of the emittance change**. For instance,
        if one provides `12` after kicking a check is done to see if the
        emittance changed by more than 12% in any plane, and if so the
        coefficients will be automatically recomputed before the next
        kick. Defaults to `None` (no checks done, no auto-recomputing).
    """

    iscollective = True  # based on alive particles, need them all here
    needs_rng = True  # TODO: random numbers involved in kicks, not sure how to use this rng yet

    def __init__(
        self,
        beam_parameters: BeamParameters,
        optics_parameters: OpticsParameters,
        auto_recompute_coefficients_percent: float = None,
        *args,
        **kwargs,
    ) -> None:
        self.beam_parameters: BeamParameters = beam_parameters
        self.optics: OpticsParameters = optics_parameters
        self.auto_recompute_coefficients_percent: float = auto_recompute_coefficients_percent
        # This one self-updates when computed, but can be overwritten by the user
        self.kick_coefficients: IBSKickCoefficients = None
        # Private flag to indicate if the coefficients need to be recomputed before the next kick
        self._need_to_recompute_coefficients: bool = False
        # Private attribute tracking the number of coefficients computations
        self._number_of_coefficients_computations: int = 0
        # And passing the rest to initialization of the xtrack.BeamElement
        super().__init__(*args, **kwargs)

    @abstractmethod
    def compute_kick_coefficients(self, particles: xt.Particles, **kwargs) -> IBSKickCoefficients:
        r"""
        Method to compute the kick coefficients. This is an abstract method
        that should be implemented in child classes based on their formalism.

        Parameters
        ----------
        particles :xtrack.Particles
            The particles to apply the IBS kicks to and compute it from.

        Returns
        -------
        IBSKickCoefficients
            An ``IBSKickCoefficients`` object with the computed kick coefficients.
        """
        raise NotImplementedError(
            "This method should be implemented in all child classes, but it hasn't been for this one."
        )

    @abstractmethod
    def _check_coefficients_presence(self) -> None:
        """
        Private method to check the relevant instance attributes
        and determine if the kick coefficients have been computed.
        This is called before applying momenta kicks.
        """
        raise NotImplementedError(
            "This method should be implemented in all child classes, but it hasn't been for this one."
        )

    @abstractmethod
    def _apply_formalism_ibs_kick(self, particles: xt.Particles, n_slices: int = 40) -> None:
        """
        Method to determine and apply IBS momenta kicks. This is an abstract
        method that should be implemented in child classes based on their
        formalism. It is the heavy-lifting part that is called in the `.track`
        method.

        Parameters
        ----------
        particles :xtrack.Particles
            The particles to apply the IBS kicks to and compute it from.
        """

    def track(self, particles: xt.Particles, n_slices: int = 40) -> None:
        """
        Method to track the particles through the IBS kick element.
        This method is called when the element is part of a `xtrack.Line`.

        Parameters
        ----------
        particles : xtrack.Particles
            The particles to apply the IBS kicks to and compute it from.
        n_slices : int
            The number of slices to use for the computation of the bunch
            longitudinal line density. Defaults to ``40``.
        """
        # ----------------------------------------------------------------------------------------------
        # Check that the kick coefficients have been computed beforehand
        self._check_coefficients_presence()
        # ----------------------------------------------------------------------------------------------
        # Check the auto-recompute flag and recompute coefficients if necessary
        if self._need_to_recompute_coefficients is True:
            LOGGER.info("Recomputing IBS kick coefficients before applying kicks")
            self.compute_kick_coefficients(particles)
            self._need_to_recompute_coefficients = False
        # ----------------------------------------------------------------------------------------------
        # Get and store pre-kick emittances if self.auto_recompute_coefficients_percent is set
        if isinstance(self.auto_recompute_coefficients_percent, (int, float)):
            # TODO (Gianni): all this assumes we are the last / first (same) element in the line
            # TODO (Gianni): if not, how can I check to get my position and compute the optics I
            # TODO (Gianni): need at this location, to get these parameters?
            _previous_bunch_length = _bunch_length(particles)
            _previous_sigma_delta = _sigma_delta(particles)
            # below we give index 0 as start / end of machine is kick location
            _previous_geom_epsx = _geom_epsx(particles, self.optics.betx[0], self.optics.dx[0])
            _previous_geom_epsy = _geom_epsy(particles, self.optics.bety[0], self.optics.dy[0])
        # ----------------------------------------------------------------------------------------------
        # Apply the kicks to the particles - the function implementation here is formalism-specific
        self._apply_formalism_ibs_kick(particles, n_slices)
        # ----------------------------------------------------------------------------------------------
        # Get post-kick emittances, check growth and set recompute flag if necessary
        # (only done if self.auto_recompute_coefficients_percent is set to a valid value)
        # fmt: off
        if isinstance(self.auto_recompute_coefficients_percent, (int, float)):
            _new_bunch_length = _bunch_length(particles)
            _new_sigma_delta = _sigma_delta(particles)
            _new_geom_epsx = _geom_epsx(particles, self.optics.betx[0], self.optics.dx[0])
            _new_geom_epsy = _geom_epsy(particles, self.optics.bety[0], self.optics.dy[0])
            # If there is an increase / decrease > than self.auto_recompute_coefficients_percent %
            # in any plane, this check function will set the recompute flag
            self._check_threshold_bypass(
                _previous_geom_epsx, _previous_geom_epsy, _previous_sigma_delta, _previous_bunch_length,
                _new_geom_epsx, _new_geom_epsy, _new_sigma_delta, _new_bunch_length,
                self.auto_recompute_coefficients_percent
            )
        # fmt: on

    def _check_threshold_bypass(
        self,
        epsx: float,
        epsy: float,
        sigma_delta: float,
        bunch_length: float,
        new_epsx: float,
        new_epsy: float,
        new_sigma_delta: float,
        new_bunch_length: float,
        threshold_percent: float,
    ) -> None:
        """
        Checks if the new values exceed a 'threshold_percent'%
        relative change to the initial ones provided. If so, sets
        the `self._need_to_recompute_coefficients` flag to `True`.
        """
        if (  # REMEMBER: threshold is a percentage so we need to divide it by 100
            abs(_percent_change(epsx, new_epsx)) > threshold_percent / 100
            or abs(_percent_change(epsy, new_epsy)) > threshold_percent / 100
            or abs(_percent_change(sigma_delta, new_sigma_delta)) > threshold_percent / 100
            or abs(_percent_change(bunch_length, new_bunch_length)) > threshold_percent / 100
        ):
            LOGGER.debug(
                f"One plane's emittance changed by more than {threshold_percent}%, "
                "setting flag to recompute coefficients before next kick."
            )
            self._need_to_recompute_coefficients = True


# ----- Kick Classes for Specific Formalism ----- #

# TODO (Gianni): Will need help to untangle the big mess of xobjects stuff,
# where should what be implemented, etc.


class IBSSimpleKick(IBSKick):
    """TODO: WRITE"""

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

    def to_dict(self) -> None:
        """Raises an error as the line should be saved without the IBS kick element."""
        raise NotImplementedError("IBS kick elements should not be saved as part of the line")

    @property
    def analytical_ibs(self) -> AnalyticalIBS:
        """Analytical IBS implementation class for growth rates calculation."""
        return self._analytical_ibs

    @analytical_ibs.setter
    def analytical_ibs(self, value: AnalyticalIBS) -> None:
        """Setter so that `.beam_params` and `.optics` are updated when set."""
        # fmt: off
        LOGGER.debug("Overwriting analytical ibs implementation used for growth rates calculation")
        self._analytical_ibs: AnalyticalIBS = value
        LOGGER.debug("Re-pointing instance's beam & optics params to that of the new analytical")
        self.beam_parameters = self.analytical_ibs.beam_parameters
        self.optics = self.analytical_ibs.optics
        # fmt: on

    def compute_kick_coefficients(self, particles: xt.Particles, **kwargs) -> IBSKickCoefficients:
        ...

    def _check_coefficients_presence(self) -> None:
        ...

    def _apply_formalism_ibs_kick(self, particles: xt.Particles, n_slices: int = 40) -> None:
        ...


class IBSKineticKick(IBSKick):
    """TODO: WRITE"""

    _xofields = {
        "beam_parameters": BeamParameters,
        "optics": OpticsParameters,
        "diffusion_coefficients": DiffusionCoefficients,
        "friction_coefficients": FrictionCoefficients,
        "kick_coefficients": IBSKickCoefficients,
    }

    def __init__(self, beam_params: BeamParameters, optics_params: OpticsParameters) -> None:
        self.xoinitialize(
            beam_parameters=beam_params,
            optics=optics_params,
            diffusion_coefficients=None,
            friction_coefficients=None,
            kick_coefficients=None,
        )

    def to_dict(self) -> None:
        """Raises an error as the line should be saved without the IBS kick element."""
        raise NotImplementedError("IBS kick elements should not be saved as part of the line")

    def compute_kick_coefficients(self, particles: xt.Particles, **kwargs) -> IBSKickCoefficients:
        ...

    def _check_coefficients_presence(self) -> None:
        ...

    def _apply_formalism_ibs_kick(self, particles: xt.Particles, n_slices: int = 40) -> None:
        ...
