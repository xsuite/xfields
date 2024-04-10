# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from __future__ import annotations  # important for sphinx to alias ArrayLike

import warnings
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Callable, Tuple

import numpy as np
import xobjects as xo
from numpy.typing import ArrayLike
from scipy.constants import c, hbar
from scipy.integrate import quad, quad_vec
from scipy.interpolate import interp1d
from scipy.special import elliprd

from xfields.ibs._formulary import _percent_change, phi
from xfields.ibs._inputs import BeamParameters, OpticsParameters

LOGGER = getLogger(__name__)

# ----- Some classes to store results (as xo.HybridClass) ----- #


# Renamed this from NagaitsevIntegrals to EllipticIntegrals
class EllipticIntegrals(xo.HybridClass):
    """
    Holds the result of the symmetric elliptic integrals
    of the second kind for each plane, which we compute
    in Nagaitsev formalism.

    Attributes:
    -----------
    Ix : float
        Horizontal elliptic integral result.
    Iy : float
        Vertical elliptic integral result.
    Iz : float
        Longitudinal elliptic integral result.
    """

    _xofields = {
        "Ix": xo.Float64,
        "Iy": xo.Float64,
        "Iz": xo.Float64,
    }

    def __init__(self, Ix: float, Iy: float, Iz: float) -> None:
        """Init with the results of the integrals."""
        self.xoinitialize(Ix=Ix, Iy=Iy, Iz=Iz)

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return the integrals as a tuple."""
        return (self.Ix, self.Iy, self.Iz)


class IBSGrowthRates(xo.HybridClass):
    """
    Holds IBS growth rates in each plane, named ``Tx``,
    ``Ty``, and ``Tz``. By growth rate we mean the 1/tau
    values, expressed in [1/s].

    Attributes:
    -----------
    Tx : float
        Horizontal IBS growth rate, in [s^-1].
    Ty : float
        Vertical IBS growth rate, in [s^-1].
    Tz : float
        Longitudinal IBS growth rate, in [s^-1].
    """

    _xofields = {
        "Tx": xo.Float64,
        "Ty": xo.Float64,
        "Tz": xo.Float64,
    }

    def __init__(self, Tx: float, Ty: float, Tz: float) -> None:
        """Init with given values."""
        self.xoinitialize(Tx=Tx, Ty=Ty, Tz=Tz)

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return the growth rates as a tuple."""
        return (self.Tx, self.Ty, self.Tz)

    def inversed(self):
        """
        Returns a tuple with the inverse of the
        growth rates, the tau values in [s].
        """
        return (1 / self.Tx, 1 / self.Ty, 1 / self.Tz)


# ----- Some private classes (as xo.HybridClass) ----- #


class _SynchrotronRadiationInputs(xo.HybridClass):
    """
    Holds SR input info for emittance evolution.

    Attributes:
    -----------
    equilibrium_epsx : float
        Horizontal equilibrium emittance from SR and QE in [m].
    equilibrium_epsy : float
        Vertical equilibrium emittance from SR and QE in [m].
    equilibrium_sigma_delta : float
        Longitudinal equilibrium emittance from SR and QE in [m].
    tau_x : float
        Horizontal damping time from SR, in [s].
    tau_y : float
        Vertical damping time from SR, in [s].
    tau_z : float
        Longitudinal damping time from SR, in [s].
    """

    _xofields = {
        "equilibrium_epsx": xo.Float64,
        "equilibrium_epsy": xo.Float64,
        "equilibrium_sigma_delta": xo.Float64,
        "tau_x": xo.Float64,
        "tau_y": xo.Float64,
        "tau_z": xo.Float64,
    }

    def __init__(
        self,
        equilibrium_epsx: float,
        equilibrium_epsy: float,
        equilibrium_sigma_delta: float,
        tau_x: float,
        tau_y: float,
        tau_z: float,
    ) -> None:
        """Init with the results of the integrals."""
        self.xoinitialize(
            equilibrium_epsx=equilibrium_epsx,
            equilibrium_epsy=equilibrium_epsy,
            equilibrium_sigma_delta=equilibrium_sigma_delta,
            tau_x=tau_x,
            tau_y=tau_y,
            tau_z=tau_z,
        )


class _ReferenceValues(xo.HybridClass):
    """
    Holds reference values for checks performed in
    auto-recompute behaviour.

    Attributes:
    -----------
    epsx : float
        Horizontal emittance in [m].
    epsy : float
        Vertical emittance in [m].
    sigma_delta : float
        Momentum spread.
    bunch_length : float
        Bunch length in [m].
    """

    _xofields = {
        "epsx": xo.Float64,
        "epsy": xo.Float64,
        "sigma_delta": xo.Float64,
        "bunch_length": xo.Float64,
    }

    def __init__(self, epsx: float, epsy: float, sigma_delta: float, bunch_length: float) -> None:
        """Init with the results of the integrals."""
        self.xoinitialize(
            epsx=epsx,
            epsy=epsy,
            sigma_delta=sigma_delta,
            bunch_length=bunch_length,
        )


# ----- Abstract Base Class to Inherit from ----- #


class AnalyticalIBS(ABC):
    """
    Abstract base class for analytical IBS calculations, from which
    all formalism implementations should inherit.

    Attributes:
    -----------
    beam_parameters : BeamParameters
        The necessary beam parameters to use for calculations.
    optics_parameters : OpticsParameters
        The necessary optics parameters to use for calculations.
    ibs_growth_rates : IBSGrowthRates
        The computed IBS growth rates. This self-updates when
        they are computed with the ``.growth_rates`` method.
    """

    def __init__(self, beam_parameters: BeamParameters, optics_parameters: OpticsParameters) -> None:
        self.beam_parameters: BeamParameters = beam_parameters
        self.optics: OpticsParameters = optics_parameters
        # This one self-updates when computed, but can be overwritten by the user
        self.ibs_growth_rates: IBSGrowthRates = None
        # The following are private attributes for growth rates auto-recomputing
        self._refs: _ReferenceValues = None  # updates when growth rates are computed
        self._number_of_growth_rates_computations: int = 0  # increments when growth rates are computed

    # TODO: adapt citations and admonitions etc
    def coulomb_log(
        self,
        epsx: float,
        epsy: float,
        sigma_delta: float,
        bunch_length: float,
        bunched: bool = True,
        normalized_emittances: bool = False,
    ) -> float:
        r"""
        Calculates the Coulomb logarithm based on the beam parameters and optics the class
        was initiated with. For a good introductory resource on the Coulomb Log, see:
        https://docs.plasmapy.org/en/stable/notebooks/formulary/coulomb.html.

        .. note::
            This function follows the formulae in :cite:`AIP:Anderson:Physics_Vade_Mecum`. The
            Coulomb log is computed as :math:`\ln \left( \Lambda \right) = \ln(r_{max} / r_{min})`.
            Here :math:`r_{max}` denotes the smaller of :math:`\sigma_x` and the Debye length; while
            :math:`r_{min}` is the larger of the classical distance of closest approach and the
            quantum diffraction limit from the nuclear radius. It is the calculation that is done by
            ``MAD-X`` (see the `twclog` subroutine in the `MAD-X/src/ibsdb.f90` source file).

        .. note::
            Both geometric or normalized emittances can be given as input to this function, and it is assumed
            the user provides geomettric emittances. If normalized ones are given the `normalized_emittances`
            parameter should be set to `True` (it defaults to `False`). Internally, a conversion is done to
            geometric emittances, which are used in the computations.

        Parameters
        ----------
        epsx : float
            Horizontal (geometric or normalized) emittance in [m].
        epsy : float
            Vertical (geometric or normalized) emittance in [m].
        sigma_delta : float
            The momentum spread.
        bunch_length : float
            The bunch length in [m].
        bunched : bool
            Whether the beam is bunched or not (coasting). Defaults to `True`.
        normalized_emittances : bool
            Whether the provided emittances are normalized or not. Defaults to
            `False` (assumes geometric emittances).

        Returns
        -------
        float
            The dimensionless Coulomb logarithm :math:`\ln \left( \Lambda \right)`.
        """
        LOGGER.debug("Computing Coulomb logarithm for defined beam and optics parameters")
        # ----------------------------------------------------------------------------------------------
        # Make sure we are working with geometric emittances
        geom_epsx = epsx if normalized_emittances is False else self._geometric_emittance(epsx)
        geom_epsy = epsy if normalized_emittances is False else self._geometric_emittance(epsy)
        # ----------------------------------------------------------------------------------------------
        # Interpolated beta and dispersion functions for the average calculation below
        LOGGER.debug("Interpolating beta and dispersion functions")
        _bxb = interp1d(self.optics.s, self.optics.betx)
        _byb = interp1d(self.optics.s, self.optics.bety)
        _dxb = interp1d(self.optics.s, self.optics.dx)
        _dyb = interp1d(self.optics.s, self.optics.dy)
        # ----------------------------------------------------------------------------------------------
        # Computing "average" of these functions - better here than a simple np.mean
        # calculation because the latter doesn't take in consideration element lengths
        # and can be skewed by some very high peaks in the optics
        with warnings.catch_warnings():  # Catch and ignore the scipy.integrate.IntegrationWarning
            warnings.simplefilter("ignore", category=UserWarning)
            _bx_bar = quad(_bxb, self.optics.s[0], self.optics.s[-1])[0] / self.optics.circumference
            _by_bar = quad(_byb, self.optics.s[0], self.optics.s[-1])[0] / self.optics.circumference
            _dx_bar = quad(_dxb, self.optics.s[0], self.optics.s[-1])[0] / self.optics.circumference
            _dy_bar = quad(_dyb, self.optics.s[0], self.optics.s[-1])[0] / self.optics.circumference
        # ----------------------------------------------------------------------------------------------
        # Calculate transverse temperature as 2*P*X, i.e. assume the transverse energy is temperature/2
        # fmt: off
        Etrans = (
            5e8 * (self.beam_parameters.gamma_rel
                   * self.beam_parameters.total_energy_eV * 1e-9  # total energy needed in GeV
                   - self.beam_parameters.particle_mass_eV * 1e-9  # particle mass needed in GeV
                )
            * (geom_epsx / _bx_bar)
        )
        # fmt: on
        TempeV = 2.0 * Etrans
        # ----------------------------------------------------------------------------------------------
        # Compute sigmas in each dimension (start from sigma_delta to get sige needed in the formula)
        sigma_x_cm = 100 * np.sqrt(
            geom_epsx * _bx_bar + (_dx_bar * sigma_delta * self.beam_parameters.beta_rel**2) ** 2
        )
        sigma_y_cm = 100 * np.sqrt(
            geom_epsy * _by_bar + (_dy_bar * sigma_delta * self.beam_parameters.beta_rel**2) ** 2
        )
        sigma_t_cm = 100 * bunch_length
        # ----------------------------------------------------------------------------------------------
        # Calculate beam volume to get density (in cm^{-3}), then Debye length
        if bunched is True:  # bunched beam
            volume = 8.0 * np.sqrt(np.pi**3) * sigma_x_cm * sigma_y_cm * sigma_t_cm
        else:  # coasting beam
            volume = 4.0 * np.pi * sigma_x_cm * sigma_y_cm * 100 * self.optics.circumference
        density = self.beam_parameters.n_part / volume
        debye_length = 743.4 * np.sqrt(TempeV / density) / abs(self.beam_parameters.particle_charge)
        # ----------------------------------------------------------------------------------------------
        # Calculate 'rmin' as larger of classical distance of closest approach or quantum mechanical
        # diffraction limit from nuclear radius
        rmincl = 1.44e-7 * self.beam_parameters.particle_charge**2 / TempeV
        rminqm = (
            hbar * c * 1e5 / (2.0 * np.sqrt(2e-3 * Etrans * self.beam_parameters.particle_mass_eV * 1e-9))
        )  # particle mass needed in GeV
        # ----------------------------------------------------------------------------------------------
        # Now compute the impact parameters and finally Coulomb logarithm
        bmin = max(rmincl, rminqm)
        bmax = min(sigma_x_cm, debye_length)
        return np.log(bmax / bmin)

    @abstractmethod
    def growth_rates(
        self,
        epsx: float,
        epsy: float,
        sigma_delta: float,
        bunch_length: float,
        bunched: bool = True,
        normalized_emittances: bool = False,
    ) -> IBSGrowthRates:
        r"""
        Method to compute the IBS growth rates. This is an abstract method
        that should be implemented in child classes based on their formalism.

        Parameters
        ----------
        epsx : float
            Horizontal (geometric or normalized) emittance in [m].
        epsy : float
            Vertical (geometric or normalized) emittance in [m].
        sigma_delta : float
            The momentum spread.
        bunch_length : float
            The bunch length in [m].
        bunched : bool
            Whether the beam is bunched or not (coasting). Defaults to `True`.
        normalized_emittances : bool
            Whether the provided emittances are normalized or not. Defaults to
            `False` (assumes geometric emittances).

        Returns
        -------
        IBSGrowthRates
            An ``IBSGrowthRates`` object with the computed growth rates.
        """
        raise NotImplementedError(
            "This method should be implemented in all child classes, but it hasn't been for this one."
        )

    def emittance_evolution(
        self,
        epsx: float,
        epsy: float,
        sigma_delta: float,
        bunch_length: float,
        dt: float = None,
        normalized_emittances: bool = False,
        auto_recompute_rates_percent: float = None,
        **kwargs,
    ) -> Tuple[float, float, float, float]:
        r"""
        Analytically computes the new emittances after a given time step `dt` has
        ellapsed, from provided values, based on the ``IBS`` growth rates.

        .. hint::
            The calculation is an exponential growth based on the rates :math:`T_{x,y,z}`. It goes
            according to the following, where :math:`N` represents the time step:

            .. math::

                T_{x,y,z} &= 1 / \tau_{x,y,z}

                \varepsilon_{x,y}^{N+1} &= \varepsilon_{x,y}^{N} * e^{t / \tau_{x,y}}

                \sigma_{\delta, z}^{N+1} &= \sigma_{\delta, z}^{N} * e^{t / 2 \tau_{z}}

        .. note::
            Both geometric or normalized emittances can be given as input to this function, and it is assumed
            the user provides geomettric emittances. If normalized ones are given the `normalized_emittances`
            parameter should be set to `True` (it defaults to `False`). Internally, a conversion is done to
            geometric emittances, which are used in the computations. The returned emittances correspond to
            the type of those provided: if given normalized emittances this function will return values that
            correspond to the new normalized emittances.


        .. admonition:: Synchrotron Radiation

            Synchrotron Radiation can play a significant role in the evolution of the emittances
            in certain scenarios, particularly for leptons. One can include the contribution of
            SR to this calculation by providing several keyword arguments corresponding to the
            equilibrium emittances and damping times from SR and quantum excitation. See the list
            of expected kwargs below. A :ref:`dedicated section in the FAQ <xibs-faq-sr-inputs>`
            provides information on how to obtain these values from ``Xsuite`` or ``MAD-X``.

            In case this contribution is included, then the calculation is modified from the one
            shown above, and goes according to :cite:`BOOK:Wolski:Beam_dynamics` (Eq (13.64)) or
            :cite:`CAS:Martini:IBS_Anatomy_Theory` (Eq (135)):

            .. math::

                T_{x,y,z} &= 1 / \tau_{x,y,z}^{\mathrm{IBS}}

                \varepsilon_{x,y}^{N+1} &= \left[ - \varepsilon_{x,y}^{\mathrm{SR}eq} + \left( \varepsilon_{x,y}^{\mathrm{SR}eq} + \frac{\varepsilon_{x,y}^{N}}{2 \tau_{x,y}^{\mathrm{IBS}}} \tau_{x,y}^{\mathrm{SR}} - 1 \right) * e^{2 t \left( \frac{1}{2 \tau_{x,y}^{\mathrm{IBS}}} - \frac{1}{\tau_{x,y}^{\mathrm{SR}}} \right)} \right] / \left( \frac{\tau_{x,y}^{\mathrm{SR}}}{2 \tau_{x,y}^{\mathrm{IBS}}} - 1 \right)

                {\sigma_{\delta, z}^{N+1}}^2 &= \left[ - {\sigma_{\delta, z}^{\mathrm{SR}eq}}^2 + \left( {\sigma_{\delta, z}^{\mathrm{SR}eq}}^2 + \frac{{\sigma_{\delta, z}^{N}}^2}{2 \tau_{z}^{\mathrm{IBS}}} \tau_{z}^{\mathrm{SR}} - 1 \right) * e^{2 t \left( \frac{1}{2 \tau_{z}^{\mathrm{IBS}}} - \frac{1}{\tau_{z}^{\mathrm{SR}}} \right)} \right] / \left( \frac{\tau_{z}^{\mathrm{SR}}}{2 \tau_{z}^{\mathrm{IBS}}} - 1 \right)


        Parameters
        ----------
        epsx : float
            Horizontal (geometric or normalized) emittance in [m].
        epsy : float
            Vertical (geometric or normalized) emittance in [m].
        sigma_delta : float
            The momentum spread.
        bunch_length : float
            The bunch length in [m].
        dt : float, optional:
            The time interval to use, in [s]. Defaults to the revolution period of the
            machine, :math:`1 / f_{rev}`, for the evolution in a single turn.
        normalized_emittances : bool
            Whether the provided emittances are normalized or not. Defaults to
            `False` (assumes geometric emittances).
        auto_recompute_rates_percent : float, optional
            If given, a check is performed to determine if an update of the growth rates is
            necessary, in which case it will be done before computing the emittance evolutions.
            **Please provide a value as a percentage of the emittance change since the last
            update of the growth rates**. For instance, if one provides `12`, a check is made to
            see if any quantity would have changed by more than 12% compared to reference values
            stored at the last growth rates update, and if so the growth rates are automatically
            recomputed to be as up-to-date as possible before returning the new values. Defaults
            to `None` (no checks done, no auto-recomputing).
        **kwargs:
            If keyword arguments are provided, they are considered inputs for the inclusion
            of synchrotron radiation in the calculation, and the following are expected,
            case-insensitively:
                sr_equilibrium_epsx : float
                    The horizontal equilibrium emittance from synchrotron radiation
                    and quantum excitation, in [m]. Should be the same type (geometric
                    or normalized) as `epsx` and `epsy`.
                sr_equilibrium_epsy : float
                    The vertical equilibrium emittance from synchrotron radiation and
                    quantum excitation, in [m]. Should be the same type (geometric or
                    normalized) as `epsx` and `epsy`.
                sr_equilibrium_sigma_delta : float
                    The equilibrium momentum spread from synchrotron radiation and
                    quantum excitation.
                sr_tau_x : float
                    The horizontal damping time from synchrotron radiation, in [s]
                    (should be the same unit as `dt`).
                sr_tau_y : float
                    The vertical damping time from synchrotron radiation, in [s]
                    (should be the same unit as `dt`).
                sr_tau_z : float
                    The longitudinal damping time from synchrotron radiation, in [s]
                    (should be the same unit as `dt`).

        Returns
        -------
        Tuple[float, float, float, float]
            A tuple with the new horizontal & vertical geometric emittances, the new
            momentum spread and the new bunch length, after the time step has ellapsed.
        """
        # ----------------------------------------------------------------------------------------------
        # Check that the IBS growth rates have been computed beforehand - compute if not
        if self.ibs_growth_rates is None:
            LOGGER.debug("Attempted to compute emittance evolution without growth rates, computing them.")
            bunched = kwargs.get("bunched", True)  # get the bunched value if provided
            self.growth_rates(epsx, epsy, sigma_delta, bunch_length, bunched, normalized_emittances)
        LOGGER.info("Computing new emittances from IBS growth rates for defined beam and optics parameters")
        # ----------------------------------------------------------------------------------------------
        # Check the kwargs and potentially get the arguments to include synchrotron radiation
        include_synchrotron_radiation = False
        if len(kwargs.keys()) >= 1:  # lets' not check with 'is not None' since default {} kwargs is not None
            LOGGER.debug("Kwargs present, assuming synchrotron radiation is to be included")
            include_synchrotron_radiation = True
            sr_inputs: _SynchrotronRadiationInputs = self._get_synchrotron_radiation_kwargs(**kwargs)
        # ----------------------------------------------------------------------------------------------
        # Make sure we are working with geometric emittances (also for SR inputs if given)
        geom_epsx = epsx if normalized_emittances is False else self._geometric_emittance(epsx)
        geom_epsy = epsy if normalized_emittances is False else self._geometric_emittance(epsy)
        if include_synchrotron_radiation is True:
            sr_eq_geom_epsx = (
                sr_inputs.equilibrium_epsx
                if normalized_emittances is False
                else self._geometric_emittance(sr_inputs.equilibrium_epsx)
            )
            sr_eq_geom_epsy = (
                sr_inputs.equilibrium_epsy
                if normalized_emittances is False
                else self._geometric_emittance(sr_inputs.equilibrium_epsy)
            )
            sr_eq_sigma_delta = sr_inputs.equilibrium_sigma_delta
        # ----------------------------------------------------------------------------------------------
        # Set the time step to 1 / frev if not provided
        if dt is None:
            LOGGER.debug("No time step provided, defaulting to 1 / frev")
            dt = 1 / self.optics.revolution_frequency
        # ----------------------------------------------------------------------------------------------
        # Compute new emittances and return them. Here we multiply because T = 1 / tau
        if include_synchrotron_radiation is False:  # the basic calculation
            new_epsx, new_epsy, new_sigma_delta, new_bunch_length = self._evolution_without_sr(
                geom_epsx, geom_epsy, sigma_delta, bunch_length, dt
            )
        else:  # the modified calculation with Synchrotron Radiation contribution
            new_epsx, new_epsy, new_sigma_delta, new_bunch_length = self._evolution_with_sr(
                geom_epsx,
                geom_epsy,
                sigma_delta,
                bunch_length,
                dt,
                sr_eq_geom_epsx,
                sr_eq_geom_epsy,
                sr_eq_sigma_delta,
                sr_inputs.tau_x,
                sr_inputs.tau_y,
                sr_inputs.tau_z,
            )
        # ----------------------------------------------------------------------------------------------
        # If auto-recompute is on and the given threshold is exceeded, recompute the growth rates
        if isinstance(auto_recompute_rates_percent, (int, float)):
            if self._bypassed_threshold(
                new_epsx, new_epsy, new_sigma_delta, new_bunch_length, auto_recompute_rates_percent
            ):
                LOGGER.debug(
                    f"One value would change by more than {auto_recompute_rates_percent}% compared to last "
                    "update of the growth rates, updating growth rates before re-computing evolutions."
                )
                bunched = kwargs.get("bunched", True)  # get the bunched value if provided
                self.growth_rates(epsx, epsy, sigma_delta, bunch_length, bunched, normalized_emittances)
                # And now we need to recompute the evolutions since the growth rates have been updated
                if include_synchrotron_radiation is False:  # the basic calculation
                    new_epsx, new_epsy, new_sigma_delta, new_bunch_length = self._evolution_without_sr(
                        geom_epsx, geom_epsy, sigma_delta, bunch_length, dt
                    )
                else:  # the modified calculation with Synchrotron Radiation contribution
                    new_epsx, new_epsy, new_sigma_delta, new_bunch_length = self._evolution_with_sr(
                        geom_epsx,
                        geom_epsy,
                        sigma_delta,
                        bunch_length,
                        dt,
                        sr_eq_geom_epsx,
                        sr_eq_geom_epsy,
                        sr_eq_sigma_delta,
                        sr_inputs.tau_x,
                        sr_inputs.tau_y,
                        sr_inputs.tau_z,
                    )
        # ----------------------------------------------------------------------------------------------
        # Make sure we return the same type of emittances as the user provided
        new_epsx = new_epsx if normalized_emittances is False else self._normalized_emittance(new_epsx)
        new_epsy = new_epsy if normalized_emittances is False else self._normalized_emittance(new_epsy)
        # ----------------------------------------------------------------------------------------------
        return float(new_epsx), float(new_epsy), float(new_sigma_delta), float(new_bunch_length)
