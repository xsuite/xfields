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
import xtrack as xt
from numpy.typing import ArrayLike
from scipy.constants import c, hbar
from scipy.integrate import quad, quad_vec
from scipy.interpolate import interp1d
from scipy.special import elliprd

from xfields.ibs._formulary import phi

LOGGER = getLogger(__name__)

# ----- Some classes to store results (as xo.HybridClass) ----- #


class NagaitsevIntegrals(xo.HybridClass):
    """
    Holds the result of the Nagaitsev integrals for each plane,
    which we compute in Nagaitsev formalism. They correspond to
    the integrals inside of Eq (32), (31) and (30) in
    :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`,
    respectively.

    Attributes
    ----------
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
        return float(self.Ix), float(self.Iy), float(self.Iz)


class IBSGrowthRates(xo.HybridClass):
    """
    Holds IBS growth rates in each plane, named ``Tx``,
    ``Ty``, and ``Tz``. By growth rate we mean the 1/tau
    values, expressed in [s^-1].

    Attributes
    ----------
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
        return float(self.Tx), float(self.Ty), float(self.Tz)

    def as_tau(self) -> Tuple[float, float, float]:
        """
        Returns a tuple with the inverse of the
        growth rates: the tau values in [s].
        """
        return float(1 / self.Tx), float(1 / self.Ty), float(1 / self.Tz)


# ----- Abstract Base Class to Inherit from ----- #


class AnalyticalIBS(ABC):
    """
    Abstract base class for analytical IBS calculations, from which
    all formalism implementations should inherit.

    Attributes
    ----------
    ibs_growth_rates : IBSGrowthRates
        The computed IBS growth rates. This self-updates when
        they are computed with the `.growth_rates` method.
    """

    def __init__(self, twiss: xt.TwissTable) -> None:
        """Initialize the class.

        Parameters
        ----------
        twiss : xtrack.TwissTable
            Twiss results of the `xtrack.Line` configuration.
        """
        self._twiss = twiss
        self._particle = twiss.particle_on_co
        # This one self-updates when computed, but can be overwritten by the user
        self.ibs_growth_rates: IBSGrowthRates = None

    def coulomb_log(
        self,
        gemitt_x: float = None,
        nemitt_x: float = None,
        gemitt_y: float = None,
        nemitt_y: float = None,
        sigma_delta: float = None,
        bunch_length: float = None,
        total_beam_intensity: int = None,
        bunched: bool = True,
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

        Parameters
        ----------
        gemitt_x : float, optional
            Horizontal geometric emittance in [m]. Either this
            parameter or `nemitt_x` is required.
        nemitt_x : float, optional
            Horizontal normalized emittance in [m]. Either this
            parameter or `gemitt_x` is required.
        gemitt_y : float, optional
            Vertical geometric emittance in [m]. Either this
            parameter or `nemitt_y` is required.
        nemitt_y : float, optional
            Vertical normalized emittance in [m]. Either this
            parameter or `gemitt_y` is required.
        sigma_delta : float
            The momentum spread.
        bunch_length : float
            The bunch length in [m].
        total_beam_intensity : int
            The beam intensity to assume.
        bunched : bool
            Whether the beam is bunched or not (coasting). Defaults to `True`.

        Returns
        -------
        float
            The dimensionless Coulomb logarithm :math:`\ln \left( \Lambda \right)`.
        """
        LOGGER.debug("Computing Coulomb logarithm for defined beam and optics parameters")
        # ----------------------------------------------------------------------------------------------
        # Perform checks on exclusive parameters and make sure we have what we need
        assert sigma_delta is not None, "Must provide 'sigma_delta'"
        assert bunch_length is not None, "Must provide 'bunch_length'"
        assert total_beam_intensity is not None, "Must provide 'total_beam_intensity'"
        assert any([gemitt_x, nemitt_x]), "Must provide either 'gemitt_x' or 'nemitt_x'"
        assert any([gemitt_y, nemitt_y]), "Must provide either 'gemitt_y' or 'nemitt_y'"
        if gemitt_x is not None:
            assert nemitt_x is None, "Cannot provide both 'gemitt_x' and 'nemitt_x'"
        if gemitt_y is not None:
            assert nemitt_y is None, "Cannot provide both 'gemitt_y' and 'nemitt_y'"
        if nemitt_x is not None:
            assert gemitt_x is None, "Cannot provide both 'gemitt_x' and 'nemitt_x'"
            gemitt_x = self._gemitt(nemitt_x)
        if nemitt_y is not None:
            assert gemitt_y is None, "Cannot provide both 'gemitt_y' and 'nemitt_y'"
            gemitt_y = self._gemitt(nemitt_y)
        # ----------------------------------------------------------------------------------------------
        # Interpolated beta and dispersion functions for the average calculation below
        LOGGER.debug("Interpolating beta and dispersion functions")
        _bxb = interp1d(self._twiss.s, self._twiss.betx)
        _byb = interp1d(self._twiss.s, self._twiss.bety)
        _dxb = interp1d(self._twiss.s, self._twiss.dx)
        _dyb = interp1d(self._twiss.s, self._twiss.dy)
        # ----------------------------------------------------------------------------------------------
        # Computing "average" of these functions - better here than a simple np.mean
        # calculation because the latter doesn't take in consideration element lengths
        # and can be skewed by some very high peaks in the optics
        with warnings.catch_warnings():  # Catch and ignore the scipy.integrate.IntegrationWarning
            warnings.simplefilter("ignore", category=UserWarning)
            _bx_bar = quad(_bxb, self._twiss.s[0], self._twiss.s[-1])[0] / self._twiss.circumference
            _by_bar = quad(_byb, self._twiss.s[0], self._twiss.s[-1])[0] / self._twiss.circumference
            _dx_bar = quad(_dxb, self._twiss.s[0], self._twiss.s[-1])[0] / self._twiss.circumference
            _dy_bar = quad(_dyb, self._twiss.s[0], self._twiss.s[-1])[0] / self._twiss.circumference
        # ----------------------------------------------------------------------------------------------
        # Calculate transverse temperature as 2*P*X, i.e. assume the transverse energy is temperature/2
        # We need the total energy and the particle mass in GeV hence the 1e-9 below
        mass0_GeV = self._particle.mass0 * 1e-9
        gamma0 = self._twiss.gamma0
        energy_GeV = self._particle.energy[0] * 1e-9
        Etrans = 5e8 * (gamma0 * energy_GeV - mass0_GeV) * (gemitt_x / _bx_bar)
        TempeV = 2.0 * Etrans
        # ----------------------------------------------------------------------------------------------
        # Compute sigmas in each dimension (start from sigma_delta to get sige needed in the formula)
        sigma_x_cm = 100 * np.sqrt(gemitt_x * _bx_bar + (_dx_bar * sigma_delta * self._twiss.beta0**2) ** 2)
        sigma_y_cm = 100 * np.sqrt(gemitt_y * _by_bar + (_dy_bar * sigma_delta * self._twiss.beta0**2) ** 2)
        sigma_t_cm = 100 * bunch_length
        # ----------------------------------------------------------------------------------------------
        # Calculate beam volume to get density (in cm^{-3}), then Debye length
        if bunched is True:  # bunched beam
            volume = 8.0 * np.sqrt(np.pi**3) * sigma_x_cm * sigma_y_cm * sigma_t_cm
        else:  # coasting beam
            volume = 4.0 * np.pi * sigma_x_cm * sigma_y_cm * 100 * self._twiss.circumference
        density = total_beam_intensity / volume
        debye_length = 743.4 * np.sqrt(TempeV / density) / abs(self._particle.q0)
        # ----------------------------------------------------------------------------------------------
        # Calculate 'rmin' as larger of classical distance of closest approach or quantum mechanical
        # diffraction limit from nuclear radius. Particle mass needed in GeV hence the 1e-9 below
        rmincl = 1.44e-7 * self._particle.q0**2 / TempeV
        rminqm = hbar * c * 1e5 / (2.0 * np.sqrt(2e-3 * Etrans * self._particle.mass0 * 1e-9))
        # ----------------------------------------------------------------------------------------------
        # Now compute the impact parameters and finally Coulomb logarithm
        bmin = max(rmincl, rminqm)
        bmax = min(sigma_x_cm, debye_length)
        return np.log(bmax / bmin)

    @abstractmethod
    def growth_rates(
        self,
        gemitt_x: float = None,
        nemitt_x: float = None,
        gemitt_y: float = None,
        nemitt_y: float = None,
        sigma_delta: float = None,
        bunch_length: float = None,
        total_beam_intensity: int = None,
        bunched: bool = True,
    ) -> IBSGrowthRates:
        r"""
        Method to compute the IBS growth rates. This is an abstract method
        that should be implemented in child classes based on their formalism.

        Parameters
        ----------
        gemitt_x : float, optional
            Horizontal geometric emittance in [m]. Either this
            parameter or `nemitt_x` is required.
        nemitt_x : float, optional
            Horizontal normalized emittance in [m]. Either this
            parameter or `gemitt_x` is required.
        gemitt_y : float, optional
            Vertical geometric emittance in [m]. Either this
            parameter or `nemitt_y` is required.
        nemitt_y : float, optional
            Vertical normalized emittance in [m]. Either this
            parameter or `gemitt_y` is required.
        sigma_delta : float
            The momentum spread.
        bunch_length : float
            The bunch length in [m].
        total_beam_intensity : int
            The beam intensity to assume.
        bunched : bool
            Whether the beam is bunched or not (coasting). Defaults to `True`.

        Returns
        -------
        IBSGrowthRates
            An ``IBSGrowthRates`` object with the computed growth rates.
        """
        raise NotImplementedError(
            "This method should be implemented in all child classes, but it hasn't been for this one."
        )

    def _nemitt(self, gemitt: float) -> float:
        """
        Computes normalized emittance from the geometric one, using relativistic
        beta and gamma from the the instance's beam parameters attribute.

        Parameters
        ----------
        gemitt : float
            The geometric emittance in [m].

        Returns
        -------
        float
            The corresponding normalized emittance in [m].
        """
        return gemitt * self._twiss.beta0 * self._twiss.gamma0

    def _gemitt(self, nemitt: float) -> float:
        """
        Computes geometric emittance from the normalized one, using relativistic
        beta and gamma from the the instance's beam parameters attribute.

        Parameters
        ----------
        nemitt : float
            The normalized emittance in [m].

        Returns
        -------
        float
            The corresponding geometric emittance in [m].
        """
        return nemitt / (self._twiss.beta0 * self._twiss.gamma0)


# ----- Analytical Classes for Specific Formalism ----- #


class NagaitsevIBS(AnalyticalIBS):
    r"""
    Analytical implementation to compute IBS growth
    rates according to S. Nagaitsev's formalism (see
    :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`).

    Please keep in mind that this formalism will give an inaccurate
    vertical growth rate in the presence of vertical dispersion. In
    such a case, prefer the Bjorken-Mtingwa formalism instead. See
    the `BjorkenMtingwaIBS` class.

    Attributes
    ----------
    ibs_growth_rates : IBSGrowthRates
        The computed IBS growth rates. This self-updates when
        they are computed with the `.growth_rates` method.
    nagaitsev_integrals : NagaitsevIntegrals
        The computed Nagaitsev integrals. This self-updates when
        they are computed with the `.integrals` method.
    """

    def __init__(self, twiss: xt.TwissTable) -> None:
        """Initialize the class.

        Parameters
        ----------
        twiss : xtrack.TwissTable
            Twiss results of the `xtrack.Line` configuration.
        """
        super().__init__(twiss)
        # This self-updates when computed, but can be overwritten by the user
        self.nagaitsev_integrals: NagaitsevIntegrals = None

    def integrals(
        self,
        gemitt_x: float = None,
        nemitt_x: float = None,
        gemitt_y: float = None,
        nemitt_y: float = None,
        sigma_delta: float = None,
    ) -> NagaitsevIntegrals:
        r"""
        Computes the "Nagaitsev" integrals for the lattice, named
        :math:`I_x, I_y` and :math:`I_z` in this code base.

        These correspond to the integrals inside of Eq (32), (31) and
        (30) in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`,
        respectively. The instance attribute `self.elliptic_integrals` is
        automatically updated with the results of this method. They are used to
        calculate the growth rates.

        .. hint::
            The calculation is done according to the following steps, which are related to different
            equations in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`:

                - Computes various intermediate terms and then :math:`a_x, a_y, a_s, a_1` and :math:`a_2` constants from Eq (18-21).
                - Computes the eigenvalues :math:`\lambda_1, \lambda_2` of the :math:`\bf{A}` matrix (:math:`\bf{L}` matrix in B&M) from Eq (22-24).
                - Computes the :math:`R_1, R_2` and :math:`R_3` terms from Eq (25-27) with the forms of Eq (5-6).
                - Computes the :math:`S_p, S_x` and :math:`S_{xp}` terms from Eq (33-35).
                - Computes and returns the integrals terms in Eq (30-32).

        Parameters
        ----------
        gemitt_x : float, optional
            Horizontal geometric emittance in [m]. Either this
            parameter or `nemitt_x` is required.
        nemitt_x : float, optional
            Horizontal normalized emittance in [m]. Either this
            parameter or `gemitt_x` is required.
        gemitt_y : float, optional
            Vertical geometric emittance in [m]. Either this
            parameter or `nemitt_y` is required.
        nemitt_y : float, optional
            Vertical normalized emittance in [m]. Either this
            parameter or `gemitt_y` is required.
        sigma_delta : float
            The momentum spread.
        bunch_length : float
            The bunch length in [m].

        Returns
        -------
        NagaitsevIntegrals
            An ``NagaitsevIntegrals`` object with the computed integrals.
        """
        LOGGER.debug("Computing integrals for defined beam and optics parameters")
        # ----------------------------------------------------------------------------------------------
        # Perform checks on exclusive parameters and make sure we have what we need
        assert sigma_delta is not None, "Must provide 'sigma_delta'"
        assert any([gemitt_x, nemitt_x]), "Must provide either 'gemitt_x' or 'nemitt_x'"
        assert any([gemitt_y, nemitt_y]), "Must provide either 'gemitt_y' or 'nemitt_y'"
        if gemitt_x is not None:
            assert nemitt_x is None, "Cannot provide both 'gemitt_x' and 'nemitt_x'"
        if gemitt_y is not None:
            assert nemitt_y is None, "Cannot provide both 'gemitt_y' and 'nemitt_y'"
        if nemitt_x is not None:
            assert gemitt_x is None, "Cannot provide both 'gemitt_x' and 'nemitt_x'"
            gemitt_x = self._gemitt(nemitt_x)
        if nemitt_y is not None:
            assert gemitt_y is None, "Cannot provide both 'gemitt_y' and 'nemitt_y'"
            gemitt_y = self._gemitt(nemitt_y)
        # fmt: off
        # ----------------------------------------------------------------------------------------------
        # Computing necessary intermediate terms for the following lines
        sigx: ArrayLike = np.sqrt(self._twiss.betx * gemitt_x + (self._twiss.dx * sigma_delta)**2)
        sigy: ArrayLike = np.sqrt(self._twiss.bety * gemitt_y + (self._twiss.dy * sigma_delta)**2)
        phix: ArrayLike = phi(self._twiss.betx, self._twiss.alfx, self._twiss.dx, self._twiss.dpx)
        # Computing the constants from Eq (18-21) in Nagaitsev paper
        ax: ArrayLike = self._twiss.betx / gemitt_x
        ay: ArrayLike = self._twiss.bety / gemitt_y
        a_s: ArrayLike = ax * (self._twiss.dx**2 / self._twiss.betx**2 + phix**2) + 1 / sigma_delta**2
        a1: ArrayLike = (ax + self._twiss.gamma0**2 * a_s) / 2.0
        a2: ArrayLike = (ax - self._twiss.gamma0**2 * a_s) / 2.0
        sqrt_term = np.sqrt(a2**2 + self._twiss.gamma0**2 * ax**2 * phix**2)  # square root term in Eq (22-23) and Eq (33-35)
        # ----------------------------------------------------------------------------------------------
        # These are from Eq (22-24) in Nagaitsev paper, eigen values of A matrix (L matrix in B&M)
        lambda_1: ArrayLike = ay
        lambda_2: ArrayLike = a1 + sqrt_term
        lambda_3: ArrayLike = a1 - sqrt_term
        # ----------------------------------------------------------------------------------------------
        # These are the R_D terms to compute, from Eq (25-27) in Nagaitsev paper (at each element of the lattice)
        LOGGER.debug("Computing the symmetric elliptic integrals R1, R2 and R3")
        R1: ArrayLike = elliprd(1 / lambda_2, 1 / lambda_3, 1 / lambda_1) / lambda_1
        R2: ArrayLike = elliprd(1 / lambda_3, 1 / lambda_1, 1 / lambda_2) / lambda_2
        R3: ArrayLike = 3 * np.sqrt(lambda_1 * lambda_2 / lambda_3) - lambda_1 * R1 / lambda_3 - lambda_2 * R2 / lambda_3
        # ----------------------------------------------------------------------------------------------
        # This are the terms from Eq (33-35) in Nagaitsev paper
        Sp: ArrayLike = (2 * R1 - R2 * (1 - 3 * a2 / sqrt_term) - R3 * (1 + 3 * a2 / sqrt_term)) * 0.5 * self._twiss.gamma0**2
        Sx: ArrayLike = (2 * R1 - R2 * (1 + 3 * a2 / sqrt_term) - R3 * (1 - 3 * a2 / sqrt_term)) * 0.5
        Sxp: ArrayLike = 3 * self._twiss.gamma0**2 * phix**2 * ax * (R3 - R2) / sqrt_term
        # ----------------------------------------------------------------------------------------------
        # These are the integrands of the integrals in Eq (30-32) in Nagaitsev paper
        Ix_integrand = self._twiss.betx / (self._twiss.circumference * sigx * sigy) * (Sx + Sp * (self._twiss.dx**2 / self._twiss.betx**2 + phix**2) + Sxp)
        Iy_integrand = self._twiss.bety / (self._twiss.circumference * sigx * sigy) * (R2 + R3 - 2 * R1)
        Iz_integrand = Sp / (self._twiss.circumference * sigx * sigy)
        # fmt: on
        # ----------------------------------------------------------------------------------------------
        # Integrating the integrands above accross the ring to get the desired results
        # This is identical to np.trapz(Ixyz_integrand, self.optics.s) but faster and somehow closer to MAD-X values
        Ix = float(np.sum(Ix_integrand[:-1] * np.diff(self._twiss.s)))
        Iy = float(np.sum(Iy_integrand[:-1] * np.diff(self._twiss.s)))
        Iz = float(np.sum(Iz_integrand[:-1] * np.diff(self._twiss.s)))
        result = NagaitsevIntegrals(Ix, Iy, Iz)
        # ----------------------------------------------------------------------------------------------
        # Self-update the instance's attributes and then return the results
        self.nagaitsev_integrals = result
        return result

    def growth_rates(
        self,
        gemitt_x: float = None,
        nemitt_x: float = None,
        gemitt_y: float = None,
        nemitt_y: float = None,
        sigma_delta: float = None,
        bunch_length: float = None,
        total_beam_intensity: int = None,
        bunched: bool = True,
    ) -> IBSGrowthRates:
        r"""
        Computes the ``IBS`` growth rates, named :math:`T_x, T_y` and :math:`T_z` in this
        code base, according to Nagaitsev's formalism. These correspond to the :math:`1 / \tau`
        terms of Eq (28) in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`. The
        instance attribute `self.ibs_growth_rates` is automatically updated with the results of
        this method when it is called.

        .. warning::
            Currently this calculation does not take into account vertical dispersion.
            Should you have any in your lattice, please use the BjorkenMtingwaIBS class
            instead, which supports it fully. Supporting vertical dispersion in
            `NagaitsevIBS` might be implemented in a future version.

        .. hint::
            The calculation is done according to the following steps, which are related to
            different equations in :cite:`PRAB:Nagaitsev:IBS_formulas_fast_numerical_evaluation`:

                - Get the Nagaitsev integrals (integrals of Eq (30-32)).
                - Computes the Coulomb logarithm for the defined beam and optics parameters.
                - Compute the rest of the constant term of Eq (30-32).
                - Compute for each plane the full result of Eq (30-32), respectively.
                - Plug these into Eq (28) and divide by either :math:`\varepsilon_x, \varepsilon_y` or :math:`\sigma_{\delta}^{2}` (as relevant) to get :math:`1 / \tau`.

            **Note:** As one can see above, this calculation is done by building on the
            Nagaitsev integrals. If these have not been computed yet, this method will
            first log a message and compute them, then compute the growth rates.

        .. admonition:: Coasting Beams

            It is possible in this formalism to get an approximation in the case of coasting
            beams by providing `bunched=False`. This will as a bunch length :math:`C / 2 \pi`
            with C the circumference (or length) of the machine, and a warning will be logged
            for the user. Additionally the appropriate adjustement will be made in the Coulomb
            logarithm calculation, and the resulting growth rates will be divided by a factor
            2 before being returned (see :cite:`ICHEA:Piwinski:IntraBeamScattering`). For fully
            accurate results in the case of coasting beams, please use the `BjorkenMtingwaIBS`
            class instead.

        Parameters
        ----------
        gemitt_x : float, optional
            Horizontal geometric emittance in [m]. Either this
            parameter or `nemitt_x` is required.
        nemitt_x : float, optional
            Horizontal normalized emittance in [m]. Either this
            parameter or `gemitt_x` is required.
        gemitt_y : float, optional
            Vertical geometric emittance in [m]. Either this
            parameter or `nemitt_y` is required.
        nemitt_y : float, optional
            Vertical normalized emittance in [m]. Either this
            parameter or `gemitt_y` is required.
        sigma_delta : float
            The momentum spread.
        bunch_length : float
            The bunch length in [m].
        total_beam_intensity : int
            The beam intensity to assume.
        bunched : bool
            Whether the beam is bunched or not (coasting). Defaults to `True`.
        normalized_emittances : bool
            Whether the provided emittances are normalized or not. Defaults to
            `False` (assumes geometric emittances).
        compute_integrals : bool
            If `True`, the elliptic integrals will be (re-)computed before the growth
            rates. Defaults to `True`.

        Returns
        -------
        IBSGrowthRates
            An ``IBSGrowthRates`` object with the computed growth rates.
        """
        LOGGER.info("Computing IBS growth rates for defined beam and optics parameters")
        # ----------------------------------------------------------------------------------------------
        # Perform checks on exclusive parameters and make sure we have what we need
        assert sigma_delta is not None, "Must provide 'sigma_delta'"
        assert bunch_length is not None, "Must provide 'bunch_length'"
        assert total_beam_intensity is not None, "Must provide 'total_beam_intensity'"
        assert any([gemitt_x, nemitt_x]), "Must provide either 'gemitt_x' or 'nemitt_x'"
        assert any([gemitt_y, nemitt_y]), "Must provide either 'gemitt_y' or 'nemitt_y'"
        if gemitt_x is not None:
            assert nemitt_x is None, "Cannot provide both 'gemitt_x' and 'nemitt_x'"
        if gemitt_y is not None:
            assert nemitt_y is None, "Cannot provide both 'gemitt_y' and 'nemitt_y'"
        if nemitt_x is not None:
            assert gemitt_x is None, "Cannot provide both 'gemitt_x' and 'nemitt_x'"
            gemitt_x = self._gemitt(nemitt_x)
        if nemitt_y is not None:
            assert gemitt_y is None, "Cannot provide both 'gemitt_y' and 'nemitt_y'"
            gemitt_y = self._gemitt(nemitt_y)
        # ----------------------------------------------------------------------------------------------
        # Adapt bunch length if the user specifies coasting beams
        if bunched is False:
            LOGGER.warning(
                "Using 'bunched=False' in this formalism makes the approximation of bunch length = C/(2*pi). "
                "Please use the BjorkenMtingwaIBS class for fully accurate results."
            )
            bunch_length: float = self._twiss.circumference / (2 * np.pi)
        # ----------------------------------------------------------------------------------------------
        # Ensure we update the integrals have been computed beforehand
        _ = self.integrals(gemitt_x=gemitt_x, gemitt_y=gemitt_y, sigma_delta=sigma_delta)
        # ----------------------------------------------------------------------------------------------
        # Get the Coulomb logarithm and the rest of the constant term in Eq (30-32)
        coulomb_logarithm = self.coulomb_log(
            gemitt_x=gemitt_x,
            gemitt_y=gemitt_y,
            sigma_delta=sigma_delta,
            bunch_length=bunch_length,
            total_beam_intensity=total_beam_intensity,
            bunched=bunched,
        )
        # Then the rest of the constant term in the equation
        radius = self._particle.get_classical_particle_radius0()
        beta0 = self._twiss.beta0
        gamma0 = self._twiss.gamma0
        numerator = total_beam_intensity * radius**2 * c
        denominator = 12 * np.pi * beta0**3 * gamma0**5 * bunch_length
        rest_of_constant_term = numerator / denominator
        full_constant_term = rest_of_constant_term * coulomb_logarithm
        # ----------------------------------------------------------------------------------------------
        # Compute the full result of Eq (30-32) for each plane | make sure to convert back to float
        Ix, Iy, Iz = self.nagaitsev_integrals.as_tuple()
        # If coasting beams, since we use bunch_length=C/(2*pi) we have to divide rates by 2 (see Piwinski)
        factor = 1.0 if bunched is True else 2.0
        Tx = float(Ix * full_constant_term / gemitt_x) / factor
        Ty = float(Iy * full_constant_term / gemitt_y) / factor
        Tz = float(Iz * full_constant_term / sigma_delta**2) / factor
        result = IBSGrowthRates(Tx, Ty, Tz)
        # ----------------------------------------------------------------------------------------------
        # Self-update the instance's attribute before returning
        self.ibs_growth_rates = result
        return result


class BjorkenMtingwaIBS(AnalyticalIBS):
    r"""
    Analytical implementation to compute IBS growth rates according
    to the `Bjorken & Mtingwa` formalism. The method follows the
    ``MAD-X`` implementation, which has corrected B&M in order to take
    in consideration vertical dispersion (see the relevant note about
    the changes at :cite:`CERN:Antoniou:Revision_IBS_MADX`).

    .. note::
        In ``MAD-X`` it is ensure that the Twiss table is centered.
        One might observe some small discrepancies against ``MAD-X``
        growth rates if not providing a centered Twiss table (by
        slicing the lattice first, for instance.)

    Attributes:
    -----------
    ibs_growth_rates : IBSGrowthRates
        The computed IBS growth rates. This self-updates when
        they are computed with the `.growth_rates` method.
    """

    def __init__(self, twiss: xt.TwissTable) -> None:
        """Initialize the class.

        Parameters
        ----------
        twiss : xtrack.TwissTable
            Twiss results of the `xtrack.Line` configuration.
        """
        super().__init__(twiss)

    def _Gamma(
        self,
        gemitt_x: float,
        gemitt_y: float,
        sigma_delta: float,
        bunch_length: float,
        bunched: bool = True,
    ) -> float:
        r"""
        Computes :math:`\Gamma`, the 6-dimensional invariant phase space
        volume of a beam.

        Parameters
        ----------
        gemitt_x : float
            Horizontal geometric emittance in [m].
        gemitt_y : float
            Vertical geometric emittance in [m].
        sigma_delta : float
            The momentum spread.
        bunch_length : float
            The bunch length in [m].
        bunched : bool
            Whether the beam is bunched or not (coasting). Defaults to `True`.

        Returns
        -------
        float
            The computed :math:`\Gamma` value.
        """
        # Below we give mass in MeV like in .growth_rates() (the m^3 terms cancel out)
        if bunched is True:
            return (
                (2 * np.pi) ** 3
                * (self._twiss.beta0 * self._twiss.gamma0) ** 3
                * (self._particle.mass0 * 1e-3) ** 3
                * gemitt_x
                * gemitt_y
                * sigma_delta
                * bunch_length
            )
        else:  # coasting beam
            return (
                4
                * np.pi ** (5 / 2)
                * (self._twiss.beta0 * self._twiss.gamma0) ** 3
                * (self._particle.mass0 * 1e-3) ** 3
                * gemitt_x
                * gemitt_y
                * sigma_delta
                * self._twiss.circumference
            )

    def _a(self, gemitt_x: float, gemitt_y: float, sigma_delta: float) -> ArrayLike:
        """
        Computes the `a` term of Table 1 in the MAD-X note.

        If comparing to the MAD-X Fortran code, this corresponds to (and
        was benchmarked against) the `a` variable there.

        Parameters
        ----------
        gemitt_x : float
            Horizontal geometric emittance in [m].
        gemitt_y : float
            Vertical geometric emittance in [m].
        sigma_delta : float
            The momentum spread.

        Returns
        -------
        ArrayLike
            An array with the `a` term, at each element in the lattice.
        """
        # ----------------------------------------------------------------------------------------------
        # We compute (once) some convenience terms used a lot in the equations, for efficiency & clarity
        beta: float = self._twiss.beta0  # relativistic beta
        gamma: float = self._twiss.gamma0  # relativistic gamma
        betx_over_epsx: ArrayLike = self._twiss.betx / gemitt_x  # beta_x / eps_x term
        bety_over_epsy: ArrayLike = self._twiss.bety / gemitt_y  # beta_y / eps_y term
        # ----------------------------------------------------------------------------------------------
        # Adjust dispersion and dispersion prime by multiplied by relativistic beta, in order to be in the
        # deltap and not the pt frame (default in MAD-X / xsuite). Necessary for non-relativistic beams
        LOGGER.debug("Adjusting Dx, Dy, Dpx, Dpy to be in the pt frame")
        Dx: ArrayLike = self._twiss.dx * beta
        Dy: ArrayLike = self._twiss.dy * beta
        Dpx: ArrayLike = self._twiss.dpx * beta
        Dpy: ArrayLike = self._twiss.dpy * beta
        # ----------------------------------------------------------------------------------------------
        # Computing Phi_{x,y} amd H_{x,y} as defined in Eq (6) and Eq (7) of the note
        LOGGER.debug("Computing Phi_x, Phi_y, H_x and H_y at all elements")
        phix: ArrayLike = phi(self._twiss.betx, self._twiss.alfx, Dx, Dpx)
        phiy: ArrayLike = phi(self._twiss.bety, self._twiss.alfy, Dy, Dpy)
        Hx: ArrayLike = (Dx**2 + self._twiss.betx**2 * phix**2) / self._twiss.betx
        Hy: ArrayLike = (Dy**2 + self._twiss.bety**2 * phiy**2) / self._twiss.bety
        # ----------------------------------------------------------------------------------------------
        a: ArrayLike = (
            gamma**2 * (Hx / gemitt_x + Hy / gemitt_y)
            + gamma**2 / (sigma_delta**2)
            + (betx_over_epsx + bety_over_epsy)
        )
        return a

    def _b(self, gemitt_x: float, gemitt_y: float, sigma_delta: float) -> ArrayLike:
        """
        Computes the `b` term of Table 1 in the MAD-X note.

        If comparing to the MAD-X Fortran code, this corresponds to (and
        was benchmarked against) the `b` variable there.

        Parameters
        ----------
        gemitt_x : float
            Horizontal geometric emittance in [m].
        gemitt_y : float
            Vertical geometric emittance in [m].
        sigma_delta : float
            The momentum spread.

        Returns
        -------
        ArrayLike
            An array with the `b` term, at each element in the lattice.
        """
        # ----------------------------------------------------------------------------------------------
        # We compute (once) some convenience terms used a lot in the equations, for efficiency & clarity
        beta: float = self._twiss.beta0  # relativistic beta
        gamma: float = self._twiss.gamma0  # relativistic gamma
        betxbety: ArrayLike = self._twiss.betx * self._twiss.bety  # beta_x * beta_y term
        epsxepsy: ArrayLike = gemitt_x * gemitt_y  # eps_x * eps_y term
        betx_over_epsx: ArrayLike = self._twiss.betx / gemitt_x  # beta_x / eps_x term
        bety_over_epsy: ArrayLike = self._twiss.bety / gemitt_y  # beta_y / eps_y term
        # ----------------------------------------------------------------------------------------------
        # Adjust dispersion and dispersion prime by multiplied by relativistic beta, in order to be in the
        # deltap and not the pt frame (default in MAD-X / xsuite). Necessary for non-relativistic beams
        LOGGER.debug("Adjusting Dx, Dy, Dpx, Dpy to be in the pt frame")
        Dx: ArrayLike = self._twiss.dx * beta
        Dy: ArrayLike = self._twiss.dy * beta
        Dpx: ArrayLike = self._twiss.dpx * beta
        Dpy: ArrayLike = self._twiss.dpy * beta
        # ----------------------------------------------------------------------------------------------
        # Computing Phi_{x,y} as defined in Eq (6) of the note
        LOGGER.debug("Computing Phi_x, Phi_y, H_x and H_y at all elements")
        phix: ArrayLike = phi(self._twiss.betx, self._twiss.alfx, Dx, Dpx)
        phiy: ArrayLike = phi(self._twiss.bety, self._twiss.alfy, Dy, Dpy)
        # ----------------------------------------------------------------------------------------------
        b: ArrayLike = (
            (betx_over_epsx + bety_over_epsy)
            * (
                (gamma**2 * Dx**2) / (gemitt_x * self._twiss.betx)
                + (gamma**2 * Dy**2) / (gemitt_y * self._twiss.bety)
                + gamma**2 / sigma_delta**2
            )
            + betxbety * gamma**2 * (phix**2 + phiy**2) / (epsxepsy)
            + (betxbety / epsxepsy)
        )
        return b

    def _c(self, gemitt_x: float, gemitt_y: float, sigma_delta: float) -> ArrayLike:
        """
        Computes the `c` term of Table 1 in the MAD-X note.

        If comparing to the MAD-X Fortran code, this corresponds to (and
        was benchmarked against) the `cprime` variable there.

        Parameters
        ----------
        gemitt_x : float
            Horizontal geometric emittance in [m].
        gemitt_y : float
            Vertical geometric emittance in [m].
        sigma_delta : float
            The momentum spread.

        Returns
        -------
        ArrayLike
            An array with the `c` term, at each element in the lattice.
        """
        # ----------------------------------------------------------------------------------------------
        # We compute (once) some convenience terms used a lot in the equations, for efficiency & clarity
        beta: float = self._twiss.beta0  # relativistic beta
        gamma: float = self._twiss.gamma0  # relativistic gamma
        betxbety: ArrayLike = self._twiss.betx * self._twiss.bety  # beta_x * beta_y term
        epsxepsy: ArrayLike = gemitt_x * gemitt_y  # eps_x * eps_y term
        # ----------------------------------------------------------------------------------------------
        # Adjust dispersion and dispersion prime by multiplied by relativistic beta, in order to be in the
        # deltap and not the pt frame (default in MAD-X / xsuite). Necessary for non-relativistic beams
        LOGGER.debug("Adjusting Dx, Dy, Dpx, Dpy to be in the pt frame")
        Dx: ArrayLike = self._twiss.dx * beta
        Dy: ArrayLike = self._twiss.dy * beta
        # ----------------------------------------------------------------------------------------------
        c: ArrayLike = (betxbety / (epsxepsy)) * (
            (gamma**2 * Dx**2) / (gemitt_x * self._twiss.betx)
            + (gamma**2 * Dy**2) / (gemitt_y * self._twiss.bety)
            + gamma**2 / sigma_delta**2
        )
        return c

    def _ax(self, gemitt_x: float, gemitt_y: float, sigma_delta: float) -> ArrayLike:
        """
        Computes the `ax` term of Table 1 in the MAD-X note.

        If comparing to the MAD-X Fortran code, this corresponds to (and
        was benchmarked against) the `tx1 * cprime / bracket_x` terms there.

        Parameters
        ----------
        gemitt_x : float
            Horizontal geometric emittance in [m].
        gemitt_y : float
            Vertical geometric emittance in [m].
        sigma_delta : float
            The momentum spread.

        Returns
        -------
        ArrayLike
            An array with the `ax` term, at each element in the lattice.
        """
        # ----------------------------------------------------------------------------------------------
        # We define new shorter names for a lot of arrays, for clarity of the expressions below
        betx: ArrayLike = self._twiss.betx  # horizontal beta-functions
        bety: ArrayLike = self._twiss.bety  # vertical beta-functions
        sigd: float = sigma_delta  # momentum spread
        # ----------------------------------------------------------------------------------------------
        # We compute (once) some convenience terms used a lot in the equations, for efficiency & clarity
        beta: float = self._twiss.beta0  # relativistic beta
        gamma: float = self._twiss.gamma0  # relativistic gamma
        betx_over_epsx: ArrayLike = betx / gemitt_x  # beta_x / eps_x term
        bety_over_epsy: ArrayLike = bety / gemitt_y  # beta_y / eps_y term
        # ----------------------------------------------------------------------------------------------
        # Adjust dispersion and dispersion prime by multiplied by relativistic beta, in order to be in the
        # deltap and not the pt frame (default in MAD-X / xsuite). Necessary for non-relativistic beams
        LOGGER.debug("Adjusting Dx, Dy, Dpx, Dpy to be in the pt frame")
        Dx: ArrayLike = self._twiss.dx * beta
        Dy: ArrayLike = self._twiss.dy * beta
        Dpx: ArrayLike = self._twiss.dpx * beta
        Dpy: ArrayLike = self._twiss.dpy * beta
        # ----------------------------------------------------------------------------------------------
        # Computing Phi_{x,y} amd H_{x,y} as defined in Eq (6) and Eq (7) of the note
        LOGGER.debug("Computing Phi_x, Phi_y, H_x and H_y at all elements")
        phix: ArrayLike = phi(self._twiss.betx, self._twiss.alfx, Dx, Dpx)
        phiy: ArrayLike = phi(self._twiss.bety, self._twiss.alfy, Dy, Dpy)
        Hx: ArrayLike = (Dx**2 + self._twiss.betx**2 * phix**2) / self._twiss.betx
        Hy: ArrayLike = (Dy**2 + self._twiss.bety**2 * phiy**2) / self._twiss.bety
        # ----------------------------------------------------------------------------------------------
        ax: ArrayLike = (
            2 * gamma**2 * (Hx / gemitt_x + Hy / gemitt_y + 1 / sigd**2)
            - (betx * Hy) / (Hx * gemitt_y)
            + (betx / (Hx * gamma**2)) * (2 * betx_over_epsx - bety_over_epsy - gamma**2 / sigd**2)
            - 2 * betx_over_epsx
            - bety_over_epsy
            + (betx / (Hx * gamma**2)) * (6 * betx_over_epsx * gamma**2 * phix**2)
        )
        return ax

    def _bx(self, gemitt_x: float, gemitt_y: float, sigma_delta: float) -> ArrayLike:
        """
        Computes the `bx` term of Table 1 in the MAD-X note.

        If comparing to the MAD-X Fortran code, this corresponds to (and
        was benchmarked against) the `tx2 * cprime / bracket_x` terms there.

        Parameters
        ----------
        gemitt_x : float
            Horizontal geometric emittance in [m].
        gemitt_x : float
            Vertical geometric emittance in [m].
        sigma_delta : float
            The momentum spread.

        Returns
        -------
        ArrayLike
            An array with the `bx` term, at each element in the lattice.
        """
        # ----------------------------------------------------------------------------------------------
        # We define new shorter names for a lot of arrays, for clarity of the expressions below
        betx: ArrayLike = self._twiss.betx  # horizontal beta-functions
        bety: ArrayLike = self._twiss.bety  # vertical beta-functions
        sigd: float = sigma_delta  # momentum spread
        # ----------------------------------------------------------------------------------------------
        # We compute (once) some convenience terms used a lot in the equations, for efficiency & clarity
        beta: float = self._twiss.beta0  # relativistic beta
        gamma: float = self._twiss.gamma0  # relativistic gamma
        betx_over_epsx: ArrayLike = betx / gemitt_x  # beta_x / eps_x term
        bety_over_epsy: ArrayLike = bety / gemitt_y  # beta_y / eps_y term
        # ----------------------------------------------------------------------------------------------
        # Adjust dispersion and dispersion prime by multiplied by relativistic beta, in order to be in the
        # deltap and not the pt frame (default in MAD-X / xsuite). Necessary for non-relativistic beams
        LOGGER.debug("Adjusting Dx, Dy, Dpx, Dpy to be in the pt frame")
        Dx: ArrayLike = self._twiss.dx * beta
        Dy: ArrayLike = self._twiss.dy * beta
        Dpx: ArrayLike = self._twiss.dpx * beta
        Dpy: ArrayLike = self._twiss.dpy * beta
        # ----------------------------------------------------------------------------------------------
        # Computing Phi_{x,y} amd H_{x,y} as defined in Eq (6) and Eq (7) of the note
        LOGGER.debug("Computing Phi_x, Phi_y, H_x and H_y at all elements")
        phix: ArrayLike = phi(self._twiss.betx, self._twiss.alfx, Dx, Dpx)
        phiy: ArrayLike = phi(self._twiss.bety, self._twiss.alfy, Dy, Dpy)
        Hx: ArrayLike = (Dx**2 + self._twiss.betx**2 * phix**2) / self._twiss.betx
        Hy: ArrayLike = (Dy**2 + self._twiss.bety**2 * phiy**2) / self._twiss.bety
        # ----------------------------------------------------------------------------------------------
        bx: ArrayLike = (
            (betx_over_epsx + bety_over_epsy)
            * (gamma**2 * Hx / gemitt_x + gamma**2 * Hy / gemitt_y + gamma**2 / sigd**2)
            - gamma**2 * (betx_over_epsx**2 * phix**2 + bety_over_epsy**2 * phiy**2)
            + betx_over_epsx * (betx_over_epsx - 4 * bety_over_epsy)
            + (betx / (Hx * gamma**2))
            * (
                (gamma**2 / sigd**2) * (betx_over_epsx - 2 * bety_over_epsy)
                + betx_over_epsx * bety_over_epsy
                + 6 * betx_over_epsx * bety_over_epsy * gamma**2 * phix**2
                + gamma**2 * (2 * bety_over_epsy**2 * phiy**2 - betx_over_epsx**2 * phix**2)
            )
            + ((betx * Hy) / (gemitt_y * Hx)) * (betx_over_epsx - 2 * bety_over_epsy)
        )
        return bx

    def _ay(self, gemitt_x: float, gemitt_y: float, sigma_delta: float) -> ArrayLike:
        """
        Computes the `ay` term of Table 1 in the MAD-X note.

        If comparing to the MAD-X Fortran code, this corresponds to (and
        was benchmarked against) the `ty1 * cprime` terms there.

        Parameters
        ----------
        gemitt_x : float
            Horizontal geometric emittance in [m].
        gemitt_y : float
            Vertical geometric emittance in [m].
        sigma_delta : float
            The momentum spread.

        Returns
        -------
        ArrayLike
            An array with the `ay` term, at each element in the lattice.
        """
        # ----------------------------------------------------------------------------------------------
        # We compute (once) some convenience terms used a lot in the equations, for efficiency & clarity
        beta: float = self._twiss.beta0  # relativistic beta
        gamma: float = self._twiss.gamma0  # relativistic gamma
        betx_over_epsx: ArrayLike = self._twiss.betx / gemitt_x  # beta_x / eps_x term
        bety_over_epsy: ArrayLike = self._twiss.bety / gemitt_y  # beta_y / eps_y term
        # ----------------------------------------------------------------------------------------------
        # Adjust dispersion and dispersion prime by multiplied by relativistic beta, in order to be in the
        # deltap and not the pt frame (default in MAD-X / xsuite). Necessary for non-relativistic beams
        LOGGER.debug("Adjusting Dx, Dy, Dpx, Dpy to be in the pt frame")
        Dx: ArrayLike = self._twiss.dx * beta
        Dy: ArrayLike = self._twiss.dy * beta
        Dpx: ArrayLike = self._twiss.dpx * beta
        Dpy: ArrayLike = self._twiss.dpy * beta
        # ----------------------------------------------------------------------------------------------
        # Computing Phi_{x,y} amd H_{x,y} as defined in Eq (6) and Eq (7) of the note
        LOGGER.debug("Computing Phi_x, Phi_y, H_x and H_y at all elements")
        phix: ArrayLike = phi(self._twiss.betx, self._twiss.alfx, Dx, Dpx)
        phiy: ArrayLike = phi(self._twiss.bety, self._twiss.alfy, Dy, Dpy)
        Hx: ArrayLike = (Dx**2 + self._twiss.betx**2 * phix**2) / self._twiss.betx
        Hy: ArrayLike = (Dy**2 + self._twiss.bety**2 * phiy**2) / self._twiss.bety
        # ----------------------------------------------------------------------------------------------
        ay: ArrayLike = (
            -(gamma**2)
            * (
                Hx / gemitt_x
                + 2 * Hy / gemitt_y
                + (self._twiss.betx * Hy) / (self._twiss.bety * gemitt_x)
                + 1 / sigma_delta**2
            )
            + 2 * gamma**4 * Hy / self._twiss.bety * (Hy / gemitt_y + Hx / gemitt_x)
            + 2 * gamma**4 * Hy / (self._twiss.bety * sigma_delta**2)
            - (betx_over_epsx - 2 * bety_over_epsy)
            + (6 * bety_over_epsy * gamma**2 * phiy**2)
        )
        return ay

    def _by(self, gemitt_x: float, gemitt_y: float, sigma_delta: float) -> ArrayLike:
        """
        Computes the `by` term of Table 1 in the MAD-X note.

        If comparing to the MAD-X Fortran code, this corresponds to (and
        was benchmarked against) the `ty2 * cprime` terms there.

        Parameters
        ----------
        gemitt_x : float
            Horizontal geometric emittance in [m].
        gemitt_y : float
            Vertical geometric emittance in [m].
        sigma_delta : float
            The momentum spread.

        Returns
        -------
        ArrayLike
            An array with the `by` term, at each element in the lattice.
        """
        # ----------------------------------------------------------------------------------------------
        # We compute (once) some convenience terms used a lot in the equations, for efficiency & clarity
        beta: float = self._twiss.beta0  # relativistic beta
        gamma: float = self._twiss.gamma0  # relativistic gamma
        betxbety: ArrayLike = self._twiss.betx * self._twiss.bety  # beta_x * beta_y term
        epsxepsy: ArrayLike = gemitt_x * gemitt_y  # eps_x * eps_y term
        betx_over_epsx: ArrayLike = self._twiss.betx / gemitt_x  # beta_x / eps_x term
        bety_over_epsy: ArrayLike = self._twiss.bety / gemitt_y  # beta_y / eps_y term
        # ----------------------------------------------------------------------------------------------
        # Adjust dispersion and dispersion prime by multiplied by relativistic beta, in order to be in the
        # deltap and not the pt frame (default in MAD-X / xsuite). Necessary for non-relativistic beams
        LOGGER.debug("Adjusting Dx, Dy, Dpx, Dpy to be in the pt frame")
        Dx: ArrayLike = self._twiss.dx * beta
        Dy: ArrayLike = self._twiss.dy * beta
        Dpx: ArrayLike = self._twiss.dpx * beta
        Dpy: ArrayLike = self._twiss.dpy * beta
        # ----------------------------------------------------------------------------------------------
        # Computing Phi_{x,y} amd H_{x,y} as defined in Eq (6) and Eq (7) of the note
        LOGGER.debug("Computing Phi_x, Phi_y, H_x and H_y at all elements")
        phix: ArrayLike = phi(self._twiss.betx, self._twiss.alfx, Dx, Dpx)
        phiy: ArrayLike = phi(self._twiss.bety, self._twiss.alfy, Dy, Dpy)
        Hx: ArrayLike = (Dx**2 + self._twiss.betx**2 * phix**2) / self._twiss.betx
        Hy: ArrayLike = (Dy**2 + self._twiss.bety**2 * phiy**2) / self._twiss.bety
        # ----------------------------------------------------------------------------------------------
        by: ArrayLike = (
            gamma**2 * (bety_over_epsy - 2 * betx_over_epsx) * (Hx / gemitt_x + 1 / sigma_delta**2)
            + gamma**2 * Hy / gemitt_y * (bety_over_epsy - 4 * betx_over_epsx)
            + (betxbety / epsxepsy)
            + gamma**2 * (2 * betx_over_epsx**2 * phix**2 - bety_over_epsy**2 * phiy**2)
            + gamma**4
            * Hy
            / self._twiss.bety
            * (betx_over_epsx + bety_over_epsy)
            * (Hy / gemitt_y + 1 / sigma_delta**2)
            + gamma**4 * Hx * Hy / (self._twiss.bety * gemitt_x) * (betx_over_epsx + bety_over_epsy)
            - gamma**4
            * Hy
            / self._twiss.bety
            * (betx_over_epsx**2 * phix**2 + bety_over_epsy**2 * phiy**2)
            + 6 * gamma**2 * phiy**2 * betx_over_epsx * bety_over_epsy
        )
        return by

    def _az(self, gemitt_x: float, gemitt_y: float, sigma_delta: float) -> ArrayLike:
        """
        Computes the `az` term of Table 1 in the MAD-X note.

        If comparing to the MAD-X Fortran code, this corresponds to (and
        was benchmarked against) the `tl1 * cprime` terms there.

        Parameters
        ----------
        gemitt_x : float
            Horizontal geometric emittance in [m].
        gemitt_y : float
            Vertical geometric emittance in [m].
        sigma_delta : float
            The momentum spread.

        Returns
        -------
        ArrayLike
            An array with the `az` term, at each element in the lattice.
        """
        # ----------------------------------------------------------------------------------------------
        # We compute (once) some convenience terms used a lot in the equations, for efficiency & clarity
        beta: float = self._twiss.beta0  # relativistic beta
        gamma: float = self._twiss.gamma0  # relativistic gamma
        betx_over_epsx: ArrayLike = self._twiss.betx / gemitt_x  # beta_x / eps_x term
        bety_over_epsy: ArrayLike = self._twiss.bety / gemitt_y  # beta_y / eps_y term
        # ----------------------------------------------------------------------------------------------
        # Adjust dispersion and dispersion prime by multiplied by relativistic beta, in order to be in the
        # deltap and not the pt frame (default in MAD-X / xsuite). Necessary for non-relativistic beams
        LOGGER.debug("Adjusting Dx, Dy, Dpx, Dpy to be in the pt frame")
        Dx: ArrayLike = self._twiss.dx * beta
        Dy: ArrayLike = self._twiss.dy * beta
        Dpx: ArrayLike = self._twiss.dpx * beta
        Dpy: ArrayLike = self._twiss.dpy * beta
        # ----------------------------------------------------------------------------------------------
        # Computing Phi_{x,y} amd H_{x,y} as defined in Eq (6) and Eq (7) of the note
        LOGGER.debug("Computing Phi_x, Phi_y, H_x and H_y at all elements")
        phix: ArrayLike = phi(self._twiss.betx, self._twiss.alfx, Dx, Dpx)
        phiy: ArrayLike = phi(self._twiss.bety, self._twiss.alfy, Dy, Dpy)
        Hx: ArrayLike = (Dx**2 + self._twiss.betx**2 * phix**2) / self._twiss.betx
        Hy: ArrayLike = (Dy**2 + self._twiss.bety**2 * phiy**2) / self._twiss.bety
        # ----------------------------------------------------------------------------------------------
        az: ArrayLike = (
            2 * gamma**2 * (Hx / gemitt_x + Hy / gemitt_y + 1 / sigma_delta**2)
            - betx_over_epsx
            - bety_over_epsy
        )
        return az

    def _bz(self, gemitt_x: float, gemitt_y: float, sigma_delta: float) -> ArrayLike:
        """
        Computes the `bz` term of Table 1 in the MAD-X note.

        If comparing to the MAD-X Fortran code, this corresponds to (and
        was benchmarked against) the `tl2 * cprime` terms there.

        Parameters
        ----------
        gemitt_x : float
            Horizontal geometric emittance in [m].
        gemitt_y : float
            Vertical geometric emittance in [m].
        sigma_delta : float
            The momentum spread.

        Returns
        -------
        ArrayLike
            An array with the `bz` term, at each element in the lattice.
        """
        # ----------------------------------------------------------------------------------------------
        # We compute (once) some convenience terms used a lot in the equations, for efficiency & clarity
        beta: float = self._twiss.beta0  # relativistic beta
        gamma: float = self._twiss.gamma0  # relativistic gamma
        betx_over_epsx: ArrayLike = self._twiss.betx / gemitt_x  # beta_x / eps_x term
        bety_over_epsy: ArrayLike = self._twiss.bety / gemitt_y  # beta_y / eps_y term
        # ----------------------------------------------------------------------------------------------
        # Adjust dispersion and dispersion prime by multiplied by relativistic beta, in order to be in the
        # deltap and not the pt frame (default in MAD-X / xsuite). Necessary for non-relativistic beams
        LOGGER.debug("Adjusting Dx, Dy, Dpx, Dpy to be in the pt frame")
        Dx: ArrayLike = self._twiss.dx * beta
        Dy: ArrayLike = self._twiss.dy * beta
        Dpx: ArrayLike = self._twiss.dpx * beta
        Dpy: ArrayLike = self._twiss.dpy * beta
        # ----------------------------------------------------------------------------------------------
        # Computing Phi_{x,y} amd H_{x,y} as defined in Eq (6) and Eq (7) of the note
        LOGGER.debug("Computing Phi_x, Phi_y, H_x and H_y at all elements")
        phix: ArrayLike = phi(self._twiss.betx, self._twiss.alfx, Dx, Dpx)
        phiy: ArrayLike = phi(self._twiss.bety, self._twiss.alfy, Dy, Dpy)
        Hx: ArrayLike = (Dx**2 + self._twiss.betx**2 * phix**2) / self._twiss.betx
        Hy: ArrayLike = (Dy**2 + self._twiss.bety**2 * phiy**2) / self._twiss.bety
        # ----------------------------------------------------------------------------------------------
        bz: ArrayLike = (
            (betx_over_epsx + bety_over_epsy) * gamma**2 * (Hx / gemitt_x + Hy / gemitt_y + 1 / sigma_delta**2)
            - 2 * betx_over_epsx * bety_over_epsy
            - gamma**2 * (betx_over_epsx**2 * phix**2 + bety_over_epsy**2 * phiy**2)
        )
        return bz

    def _constants(
        self,
        gemitt_x: float,
        gemitt_y: float,
        sigma_delta: float,
        bunch_length: float,
        total_beam_intensity: int,
        bunched: bool = True,
    ) -> Tuple[float, ArrayLike, ArrayLike, float]:
        r"""
        Computes the constant terms of Eq (8) in the MAD-X note
        :cite:`CERN:Antoniou:Revision_IBS_MADX`. Returned are four terms:
        first the constant common to all planes, then the horizontal,
        vertical and longitudinal terms (in brackets in Eq (8)).

        The common constant and the longitudinal constant are floats. The
        horizontal and vertical terms are arrays, with one value per element
        in the lattice (as they depend on :math:`H_x` and :math:`\beta_y`,
        respectively).

        Parameters
        ----------
        gemitt_x : float
            Horizontal geometric emittance in [m].
        gemitt_y : float
            Vertical geometric emittance in [m].
        sigma_delta : float
            The momentum spread.
        bunch_length : float
            The bunch length in [m].
        total_beam_intensity : int
            The beam intensity to assume.
        bunched : bool
            Whether the beam is bunched or not (coasting). Defaults to `True`.

        Returns
        -------
        Tuple[float, ArrayLike, ArrayLike, float]
            Four variables corresponding to the common, horizontal, vertical and
            longitudinal 'constants' of Eq (8) in :cite:`CERN:Antoniou:Revision_IBS_MADX`.
            The horizontal and vertical ones are arrays, with a value per element.
        """
        # ----------------------------------------------------------------------------------------------
        # fmt: off
        # We define new shorter names for a lot of arrays, for clarity of the expressions below
        betx: ArrayLike = self._twiss.betx  # horizontal beta-functions
        bety: ArrayLike = self._twiss.bety  # vertical beta-functions
        alfx: ArrayLike = self._twiss.alfx  # horizontal alpha-functions
        # ----------------------------------------------------------------------------------------------
        # We compute (once) some convenience terms used a lot in the equations, for efficiency & clarity
        beta: float = self._twiss.beta0  # relativistic beta
        gamma: float = self._twiss.gamma0  # relativistic gamma
        bety_over_epsy: ArrayLike = bety / gemitt_y  # beta_y / eps_y term
        # ----------------------------------------------------------------------------------------------
        # Adjust dispersion and dispersion prime by multiplied by relativistic beta, in order to be in the
        # deltap and not the pt frame (default in MAD-X / xsuite). Necessary for non-relativistic beams
        LOGGER.debug("Adjusting Dx, Dy, Dpx, Dpy to be in the pt frame")
        Dx: ArrayLike = self._twiss.dx * beta
        Dpx: ArrayLike = self._twiss.dpx * beta
        # ----------------------------------------------------------------------------------------------
        # Computing Phi_x amd H_x as defined in Eq (6) and Eq (7) of the note
        LOGGER.debug("Computing Phi_x, Phi_y, H_x and H_y at all elements")
        phix: ArrayLike = phi(betx, alfx, Dx, Dpx)
        Hx: ArrayLike = (Dx**2 + betx**2 * phix**2) / betx
        # ----------------------------------------------------------------------------------------------
        # Compute the Coulomb logarithm and the common constant term in Eq (8) (the first fraction)
        # Below we give mass in MeV like in .growth_rates() (the m^3 terms cancel out)
        coulomb_logarithm = self.coulomb_log(
            gemitt_x=gemitt_x,
            gemitt_y=gemitt_y,
            sigma_delta=sigma_delta,
            bunch_length=bunch_length,
            total_beam_intensity=total_beam_intensity,
            bunched=bunched,
        )
        common_constant_term: float = (
            np.pi**2
            * self._particle.get_classical_particle_radius0()**2
            * c
            * (self._particle.mass0 * 1e-3)** 3
            * total_beam_intensity
            * coulomb_logarithm
            / (self._twiss.gamma0 * self._Gamma(gemitt_x, gemitt_y, sigma_delta, bunch_length, bunched))
        )
        # ----------------------------------------------------------------------------------------------
        # fmt: on
        # Compute the plane-dependent constants (in brackets) for each plane of Eq (8) in the MAD-X note
        const_x: ArrayLike = gamma**2 * Hx / gemitt_x
        const_y: ArrayLike = bety_over_epsy
        const_z: float = gamma**2 / sigma_delta**2
        # ----------------------------------------------------------------------------------------------
        # Return the four terms now - they are Tuple[float, ArrayLike, ArrayLike, float]
        return common_constant_term, const_x, const_y, const_z

    def growth_rates(
        self,
        gemitt_x: float = None,
        nemitt_x: float = None,
        gemitt_y: float = None,
        nemitt_y: float = None,
        sigma_delta: float = None,
        bunch_length: float = None,
        total_beam_intensity: int = None,
        bunched: bool = True,
        integration_intervals: int = 17,
    ) -> IBSGrowthRates:
        r"""
        Computes the ``IBS`` growth rates, named :math:`T_x, T_y` and :math:`T_z` in this
        code base, according to the Bjorken & Mtingwa formalism. These correspond to the
        (averaged) :math:`1 / \tau` terms of Eq (8) in :cite:`CERN:Antoniou:Revision_IBS_MADX`.
        The instance attribute `self.ibs_growth_rates` is automatically updated with the result
        of this method when it is called.

        .. warning::
            In ``MAD-X`` it is ensure that the Twiss table is centered.
            One might observe some small discrepancies against ``MAD-X``
            growth rates if not providing a centered Twiss table (by
            slicing the lattice first, for instance.)

        .. hint::
            The calculation is done according to the following steps, which are related
            to different equations in :cite:`CERN:Antoniou:Revision_IBS_MADX`:

                - Adjusts the :math:`D_x, D_y, D^{\prime}_{x}, D^{\prime}_{y}` terms (multiply by :math:`\beta_{rel}`) to be in the :math:`pt` frame.
                - Computes the various terms from Table 1 of the MAD-X note.
                - Computes the Coulomb logarithm and the common constant term (first fraction) of Eq (8).
                - Defines the integrands of integrals in Eq (8) of the MAD-X note.
                - Defines sub-intervals and integrates the above over all of them, getting growth rates at each element in the lattice.
                - Averages the results over the full circumference of the machine.

        Parameters
        ----------
        gemitt_x : float, optional
            Horizontal geometric emittance in [m]. Either this
            parameter or `nemitt_x` is required.
        nemitt_x : float, optional
            Horizontal normalized emittance in [m]. Either this
            parameter or `gemitt_x` is required.
        gemitt_y : float, optional
            Vertical geometric emittance in [m]. Either this
            parameter or `nemitt_y` is required.
        nemitt_y : float, optional
            Vertical normalized emittance in [m]. Either this
            parameter or `gemitt_y` is required.
        sigma_delta : float
            The momentum spread.
        bunch_length : float
            The bunch length in [m].
        total_beam_intensity : int
            The beam intensity to assume.
        bunched : bool
            Whether the beam is bunched or not (coasting). Defaults to `True`.
        integration_intervals : int
            The number of sub-intervals to use when integrating the integrands of
            Eq (8) of the MAD-X note. DO NOT change this parameter unless you know
            exactly what you are doing, as you might affect convergence. Defaults
            to 17.

        Returns
        -------
        IBSGrowthRates
            An ``IBSGrowthRates`` object with the computed growth rates.
        """
        LOGGER.info("Computing IBS growth rates for defined beam and optics parameters")
        # ----------------------------------------------------------------------------------------------
        # Perform checks on exclusive parameters and make sure we have what we need
        assert sigma_delta is not None, "Must provide 'sigma_delta'"
        assert bunch_length is not None, "Must provide 'bunch_length'"
        assert total_beam_intensity is not None, "Must provide 'total_beam_intensity'"
        assert any([gemitt_x, nemitt_x]), "Must provide either 'gemitt_x' or 'nemitt_x'"
        assert any([gemitt_y, nemitt_y]), "Must provide either 'gemitt_y' or 'nemitt_y'"
        if gemitt_x is not None:
            assert nemitt_x is None, "Cannot provide both 'gemitt_x' and 'nemitt_x'"
        if gemitt_y is not None:
            assert nemitt_y is None, "Cannot provide both 'gemitt_y' and 'nemitt_y'"
        if nemitt_x is not None:
            assert gemitt_x is None, "Cannot provide both 'gemitt_x' and 'nemitt_x'"
            gemitt_x = self._gemitt(nemitt_x)
        if nemitt_y is not None:
            assert gemitt_y is None, "Cannot provide both 'gemitt_y' and 'nemitt_y'"
            gemitt_y = self._gemitt(nemitt_y)
        # ----------------------------------------------------------------------------------------------
        # Getting the arrays from Table 1 of the MAD-X note (all implemented to ask for gemitt_[xy])
        # fmt: off
        LOGGER.debug("Computing terms from Table 1 of the MAD-X note")
        a: ArrayLike = self._a(gemitt_x, gemitt_y, sigma_delta)    # This is 'a' in MAD-X fortran code
        b: ArrayLike = self._b(gemitt_x, gemitt_y, sigma_delta)    # This is 'b' in MAD-X fortran code
        c: ArrayLike = self._c(gemitt_x, gemitt_y, sigma_delta)    # This is 'cprime' in MAD-X fortran code
        ax: ArrayLike = self._ax(gemitt_x, gemitt_y, sigma_delta)  # This is 'tx1 * cprime / bracket_x' in MAD-X fortran code
        bx: ArrayLike = self._bx(gemitt_x, gemitt_y, sigma_delta)  # This is 'tx2 * cprime / bracket_x' in MAD-X fortran code
        ay: ArrayLike = self._ay(gemitt_x, gemitt_y, sigma_delta)  # This is 'ty1 * cprime' in MAD-X fortran code
        by: ArrayLike = self._by(gemitt_x, gemitt_y, sigma_delta)  # This is 'ty2 * cprime' in MAD-X fortran code
        az: ArrayLike = self._az(gemitt_x, gemitt_y, sigma_delta)  # This is 'tl1 * cprime' in MAD-X fortran code
        bz: ArrayLike = self._bz(gemitt_x, gemitt_y, sigma_delta)  # This is 'tl2 * cprime' in MAD-X fortran code                                   
        # fmt: on
        # ----------------------------------------------------------------------------------------------
        # Getting the constant term and the bracket terms from Eq (8) of the MAD-X note
        LOGGER.debug("Computing common constant term and bracket terms from Eq (8) of the MAD-X note")
        common_constant_term, bracket_x, bracket_y, bracket_z = self._constants(
            gemitt_x, gemitt_y, sigma_delta, bunch_length, total_beam_intensity, bunched
        )
        # ----------------------------------------------------------------------------------------------
        # Defining the integrands from Eq (8) of the MAD-X note, for each plane (remember these functions
        # are vectorised since a, b, c, ax, bx, ay, by are all arrays). The bracket terms are included!
        LOGGER.debug("Defining integrands of Eq (8) of the MAD-X note")

        def Ix_integrand_vectorized(_lambda: float) -> ArrayLike:
            """
            Vectorized function for the integrand of horizontal term of Eq (8)
            in MAD-X note, but it includes the square bracket term.
            """
            numerator: ArrayLike = bracket_x * np.sqrt(_lambda) * (ax * _lambda + bx)
            denominator: ArrayLike = (_lambda**3 + a * _lambda**2 + b * _lambda + c) ** (3 / 2)
            return numerator / denominator

        def Iy_integrand_vectorized(_lambda: float) -> ArrayLike:
            """
            Vectorized function for the integrand of vertical term of Eq (8)
            in MAD-X note, but it includes the square bracket term.
            """
            numerator: ArrayLike = bracket_y * np.sqrt(_lambda) * (ay * _lambda + by)
            denominator: ArrayLike = (_lambda**3 + a * _lambda**2 + b * _lambda + c) ** (3 / 2)
            return numerator / denominator

        def Iz_integrand_vectorized(_lambda: float) -> ArrayLike:
            """
            Vectorized function for the integrand of longitudinal term of
            Eq (8) in MAD-X note, but it includes the square bracket term.
            """
            numerator: ArrayLike = bracket_z * np.sqrt(_lambda) * (az * _lambda + bz)
            denominator: ArrayLike = (_lambda**3 + a * _lambda**2 + b * _lambda + c) ** (3 / 2)
            return numerator / denominator

        # ----------------------------------------------------------------------------------------------
        # Defining a function to perform the integration, which is done sub-interval by sub-interval
        def calculate_integral_vectorized(func: Callable) -> ArrayLike:
            """
            Computes integral of Eq (8) of the MAD-X note, when provided with the integrand.
            The integrand being a vectorized function, this returns an array with the result
            of the integral at each element.

            This computation defines several intervals on which to perform the integration on,
            and performs the integration of the provided function on each one. At each step,
            we add the intermediate values to the final result, which is returned.

            Parameters
            ----------
            func: Callable
                Vectorized function defining the integrand.

            Returns
            -------
            ArrayLike
                An array with the result of the integration at each element in the lattice.
            """
            nb_elements: int = ax.size
            result: ArrayLike = np.zeros(nb_elements)

            # The following two hold the values for starts and ends of sub-intervals on which to integrate
            interval_starts = np.array([10**i for i in np.arange(0, int(integration_intervals) - 1)])
            interval_ends = np.array([10**i for i in np.arange(1, int(integration_intervals))])

            # Now we loop over the intervals and integrate the function on each one, using scipy
            # We add the intermediate integration result of each interval to our final result
            for start, end in zip(interval_starts, interval_ends):
                integrals, _ = quad_vec(func, start, end)  # integrals is an array
                result += integrals
            return result

        # ----------------------------------------------------------------------------------------------
        # Now we loop over the lattice and compute the integrals at each element
        LOGGER.debug("Computing integrals of Eq (8) of the MAD-X note - at each element in the lattice")
        Tx_array: ArrayLike = calculate_integral_vectorized(Ix_integrand_vectorized)
        Ty_array: ArrayLike = calculate_integral_vectorized(Iy_integrand_vectorized)
        Tz_array: ArrayLike = calculate_integral_vectorized(Iz_integrand_vectorized)
        # ----------------------------------------------------------------------------------------------
        # Don't forget to multiply by the common constant term here
        LOGGER.debug("Including common constant term of Eq (8) of the MAD-X note")
        Tx_array *= common_constant_term
        Ty_array *= common_constant_term
        Tz_array *= common_constant_term
        # ----------------------------------------------------------------------------------------------
        # For a better average, interpolate these intermediate growth rates through the lattice
        LOGGER.debug("Interpolating intermediate growth rates through the lattice")
        _tx = interp1d(self._twiss.s, Tx_array)
        _ty = interp1d(self._twiss.s, Ty_array)
        _tz = interp1d(self._twiss.s, Tz_array)
        # ----------------------------------------------------------------------------------------------
        # And now cmpute the final growth rates for each plane as an average of these interpolated
        # functions over the whole lattice - also ensure conversion to float afterwards!
        LOGGER.debug("Getting average growth rates over the lattice")
        with warnings.catch_warnings():  # Catch and ignore the scipy.integrate.IntegrationWarning
            warnings.simplefilter("ignore", category=UserWarning)
            Tx: float = float(quad(_tx, self._twiss.s[0], self._twiss.s[-1])[0] / self._twiss.circumference)
            Ty: float = float(quad(_ty, self._twiss.s[0], self._twiss.s[-1])[0] / self._twiss.circumference)
            Tz: float = float(quad(_tz, self._twiss.s[0], self._twiss.s[-1])[0] / self._twiss.circumference)
        result = IBSGrowthRates(Tx, Ty, Tz)
        # ----------------------------------------------------------------------------------------------
        # Self-update the instance's attribute before returning
        self.ibs_growth_rates = result
        return result
