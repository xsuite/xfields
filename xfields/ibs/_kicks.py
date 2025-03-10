# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from __future__ import annotations  # important for sphinx to alias ArrayLike

from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
import xobjects as xo
from scipy.constants import c
from scipy.special import elliprd

from xfields.ibs._analytical import BjorkenMtingwaIBS
from xfields.ibs._formulary import (
    _assert_accepted_context,
    _beam_intensity,
    _bunch_length,
    _current_turn,
    _gemitt_x,
    _gemitt_y,
    _mean_px,
    _mean_py,
    _sigma_delta,
    _sigma_px,
    _sigma_py,
    _sigma_x,
    _sigma_y,
    phi,
)

if TYPE_CHECKING:
    import xtrack as xt
    from numpy.typing import ArrayLike

    from xfields.ibs._analytical import IBSAmplitudeGrowthRates, IBSEmittanceGrowthRates

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
        Horizontal diffusion coefficient, in [s^-1].
    Dy : float
        Vertical diffusion coefficient, in [s^-1].
    Dz : float
        Longitudinal diffusion coefficient, in [s^-1].
    """

    _xofields = {
        "Dx": xo.Float64,
        "Dy": xo.Float64,
        "Dz": xo.Float64,
    }

    def __init__(self, Dx: float, Dy: float, Dz: float) -> None:
        """Init by providing the diffusion coefficients."""
        self.xoinitialize(Dx=Dx, Dy=Dy, Dz=Dz)

    def as_tuple(self) -> tuple[float, float, float]:
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
        Horizontal friction coefficient, in [s^-1].
    Fy : float
        Vertical friction coefficient, in [s^-1].
    Fz : float
        Longitudinal friction coefficient, in [s^-1].
    """

    _xofields = {
        "Fx": xo.Float64,
        "Fy": xo.Float64,
        "Fz": xo.Float64,
    }

    def __init__(self, Fx: float, Fy: float, Fz: float) -> None:
        """Init by providing the friction coefficients."""
        self.xoinitialize(Fx=Fx, Fy=Fy, Fz=Fz)

    def as_tuple(self) -> tuple[float, float, float]:
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
        Horizontal kick coefficient, in [s^-1].
    Ky : float
        Vertical kick coefficient, in [s^-1].
    Kz : float
        Longitudinal kick coefficient, in [s^-1].
    """

    _xofields = {
        "Kx": xo.Float64,
        "Ky": xo.Float64,
        "Kz": xo.Float64,
    }

    def __init__(self, Kx: float, Ky: float, Kz: float) -> None:
        """Init by providing the kick coefficients."""
        self.xoinitialize(Kx=Kx, Ky=Ky, Kz=Kz)

    def as_tuple(self) -> tuple[float, float, float]:
        """Return the growth rates as a tuple."""
        return float(self.Kx), float(self.Ky), float(self.Kz)


# ----- Useful Functions ----- #


# TODO: someday replace this with what Gianni is working on in xfields.longitudinal_profiles
def line_density(particles: xt.Particles, num_slices: int) -> ArrayLike:
    """
    Returns the longitudinal "line density" of the provided `xtrack.Particles`.
    It is used as a weighing factor for the application of IBS kicks, so that
    particles in the denser parts of the bunch will receive a larger kick, and
    vice versa.

    Parameters
    ----------
    particles : xtrack.Particles
        The `xtrack.Particles` object to compute the line density for.
    num_slices : int
        The number of slices to use for the computation of the bins.

    Returns
    -------
    ArrayLike
        An array with the weight value for each particle, to be used
        as a weight in the kicks application. This array is on the
        context device of the particles.
    """
    # ----------------------------------------------------------------------------------------------
    # Get the nplike_lib from the particles' context, to compute on the context device
    nplike = particles._context.nplike_lib
    # ----------------------------------------------------------------------------------------------
    # Determine properties from longitudinal particles distribution: cuts, slice width, bunch length
    LOGGER.debug("Determining longitudinal particles distribution properties")
    zeta: ArrayLike = particles.zeta[particles.state > 0]  # only consider active particles
    z_cut_head: float = nplike.max(zeta)  # z cut at front of bunch
    z_cut_tail: float = nplike.min(zeta)  # z cut at back of bunch
    slice_width: float = (z_cut_head - z_cut_tail) / num_slices  # slice width
    # ----------------------------------------------------------------------------------------------
    # Determine bin edges and bin centers for the distribution
    LOGGER.debug("Determining bin edges and bin centers for the distribution")
    bin_edges: ArrayLike = nplike.linspace(
        z_cut_tail - 1e-7 * slice_width,
        z_cut_head + 1e-7 * slice_width,
        num=num_slices + 1,
        dtype=np.float64,
    )
    bin_centers: ArrayLike = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    # ----------------------------------------------------------------------------------------------
    # Compute histogram on longitudinal distribution then compute and return normalized line density
    counts_normed, bin_edges = nplike.histogram(zeta, bin_edges, density=True)
    return nplike.interp(zeta, bin_centers, counts_normed)


# ----- Parent Class to Identify the IBS Kicks ----- #


class IBSKick:
    """
    General class for IBS kicks to inherit from.
    """

    iscollective = True  # based on alive particles, need them all here

    def to_dict(self) -> None:
        """Raises an error as the line should be saved without the IBS kick element."""
        raise NotImplementedError("IBS kick elements should not be saved as part of the line")


# ----- Simple Kick Implementation ----- #


class IBSAnalyticalKick(IBSKick):
    r"""
    Beam element to apply IBS effects to particles during tracking according to
    the formalism introduced in :cite:`PRAB:Bruce:Simple_IBS_Kicks`. It provides
    momenta kicks based on analytical growth rates, weighted by the longitudinal
    line density of the particles and including a random component.

    The element starts off by default (will not affect particles) and has to be
    configured through the `line.configure_intrabeam_scattering` method.

    Warnings
    --------
        This formalism is only valid **above** transition energy. The implemented
        weighted random-component momentum kick depends on the square root of the
        growth rate, which is set to 0 if it is negative. Below transition it is
        common to observe negative growth rates and emittance shrinkage, which this
        kick would not be reprensentative of. A message is logged to inform the user
        when this happens. For machines below transition energy, the kinetic formalism
        should be used instead: see the `IBSKineticKick` class).

    Attributes
    ----------
    num_slices : int
        The number of slices used for the computation of the bunch's
        longitudinal line density.
    formalism : str
        The formalism used for the computation of the growth rates.
    update_every : int
        The frequency at which to recompute the kick coefficients, in
        number of turns. They will be computed at the first turn of
        tracking, and then every `update_every` turns afterwards.
    kick_coefficients : IBSKickCoefficients
        The computed kick coefficients. This self-updates when they
        are computed with the `.compute_kick_coefficients` method.
    """

    def __init__(self, formalism: str, num_slices: int) -> None:
        """
        Initialize the Simple IBS kick element. It is off
        by default and will have to be configured (see the
        line.configure_intrabeam_scattering method).

        Parameters
        ----------
        formalism : str
            Which formalism to use for the computation of the growth
            rates. Can be ``Nagaitsev`` or ``Bjorken-Mtingwa`` (also
            accepts ``B&M``), case-insensitively.
        num_slices : int
            The number of slices used for the computation of
            the bunch's longitudinal line density.
        """
        assert formalism.lower() in ("nagaitsev", "bjorken-mtingwa", "b&m")
        self.num_slices = num_slices
        self.formalism = formalism
        self.kick_coefficients: IBSKickCoefficients = None
        # The following are needed but start unset. They are set
        # when calling line.configure_intrabeam_scattering()
        self.update_every: int = None
        self._name: str = None
        self._twiss: xt.TwissTable = None
        self._scale_strength: float = 0  # by default element does not "track"

    def _coefficients_need_recompute(self, particles: xt.Particles) -> bool:
        """
        Called to determine if the kick coefficients need to be recomputed before
        applying kicks. This sets an internal flag. Coefficients need recomputing
        if they are `None` or if the current turn is a multiple of the frequency
        at which to recompute them.

        Parameters
        ----------
        particles : xtrack.Particles
            The particles to apply the IBS kicks to and compute it from.

        Returns
        -------
        bool
            Whether the coefficients need to be recomputed.
        """
        # ----------------------------------------------------------------------------------------------
        # Check coefficients existence and if current turn is a multiple of the frequency to recompute
        if self.kick_coefficients is None or _current_turn(particles) % self.update_every == 0:
            return True
        return False

    def compute_kick_coefficients(self, particles: xt.Particles, **kwargs) -> IBSKickCoefficients:
        r"""
        Computes the ``IBS`` kick coefficients, named :math:`K_x, K_y`
        and :math:`K_z` in this code base, from analytical growth rates.
        The coefficients correspond to the right-hand side of Eq (8) in
        :cite:`PRAB:Bruce:Simple_IBS_Kicks` without the line density
        :math:`\rho_t(t)` term, nor the random component :math:`r`.

        This coefficient corresponds to the scaling of the generated random
        distribution :math:`r` used for the momenta kicks and is expressed as
        :math:`K_u = \sigma_{p_u} \sqrt{2 T^{-1}_{IBS_u} T_{rev} \sigma_t \sqrt{\pi}}`.

        Notes
        -----
            The calculation is done according to the following steps, which are related
            to different terms in Eq (8) of :cite:`PRAB:Bruce:Simple_IBS_Kicks`:

                - Computes various properties from the non-lost particles in the bunch (:math:`\sigma_{x,y,\delta,t}`).
                - Computes the standard deviation of momenta for each plane (:math:`\sigma_{p_u}`).
                - Computes the constant term :math:`\sqrt{2 T_{rev} \sqrt{\pi}}`.
                - Computes the analytical growth rates :math:`T_{x,y,z}` (:math:`T^{-1}_{IBS_u}` in Eq (8)).
                - Computes, stores and returns the kick coefficients.

        Parameters
        ----------
        particles : xtrack.Particles
            The particles to apply the IBS kicks to and compute it from.
        **kwargs : dict, optional
            Keyword arguments will be passed to the growth rates calculation.
            Note that `gemitt_x`, `gemitt_y`, `sigma_delta`, and `bunch_length`
            are already provided.

        Returns
        -------
        IBSKickCoefficients
            An ``IBSKickCoefficients`` object with the computed kick coefficients.
        """
        # ----------------------------------------------------------------------------------------------
        # This full computation (apart from getting properties from the particles object) is done
        # on the CPU, as the growth rates are computed on CPU. The rest of the computing is made of
        # small operations, better to do these on CPU than transfer to GPU and finish there. The full
        # computing is dominated by the growth rates computation in any case.
        # ----------------------------------------------------------------------------------------------
        # Make sure the particles' context will not cause issues
        _assert_accepted_context(particles._context)
        # ----------------------------------------------------------------------------------------------
        # Compute the momentum spread, bunch length and (geometric) emittances from the Particles object
        # fmt: off
        LOGGER.debug("Computing emittances, momentum spread and bunch length from particles")
        beam_intensity: float = _beam_intensity(particles)
        bunch_length: float = _bunch_length(particles)
        sigma_delta: float = _sigma_delta(particles)
        gemitt_x: float = _gemitt_x(particles, self._twiss["betx", self._name], self._twiss["dx", self._name])
        gemitt_y: float = _gemitt_y(particles, self._twiss["bety", self._name], self._twiss["dy", self._name])
        # ----------------------------------------------------------------------------------------------
        # Computing standard deviation of (normalized) momenta, corresponding to sigma_{pu} in Eq (8) of reference
        # Normalized: for momentum we have to multiply with 1/sqrt(gamma) = sqrt(beta) / sqrt(1 + alpha^2), and the
        # sqrt(beta) is included in the std of p[xy]. If bunch is rotated, the std takes from the "other plane" so
        # we take the normalized momenta to compensate.
        sigma_px_normalized: float = _sigma_px(particles, self._twiss["dpx", self._name]) / np.sqrt(1 + self._twiss["alfx", self._name] ** 2)
        sigma_py_normalized: float = _sigma_py(particles, self._twiss["dpy", self._name]) / np.sqrt(1 + self._twiss["alfy", self._name] ** 2)
        # ----------------------------------------------------------------------------------------------
        # Determine the "scaling factor", corresponding to 2 * sigma_t * sqrt(pi) in Eq (8) of reference
        scaling_factor: float = float(2 * np.sqrt(np.pi) * bunch_length)
        # ----------------------------------------------------------------------------------------------
        # Computing the analytical IBS growth rates through the instance's set TwissTable. Note that since
        # we have no way of detecting coasting beams we assume bunched and take the rms bunch length as is
        # (but we don't get the correction factor in the coulomb logarithm...)
        amp_growth_rates: IBSAmplitudeGrowthRates = self._twiss.get_ibs_growth_rates(
            formalism=self.formalism,
            gemitt_x=float(gemitt_x),
            gemitt_y=float(gemitt_y),
            sigma_delta=float(sigma_delta),
            bunch_length=float(bunch_length),
            total_beam_intensity=beam_intensity,
            **kwargs,
        )
        # ----------------------------------------------------------------------------------------------
        # In Xsuite we give growth rates in amplitude convention but the theory of R. Bruce assumes
        # the growth rates to be in emittance convention, so we do the conversion here.
        emit_growth_rates: IBSEmittanceGrowthRates = amp_growth_rates.to_emittance_growth_rates()
        # ----------------------------------------------------------------------------------------------
        # Making sure we do not have negative growth rates (see class docstring warning for detail)
        # In paper the growth rates are noted with a T so I stick to that
        Tx: float = 0.0 if emit_growth_rates.Kx < 0 else float(emit_growth_rates.Kx)
        Ty: float = 0.0 if emit_growth_rates.Ky < 0 else float(emit_growth_rates.Ky)
        Tz: float = 0.0 if emit_growth_rates.Kz < 0 else float(emit_growth_rates.Kz)
        if any(rate == 0 for rate in (Tx, Ty, Tz)):
            LOGGER.debug("At least one IBS growth rate was negative, and was set to 0")
        # ----------------------------------------------------------------------------------------------
        # Compute the kick coefficients - see function docstring for exact definition.
        # For the longitudinal plane, since the values are computed from ΔP/P but applied to the ΔE/E
        # (the particles.delta in Xsuite), we multiply by beta0**2 to adapt
        LOGGER.debug("Computing simple kick coefficients")
        beta0 = self._twiss.beta0
        revolution_frequency: float = 1 / self._twiss.T_rev0
        Kx = float(sigma_px_normalized * np.sqrt(2 * scaling_factor * Tx / revolution_frequency))
        Ky = float(sigma_py_normalized * np.sqrt(2 * scaling_factor * Ty / revolution_frequency))
        Kz = float(sigma_delta * np.sqrt(2 * scaling_factor * Tz / revolution_frequency) * beta0**2)
        result = IBSKickCoefficients(Kx, Ky, Kz)
        # ----------------------------------------------------------------------------------------------
        # Self-update the instance's attributes and then return the results
        self.kick_coefficients = result
        return result

    def track(self, particles: xt.Particles) -> None:
        """
        Method to determine and apply IBS momenta kicks based on the provided
        `xtrack.Particles`. The kicks are implemented according to Eq (8) of
        :cite:`PRAB:Bruce:Simple_IBS_Kicks`.

        Parameters
        ----------
        particles : xtrack.Particles
            The particles to apply the IBS kicks to and compute it from.
        """
        # ----------------------------------------------------------------------------------------------
        # Intercept here if the kick element is "off" and has no scale strength - do not compute for nothing
        if self._scale_strength == 0:
            return
        # ----------------------------------------------------------------------------------------------
        # Check if coefficients need to be recomputed before applying kicks & recompute if necessary
        if self._coefficients_need_recompute(particles) is True:
            LOGGER.debug("Recomputing simple kick coefficients before applying kicks")
            self.compute_kick_coefficients(particles)
        # ----------------------------------------------------------------------------------------------
        # Get the nplike_lib from the particles' context, to compute on the context device
        _assert_accepted_context(particles._context)
        nplike = particles._context.nplike_lib
        # ----------------------------------------------------------------------------------------------
        # Compute the line density - this is the rho_t(t) term in Eq (8) of reference
        rho_t: ArrayLike = line_density(particles, self.num_slices)  # on context | does not include _factor
        # ----------------------------------------------------------------------------------------------
        # Determining size of arrays for kicks to apply: only the non-lost particles in the bunch
        _size: int = particles.px[particles.state > 0].shape[0]  # same for py and delta
        # ----------------------------------------------------------------------------------------------
        # Determining kicks - this corresponds to the full result of Eq (8) of reference (\Delta p_u)
        LOGGER.debug("Determining kicks to apply")
        rng = nplike.random.default_rng()
        Kx, Ky, Kz = self.kick_coefficients.as_tuple()  # floats
        delta_px: ArrayLike = rng.standard_normal(_size) * Kx * np.sqrt(rho_t)  # on context
        delta_py: ArrayLike = rng.standard_normal(_size) * Ky * np.sqrt(rho_t)  # on context
        delta_delta: ArrayLike = rng.standard_normal(_size) * Kz * np.sqrt(rho_t)  # on context
        # ----------------------------------------------------------------------------------------------
        # Apply the kicks to the particles, applying the momenta deltas
        LOGGER.debug("Applying momenta kicks to the particles (on px, py and delta properties)")
        particles.px[particles.state > 0] += delta_px
        particles.py[particles.state > 0] += delta_py
        particles.delta[particles.state > 0] += delta_delta


# ----- Kinetic Kick Implementation ----- #


class IBSKineticKick(IBSKick):
    r"""
    Beam element to apply IBS effects to particles during tracking according to
    the formalism introduced in :cite:`NuclInstr:Zenkevich:Kinetic_IBS`. It provides
    momenta kicks based on analytical growth rates, weighted by the longitudinal
    line density of the particles and including a random component.

    The element starts off by default (will not affect particles) and has to be
    configured through the `line.configure_intrabeam_scattering` method.

    Attributes
    ----------
    num_slices : int
        The number of slices used for the computation of the bunch's
        longitudinal line density.
    update_every : int
        The frequency at which to recompute the kick coefficients, in
        number of turns. They will be computed at the first turn of
        tracking, and then every `update_every` turns afterwards.
    diffusion_coefficients : DiffusionCoefficients
        The computed diffusion coefficients, from the kinetic theory.
        This attribute self-updates when they are computed with the
        `.compute_kinetic_coefficients` method.
    friction_coefficients : FrictionCoefficients
        The computed friction coefficients, from the kinetic theory.
        This attribute self-updates when they are computed with the
        `.compute_kinetic_coefficients` method.
    """

    def __init__(self, num_slices: int) -> None:
        """
        Initialize the Simple IBS kick element. It is off
        by default and will have to be configured (see the
        line.configure_intrabeam_scattering method).

        Parameters
        ----------
        num_slices : int
            The number of slices used for the computation of
            the bunch's longitudinal line density.
        """
        self.num_slices = num_slices
        self.diffusion_coefficients: DiffusionCoefficients = None
        self.friction_coefficients: FrictionCoefficients = None
        # The following are needed but start unset. They are set
        # when calling line.configure_intrabeam_scattering()
        self.update_every: int = None
        self._name: str = None
        self._twiss: xt.TwissTable = None
        self._scale_strength: float = 0  # by default element does not "track"

    def _coefficients_need_recompute(self, particles: xt.Particles) -> bool:
        """
        Called to determine if the kick coefficients need to be recomputed before
        applying kicks. This sets an internal flag. Coefficients need recomputing
        if they are `None` or if the current turn is a multiple of the frequency
        at which to recompute them.

        Parameters
        ----------
        particles : xtrack.Particles
            The particles to apply the IBS kicks to and compute it from.

        Returns
        -------
        bool
            Whether the coefficients need to be recomputed.
        """
        # ----------------------------------------------------------------------------------------------
        # Check coefficients existence and if current turn is a multiple of the frequency to recompute
        if (
            self.diffusion_coefficients is None
            or self.friction_coefficients is None
            or _current_turn(particles) % self.update_every == 0
        ):
            return True
        return False

    def compute_kinetic_coefficients(
        self, particles: xt.Particles
    ) -> tuple[DiffusionCoefficients, FrictionCoefficients]:
        r"""
        Computes the ``IBS`` friction coefficients (named :math:`D_x, D_y`
        and :math:`D_z` in this code base) and friction coefficients (named
        :math:`F_x, F_y` and :math:`F_z`) from the kinetic theory introduced
        in :cite:`NuclInstr:Zenkevich:Kinetic_IBS`. These are computed from
        terms of Nagaitsev's theory for faster evaluation, according to the
        derivations done in :cite:arXiv:`Zampetakis:Interplay_SC_IBS_LHC`.

        Notes
        -----
            The calculation is done according to the following steps, which are related
            to the derivations found in :cite:arXiv:`Zampetakis:Interplay_SC_IBS_LHC`
            (in which the generalized diffusion and friction coefficients are used):

                - Computes various terms from the Nagaitsev formalism
                - Computes the intermediate :math:`D_{xx}, D_{xz}, D_{yy}` and :math:`D_{zz}` terms from Eq (39-41, 44)
                - Computes the intermediate :math:`K_x, K_y` and :math:`K_z` terms from Eq (42-44)
                - Computes the diffusion coefficients :math:`D_{x,y,z}` from Eq (45-47)
                - Computes the friction coefficients :math:`F_{x,y,z}` from Eq (48-50)

        Parameters
        ----------
        particles : xtrack.Particles
            The particles to apply the IBS kicks to and compute it from.

        Returns
        -------
        DiffusionCoefficients, FrictionCoefficients
            A tuple with the computed kinetic coefficients slotted into a
            ``DiffusionCoefficients``  and a ``FrictionCoefficients`` objects.
        """
        # ----------------------------------------------------------------------------------------------
        # This full computation (apart from getting properties from the particles object) is done
        # on the CPU, as no GPU context provides scipy's elliptic integrals. It is then better to
        # do the rest of the computation on CPU than to transfer arrays to GPU and finish it there.
        # ----------------------------------------------------------------------------------------------
        # Compute (geometric) emittances, momentum spread, bunch length and sigmas from the Particles
        # fmt: off
        LOGGER.debug("Computing emittances, momentum spread and bunch length from particles")
        gemitt_x: float = _gemitt_x(particles, self._twiss["betx", self._name], self._twiss["dx", self._name])
        gemitt_y: float = _gemitt_y(particles, self._twiss["bety", self._name], self._twiss["dy", self._name])
        total_beam_intensity: float = _beam_intensity(particles)
        bunch_length: float = _bunch_length(particles)
        sigma_delta: float = _sigma_delta(particles)
        sigma_x: float = _sigma_x(particles)
        sigma_y: float = _sigma_y(particles)
        # fmt: on
        # ----------------------------------------------------------------------------------------------
        # Allocating some properties to simple variables for readability
        beta0: float = self._twiss.beta0
        gamma0: float = self._twiss.gamma0
        radius: float = self._twiss.particle_on_co.get_classical_particle_radius0()
        circumference: float = self._twiss.circumference
        betx = self._twiss.betx
        bety = self._twiss.bety
        dx = self._twiss.dx
        # ----------------------------------------------------------------------------------------------
        # Compute the Coulomb logarithm and then the common constant term of Eq (45-50). Note that since
        # we have no way of detecting coasting beams we assume bunched and take the rms bunch length as is
        # (but we don't get the correction factor in the coulomb logarithm...)
        coulomb_logarithm: float = BjorkenMtingwaIBS(self._twiss).coulomb_log(
            gemitt_x=gemitt_x,
            gemitt_y=gemitt_y,
            sigma_delta=sigma_delta,
            bunch_length=bunch_length,
            total_beam_intensity=total_beam_intensity,
        )
        const_numerator: float = total_beam_intensity * radius**2 * c * coulomb_logarithm
        const_denominator: float = 12 * np.pi * beta0**3 * gamma0**5 * bunch_length
        full_constant_term: float = const_numerator / const_denominator
        # ----------------------------------------------------------------------------------------------
        # Computing the constants from Eq (18-21) in Nagaitsev's paper
        phix: ArrayLike = phi(betx, self._twiss.alfx, dx, self._twiss.dpx)
        ax: ArrayLike = betx / gemitt_x
        ay: ArrayLike = bety / gemitt_y
        a_s: ArrayLike = ax * (dx**2 / betx**2 + phix**2) + 1 / sigma_delta**2
        a1: ArrayLike = (ax + gamma0**2 * a_s) / 2.0
        a2: ArrayLike = (ax - gamma0**2 * a_s) / 2.0
        sqrt_term = np.sqrt(a2**2 + gamma0**2 * ax**2 * phix**2)  # qx in Michalis paper
        # ----------------------------------------------------------------------------------------------
        # Compute the eigen values of A matrix (L matrix in B&M) then elliptic integrals R1, R2 and R3
        # They are arrays with one value per element in the lattice
        # fmt: off
        lambda_1: ArrayLike = ay
        lambda_2: ArrayLike = a1 + sqrt_term
        lambda_3: ArrayLike = a1 - sqrt_term
        R1: ArrayLike = elliprd(1 / lambda_2, 1 / lambda_3, 1 / lambda_1) / lambda_1
        R2: ArrayLike = elliprd(1 / lambda_3, 1 / lambda_1, 1 / lambda_2) / lambda_2
        R3: ArrayLike = 3 * np.sqrt(lambda_1 * lambda_2 / lambda_3) - lambda_1 * R1 / lambda_3 - lambda_2 * R2 / lambda_3
        # ----------------------------------------------------------------------------------------------
        # Computing the Dxx, Dxz, Dyy, Dzz terms from Eq (39-41, 44)
        Dxx: ArrayLike = 0.5 * (2 * R1 + R2 * (1 - a2 / sqrt_term) + R3 * (1 + a2 / sqrt_term))
        Dxz: ArrayLike = 3 * gamma0**2 * phix**2 * ax * (R3 - R2) / sqrt_term
        Dyy: ArrayLike = R2 + R3
        Dzz: ArrayLike = 0.5 * gamma0**2 * (2 * R1 + R2 * (1 + a2 / sqrt_term) + R3 * (1 - a2 / sqrt_term))
        # ----------------------------------------------------------------------------------------------
        # Computing the Kx, Ky, Kz terms from Eq (42-44)
        # TODO: remove comments below when Michalis has a new paper version online. Due to the factor
        # 2 in Eq (9) it is included directly in here, but this is badly defined in the paper
        Kx: ArrayLike = 1.0 * (R2 * (1 + a2 / sqrt_term) + R3 * (1 - a2 / sqrt_term))  # Eq says 0.5
        Ky: ArrayLike = 2 * R1  # Eq says 1
        Kz: ArrayLike = 1.0 * gamma0**2 * (R2 * (1 - a2 / sqrt_term) + R3 * (1 + a2 / sqrt_term))  # Eq says 0.5
        # ----------------------------------------------------------------------------------------------
        # Computing integrands for the diffusion and friction terms from Eq (45-50)
        # TODO: missing bet[xy] terms in paper are typos, remove this comment when new version is out
        int_denominator: float = circumference * sigma_x * sigma_y  # common denominator to all integrands
        Dx_integrand: ArrayLike = betx * (Dxx + (dx**2 / betx**2 + phix**2) * Dzz + Dxz) / int_denominator
        Dy_integrand: ArrayLike = bety * Dyy / int_denominator
        Dz_integrand: ArrayLike = Dzz / int_denominator
        Fx_integrand: ArrayLike = betx * (Kx + (dx**2 / betx**2 + phix**2) * Kz) / int_denominator
        Fy_integrand: ArrayLike = bety * Ky / int_denominator
        Fz_integrand: ArrayLike = Kz / int_denominator
        # fmt: on
        # ----------------------------------------------------------------------------------------------
        # Integrating them to obtain the final diffusion and friction coefficients (full Eq (45-50))
        ds: ArrayLike = np.diff(self._twiss.s)
        Dx: float = np.sum(Dx_integrand[:-1] * ds) * full_constant_term / gemitt_x
        Dy: float = np.sum(Dy_integrand[:-1] * ds) * full_constant_term / gemitt_y
        Dz: float = np.sum(Dz_integrand[:-1] * ds) * full_constant_term / sigma_delta**2
        Fx: float = np.sum(Fx_integrand[:-1] * ds) * full_constant_term / gemitt_x
        Fy: float = np.sum(Fy_integrand[:-1] * ds) * full_constant_term / gemitt_y
        Fz: float = np.sum(Fz_integrand[:-1] * ds) * full_constant_term / sigma_delta**2
        # ----------------------------------------------------------------------------------------------
        # Self-update the instance's attributes and then return the results
        self.diffusion_coefficients = DiffusionCoefficients(Dx, Dy, Dz)
        self.friction_coefficients = FrictionCoefficients(Fx, Fy, Fz)
        return self.diffusion_coefficients, self.friction_coefficients

    def track(self, particles: xt.Particles) -> None:
        """
        Method to determine and apply IBS momenta kicks based on the provided
        `xtrack.Particles`. The kicks are implemented according to Eq (19) of
        :cite:arXiv:`Zampetakis:Interplay_SC_IBS_LHC`.

        Parameters
        ----------
        particles : xtrack.Particles
            The particles to apply the IBS kicks to and compute it from.
        """
        # ----------------------------------------------------------------------------------------------
        # Intercept here if the kick element is "off" and has no scale strength - do not compute for nothing
        if self._scale_strength == 0:
            return
        # ----------------------------------------------------------------------------------------------
        # Check if coefficients need to be recomputed before applying kicks & recompute if necessary
        if self._coefficients_need_recompute(particles) is True:
            LOGGER.debug("Recomputing simple kick coefficients before applying kicks")
            self.compute_kinetic_coefficients(particles)
        # ----------------------------------------------------------------------------------------------
        # Get the nplike_lib from the particles' context, to compute on the context device
        _assert_accepted_context(particles._context)
        nplike = particles._context.nplike_lib
        # ----------------------------------------------------------------------------------------------
        # Compute delta_t for the turn and the line density (rho(z) term in Eq (19) of reference)
        dt: float = self._twiss.T_rev0
        rho_z: ArrayLike = line_density(particles, self.num_slices)  # on context
        # ----------------------------------------------------------------------------------------------
        # TODO: Michalis wrote rho(z) in the paper but the way he had implemented it, it is actually
        # 2 * sqrt(pi) * sigma_t * rho(z) so I apply the factor here. Remove comment when new paper is out
        bunch_length: float = _bunch_length(particles)
        factor: float = float(bunch_length * 2 * np.sqrt(np.pi))
        rho_z = factor * rho_z
        # fmt: off
        # ----------------------------------------------------------------------------------------------
        # Computing standard deviation of (normalized) momenta, corresponding to sigma_{pu} in Eq (8) of
        # reference. Normalized: for momentum we have to multiply with 1 / sqrt(gamma) which is equal to
        # sqrt(beta) / sqrt(1 + alpha^2), and the sqrt(beta) is included in the std of p[xy]. If the bunch
        # is rotated, the stdev takes from the "other plane" so we take the normalized momenta to compensate.
        sigma_delta: float = _sigma_delta(particles)                                                                                               # on context
        sigma_px_normalized: float = _sigma_px(particles, self._twiss["dpx", self._name]) / nplike.sqrt(1 + self._twiss["alfx", self._name] ** 2)  # on context
        sigma_py_normalized: float = _sigma_py(particles, self._twiss["dpy", self._name]) / nplike.sqrt(1 + self._twiss["alfy", self._name] ** 2)  # on context
        # ----------------------------------------------------------------------------------------------
        # Determining the Friction kicks (momenta change from friction forces)
        # Friction term is in absolute value and depends on the momentum. If we have a distribution
        # the friction term is with respect to the center -> if the beam is off-center we need to
        # compensate for this, so we use deviation of particle p[xy] from distribution mean of p[xy]
        LOGGER.debug("Determining friction kicks")
        dev_px: ArrayLike = particles.px[particles.state > 0] - _mean_px(particles, self._twiss["dpx", self._name])      # on context
        dev_py: ArrayLike = particles.py[particles.state > 0] - _mean_py(particles, self._twiss["dpy", self._name])      # on context
        dev_delta: ArrayLike = particles.delta[particles.state > 0] - nplike.mean(particles.delta[particles.state > 0])  # on context
        Fx, Fy, Fz = self.friction_coefficients.as_tuple()  # floats
        delta_px_friction: ArrayLike = -Fx * dev_px * dt * rho_z        # on context
        delta_py_friction: ArrayLike = -Fy * dev_py * dt * rho_z        # on context
        delta_delta_friction: ArrayLike = -Fz * dev_delta * dt * rho_z  # on context
        # ----------------------------------------------------------------------------------------------
        # Determining the Diffusion kicks (momenta change from diffusion forces)
        LOGGER.debug("Determining diffusion kicks")
        Dx, Dy, Dz = self.diffusion_coefficients.as_tuple()  # floats
        rng = nplike.random.default_rng()
        _size = particles.px[particles.state > 0].shape[0]  # same for py and delta, it's the alive particles
        # TODO: the factor 2 here is missing in the paper atm and it's a typo from Michalis (remove comment when new paper is out)
        delta_px_diffusion: ArrayLike = sigma_px_normalized * nplike.sqrt(2 * dt * Dx * rho_z) * rng.standard_normal(_size)  # on context
        delta_py_diffusion: ArrayLike = sigma_py_normalized * nplike.sqrt(2 * dt * Dy * rho_z) * rng.standard_normal(_size)  # on context
        delta_delta_diffusion: ArrayLike = sigma_delta * nplike.sqrt(2 * dt * Dz * rho_z) * rng.standard_normal(_size)       # on context
        # ----------------------------------------------------------------------------------------------
        # Applying the momenta kicks to the particles
        LOGGER.debug("Applying momenta kicks to the particles (on px, py and delta properties)")
        particles.px[particles.state > 0] += delta_px_friction + delta_px_diffusion
        particles.py[particles.state > 0] += delta_py_friction + delta_py_diffusion
        particles.delta[particles.state > 0] += delta_delta_friction + delta_delta_diffusion
