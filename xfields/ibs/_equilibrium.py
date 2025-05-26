# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xobjects as xo
import xtrack as xt
from xobjects.general import _print
from xtrack import Table

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import ArrayLike

    from xfields.ibs._analytical import IBSAmplitudeGrowthRates

LOGGER = logging.getLogger(__name__)


# ----- Some classes to store results (as xo.HybridClass) ----- #


class EmittanceTimeDerivatives(xo.HybridClass):
    """
    Holds emittance evolution rates named ``dex``,
    ``dey``, and ``dez``. The values are expressed
    in [m.s^-1].

    Attributes
    ----------
    dex : float
        Horizontal geometric emittance time
        derivative, in [m.s^-1].
    dey : float
        Vertical geometric emittance time
        derivative, in [m.s^-1].
    dez : float
        Longitudinal geometric emittance time
        derivative, in [m.s^-1].
    """

    _xofields = {
        "dex": xo.Float64,
        "dey": xo.Float64,
        "dez": xo.Float64,
    }

    def __init__(self, dex: float, dey: float, dez: float) -> None:
        """Init with given values."""
        self.xoinitialize(dex=dex, dey=dey, dez=dez)

    def as_tuple(self) -> tuple[float, float, float]:
        """Return the growth rates as a tuple."""
        return float(self.dex), float(self.dey), float(self.dez)


def _ibs_rates_and_emittance_derivatives(
    twiss: xt.TwissTable,
    formalism: Literal["Nagaitsev", "Bjorken-Mtingwa", "B&M"],
    total_beam_intensity: int,
    gemitt_x: float,
    gemitt_y: float,
    gemitt_zeta: float,
    longitudinal_emittance_ratio: float,
    **kwargs,
) -> tuple[IBSAmplitudeGrowthRates, EmittanceTimeDerivatives]:
    """
    Compute the IBS growth rates and emittance time derivatives from
    the effect of both IBS and SR. Time derivatives are computed as
    expressed in 'A Code for Calculating Intrabeam Scattering and
    Beam Lifetime', C.H.Kim, LBNL-40296, 1997.

    Parameters
    ----------
    twiss : xtrack.TwissTable
        Twiss results of the `xtrack.Line` configuration.
    formalism : str
        Which formalism to use for the computation of the IBS growth rates.
        Can be ``Nagaitsev`` or ``Bjorken-Mtingwa`` (also accepts ``B&M``),
        case-insensitively.
    total_beam_intensity : int
        The bunch intensity, in [particles per bunch].
    gemitt_x : float
        Horizontal geometric emittance in [m].
    gemitt_y : float
        Vertical geometric emittance in [m].
    gemitt_zeta : float
        Longitudinal geometric emittance in [m].
    longitudinal_emittance_ratio : float, optional
        Ratio of the RMS bunch length to the RMS momentum spread. If provided,
        allows accounting for a perturbed longitudinal distrubtion due to
        bunch lengthening or a microwave instability. Default is None.
    **kwargs : dict
        Keyword arguments are passed to the growth rates computation method of
        the chosen IBS formalism implementation. See the formalism classes in
        the ``xfields.ibs._analytical`` for more details.

    Returns
    -------
    tuple[IBSAmplitudeGrowthRates, EmittanceTimeDerivatives]
        Both the computed IBS growth rates and the emittance time derivatives
        from the contributions of SR and IBS, each in a specific container
        (namely ``IBSAmplitudeGrowthRates`` and ``EmittanceTimeDerivatives``,
        respectively).
    """
    LOGGER.debug("Computing IBS growth rates and emittance time derivatives.")
    # ---------------------------------------------------------------------------------------------
    # Check for valid emittance values
    assert gemitt_x > 0.0, "Horizontal emittance should be larger than zero"
    assert gemitt_y > 0.0, "Vertical emittance should be larger than zero"
    assert gemitt_zeta > 0.0, "Longitudinal emittance should be larger than zero"
    # ---------------------------------------------------------------------------------------------
    # Compute relevant longitudinal parameters for the bunch (needed for IBS growth rates)
    LOGGER.debug("Computing longitudinal parameters for the bunch.")
    sigma_zeta = (gemitt_zeta * longitudinal_emittance_ratio) ** 0.5  # in [m]
    sigma_delta = (gemitt_zeta / longitudinal_emittance_ratio) ** 0.5  # in [-]
    # ---------------------------------------------------------------------------------------------
    # Ask to compute the IBS growth rates (this function logs so no need to do it here)
    # These are in amplitude convention in xsuite (like SR damping constants)
    ibs_growth_rates = twiss.get_ibs_growth_rates(
        formalism=formalism,
        total_beam_intensity=total_beam_intensity,
        gemitt_x=gemitt_x,
        gemitt_y=gemitt_y,
        sigma_delta=sigma_delta,
        bunch_length=sigma_zeta,  # 1 sigma_{zeta,RMS} bunch length
        **kwargs,
    )
    # ---------------------------------------------------------------------------------------------
    # Computing the emittance time derivatives analytically. These are Eq (1-3) of the reference
    LOGGER.debug("Computing emittance time derivatives analytically.")
    depsilon_x_dt = (
        -2 * twiss.damping_constants_s[0] * (gemitt_x - twiss.eq_gemitt_x)
        + 2 * ibs_growth_rates.Kx * gemitt_x
    )
    depsilon_y_dt = (
        -2 * twiss.damping_constants_s[1] * (gemitt_y - twiss.eq_gemitt_y)
        + 2 * ibs_growth_rates.Ky * gemitt_y
    )
    depsilon_z_dt = (
        -2 * twiss.damping_constants_s[2] * (gemitt_zeta - twiss.eq_gemitt_zeta)
        + 2 * ibs_growth_rates.Kz * gemitt_zeta
    )
    # ---------------------------------------------------------------------------------------------
    # And return the results
    return (
        ibs_growth_rates,
        EmittanceTimeDerivatives(dex=depsilon_x_dt, dey=depsilon_y_dt, dez=depsilon_z_dt),
    )


# ----- Public API (integrated as method in TwissTable) ----- #


def get_ibs_and_synrad_emittance_evolution(
    twiss: xt.TwissTable,
    formalism: Literal["Nagaitsev", "Bjorken-Mtingwa", "B&M"],
    total_beam_intensity: int,
    gemitt_x: float | None = None,
    nemitt_x: float | None = None,
    gemitt_y: float | None = None,
    nemitt_y: float | None = None,
    gemitt_zeta: float | None = None,
    nemitt_zeta: float | None = None,
    overwrite_sigma_zeta: float | None = None,
    overwrite_sigma_delta: float | None = None,
    emittance_coupling_factor: float = 0,
    emittance_constraint: Literal["coupling", "excitation"] | None = "coupling",
    rtol: float = 1e-6,
    tstep: float | None = None,
    max_steps: float | None = None,
    verbose: bool = True,
    **kwargs,
) -> Table:
    """
    Compute the evolution of emittances due to Synchrotron Radiation
    and Intra-Beam Scattering until convergence to equilibrium values.
    The equilibrium state is determined by an iterative process which
    consists in computing the IBS growth rates and the emittance time
    derivatives, then computing the emittances at the next time step,
    potentially including the effect of transverse constraints, and
    checking for convergence. The convergence criteria can be chosen
    by the user.

    Transverse emittances can be constrained to follow two scenarios:
        - An emittance exchange originating from betatron coupling.
        - A vertical emittance originating from an excitation.

    The impact from the longitudinal impedance (e.g. bunch lengthening
    or microwave instability) can be accounted for by specifying the RMS
    bunch length and momentum spread.

    Notes
    -----
        It is required that radiation has been configured in the line,
        and that the `TwissTable` holds information on the equilibrium
        state from Synchrotron Radiation. This means calling first
        `line.configure_radiation(model="mean")` and then the `.twiss()`
        method with `eneloss_and_damping=True`. Please refer to the Twiss
        user guide in the `xsuite` documentation for more information.

    Warning
    -------
        If the user does not provide a starting emittance, the program
        defaults to using the SR equilibrium value from the `TwissTable`,
        which is a reasonable defaults for light sources. If a constraint
        is provided via `emittance_constraint`  the starting emittances are
        re-computed to respect that constraint (this is logged to the user).

        If the user does provide starting emittances **and** a constraint, it
        is up to the user to make sure these provided values are consistent
        with the provided constraint!

    Parameters
    ----------
    twiss : xtrack.TwissTable
        Twiss results of the `xtrack.Line` configuration.
    formalism : str
        Which formalism to use for the computation of the IBS growth rates.
        Can be ``Nagaitsev`` or ``Bjorken-Mtingwa`` (also accepts ``B&M``),
        case-insensitively.
    total_beam_intensity : int
        The bunch intensity, in [particles per bunch].
    gemitt_x : float, optional
        Starting horizontal geometric emittance, in [m]. If neither this nor
        the normalized one is provided, the SR equilibrium value from the
        provided `TwissTable` is used.
    nemitt_x : float, optional
        Starting horizontal normalized emittance, in [m]. If neither this nor
        the geometric one is provided, the SR equilibrium value from the
        provided `TwissTable` is used.
    gemitt_y : float, optional
        Starting vertical geometric emittance, in [m]. If neither this nor
        the normalized one is provided, the SR equilibrium value from the
        provided `TwissTable` is used.
    nemitt_y : float, optional
        Starting vertical normalized emittance, in [m]. If neither this nor
        the geometric one is provided, the SR equilibrium value from the
        provided `TwissTable` is used.
    gemitt_zeta : float, optional
        Starting longitudinal geometric emittance, in [m]. If neither this
        nor the normalized one is provided, the SR equilibrium value from the
        provided `TwissTable` is used.
    nemitt_zeta : float, optional
        Starting longitudinal normalized emittance, in [m]. If neither this
        nor the geometric one is provided, the SR equilibrium value from the
        provided `TwissTable` is used.
    emittance_coupling_factor : float, optional
        The ratio of perturbed transverse emittances due to betatron coupling.
        If a value is provided, it is taken into account for the evolution of
        emittances and induces an emittance sharing between the two planes.
        See the next parameter for possible scenarios and how this value is
        used. Defaults to 0.
    emittance_constraint : str, optional
        If an accepted value is provided, enforces constraints on the transverse
        emittances. Can be either "coupling" or "excitation", case-insensitively.
        Defaults to "coupling".
          - If `coupling`, vertical emittance is the result of linear coupling. In
            this case both the vertical and horizontal emittances are altered and
            determined based on the value of `emittance_coupling_factor` and the
            damping partition numbers. If the horizontal and vertical partition
            numbers are equal then the total transverse emittance is preserved.
          - If `excitation`, vertical emittance is the result of an excitation
            (e.g. from a feedback system) and is determined from the horizontal
            emittance based on the value of `emittance_coupling_factor`. In this
            case the total transverse emittance is NOT preserved.
        Providing `None` allows one to study a scenario without constraint. Note
        that as `emittance_coupling_factor` defaults to 0, the constraint has no
        effect unless a non-zero factor is provided.
    overwrite_sigma_zeta : float, optional
        The RMS bunch length, in [m]. If provided, overwrites the one computed from
        the longitudinal emittance and forces a recompute of the longitudinal
        emittance. Defaults to `None`.
    overwrite_sigma_delta : float, optional
        The RMS momentum spread of the bunch. If provided, overwrites the one
        computed from the longitudinal emittance and forces a recompute of the
        longitudinal emittance. Defaults to `None`.
    rtol : float, optional
        Relative tolerance to determine when convergence is reached: if the relative
        difference between the computed emittances and those at the previous step is
        below `rtol`, then convergence is considered achieved. Defaults to 1e-6.
    tstep : float, optional
        Time step to use for each iteration, in [s]. If not provided, an
        adaptive time step is computed based on the IBS growth rates and
        the damping constants. Defaults to `None`.
    max_steps : float, optional
        The maximum number of iterations to perform before stopping the iterative
        process. If not provided, the process continues until it reaches convergence
        (according to the provided `rtol`). Defaults to `None`.
    verbose : bool, optional
        Whether to print out information on the current iteration step and estimated
        convergence progress. Defaults to `True`.
    **kwargs : dict
        Keyword arguments are passed to the growth rates computation method of
        the chosen IBS formalism implementation. See the formalism classes in
        the ``xfields.ibs._analytical`` for more details.

    Returns
    -------
    xtrack.Table
        The convergence calculations results. The table contains the following
        columns, as time-step by time-step quantities:
            - time: time values at which quantities are computed, in [s].
            - gemitt_x: horizontal geometric emittances, in [m].
            - nemitt_x: horizontal normalized emittances, in [m].
            - gemitt_y: vertical geometric emittances, in [m].
            - nemitt_y: vertical normalized emittances, in [m].
            - gemitt_zeta: longitudinal geometric emittances, in [m].
            - nemitt_zeta: longitudinal normalized emittances, in [m].
            - sigma_zeta: bunch lengths, in [m].
            - sigma_delta: momentum spreads, in [-].
            - Kx: horizontal IBS amplitude growth rates, in [s^-1].
            - Ky: vertical IBS amplitude growth rates, in [s^-1].
            - Kz: longitudinal IBS amplitude growth rates, in [s^-1].
        The table also contains the following global quantities:
            - damping_constants_s: radiation damping constants used, in [s].
            - partition_numbers: damping partition numbers used.
            - eq_gemitt_x: horizontal equilibrium geometric emittance from synchrotron radiation used, in [m].
            - eq_gemitt_y: vertical equilibrium geometric emittance from synchrotron radiation used, in [m].
            - eq_gemitt_zeta: longitudinal equilibrium geometric emittance from synchrotron radiation used, in [m].
            - eq_sr_ibs_gemitt_x: final horizontal equilibrium geometric emittance converged to, in [m].
            - eq_sr_ibs_nemitt_x: final horizontal equilibrium normalized emittance converged to, in [m].
            - eq_sr_ibs_gemitt_y: final vertical equilibrium geometric emittance converged to, in [m].
            - eq_sr_ibs_gemitt_y: final vertical equilibrium normalized emittance converged to, in [m].
            - eq_sr_ibs_gemitt_zeta: final longitudinal equilibrium geometric emittance converged to, in [m].
            - eq_sr_ibs_gemitt_zeta: final longitudinal equilibrium normalized emittance converged to, in [m].
    """
    # fmt: off
    # ---------------------------------------------------------------------------------------------
    # Check for the required parameters
    assert formalism is not None, "Must provide 'formalism'"  # accepted values check in called functions
    assert total_beam_intensity is not None, "Must provide 'total_beam_intensity'"
    assert rtol is not None, "Must provide 'rtol'"  # if the user sets None we would crash in weird ways
    # ---------------------------------------------------------------------------------------------
    # Check for SR equilibrium emittances, damping constants and partition numbers in the TwissTable
    _required_attrs = ["damping_constants_s", "partition_numbers", "eq_gemitt_x", "eq_gemitt_y", "eq_gemitt_zeta"]
    if any(getattr(twiss, attr, None) is None for attr in _required_attrs):
        LOGGER.error("Invalid TwissTable, does not have SR equilibrium properties. Did you configure radiation?")
        raise AttributeError(
            "The TwissTable must contain SR equilibrium emittances and damping constants. "
            "Did you activate radiation and twiss with `eneloss_and_damping=True?`"
        )
    # ---------------------------------------------------------------------------------------------
    # Check for valid value of emittance_constraint and warn if constraint provided but factor is 0
    if emittance_constraint is not None:
        _valid_constraints = ("coupling", "excitation")
        assert emittance_constraint.lower() in _valid_constraints, "Invalid 'emittance_constraint', accepted values are 'coupling' or 'excitation'."
        if emittance_coupling_factor == 0:
            LOGGER.warning("As 'emittance_coupling_factor` is zero, providing 'emittance_constraint' has no effect!")
    # ---------------------------------------------------------------------------------------------
    # First: if geometric emittances are provided we will just use those. If instead normalized
    # emittances are provided, we convert those to geometric emittances
    if nemitt_x is not None:
        assert gemitt_x is None, "Cannot provide both 'gemitt_x' and 'nemitt_x'"
        gemitt_x = nemitt_x / (twiss.beta0 * twiss.gamma0)
    if nemitt_y is not None:
        assert gemitt_y is None, "Cannot provide both 'gemitt_y' and 'nemitt_y'"
        gemitt_y = nemitt_y / (twiss.beta0 * twiss.gamma0)
    if nemitt_zeta is not None:
        assert gemitt_zeta is None, "Cannot provide both 'gemitt_zeta' and 'nemitt_zeta'"
        gemitt_zeta = nemitt_zeta / (twiss.beta0 * twiss.gamma0)
    # ---------------------------------------------------------------------------------------------
    # Take default values from the TwissTable if no initial emittances are provided. In this case we
    # need to renormalize with the emittance_constraint so we add a flag to know we need to do that
    _renormalize_transverse_emittances = False
    if gemitt_x is None and nemitt_x is None:
        LOGGER.info("No horizontal emittance provided, taking SR equilibrium value from TwissTable.")
        gemitt_x = twiss.eq_gemitt_x
        _renormalize_transverse_emittances = True
    if gemitt_y is None and nemitt_y is None:
        LOGGER.info("No vertical emittance provided, taking SR equilibrium value from TwissTable.")
        gemitt_y = twiss.eq_gemitt_y
        _renormalize_transverse_emittances = True
    if gemitt_zeta is None and nemitt_zeta is None:
        LOGGER.info("No longitudinal emittance provided, taking SR equilibrium value from TwissTable.")
        gemitt_zeta = twiss.eq_gemitt_zeta
    # We need to also check for the case where use provided emittance and it is the twiss SR eq one
    # which would be our default if not provided and then would trigger renormalization above
    if gemitt_x == twiss.eq_gemitt_x or gemitt_y == twiss.eq_gemitt_y:
        LOGGER.debug("At least one provided transverse emittance is the SR equilibrium value, renormalizing transverse emittances if constraint provided.")
        _renormalize_transverse_emittances = True
    # ---------------------------------------------------------------------------------------------
    # By now we should have a value for geometric emittances in each plane. We assign them to new
    # variables for clarity. These might be overwritten below in case we have to apply a constraint
    starting_gemitt_x: float = gemitt_x  # type: ignore (Pylance does not understand it exists)
    starting_gemitt_y: float = gemitt_y  # type: ignore
    starting_gemitt_zeta: float = gemitt_zeta  # type: ignore
    # ---------------------------------------------------------------------------------------------
    # If we need to renormalize the transverse emittances, we do so now. If emittance_coupling_factor
    # is non-zero, transverse emittances are modified accordingly (used convention is valid for arbitrary
    # damping partition numbers and emittance_coupling_factor values).
    if _renormalize_transverse_emittances is True and emittance_constraint is not None:
        # If constraint is coupling, both emittances are modified (from factor and partition numbers)
        if emittance_constraint.lower() == "coupling" and emittance_coupling_factor != 0:
            LOGGER.info("Enforcing 'coupling' constraint on transverse emittances.")
            starting_gemitt_y = starting_gemitt_x * emittance_coupling_factor / (1 + emittance_coupling_factor * twiss.partition_numbers[1] / twiss.partition_numbers[0])
            starting_gemitt_x = starting_gemitt_x / (1 + emittance_coupling_factor * twiss.partition_numbers[1] / twiss.partition_numbers[0])
        # If constraint is excitation, only vertical emittance is modified
        elif emittance_constraint.lower() == "excitation" and emittance_coupling_factor != 0:
            LOGGER.info("Enforcing 'excitation' constraint on transverse emittances.")
            starting_gemitt_y = starting_gemitt_x * emittance_coupling_factor
            starting_gemitt_x = starting_gemitt_x
    # fmt: on
    # ---------------------------------------------------------------------------------------------
    # Handle the potential longitudinal effects (bunch lengthening, microwave instability)
    # First compute bunch length and momentum spread from longitudinal emittance (see xsuite twiss doc)
    sigma_zeta = (starting_gemitt_zeta * twiss.bets0) ** 0.5
    sigma_delta = (starting_gemitt_zeta / twiss.bets0) ** 0.5
    # Now handle the scenario where the user wants to overwrite those
    if overwrite_sigma_zeta is not None:
        LOGGER.warning("'overwrite_sigma_zeta' specified.")
        sigma_zeta = overwrite_sigma_zeta
    elif overwrite_sigma_delta is not None:
        LOGGER.warning("'overwrite_sigma_delta' specified.")
        sigma_delta = overwrite_sigma_delta
    longitudinal_emittance_ratio = sigma_zeta / sigma_delta
    # If either bunch length or momentum spread was overwritten, we recompute the longidutinal emittance
    if overwrite_sigma_zeta is not None or overwrite_sigma_delta is not None:
        LOGGER.warning("At least one longitudinal property overwritten, recomputing longitudinal emittance.")
        starting_gemitt_zeta = sigma_zeta * sigma_delta
    # ---------------------------------------------------------------------------------------------
    # Log the starting emittances and the computed longitudinal properties - useful for debugging
    LOGGER.debug(f"Starting emittances are: {starting_gemitt_x:.2e}, {starting_gemitt_y:.2e}, {starting_gemitt_zeta:.2e}")
    # ---------------------------------------------------------------------------------------------
    # Initialize values for the iterative process (first time step is revolution period)
    iterations: float = 0
    tolerance: float = np.inf
    time_step: float = twiss.T_rev0
    # Structures for iterative results (time, IBS growth rates, computed emittances)
    time_deltas: list[float] = []  # stores the deltas (!), we do a cumsum at the end
    K_x: list[float] = []
    K_y: list[float] = []
    K_z: list[float] = []
    res_gemitt_x: list[float] = []
    res_gemitt_y: list[float] = []
    res_gemitt_zeta: list[float] = []
    # Starting emittances (numpy array since we compute the next ones from these)
    current_emittances: ArrayLike = np.array([starting_gemitt_x, starting_gemitt_y, starting_gemitt_zeta])
    # ---------------------------------------------------------------------------------------------
    # Start the iterative process until convergence:
    # - Compute IBS rates and emittance time derivatives
    # - Compute new emittances using the time derivatives and time step
    # - Enforce transverse constraints if specified
    # - Store all intermediate results for this time step
    # - Compute tolerance and check for convergence
    while tolerance > rtol:
        # --------------------------------------------------------------------------
        if max_steps is not None and iterations >= max_steps:
            _print(f"Reached maximum number of iterations ({max_steps}), stopping.")
            break
        # --------------------------------------------------------------------------
        LOGGER.debug(f"Current geometric emittances (x, y, zeta): {current_emittances}")
        # --------------------------------------------------------------------------
        # Display estimated convergence progress if asked
        if verbose is True:
            _print(f"Iteration {iterations} - convergence = {100 * rtol / tolerance:.2g}%    ", end="\r")
        # --------------------------------------------------------------------------
        # Compute IBS growth rates and emittance derivatives (and unpack)
        # These are in amplitude convention in xsuite (like SR damping constants)
        ibs_growth_rates, emittance_derivatives = _ibs_rates_and_emittance_derivatives(
            twiss=twiss,
            formalism=formalism,
            total_beam_intensity=total_beam_intensity,
            gemitt_x=current_emittances[0],
            gemitt_y=current_emittances[1],
            gemitt_zeta=current_emittances[2],
            longitudinal_emittance_ratio=longitudinal_emittance_ratio,
            **kwargs,
        )
        ibs_growth_rates = ibs_growth_rates.as_tuple()
        emittance_derivatives = emittance_derivatives.as_tuple()
        # --------------------------------------------------------------------------
        # Update current emittances - add the time step * emittance time derivatives
        current_emittances += np.array(emittance_derivatives) * time_step
        # --------------------------------------------------------------------------
        # Enforce transverse constraint if specified (valid with a factor value of 0)
        if emittance_constraint is not None:
            if emittance_constraint.lower() == "coupling":
                forced_gemitt_x = (current_emittances[0] + current_emittances[1]) / (1 + emittance_coupling_factor)
                forced_gemitt_y = forced_gemitt_x * emittance_coupling_factor
                current_emittances[0] = forced_gemitt_x
                current_emittances[1] = forced_gemitt_y
            elif emittance_constraint.lower() == "excitation":
                forced_gemitt_y = current_emittances[0] * emittance_coupling_factor
                current_emittances[1] = forced_gemitt_y
        LOGGER.debug(f"Stored emittances after time step: {current_emittances}")
        # --------------------------------------------------------------------------
        # Store the current values for this time step
        time_deltas.append(time_step)
        K_x.append(ibs_growth_rates[0])
        K_y.append(ibs_growth_rates[1])
        K_z.append(ibs_growth_rates[2])
        res_gemitt_x.append(current_emittances[0])
        res_gemitt_y.append(current_emittances[1])
        res_gemitt_zeta.append(current_emittances[2])
        # --------------------------------------------------------------------------
        # Compute tolerance (but not at first step since there is no previous value)
        # and store current emittances for tolerance computation at next iteration
        if iterations > 0:
            tolerance = np.max(np.abs((current_emittances - previous_emittances) / previous_emittances))  # type: ignore  # noqa: F821
        previous_emittances = current_emittances.copy()  # noqa: F841  (used next loop)
        # --------------------------------------------------------------------------
        # Update time step for the next iteration and increase counter. If the user
        # provided a time step value we use it, otherwise it is adaptive and set to
        # 1% of the max damping time between IBS and SR across all planes
        time_step = tstep if tstep is not None else 0.01 / np.max((ibs_growth_rates, twiss.damping_constants_s))
        LOGGER.debug(f"Time step for next iteration: {time_step:.2e}")
        iterations += 1
    # ----------------------------------------------------------------------------------------------
    # We have exited the loop, we have converged. Construct a Table with the results and return it
    _print(f"Reached equilibrium in {iterations} iterations, with tolerance={tolerance:.2e} (vs rtol={rtol:.2e})")
    result_table = Table(
        data={
            "time": np.cumsum(time_deltas),
            "gemitt_x": np.array(res_gemitt_x),
            "nemitt_x": np.array(res_gemitt_x) * twiss.beta0 * twiss.gamma0,
            "gemitt_y": np.array(res_gemitt_y),
            "nemitt_y": np.array(res_gemitt_y) * twiss.beta0 * twiss.gamma0,
            "gemitt_zeta": np.array(res_gemitt_zeta),
            "nemitt_zeta": np.array(res_gemitt_zeta) * twiss.bets0 * twiss.gamma0,
            "sigma_zeta": np.sqrt(np.array(res_gemitt_zeta) * longitudinal_emittance_ratio),
            "sigma_delta": np.sqrt(np.array(res_gemitt_zeta) / longitudinal_emittance_ratio),
            "Kx": np.array(K_x),
            "Ky": np.array(K_y),
            "Kz": np.array(K_z),
        },
        index="time",
    )
    # Provide global quantities as well
    result_table._data.update(
        {
            "damping_constants_s": twiss.damping_constants_s,
            "partition_numbers": twiss.partition_numbers,
            "eq_gemitt_x": twiss.eq_gemitt_x,
            "eq_gemitt_y": twiss.eq_gemitt_y,
            "eq_gemitt_zeta": twiss.eq_gemitt_zeta,
            "eq_sr_ibs_gemitt_x": res_gemitt_x[-1],
            "eq_sr_ibs_nemitt_x": res_gemitt_x[-1] * twiss.beta0 * twiss.gamma0,
            "eq_sr_ibs_gemitt_y": res_gemitt_y[-1],
            "eq_sr_ibs_nemitt_y": res_gemitt_y[-1] * twiss.beta0 * twiss.gamma0,
            "eq_sr_ibs_gemitt_zeta": res_gemitt_zeta[-1],
            "eq_sr_ibs_nemitt_zeta": res_gemitt_zeta[-1] * twiss.beta0 * twiss.gamma0,
        }
    )
    return result_table
