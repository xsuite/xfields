# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from __future__ import annotations

import logging
import warnings
from typing import Literal

import numpy as np
import xobjects as xo
import xtrack as xt
from xtrack import Table

from xfields.ibs._analytical import IBSGrowthRates

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
    input_emittances: tuple[float, float, float],
    longitudinal_emittance_ratio: float = None,
    **kwargs,
) -> tuple[IBSGrowthRates, EmittanceTimeDerivatives]:
    """
    Compute the IBS growth rates and emittance time derivatives from
    the effect of both IBS and SR.

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
    input_emittances : tuple[float, float, float]
        The bunch's starting geometric emittances in the horizontal,
        vertical and longitudinal planes, in [m].
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
    tuple[IBSGrowthRates, EmittanceTimeDerivatives]
        Both the computed IBS growth rates and the emittance time derivatives
        from the contributions of SR and IBS, each in a specific container
        object (namely ``IBSGrowthRates`` and ``EmittanceTimeDerivatives``,
        respectively).
    """
    LOGGER.debug("Computing IBS growth rates and emittance time derivatives.")
    # ----------------------------------------------------------------------------------------------
    # TODO: bunch emittances - ask for the three separately and update docstring
    input_emittance_x, input_emittance_y, input_emittance_z = input_emittances
    assert input_emittance_x > 0.0, (
        "'input_emittance_x' should be larger than zero, try providing 'initial_emittances'"
    )
    assert input_emittance_y > 0.0, (
        "'input_emittance_y' should be larger than zero, try providing 'initial_emittances'"
    )
    # ----------------------------------------------------------------------------------------------
    # TODO: check for SR eq emittances etc in twiss table (or in public func?) Could be:
    # if None in (
    #     getattr(twiss, "eq_gemitt_x", None),
    #     getattr(twiss, "eq_gemitt_y", None),
    #     getattr(twiss, "eq_gemitt_zeta", None),
    #     getattr(twiss, "damping_constants_s", None),
    # ):
    #     LOGGER.error("Invalid TwissTable, does not have SR equilibrium properties.")
    #     raise AttributeError(
    #         "The TwissTable must contain SR equilibrium emittances and damping constants. "
    #         "Did you activate radiation and twiss with eneloss_and_damping=True?"
    #     )
    # ----------------------------------------------------------------------------------------------
    # Compute relevant longitudinal parameters for the bunch (needed for IBS growth rates)
    LOGGER.debug("Computing longitudinal parameters for the bunch.")
    sigma_zeta = (input_emittance_z * longitudinal_emittance_ratio) ** 0.5  # in [m]
    sigma_delta = (input_emittance_z / longitudinal_emittance_ratio) ** 0.5  # in [-]
    # ----------------------------------------------------------------------------------------------
    # Ask to compute the IBS growth rates (this function logs so no need to do it here)
    ibs_growth_rates = twiss.get_ibs_growth_rates(
        formalism=formalism,
        total_beam_intensity=total_beam_intensity,
        gemitt_x=input_emittance_x,
        gemitt_y=input_emittance_y,
        sigma_delta=sigma_delta,
        bunch_length=sigma_zeta,  # 1 sigma_{zeta,RMS} bunch length
        **kwargs,
    )
    # ----------------------------------------------------------------------------------------------
    # Computing the emittance time derivatives analytically.
    # TODO: ADD A REF TO THE FORMULA HERE
    # TODO: replace input_emittance_[xyz] by gemitt_[xyz] once they are parameters
    LOGGER.debug("Computing emittance time derivatives analytically.")
    depsilon_x_dt = (
        -2 * twiss.damping_constants_s[0] * (input_emittance_x - twiss.eq_gemitt_x)
        + ibs_growth_rates.Tx * input_emittance_x
    )
    depsilon_y_dt = (
        -2 * twiss.damping_constants_s[1] * (input_emittance_y - twiss.eq_gemitt_y)
        + ibs_growth_rates.Ty * input_emittance_y
    )
    depsilon_z_dt = (
        -2 * twiss.damping_constants_s[2] * (input_emittance_z - twiss.eq_gemitt_zeta)
        + ibs_growth_rates.Tz * input_emittance_z
    )
    # ----------------------------------------------------------------------------------------------
    # And return the results
    return (
        ibs_growth_rates,
        EmittanceTimeDerivatives(dex=depsilon_x_dt, dey=depsilon_y_dt, dez=depsilon_z_dt),
    )


# TODO: find a better, more explicit name for this
def compute_emittance_evolution(
    twiss: xt.TwissTable,
    formalism: Literal["Nagaitsev", "Bjorken-Mtingwa", "B&M"],
    total_beam_intensity: int,
    # TODO: want to force providing gemitt_x, gemitt_y & gemitt_zeta instead of this initial_emittances (can allow nemitt_x and add checks / conversions)
    initial_emittances: tuple[float, float, float] = None,
    # gemitt_x: float = None,
    # nemitt_x: float = None,
    # gemitt_y: float = None,
    # nemitt_y: float = None,
    # gemitt_zeta: float = None,
    # nemitt_zeta: float = None,
    emittance_coupling_factor: float = 0,
    emittance_constraint: Literal["coupling", "excitation"] = "coupling",
    # TODO: move these two (still optional) below gemitt_x, gemitt_y, gemitt_zeta when modify that
    overwrite_sigma_zeta: float = None,
    overwrite_sigma_delta: float = None,
    rtol: float = 1e-6,
    verbose: bool = True,
    **kwargs,
):
    # TODO: rework this main chunk of docstring
    """
    Compute the evolution of beam emittances due to IBS until convergence.
    By default, the function assumes the emittances from the Twiss object.
    They can also be specified as well as different natural emittances.
    The emittance evolution can be constrained to follow two scenarios:
        - A vertical emittance originating from linear coupling.
        - A vertical emittance originating from an excitation.
    The impact from the longitudinal impedance (e.g. bunch lengthening or
    microwave instability) can be accounted for by specifying the RMS bunch
    length and momentum spread.

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
    initial_emittances : tuple[float, float, float], optional
        The bunch's starting geometric emittances in the horizontal,
        vertical and longitudinal planes, in [m]. If not provided, the
        SR equilibrium emittances from the TwissTable are used. Defaults
        to `None`.
    emittance_coupling_factor : float, optional
        The ratio of perturbed transverse emittances due to betatron coupling.
        If a value is provided, it is taken into account for the evolution of
        emittances and induced an emittance sharing between the two planes.
        See the next parameter for possible scenarios and how this value is
        used. Defaults to 0.
    emittance_constraint : str, optional
        If an accepted value is provided, enforces constraints on the transverse
        emittances. Can be either "coupling" or "excitation", case-insensitively.
        Defaults to "coupling".
          - If `coupling`, vertical emittance is the result of linear coupling. In
            this case both the vertical and horizontal emittances are altered and
            determined based on the value of `emittance_coupling_factor` such that
            the total transverse IS preserved.
          - If `excitation`, vertical emittance is the result of an excitation
            (e.g. from a feedback system) and is determined from the horizontal
            emittance based on the value of `emittance_coupling_factor`. In this
            case the total transverse emittance is NOT preserved.
        This has no effect by default as `emittance_coupling_factor` defaults to 0.
    overwrite_sigma_zeta : float, optional
        The RMS bunch length. If provided, overwrites the one computed from
        the longitudinal emittance. Defaults to `None`.
    overwrite_sigma_delta : float, optional
        The RMS momentum spread of the bunch. If provided, overwrites the one
        computed from the longitudinal emittance. Defaults to `None`.
    rtol : float, optional
        Relative tolerance to determine when convergence is reached: if the relative
        difference between the computed emittances and those at the previous step is
        below `rtol`, then convergence is considered achieved. Defaults to 1e-6.
    verbose : bool, optional
        Whether to print out information on the current iteration step and estimated
        convergence progress. Defaults to `False`.
    **kwargs : dict
        Keyword arguments are passed to the growth rates computation method of
        the chosen IBS formalism implementation. See the formalism classes in
        the ``xfields.ibs._analytical`` for more details.

    Returns
    -------
    time : numpy.ndarray
        Computed time steps [s].
    emittances_x : list of float
        Horizontal emittance values computed over all the time steps.
    emittances_y : list of float
        Vertical emittance values computed over all the time steps.
    emittances_z : list of float
        Longitudinal emittance values computed over all the time steps.
    T_x : list of float
        Horizontal IBS growth rates computed over all the time steps.
    T_y : list of float
        Vertical IBS growth rates computed over all the time steps.
    T_z : list of float
        Longitudinal IBS growth rates computed over all the time steps.
    """
    # Returns
    # -------
    # xtrack.TwissTable
    #     The convergence calculations results. The table contains the following
    #     columns, as time-step by time-step quantities:
    #         - time: time values at which quantities are computed, in [s]
    #         - gemitt_x: horizontal geometric emittance values, in [m]
    #         - gemitt_y: vertical geometric emittance values, in [m]
    #         - gemitt_zeta: longitudinal geometric emittance values, in [m]
    #         - Tx: horizontal IBS growth rate, in [s^-1]
    #         - Ty: vertical IBS growth rate, in [s^-1]
    #         - Tz: longitudinal IBS growth rate, in [s^-1]
    #     The table also contains the following global quantities:
    #         - damping_constants_s: radiation damping constants per second used for the calculations
    #         - partition_numbers: damping partition numbers used for the calculations
    #         - eq_gemitt_x: final horizontal equilibrium geometric emittance converged to, in [m]
    #         - eq_gemitt_y: final vertical equilibrium geometric emittance converged to, in [m]
    #         - eq_gemitt_zeta: final longitudinal equilibrium geometric emittance converged to, in [m]
    #         - sr_eq_gemitt_x: horizontal equilibrium geometric emittance from synchrotron radiation used, in [m]
    #         - sr_eq_gemitt_y: vertical equilibrium geometric emittance from synchrotron radiation used, in [m]
    #         - eq_nemitt_zeta: longitudinal equilibrium normalized emittance from synchrotron radiation used, in [m]
    # ----------------------------------------------------------------------------------------------
    # TODO: Perform check for valid value of emittance_constraint but no value of emittance coupling factor
    # ----------------------------------------------------------------------------------------------
    # Handle initial transverse emittances and potential effect of coupling / excitation constraints
    # TODO: I don't like this, would rather force the user to provide gemitt_x, gemitt_y & gemitt_zeta
    if initial_emittances is None:
        print("Emittances from the Twiss object are being used.")
        emittance_x, emittance_y, emittance_z = (
            twiss.eq_gemitt_x,
            twiss.eq_gemitt_y,
            twiss.eq_gemitt_zeta,
        )
        # If emittance_coupling_factor is non zero, then natural emittance is
        # modified accordingly
        # fmt: off
        if emittance_coupling_factor != 0 and emittance_constraint.lower() == "coupling":
            # The convention used is valid for arbitrary damping partition numbers and emittance_coupling_factor.
            emittance_y = emittance_x * emittance_coupling_factor / (1 + emittance_coupling_factor * twiss.partition_numbers[1] / twiss.partition_numbers[0])
            emittance_x *= 1 / (1 + emittance_coupling_factor * twiss.partition_numbers[1] / twiss.partition_numbers[0])

        if emittance_coupling_factor != 0 and emittance_constraint.lower() == "excitation":
            # The convention used only enforce a constraint on the vertical
            # emittance
            emittance_y = emittance_x * emittance_coupling_factor
    else:
        emittance_x, emittance_y, emittance_z = initial_emittances
        # fmt: on
    # ----------------------------------------------------------------------------------------------
    # Handle initial longitudinal emittance and potential effect of bunch lengthening
    sigma_zeta = (emittance_z * twiss.bets0) ** 0.5
    sigma_delta = (emittance_z / twiss.bets0) ** 0.5
    if overwrite_sigma_zeta is not None:
        warnings.warn("'overwrite_sigma_zeta' is specified, make sure it remains consistent with 'initial_emittances'.")
        sigma_zeta = overwrite_sigma_zeta
    elif overwrite_sigma_delta is not None:
        warnings.warn("'sigma_delta' is specified, make sure it remains consistent with 'initial_emittances'.")
        sigma_delta = overwrite_sigma_delta
    longitudinal_emittance_ratio = sigma_zeta / sigma_delta
    if overwrite_sigma_zeta is not None or overwrite_sigma_delta is not None:
        assert initial_emittances is not None, (
            "Input of 'overwrite_sigma_zeta' or 'overwrite_sigma_delta' provided, but "
            "not of 'initial_emittances'. Please provide 'initial_emittances'."
        )
    # ----------------------------------------------------------------------------------------------
    # Start structures to store the iterative results until convergence
    tolerance = np.inf
    time_step = twiss.T_rev0  # Initial time step is the revolution period

    time_deltas = []
    res_gemitt_x, res_gemitt_y, res_gemitt_zeta = [], [], []
    T_x, T_y, T_z = [], [], []
    current_emittances = np.array([emittance_x, emittance_y, emittance_z])
    iterations = 0  # Iteration counter
    # ----------------------------------------------------------------------------------------------
    # Start the iterative process until convergence:
    # - Compute IBS rates and emittance time derivatives
    # - Update emittances using the time derivatives and time step
    # - Enforce transverse / longitudinal constraints if specified
    # - Store all intermediate results for this time step
    # - Compute tolerance and check for convergence
    while tolerance > rtol:
        if verbose is True:  # Display estimated convergence progress if asked
            xo.general._print(f"Iteration {iterations} - convergence = {100 * rtol / tolerance:.1f}%", end="\r")

        # Compute IBS growth rates and emittance derivatives
        ibs_growth_rates, emittance_derivatives = _ibs_rates_and_emittance_derivatives(
            twiss=twiss,
            formalism=formalism,
            total_beam_intensity=total_beam_intensity,
            input_emittances=current_emittances,
            longitudinal_emittance_ratio=longitudinal_emittance_ratio,
            **kwargs,
        )
        # Make sure we have them as tuples for below
        ibs_growth_rates = ibs_growth_rates.as_tuple()
        emittance_derivatives = emittance_derivatives.as_tuple()

        # Update current emittances - add the time step * emittance time derivatives
        current_emittances += np.array(emittance_derivatives) * time_step

        # Enforce constraints if specified
        if emittance_constraint.lower() == "coupling":
            forced_emittance_x = (current_emittances[0] + current_emittances[1]) / (
                1 + emittance_coupling_factor
            )
            forced_emittance_y = forced_emittance_x * emittance_coupling_factor
            current_emittances[0] = forced_emittance_x
            current_emittances[1] = forced_emittance_y

        if emittance_constraint.lower() == "excitation":
            forced_emittance_y = current_emittances[0] * emittance_coupling_factor
            current_emittances[1] = forced_emittance_y

        # Append current values to lists
        time_deltas.append(time_step)
        res_gemitt_x.append(current_emittances[0])
        res_gemitt_y.append(current_emittances[1])
        res_gemitt_zeta.append(current_emittances[2])
        T_x.append(ibs_growth_rates[0])
        T_y.append(ibs_growth_rates[1])
        T_z.append(ibs_growth_rates[2])

        # Compute tolerance (but not at first step since there is no previous value)
        if iterations > 0:
            tolerance = np.max(np.abs((current_emittances - previous_emittances) / previous_emittances))

        # Store current emittances for the next iteration
        previous_emittances = current_emittances.copy()

        # Update time step for the next iteration
        time_step = 0.01 / np.max((ibs_growth_rates, twiss.damping_constants_s))

        iterations += 1
    # ----------------------------------------------------------------------------------------------
    # We have exited the loop, we have converged. Construct a Table with the results and return it
    print("\nConverged!")
    # result_table = Table(
    #     data={
    #         "time": np.cumsum(time),
    #         "gemitt_x": emittances_x_list,
    #         "gemitt_y": emittances_y_list,
    #         "gemitt_zeta": emittances_z_list,
    #         "Tx": Tx,
    #         "Ty": Ty,
    #         "Tz": Tz,
    #     },
    #     index="time",
    # )
    # Provide global quantities as well
    # result_table._data.update(
    #     {
    #         "damping_constants_s": twiss.damping_constants_s,
    #         "partition_numbers": twiss.partition_numbers,
    #         "eq_gemitt_x": emittances_x_list[-1],
    #         "eq_gemitt_x": emittances_y_list[-1],
    #         "eq_gemitt_x": emittances_z_list[-1],
    #         "sr_eq_gemitt_x": twiss.eq_gemitt_x,
    #         "sr_eq_gemitt_y": twiss.eq_gemitt_y,
    #         "eq_nemitt_zeta": twiss.eq_nemitt_zeta,
    #     }
    # )
    # return result_table
    return (
        np.cumsum(time_deltas),
        res_gemitt_x,
        res_gemitt_y,
        res_gemitt_zeta,
        T_x,
        T_y,
        T_z,
    )
