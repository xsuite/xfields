# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #
from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

from xfields.ibs._analytical import BjorkenMtingwaIBS, NagaitsevIBS
from xfields.ibs._kicks import IBSAnalyticalKick, IBSKick

if TYPE_CHECKING:
    from typing import Literal

    import xtrack as xt

    from xfields.ibs._analytical import IBSAmplitudeGrowthRates


LOGGER = getLogger(__name__)


# ----- API for Analytical IBS -----#


def get_intrabeam_scattering_growth_rates(
    twiss: xt.TwissTable,
    formalism: Literal["Nagaitsev", "Bjorken-Mtingwa", "B&M"],
    total_beam_intensity: int = None,
    gemitt_x: float = None,
    nemitt_x: float = None,
    gemitt_y: float = None,
    nemitt_y: float = None,
    sigma_delta: float = None,
    bunch_length: float = None,
    bunched: bool = True,
    **kwargs,
) -> IBSAmplitudeGrowthRates:
    """
    Computes IntraBeam Scattering amplitude growth rates from the
    provided `xtrack.TwissTable`. We give values in amplitude to
    be consistent with the Xsuite SR damping times.

    Parameters
    ----------
    line : xtrack.Line
        Line in which the IBS kick element will be installed.
    formalism : str
        Which formalism to use for the computation. Can be ``Nagaitsev``
        or ``Bjorken-Mtingwa`` (also accepts ``B&M``), case-insensitively.
    total_beam_intensity : int, optional
        The beam intensity. Required if `particles` is not provided.
    gemitt_x : float, optional
        Horizontal geometric emittance in [m]. If `particles` is not
        provided, either this parameter or `nemitt_x` is required.
    nemitt_x : float, optional
        Horizontal normalized emittance in [m]. If `particles` is not
        provided, either this parameter or `gemitt_x` is required.
    gemitt_y : float, optional
        Vertical geometric emittance in [m]. If `particles` is not
        provided, either this parameter or `nemitt_y` is required.
    nemitt_y : float, optional
        Vertical normalized emittance in [m]. If `particles` is not
        provided, either this parameter or `gemitt_y` is required.
    sigma_delta : float, optional
        The momentum spread. Required if `particles` is not provided.
    bunch_length : float, optional
        The bunch length in [m]. Required if `particles` is not provided.
    bunched : bool, optional
        Whether the beam is bunched or not (coasting). Defaults to `True`.
        Required if `particles` is not provided.
    **kwargs : dict
        Keyword arguments are passed to the growth rates computation method of
        the chosen IBS formalism implementation. See the formalism classes in
        the ``xfields.ibs._analytical`` for more details.

    Returns
    -------
    IBSAmplitudeGrowthRates
        An ``IBSAmplitudeGrowthRates`` object with the computed growth rates.
    """
    # ----------------------------------------------------------------------------------------------
    # Perform checks on exclusive parameters: need either particles or all emittances, etc.
    assert total_beam_intensity is not None, "Must provide 'total_beam_intensity'"
    assert sigma_delta is not None, "Must provide 'sigma_delta'"
    assert bunch_length is not None, "Must provide 'bunch_length'"
    assert any([gemitt_x, nemitt_x]), "Must provide either 'gemitt_x' or 'nemitt_x'"
    assert any([gemitt_y, nemitt_y]), "Must provide either 'gemitt_y' or 'nemitt_y'"
    # ----------------------------------------------------------------------------------------------
    # Ensure valid formalism parameter was given and determine the corresponding class
    assert formalism.lower() in ("nagaitsev", "bjorken-mtingwa", "b&m")
    if formalism.lower() == "nagaitsev":
        if np.count_nonzero(twiss.dy) != 0:
            LOGGER.warning("Vertical dispersion is present, Nagaitsev formalism does not account for it")
        ibs = NagaitsevIBS(twiss)
    else:
        ibs = BjorkenMtingwaIBS(twiss)
    # ----------------------------------------------------------------------------------------------
    # Now computing the growth rates using the IBS class and returning them
    return ibs.growth_rates(
        gemitt_x=gemitt_x,
        nemitt_x=nemitt_x,
        gemitt_y=gemitt_y,
        nemitt_y=nemitt_y,
        sigma_delta=sigma_delta,
        bunch_length=bunch_length,
        total_beam_intensity=total_beam_intensity,
        bunched=bunched,
        **kwargs,
    )


# ----- API for Kick-Based IBS -----#


def configure_intrabeam_scattering(
    line: xt.Line,
    element: IBSKick = None,
    update_every: int = None,
    **kwargs,
) -> None:
    """
    Configures the IBS kick element in the line for tracking.

    Notes
    -----
        This **should be** one of the last steps taken before tracking.
        At the very least, if steps are taken that change the lattice's
        optics after this configuration, then this function should be
        called once again.

    Parameters
    ----------
    line : xtrack.Line
        The line in which the IBS kick element was inserted.
    element : IBSKick, optional
        If provided, the element is first inserted in the line,
        before proceeding to configuration. In this case the keyword
        arguments are passed on to the `line.insert_element` method.
        This will also discard and rebuild the line tracker.
    update_every : int
        The frequency at which to recompute the kick coefficients, in
        number of turns. They will be computed at the first turn of
        tracking, and then every `update_every` turns afterwards.

    **kwargs : dict, optional
        Required if an element is provided. Keyword arguments are
        passed to the `line.insert_element()` method according to
        `line.insert_element(element=element, **kwargs)`.

    Raises
    ------
    AssertionError
        If the provided `update_every` is not a positive integer.
    AssertionError
        If more than one IBS kick element is found in the line.
    AssertionError
        If the element is an `IBSAnalyticalKick` and the line is
        operating below transition energy.
    """
    # ----------------------------------------------------------------------------------------------
    # Asserting validity of provided parameters
    assert isinstance(update_every, int), "The 'update_every' parameter must be an integer"
    assert update_every > 0, "The 'update_every' parameter must be a positive integer"
    # ----------------------------------------------------------------------------------------------
    # If the user provided a valid element, we insert it in the line first (and pass kwargs)
    if element is not None and isinstance(element, IBSKick):
        LOGGER.info("Inserting provided element in the line, passing on keyword arguments")
        if line.tracker is not None:
            # if there's a tracker we discard and will build it back identical
            _buffer = line._buffer
            line.discard_tracker()
        else:
            _buffer = None
        line.insert_element(element=element, **kwargs)
        if _buffer is not None:
            line.build_tracker(_buffer=_buffer)
    # ----------------------------------------------------------------------------------------------
    # Otherwise, from here on we assume it's there. Now we need a TwissTable for the elements
    LOGGER.info("Computing Twiss for the provided line and configuring IBS kick element")
    twiss = line.twiss(method="4d")
    # ----------------------------------------------------------------------------------------------
    # Figure out the IBS kick element and its name in the line
    inserted_ibs_kicks = {
        name: element for name, element in line.element_dict.items() if isinstance(element, IBSKick)
    }
    assert len(inserted_ibs_kicks) == 1, "Only one 'IBSKick' element should be present in the line"
    name, element = inserted_ibs_kicks.popitem()
    # ----------------------------------------------------------------------------------------------
    # Set necessary (private) attributes for the kick to function
    element.update_every = update_every
    element._name = name
    element._twiss = twiss
    element._scale_strength = 1  # element is now ON, will track
    # ----------------------------------------------------------------------------------------------
    # Handle Simple kick specificities (valid above transition only)
    if isinstance(element, IBSAnalyticalKick):
        assert twiss.slip_factor >= 0, "IBSAnalyticalKick is not valid below transition"
    LOGGER.debug("Done configuring IntraBeam Scattering kick element")
