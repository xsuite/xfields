# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #
from logging import getLogger
from typing import Literal

import numpy as np
import xtrack as xt

from xfields.ibs._analytical import BjorkenMtingwaIBS, IBSGrowthRates, NagaitsevIBS
from xfields.ibs._formulary import _beam_intensity, _bunch_length, _gemitt_x, _gemitt_y, _sigma_delta

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
    particles: xt.Particles = None,
    **kwargs,
) -> IBSGrowthRates:
    """
    Computes IntraBeam Scattering growth rates from the provided `xtrack.Line`.

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
    particles : xtrack.Particles
        The particles to circulate in the line. If provided the emittances,
        momentum spread and bunch length will be computed from the particles.
        Otherwise explicit values must be provided (see above parameters).
        Currently pending a new xtrack release, will come in the future.
    **kwargs : dict
        Keyword arguments are passed to the growth rates computation method of
        the chosen IBS formalism implementation. See the formalism classes in
        the ``xfields.ibs._analytical`` for more details.

    Returns
    -------
    IBSGrowthRates
        An ``IBSGrowthRates`` object with the computed growth rates.
    """
    # ----------------------------------------------------------------------------------------------
    # Perform checks on exclusive parameters: need either particles or all emittances, etc.
    if isinstance(particles, xt.Particles):
        # TODO: wait for production-ready functionality from xtrack to handle this
        raise NotImplementedError("Not yet implemented")
        LOGGER.info("Particles provided, will determine emittances, etc. from them")
        gemitt_x = _gemitt_x(particles, twiss.betx[0], twiss.dx[0])
        gemitt_y = _gemitt_y(particles, twiss.bety[0], twiss.dy[0])
        sigma_delta = _sigma_delta(particles)
        bunch_length = _bunch_length(particles)
        total_beam_intensity = _beam_intensity(particles)
    else:
        LOGGER.info("Using explicitely provided parameters for emittances, etc.")
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

# TODO: Favor the user creating basic kick element and inserting (see _best.py)
def install_intrabeam_scattering_kick(
    line: xt.Line,
    formalism: str,  # let's give an enum for the hint or something?
) -> None:
    """
    Installs an IntraBeam Scattering kick element at the end of the provided xtrack.Line object.

    Parameters
    ----------
    line : xtrack.Line
        Line in which the IBS kick element will be installed.
    formalism : str
        Which formalism to use for the IBS kicks. Can be "simple" (only valid above transition) or "kinetic".
    recompute_rates_every_nturns : int
        The period in [turns] with which to recompute the IBS growth rates during tracking.
    """
    # get the beam/optics params now but it might be outdated, or when the particles
    # get to the element only? Would need to change the element logic
    raise NotImplementedError("Not yet implemented")

# TODO: see _best.py for things to do in here
def configure_intrabeam_scattering(
    line: xt.Line,
    element_name: str,
    recompute_rates_every_nturns: int,
    formalism: str = None,
):
    """Configuration step for IBS parameters (like for beambeam for instance)
    where we do the twiss etc"""
    # Do a 4D twiss of the line and create opticsparameters
    # Create a beamparameters object from the particles
    # properly instantiate the kick element with the above
    raise NotImplementedError("Not yet implemented")
