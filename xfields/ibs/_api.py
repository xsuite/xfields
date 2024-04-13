# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #
from logging import getLogger
from typing import Literal, Tuple, Union

import numpy as np
import xtrack as xt

from xfields.ibs._analytical import AnalyticalIBS, BjorkenMtingwaIBS, IBSGrowthRates, NagaitsevIBS
from xfields.ibs._formulary import _bunch_length, _geom_epsx, _geom_epsy, _sigma_delta

LOGGER = getLogger(__name__)


# ----- API for Analytical IBS -----#


# TODO: return a table and we can always change the inside of container? RABBIT HOLE
def get_intrabeam_scattering_growth_rates(
    twiss: xt.TwissTable,
    formalism: Literal["Nagaitsev", "Bjorken-Mtingwa", "B&M"],
    num_particles: int = None,
    gemitt_x: float = None,
    gemitt_y: float = None,
    nemitt_x: float = None,
    nemitt_y: float = None,
    sigma_delta: float = None,
    bunch_length: float = None,
    bunched: bool = True,
    particles: xt.Particles = None,
    return_class: bool = False,
    **kwargs,
) -> Union[IBSGrowthRates, Tuple[IBSGrowthRates, AnalyticalIBS]]:
    """
    Computes IntraBeam Scattering growth rates from the provided `xtrack.Line`.

    Parameters
    ----------
    line : xtrack.Line
        Line in which the IBS kick element will be installed.
    formalism : str
        Which formalism to use for the computation. Can be ``Nagaitsev``
        or ``Bjorken-Mtingwa`` (also accepts ``B&M``), case-insensitively.
    num_particles : int, optional
        The number of particles in the beam. Required if `particles` is
        not provided.
    gemitt_x : float, optional
        Horizontal geometric emittance in [m]. If `particles` is not
        provided, either this parameter or `nemitt_x` is required.
    gemitt_y : float, optional
        Vertical geometric emittance in [m]. If `particles` is not
        provided, either this parameter or `nemitt_y` is required.
    nemitt_x : float, optional
        Horizontal normalized emittance in [m]. If `particles` is not
        provided, either this parameter or `gemitt_x` is required.
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
        Otherwise the ``epsx``, ``epsy``, ``sigma_delta`` and ``bunch_length``
        parameters must be provided.
    return_class : bool
        Whether to return the IBS class instance or not. Defaults to `False`.
    **kwargs : dict
        Keyword arguments are passed to the growth rates computation method of
        the chosen IBS formalism implementation. See the formalism classes in
        the ``xfields.ibs._analytical`` for more details.

    Returns
    -------
    IBSGrowthRates
        An ``IBSGrowthRates`` object with the computed growth rates.
    AnalyticalIBS, optional
        If ``return_class`` is `True`, the IBS class instance is also returned.
        It has knowledge of relevant beam and optics parameters, growth rates,
        and can be used to compute analytical emittance evolutions
    """
    # ----------------------------------------------------------------------------------------------
    # Perform checks on exclusive parameters: need either particles or all emittances, etc.
    # If particles parameter is an xtrack.Particles object
    if isinstance(particles, xt.Particles):
        LOGGER.info("Particles provided, will determine emittances, etc. from them")
        assert num_particles is None, "Cannot provide 'num_particles' with 'particles'"
        gemitt_x = _geom_epsx(particles)
        gemitt_y = _geom_epsy(particles)
        sigma_delta = _sigma_delta(particles)
        bunch_length = _bunch_length(particles)
        num_particles = particles._num_active_particles * particles.weight[0]  # total_intensity_particles
    # If particles is None or nonsense, we expect emittances given
    else:
        LOGGER.info("Using explicitely provided parameters for emittances, etc.")
        # The following are required
        assert num_particles is not None, "Must provide 'num_particles'"
        assert sigma_delta is not None, "Must provide 'sigma_delta'"
        assert bunch_length is not None, "Must provide 'bunch_length'"

        # At least one in these duos is required
        if gemitt_x is None and nemitt_x is None:
            raise ValueError("Must provide either 'gemitt_x' or 'nemitt_x'")
        if gemitt_y is None and nemitt_y is None:
            raise ValueError("Must provide either 'gemitt_y' or 'nemitt_y'")

        # For transverse emittances, geometric and normalized are exclusive
        if gemitt_x is not None:
            assert nemitt_x is None, "Cannot provide both 'gemitt_x' and 'nemitt_x'"

        if gemitt_y is not None:
            assert nemitt_y is None, "Cannot provide both 'gemitt_y' and 'nemitt_y'"

        beta0: float = twiss.particle_on_co.beta0
        gamma0: float = twiss.particle_on_co.gamma0

        if nemitt_x is not None:
            assert gemitt_x is None, "Cannot provide both 'gemitt_x' and 'nemitt_x'"
            gemitt_x = nemitt_x / (beta0 * gamma0)

        if nemitt_y is not None:
            assert gemitt_y is None, "Cannot provide both 'gemitt_y' and 'nemitt_y'"
            gemitt_y = nemitt_y / (beta0 * gamma0)
    # ----------------------------------------------------------------------------------------------
    # Ensure valid formalism parameter was given and determine the corresponding class
    assert formalism.lower() in ("nagaitsev", "bjorken-mtingwa", "b&m")
    if formalism.lower() == "nagaitsev":
        if np.count_nonzero(twiss.dy) != 0:
            LOGGER.warning("Vertical dispersion is present, Nagaitsev formalism does not account for it")
        ibs = NagaitsevIBS(twiss, num_particles)
    else:
        ibs = BjorkenMtingwaIBS(twiss, num_particles)
    # ----------------------------------------------------------------------------------------------
    # Now computing the growth rates using the IBS class
    growth_rates: IBSGrowthRates = ibs.growth_rates(
        epsx=gemitt_x,
        epsy=gemitt_y,
        sigma_delta=sigma_delta,
        bunch_length=bunch_length,
        bunched=bunched,
        normalized_emittances=False,
        **kwargs,
    )
    # ----------------------------------------------------------------------------------------------
    # Return the growth rates, and potentially the IBS class instance too
    return (growth_rates, ibs) if return_class is True else growth_rates


# ----- API for Kick-Based IBS -----#


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


def configure_intrabeam_scattering(
    line: xt.Line,
    particles: xt.Particles,
    recompute_rates_every_nturns: int,
):
    """Configuration step for IBS parameters (like for beambeam for instance)
    where we do the twiss etc"""
    # Do a 4D twiss of the line and create opticsparameters
    # Create a beamparameters object from the particles
    # properly instantiate the kick element with the above
    raise NotImplementedError("Not yet implemented")
