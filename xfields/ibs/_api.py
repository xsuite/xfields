# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #
from logging import getLogger
from typing import Literal, Tuple, Union

import xtrack as xt

from xfields.ibs._analytical import AnalyticalIBS, BjorkenMtingwaIBS, IBSGrowthRates, NagaitsevIBS
from xfields.ibs._formulary import _bunch_length, _geom_epsx, _geom_epsy, _sigma_delta
from xfields.ibs._inputs import BeamParameters, OpticsParameters

LOGGER = getLogger(__name__)


# ----- API for Analytical IBS -----#


def get_intrabeam_scattering_growth_rates(
    line: xt.Line,
    formalism: Literal["Nagaitsev", "Bjorken-Mtingwa", "B&M"],
    npart: int = None,
    epsx: float = None,
    epsy: float = None,
    sigma_delta: float = None,
    bunch_length: float = None,
    bunched: bool = True,
    normalized_emittances: bool = False,
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
    npart : int, optional
        The number of particles in the beam. Required if `particles` is
        not provided.
    epsx : float, optional
        Horizontal (geometric or normalized) emittance in [m]. Required
        if `particles` is not provided.
    epsy : float, optional
        Vertical (geometric or normalized) emittance in [m]. Required
        if `particles` is not provided.
    sigma_delta : float, optional
        The momentum spread. Required if `particles` is not provided.
    bunch_length : float, optional
        The bunch length in [m]. Required if `particles` is not provided.
    bunched : bool, optional
        Whether the beam is bunched or not (coasting). Defaults to `True`.
        Required if `particles` is not provided.
    normalized_emittances : bool, optional
        Whether the provided emittances are normalized or not. Defaults to
        `False` (assumes geometric emittances).  Required if `particles`
        is not provided.
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
    if isinstance(particles, xt.Particles):  # also asserts it is not None
        LOGGER.info("Particles provided, will determine emittances, etc. from them")
        assert npart is None
        assert epsx is None
        assert epsy is None
        assert sigma_delta is None
        assert bunch_length is None
        _using_particles: bool = True
        npart: int = particles._num_active_particles * particles.weight[0]  # its total_intensity_particles
        epsx: float = _geom_epsx(particles)
        epsy: float = _geom_epsy(particles)
        sigma_delta: float = _sigma_delta(particles)
        bunch_length: float = _bunch_length(particles)
        normalized_emittances: bool = False  # we computed geometric
    else:  # we expect all emittances, etc. to be provided
        LOGGER.info("Using explicitely provided parameters for emittances, etc.")
        assert npart is not None
        assert epsx is not None
        assert epsy is not None
        assert sigma_delta is not None
        assert bunch_length is not None
        assert normalized_emittances is not None
        _using_particles: bool = False
    # ----------------------------------------------------------------------------------------------
    # Get necessary beam and optics parameters from Line and/or particles
    beam_params = BeamParameters(particles) if _using_particles else BeamParameters.from_line(line, npart)
    optics_params = OpticsParameters.from_line(line)
    # ----------------------------------------------------------------------------------------------
    # Ensure valid formalism parameter was given and determine the corresponding class
    assert formalism.lower() in ("nagaitsev", "bjorken-mtingwa", "b&m")
    if formalism.lower() == "nagaitsev":
        IBS = NagaitsevIBS(beam_params, optics_params)
    else:
        IBS = BjorkenMtingwaIBS(beam_params, optics_params)
    # ----------------------------------------------------------------------------------------------
    # Now computing the growth rates using the IBS class
    growth_rates: IBSGrowthRates = IBS.growth_rates(
        epsx=epsx,
        epsy=epsy,
        sigma_delta=sigma_delta,
        bunch_length=bunch_length,
        bunched=bunched,
        normalized_emittances=normalized_emittances,
        **kwargs,
    )
    # ----------------------------------------------------------------------------------------------
    # Return the growth rates, and potentially the IBS class instance too
    if return_class is True:
        return growth_rates, IBS
    else:
        return growth_rates


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
    recompute_rates_every_nturns: int,  # will need a way to know which tracking turn we're at
):
    """Configuration step for IBS parameters (like for beambeam for instance)
    where we do the twiss etc"""
    # Do a 4D twiss of the line and create opticsparameters
    # Create a beamparameters object from the particles
    # properly instantiate the kick element with the above
    raise NotImplementedError("Not yet implemented")
