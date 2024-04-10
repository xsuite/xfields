# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xtrack as xt


def get_intrabeam_scattering_growth_rates(
    line: xt.Line,
    particles: xt.Particles,
    formalism: str,  # let's give an enum for the hint or something?
):
    """
    Computes IntraBeam Scattering growth rates from an xtrack.Line object, and the xt.Particles to circulate through it.

    Parameters
    ----------
    line : xtrack.Line
        Line in which the IBS kick element will be installed.
    particles : xtrack.Particles
        The period in [turns] with which to recompute the IBS growth rates during tracking.
    formalism : str
        Which formalism to use for the IBS kicks. Can be "simple" (only valid above transition) or "kinetic".
    """
    pass


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
    pass


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
    pass
