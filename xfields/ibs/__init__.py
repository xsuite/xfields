# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from ._analytical import BjorkenMtingwaIBS, IBSGrowthRates, NagaitsevIBS
from ._api import (
    configure_intrabeam_scattering,
    get_intrabeam_scattering_growth_rates,
    install_intrabeam_scattering_kick,
)
from ._kicks import IBSKineticKick, IBSSimpleKick

__all__ = [
    "BjorkenMtingwaIBS",
    "NagaitsevIBS",
    "configure_intrabeam_scattering",
    "get_intrabeam_scattering_growth_rates",
    "install_intrabeam_scattering_kick",
    "IBSGrowthRates",
    "IBSKineticKick",
    "IBSSimpleKick",
]

# TODO: see with / ask Gianni:
# - can the element be made aware of the line it's in and the turn number in tracking? -> SEE the at_turn from particles
# - how to maximize the use of xobjects? Currently rewrote some things as xo.HybridClass objects, is this the right way?
