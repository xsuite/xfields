# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #
import json

import numpy as np
import xtrack as xt

from xfields.ibs import get_intrabeam_scattering_growth_rates

##########################
# Load xt.Line from file #
##########################

fname_line_particles = "../../../xtrack/test_data/lhc_no_bb/line_and_particle.json"

with open(fname_line_particles, "r") as fid:
    input_data = json.load(fid)

line = xt.Line.from_json(fname_line_particles)
line.particle_ref = xt.Particles.from_dict(input_data["particle"])
tw = line.twiss(method="4d")

#####################
# Define parameters #
#####################

# Line is for LHC protons at top energy
bunch_intensity: int = int(1.8e11)
nemitt_x: float = 1.8e-6
nemitt_y: float = 1.8e-6
sigma_delta: float = 4.71e-5
bunch_length: float = 3.75e-2

###################################
# Get growth rates with Nagaitsev #
###################################

nag_growth_rates = get_intrabeam_scattering_growth_rates(
    twiss=tw,
    formalism="nagaitsev",
    total_beam_intensity=bunch_intensity,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_delta=sigma_delta,
    bunch_length=bunch_length,
    bunched=True,
)

#########################################
# Get growth rates with Bjorken-Mtingwa #
#########################################

bm_growth_rates = get_intrabeam_scattering_growth_rates(
    twiss=tw,
    formalism="bjorken-mtingwa",  # also accepts "b&m"
    total_beam_intensity=bunch_intensity,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_delta=sigma_delta,
    bunch_length=bunch_length,
    bunched=True,
)


##########################################################
# Compare: we expect Nagaitsev to be wrong in horizontal #
##########################################################

print()
print("Computed from normalized emittances:")
print("------------------------------------")
print(f"Nagaitsev:       {nag_growth_rates}")
print(f"Bjorken-Mtingwa: {bm_growth_rates}")

#####################
# Define parameters #
#####################

gemitt_x: float = 2.598e-10
gemitt_y: float = 2.598e-10

###################################
# Get growth rates with Nagaitsev #
###################################

nag_growth_rates2 = get_intrabeam_scattering_growth_rates(
    twiss=tw,
    formalism="nagaitsev",
    total_beam_intensity=bunch_intensity,
    gemitt_x=gemitt_x,
    gemitt_y=gemitt_y,
    sigma_delta=sigma_delta,
    bunch_length=bunch_length,
    bunched=True,
)

#########################################
# Get growth rates with Bjorken-Mtingwa #
#########################################

bm_growth_rates2 = get_intrabeam_scattering_growth_rates(
    twiss=tw,
    formalism="bjorken-mtingwa",  # also accepts "b&m"
    total_beam_intensity=bunch_intensity,
    gemitt_x=gemitt_x,
    gemitt_y=gemitt_y,
    sigma_delta=sigma_delta,
    bunch_length=bunch_length,
    bunched=True,
)

##########################################################
# Compare: we expect Nagaitsev to be wrong in horizontal #
##########################################################

print()
print("Computed from geometric emittances (rough equivalent):")
print("------------------------------------------------------")
print(f"Nagaitsev:       {nag_growth_rates2}")
print(f"Bjorken-Mtingwa: {bm_growth_rates2}")
