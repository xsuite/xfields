# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #
import json

import xtrack as xt

##########################
# Load xt.Line from file #
##########################

fname_line_particles = "../../../xtrack/test_data/lhc_no_bb/" \
                       "line_and_particle.json"

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

nag_growth_rates = tw.get_ibs_growth_rates(
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

bm_growth_rates = tw.get_ibs_growth_rates(
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

# Computed from normalized emittances:
# ------------------------------------
# Nagaitsev:       IBSGrowthRates(Tx=6.24e-05, Ty=-2.27e-09, Tz=0.00031)
# Bjorken-Mtingwa: IBSGrowthRates(Tx=6.21e-05, Ty=1.1e-06, Tz=0.00031)
