# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #
import xtrack as xt

from xfields.ibs import get_intrabeam_scattering_growth_rates

##########################
# Load xt.Line from file #
##########################

fname_line_particles = "../../../xtrack/test_data/sps_ions/line_and_particle.json"
line = xt.Line.from_json(fname_line_particles)
tw = line.twiss(method="4d")

#####################
# Define parameters #
#####################

# Line is for SPS ions at injection
bunch_intensity: int = int(3.5e8)
nemitt_x: float = 1.2612e-6
nemitt_y: float = 0.9081e-6
sigma_delta: float = 3.59e-4
bunch_length: float = 19.51e-2

###################################
# Get growth rates with Nagaitsev #
###################################

nag_growth_rates = get_intrabeam_scattering_growth_rates(
    twiss=tw,
    formalism="nagaitsev",
    num_particles=bunch_intensity,
    epsx=nemitt_x,
    epsy=nemitt_y,
    sigma_delta=sigma_delta,
    bunch_length=bunch_length,
    bunched=True,
    normalized_emittances=True,
)

#########################################
# Get growth rates with Bjorken-Mtingwa #
#########################################

bm_growth_rates = get_intrabeam_scattering_growth_rates(
    twiss=tw,
    formalism="bjorken-mtingwa",  # also accepts "b&m"
    num_particles=bunch_intensity,
    epsx=nemitt_x,
    epsy=nemitt_y,
    sigma_delta=sigma_delta,
    bunch_length=bunch_length,
    bunched=True,
    normalized_emittances=True,
)

###################
# Compare results #
###################

print()
print("Computed from normalized emittances:")
print("------------------------------------")
print(f"Nagaitsev:       {nag_growth_rates}")
print(f"Bjorken-Mtingwa: {bm_growth_rates}")

#####################
# Define parameters #
#####################

# Should roughly be equivalent
gemitt_x: float = 1.742e-7
gemitt_y: float = 1.254e-7

###################################
# Get growth rates with Nagaitsev #
###################################

nag_growth_rates2 = get_intrabeam_scattering_growth_rates(
    twiss=tw,
    formalism="nagaitsev",
    num_particles=bunch_intensity,
    epsx=gemitt_x,
    epsy=gemitt_y,
    sigma_delta=sigma_delta,
    bunch_length=bunch_length,
    bunched=True,
    normalized_emittances=False,  # default value
)

#########################################
# Get growth rates with Bjorken-Mtingwa #
#########################################

bm_growth_rates2 = get_intrabeam_scattering_growth_rates(
    twiss=tw,
    formalism="bjorken-mtingwa",  # also accepts "b&m"
    num_particles=bunch_intensity,
    epsx=gemitt_x,
    epsy=gemitt_y,
    sigma_delta=sigma_delta,
    bunch_length=bunch_length,
    bunched=True,
    normalized_emittances=False,  # default value
)

###################
# Compare results #
###################

print()
print("Computed from geometric emittances (rough equivalent):")
print("------------------------------------------------------")
print(f"Nagaitsev:       {nag_growth_rates2}")
print(f"Bjorken-Mtingwa: {bm_growth_rates2}")
