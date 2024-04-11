# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #
import xtrack as xt

from xfields.ibs import get_intrabeam_scattering_growth_rates

##########################
# Load xt.Line from file #
##########################

# TODO: have lines or something in this repo?
fname_line_particles = "../../../xtrack/test_data/sps_ions/line_and_particle.json"
line = xt.Line.from_json(fname_line_particles)

#####################
# Define parameters #
#####################

bunch_intensity: float = 3.5e8
nemit_x: float = 1.2612e-6
nemit_y: float = 0.9081e-6
sigma_delta: float = 3.59e-4
bunch_length: float = 19.51e-2

###################################
# Get growth rates with Nagaitsev #
###################################

nag_growth_rates = get_intrabeam_scattering_growth_rates(
    line=line,
    formalism="nagaitsev",
    npart=bunch_intensity,
    epsx=nemit_x,
    epsy=nemit_y,
    sigma_delta=sigma_delta,
    bunch_length=bunch_length,
    bunched=True,
    normalized_emittances=True,
)

###################################
# Get growth rates with Nagaitsev #
###################################

bm_growth_rates = get_intrabeam_scattering_growth_rates(
    line=line,
    formalism="bjorken-mtingwa",  # also accepts "b&m"
    npart=bunch_intensity,
    epsx=nemit_x,
    epsy=nemit_y,
    sigma_delta=sigma_delta,
    bunch_length=bunch_length,
    bunched=True,
    normalized_emittances=True,
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

# Should roughly be equivalent
geom_epsx: float = 1.742e-7
geom_epsy: float = 1.254e-7

###################################
# Get growth rates with Nagaitsev #
###################################

nag_growth_rates2 = get_intrabeam_scattering_growth_rates(
    line=line,
    formalism="nagaitsev",
    npart=bunch_intensity,
    epsx=geom_epsx,
    epsy=geom_epsy,
    sigma_delta=sigma_delta,
    bunch_length=bunch_length,
    bunched=True,
    normalized_emittances=False,  # default value
)

###################################
# Get growth rates with Nagaitsev #
###################################

bm_growth_rates2 = get_intrabeam_scattering_growth_rates(
    line=line,
    formalism="bjorken-mtingwa",  # also accepts "b&m"
    npart=bunch_intensity,
    epsx=geom_epsx,
    epsy=geom_epsy,
    sigma_delta=sigma_delta,
    bunch_length=bunch_length,
    bunched=True,
    normalized_emittances=False,  # default value
)

##########################################################
# Compare: we expect Nagaitsev to be wrong in horizontal #
##########################################################

print()
print("Computed from geometric emittances (rough equivalent):")
print("------------------------------------------------------")
print(f"Nagaitsev:       {nag_growth_rates2}")
print(f"Bjorken-Mtingwa: {bm_growth_rates2}")
