# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #
import xpart as xp
import xtrack as xt

from xfields.ibs import get_intrabeam_scattering_growth_rates

##########################
# Load xt.Line from file #
##########################

# TODO: have lines or something in this repo?
fname_line_particles = "../../../xtrack/test_data/sps_ions/line_and_particle.json"
line = xt.Line.from_json(fname_line_particles)
twiss = line.twiss(method="4d")

#####################
# Define parameters #
#####################

# Line is for SPS ions at injection
bunch_intensity: int = int(3.5e8)
nemitt_x: float = 1.2612e-6
nemitt_y: float = 0.9081e-6
sigma_delta: float = 3.59e-4
bunch_length: float = 19.51e-2

###############################
# Power accelerating cavities #
###############################

rf_voltage = 1.7e6  # 1.7MV
harmonic_number = 4653
cavity = "actcse.31632"
line[cavity].lag = 180  # 180 above transition
line[cavity].voltage = rf_voltage
line[cavity].frequency = harmonic_number / twiss.T_rev0  # H * revolution frequency

line.build_tracker()

######################
# Generate particles #
######################

particles = xp.generate_matched_gaussian_bunch(
    num_particles=10_000,
    total_intensity_particles=bunch_intensity,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=bunch_length,
    line=line,
)

###################################
# Get growth rates with Nagaitsev #
###################################

nag_growth_rates = get_intrabeam_scattering_growth_rates(
    line=line,
    formalism="nagaitsev",
    particles=particles,
    bunched=True,
)

#########################################
# Get growth rates with Bjorken-Mtingwa #
#########################################

bm_growth_rates = get_intrabeam_scattering_growth_rates(
    line=line,
    formalism="bjorken-mtingwa",  # also accepts "b&m"
    particles=particles,
    bunched=True,
)

###################
# Compare results #
###################

print()
print("Computed from particles object:")
print("-------------------------------")
print(f"Nagaitsev:       {nag_growth_rates}")
print(f"Bjorken-Mtingwa: {bm_growth_rates}")
