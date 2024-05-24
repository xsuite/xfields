# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #
import xtrack as xt

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

###################
# Compare results #
###################

print()
print("Computed from normalized emittances:")
print("------------------------------------")
print(f"Nagaitsev:       {nag_growth_rates}")
print(f"Bjorken-Mtingwa: {bm_growth_rates}")

# Computed from normalized emittances:
# ------------------------------------
# Nagaitsev:       IBSGrowthRates(Tx=0.00104, Ty=0.011, Tz=0.00803)
# Bjorken-Mtingwa: IBSGrowthRates(Tx=0.00111, Ty=0.0109, Tz=0.008)

#####################
# Define parameters #
#####################

# Should roughly be equivalent
gemitt_x: float = 1.742e-7
gemitt_y: float = 1.254e-7

###################################
# Get growth rates with Nagaitsev #
###################################

nag_growth_rates2 = tw.get_ibs_growth_rates(
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

bm_growth_rates2 = tw.get_ibs_growth_rates(
    formalism="bjorken-mtingwa",  # also accepts "b&m"
    total_beam_intensity=bunch_intensity,
    gemitt_x=gemitt_x,
    gemitt_y=gemitt_y,
    sigma_delta=sigma_delta,
    bunch_length=bunch_length,
    bunched=True,
)

###################
# Compare results #
###################

print()
print("Computed from geometric emittances (rough equivalent):")
print("------------------------------------------------------")
print(f"Nagaitsev:       {nag_growth_rates2}")
print(f"Bjorken-Mtingwa: {bm_growth_rates2}")

# Computed from geometric emittances (rough equivalent):
# ------------------------------------------------------
# Nagaitsev:       IBSGrowthRates(Tx=0.00103, Ty=0.0111, Tz=0.00803)
# Bjorken-Mtingwa: IBSGrowthRates(Tx=0.00111, Ty=0.0109, Tz=0.008)