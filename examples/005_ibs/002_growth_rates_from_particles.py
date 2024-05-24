# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #
import xpart as xp
import xtrack as xt

##########################
# Load xt.Line from file #
##########################

# This is SPS line with proton as particle ref
fname_line_particles = "../../../xtrack/test_data/sps_w_spacecharge/"\
                       "line_no_spacecharge_and_particle.json"
line = xt.Line.from_json(fname_line_particles)
tw = line.twiss(method="4d")

#####################
# Define parameters #
#####################

# Line is for SPS protons at injection
bunch_intensity: int = int(3.5e8)
nemitt_x: float = 2.5e-6
nemitt_y: float = 2.5e-6
sigma_delta: float = 9.56e-4
bunch_length: float = 8.98e-2

######################
# Generate particles #
######################

particles = xp.generate_matched_gaussian_bunch(
    num_particles=100_000,
    total_intensity_particles=bunch_intensity,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=bunch_length,
    line=line,
)

###################################
# Get growth rates with Nagaitsev #
###################################

nag_growth_rates = tw.get_ibs_growth_rates(
    formalism="nagaitsev",
    particles=particles,
    bunched=True,
)

#########################################
# Get growth rates with Bjorken-Mtingwa #
#########################################

bm_growth_rates = tw.get_ibs_growth_rates(
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

# Computed from particles object:
# -------------------------------
# Nagaitsev:       IBSGrowthRates(Tx=1.54e-06, Ty=-1.46e-07, Tz=1.65e-06)
# Bjorken-Mtingwa: IBSGrowthRates(Tx=1.54e-06, Ty=-1.48e-07, Tz=1.65e-06)

###################################
# Get growth rates with Nagaitsev #
###################################

nag_growth_rates2 = tw.get_ibs_growth_rates(
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

bm_growth_rates2 = tw.get_ibs_growth_rates(
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
print("Computed from normalized emittances (rough equivalent):")
print("-------------------------------------------------------")
print(f"Nagaitsev:       {nag_growth_rates}")
print(f"Bjorken-Mtingwa: {bm_growth_rates}")

# Computed from normalized emittances (rough equivalent):
# -------------------------------------------------------
# Nagaitsev:       IBSGrowthRates(Tx=1.54e-06, Ty=-1.46e-07, Tz=1.65e-06)
# Bjorken-Mtingwa: IBSGrowthRates(Tx=1.54e-06, Ty=-1.48e-07, Tz=1.65e-06)
