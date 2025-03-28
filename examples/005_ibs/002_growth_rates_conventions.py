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

####################
# Get growth rates #
####################

# There is no vertical dispersion so Nagaitsev
# will be correct in vertical
amp_growth_rates = tw.get_ibs_growth_rates(
    formalism="nagaitsev",
    total_beam_intensity=bunch_intensity,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_delta=sigma_delta,
    bunch_length=bunch_length,
    bunched=True,
)

##########################################################
# Converting between amplitude and emittance conventions #
##########################################################

# Notice how, when printing the returned object, it states
# the growth rates are given in amplitude convention
print(amp_growth_rates)
# IBSAmplitudeGrowthRates(Kx=0.000518, Ky=0.00552, Kz=0.00402)

# Methods are implemented to convert to the emittance convention
emit_growth_rates = amp_growth_rates.to_emittance_growth_rates()
print(emit_growth_rates)
# IBSEmittanceGrowthRates(Kx=0.00104, Ky=0.011, Kz=0.00803)

# It is also possible to convert back to the amplitude convention
print(f"Initial:         {amp_growth_rates}")
print(f"Converted twice: {emit_growth_rates.to_amplitude_growth_rates()}")
# Initial:         IBSAmplitudeGrowthRates(Kx=0.000518, Ky=0.00552, Kz=0.00402)
# Converted twice: IBSAmplitudeGrowthRates(Kx=0.000518, Ky=0.00552, Kz=0.00402)

####################################################
# Converting between growth rates and growth times #
####################################################

# Should one want the growth times, a method is available in both
# conventions to perform this conversion, although it returns a tuple
print(f"Amp times from amp rates:  {amp_growth_rates.to_amplitude_growth_times()}")
print(f"Amp times from emit rates: {emit_growth_rates.to_amplitude_growth_times()}")
# Amp times from amp rates:  (1930.7146824847905, 181.11747760500302, 248.968512633387)
# Amp times from emit rates: (1930.7146824847905, 181.11747760500302, 248.968512633387)

# And it is of course possible to get the emittance
# growth times from any of the two conventions
print(f"Emit times from amp rates:  {amp_growth_rates.to_emittance_growth_times()}")
print(f"Emit times from emit rates: {emit_growth_rates.to_emittance_growth_times()}")
# Emit times from amp rates:  (965.3573412423953, 90.55873880250151, 124.4842563166935)
# Emit times from emit rates: (965.3573412423953, 90.55873880250151, 124.4842563166935)