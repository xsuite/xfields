# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #
import xfields as xf
import xobjects as xo
import xpart as xp
import xtrack as xt

context = xo.ContextCpu(omp_num_threads="auto")

##########################
# Load xt.Line from file #
##########################

# This is SPS line with proton as particle ref
fname_line_particles = "../../../xtrack/test_data/sps_w_spacecharge/"\
                       "line_no_spacecharge_and_particle.json"
line: xt.Line = xt.Line.from_json(fname_line_particles)
line.build_tracker(_context=context)

#######################################
# Create and Install IBS Kick Element #
#######################################

# For the analytical kick formalism: kicks are computed based
# on the analytical growth rates (so it needs a formalism)
# ibs_kick = xf.IBSAnalyticalKick(formalism="nagaitsev", num_slices=50)

# For the kinetic formalism: kicks are computed based on the
# friction and diffusion terms of the kinetic theory of gases
ibs_kick = xf.IBSKineticKick(num_slices=50)

# By default the element is off until configuration. Let's install
# the kick at the end of the line and configure it. This internally
# provides the necessary information to the element
line.configure_intrabeam_scattering(
    element=ibs_kick, name="ibskick", index=-1, update_every=50
)

############################################
# Define parameters and Generate Particles #
############################################

# Line is for SPS protons at injection
bunch_intensity: int = int(3.5e8)
nemitt_x: float = 2.5e-6
nemitt_y: float = 2.5e-6
sigma_delta: float = 9.56e-4
bunch_length: float = 8.98e-2

particles = xp.generate_matched_gaussian_bunch(
    num_particles=10_000,
    total_intensity_particles=bunch_intensity,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=bunch_length,
    line=line,
    _context=context,
)

##############################################
# Track now applies an IBS kick at each turn #
##############################################

line.track(particles, num_turns=100, with_progress=5)
