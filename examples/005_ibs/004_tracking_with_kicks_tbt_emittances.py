# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #
import xfields as xf
import xobjects as xo
import xpart as xp
import xtrack as xt
from xfields.ibs._formulary import _gemitt_x, _gemitt_y, _sigma_delta, _bunch_length
import numpy as np
import matplotlib.pyplot as plt

# context = xo.ContextCupy()
context = xo.ContextCpu(omp_num_threads="auto")

##########################
# Load xt.Line from file #
##########################

fname_line_particles = "../../../xtrack/test_data/clic_dr/line.json"
line: xt.Line = xt.Line.from_json(fname_line_particles)
line.build_tracker(_context=context)
cavities = [element for element in line.elements if isinstance(element, xt.Cavity)]
for cavity in cavities:
    cavity.lag = 180
tw = line.twiss(method="4d")

#######################################
# Create and Install IBS Kick Element #
#######################################

# ibs_kick = xf.IBSKineticKick(num_slices=50)
ibs_kick = xf.IBSAnalyticalKick(formalism="nagaitsev", num_slices=50)
line.configure_intrabeam_scattering(
    element=ibs_kick, name="ibskick", index=-1, update_every=50
)

################################
# Generate Particles and Track #
################################

nturns: int = 2000
epsx, epsy, sigd, bl = [], [], [], []

particles = xp.generate_matched_gaussian_bunch(
    num_particles=2500,
    total_intensity_particles=int(4.5e9),
    nemitt_x=5.66e-7,
    nemitt_y=3.7e-9,
    sigma_z=1.58e-3,
    line=line,
    _context=context,
)

for turn in range(nturns):
    line.track(particles, num_turns=1)
    epsx.append(_gemitt_x(particles, tw.betx[0], tw.dx[0]))
    epsy.append(_gemitt_y(particles, tw.bety[0], tw.dy[0]))
    sigd.append(_sigma_delta(particles))
    bl.append(_bunch_length(particles))

#############################
# Plot turn-by-turn results #
#############################

turns = np.arange(nturns) + 1  # start from 1
epsx = np.array(epsx)
epsy = np.array(epsy)
sigd = np.array(sigd)
bl = np.array(bl)

fig, axx = plt.subplots()
axx.plot(turns, 1e10 * epsx, c="C0", label=r"$\varepsilon_x$")
axx.set_xlabel("Turn")
axx.set_ylabel(r"$\varepsilon_x$ [$10^{-10}$m]")
axy = axx.twinx()
axy.plot(turns, 1e13 * epsy, c="C1", label=r"$\varepsilon_y$")
axy.yaxis.set_label_position("right")
axy.set_ylabel(r"$\varepsilon_y$ [$10^{-13}$m]")
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.95), ncols=2)
fig.tight_layout()
fig.show()

fig2, axd = plt.subplots()
axd.plot(turns, 1e3 * sigd, c="C2", label=r"$\sigma_{\delta}$")
axd.set_xlabel("Turn")
axd.set_ylabel(r"$\sigma_{\delta}$ [$10^{-3}$]")
axb = axd.twinx()
axb.plot(turns, 1e3 * bl, c="C3", label=r"$\sigma_z$")
axb.yaxis.set_label_position("right")
axb.set_ylabel(r"$\sigma_z$ [$10^{-3}$m]")
fig2.legend(loc="upper center", bbox_to_anchor=(0.5, 0.95), ncols=2)
fig2.tight_layout()
fig2.show()
