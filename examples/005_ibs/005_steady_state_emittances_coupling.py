# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #
import matplotlib.pyplot as plt
import xtrack as xt
from scipy.constants import e

##########################
# Load xt.Line from file #
##########################

fname_line_particles = "../../../xtrack/test_data/bessy3/bessy3.json"
line = xt.Line.from_json(fname_line_particles)  # has particle_ref
line.build_tracker()

########################
# Twiss with Radiation #
########################

# We need to Twiss with Synchrotron Radiation enabled to obtain
# the SR equilibrium emittances and damping constants

line.matrix_stability_tol = 1e-2
line.configure_radiation(model="mean")
line.compensate_radiation_energy_loss()
tw = line.twiss(eneloss_and_damping=True)

######################################
# Steady-State Emittance Calculation #
######################################

bunch_intensity = 1e-9 / e  # 1C bunch charge
emittance_coupling_factor = 1  # round beam

# If not providing starting emittances, the function will
# default to the SR equilibrium emittances from the TwissTable

result = tw.compute_equilibrium_emittances_from_sr_and_ibs(
    formalism="nagaitsev",  # can also be "bjorken-mtingwa"
    total_beam_intensity=bunch_intensity,
    # gemitt_x=...,  # defaults to tw.eq_gemitt_x
    # gemitt_y=...,  # defaults to tw.eq_gemitt_x
    # gemitt_zeta=...,  # defaults to tw.eq_gemitt_zeta
    emittance_coupling_factor=emittance_coupling_factor,
    emittance_constraint="coupling",
)

# The returned object is an xtrack Table
print(result)

# Table: 1089 rows, 9 cols
# time                                  gemitt_x      gemitt_y   gemitt_zeta    sigma_zeta   sigma_delta            Kx            Ky            Kz
# 1.2104374139405318e-06              6.8355e-11    6.8355e-11   3.38163e-06    0.00344696   0.000981047       58.4843    -0.0165807       17.0321
# 9.306653395492252e-05               6.8722e-11    6.8722e-11   3.39221e-06    0.00345235   0.000982581       58.4741    -0.0165776       17.0295
# 0.00018492263049590453             6.90808e-11   6.90808e-11   3.40264e-06    0.00345765    0.00098409       57.7063     -0.016351       16.8364
# ...
# 0.09975693128092233                8.44921e-11   8.44921e-11    4.5177e-06    0.00398411    0.00113393       29.7991   -0.00861234       8.14798
# 0.09984878737746332                8.44921e-11   8.44921e-11    4.5177e-06    0.00398412    0.00113393       29.7991   -0.00861234       8.14797
# 0.0999406434740043                 8.44921e-11   8.44921e-11   4.51771e-06    0.00398412    0.00113393       29.7991   -0.00861234       8.14796

######################################
# Comparison with analytical results #
######################################

# These are analytical estimate (from the last step's IBS growth rates)
# The factor below is to be respected with the coupling constraint
factor = 1 + emittance_coupling_factor * (tw.partition_numbers[1] / tw.partition_numbers[0])
analytical_x = result.gemitt_x[0] / (1 - result.Kx[-1] / (tw.damping_constants_s[0] * factor))
analytical_y = result.gemitt_y[0] / (1 - result.Kx[-1] / (tw.damping_constants_s[0] * factor))
analytical_z = result.gemitt_zeta[0] / (1 - result.Kz[-1] / (tw.damping_constants_s[2]))

print()
print("Emittance Constraint: Coupling")
print("Horizontal steady-state emittance:")
print("---------------------------------")
print(f"Analytical: {analytical_x}")
print(f"ODE:        {result.eq_sr_ibs_gemitt_x}")
print("Vertical steady-state emittance:")
print("-------------------------------")
print(f"Analytical: {analytical_y}")
print(f"ODE:        {result.eq_sr_ibs_gemitt_y}")
print("Longitudinal steady-state emittance:")
print("-----------------------------------")
print(f"Analytical: {analytical_z}")
print(f"ODE:        {result.eq_sr_ibs_gemitt_zeta}")

# Emittance Constraint: Coupling
# Horizontal steady-state emittance:
# ---------------------------------
# Analytical: 8.450210629127059e-11
# ODE:        8.449209288849085e-11
# Vertical steady-state emittance:
# -------------------------------
# Analytical: 8.450210629127059e-11
# ODE:        8.449209288849085e-11
# Longitudinal steady-state emittance:
# -----------------------------------
# Analytical: 4.518897071412721e-06
# ODE:        4.517705696189597e-06

# The results from the table can easily be plotted to view
# at the evolution of various parameters across time steps

#!end-doc-part
# fmt: off

fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, layout="constrained")

(l1,) = ax0.plot(result.time * 1e3, result.gemitt_x * 1e12, ls="-", label=r"$\tilde{\varepsilon}_x$")
(l2,) = ax0.plot(result.time * 1e3, result.gemitt_y * 1e12, ls="--", label=r"$\tilde{\varepsilon}_y$")
l4 = ax0.axhline(analytical_x * 1e12, color="C0", ls="-.", label=r"Analytical $\varepsilon_{x}^{eq}$")
l5 = ax0.axhline(analytical_y * 1e12, color="C1", ls="-.", label=r"Analytical $\varepsilon_{y}^{eq}$")
ax0b = ax0.twinx()
(l3,) = ax0b.plot(result.time * 1e3, result.gemitt_zeta * 1e6, color="C2", label=r"$\varepsilon_z$")
l6 = ax0b.axhline(analytical_z * 1e6, color="C2", ls="-.", label=r"Analytical $\varepsilon_{\zeta}^{eq}$")
ax0.legend(handles=[l1, l2, l3, l4, l5], ncols=2)

ax1.plot(result.time * 1e3, result.Kx, label=r"$\alpha_{x}^{IBS}$")
ax1.plot(result.time * 1e3, result.Ky, label=r"$\alpha_{y}^{IBS}$")
ax1.plot(result.time * 1e3, result.Kz, label=r"$\alpha_{z}^{IBS}$")
ax1.legend()

ax1.set_xlabel("Time [ms]")
ax0.set_ylabel(r"$\tilde{\varepsilon}_{x,y}$ [pm.rad]")
ax0b.set_ylabel(r"$\varepsilon_{\zeta}$ [m]")
ax1.set_ylabel(r"$\alpha^{IBS}$ [$s^{-1}$]")
fig.align_ylabels((ax0, ax1))
plt.tight_layout()
plt.show()
