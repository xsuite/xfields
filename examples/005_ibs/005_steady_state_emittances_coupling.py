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

result = tw.get_ibs_and_synrad_emittance_evolution(
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
# time                                  gemitt_x      nemitt_x      gemitt_y      nemitt_y   gemitt_zeta   nemitt_zeta    sigma_zeta   sigma_delta            Kx            Ky            Kz
# 1.2104374139405318e-06              6.8355e-11   3.34418e-07    6.8355e-11   3.34418e-07   3.38167e-06     0.0581296    0.00344698   0.000981052       58.4838    -0.0165806       17.0318
# 9.306647341196751e-05              6.87219e-11   3.36214e-07   6.87219e-11   3.36214e-07   3.39225e-06     0.0583114    0.00345237   0.000982586       58.4736    -0.0165775       17.0292
# 0.0001849225094099945              6.90808e-11   3.37969e-07   6.90808e-11   3.37969e-07   3.40267e-06     0.0584907    0.00345767   0.000984095       57.7058    -0.0163509       16.8361
# ...
# 0.0997568655312697                  8.4492e-11   4.13367e-07    8.4492e-11   4.13367e-07   4.51774e-06     0.0776582    0.00398413    0.00113393        29.799   -0.00861232       8.14787
# 0.09984872156726772                 8.4492e-11   4.13367e-07    8.4492e-11   4.13367e-07   4.51774e-06     0.0776583    0.00398413    0.00113393        29.799   -0.00861232       8.14786
# 0.09994057760326575                8.44919e-11   4.13367e-07   8.44919e-11   4.13367e-07   4.51775e-06     0.0776584    0.00398414    0.00113393        29.799   -0.00861232       8.14784

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
# Analytical: 8.450195693021898e-11
# ODE:        8.449194372183254e-11
# Vertical steady-state emittance:
# -------------------------------
# Analytical: 8.450195693021898e-11
# ODE:        8.449194372183254e-11
# Longitudinal steady-state emittance:
# -----------------------------------
# Analytical: 4.518939710729963e-06
# ODE:        4.5177482475985255e-06

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

ax1.plot(result.time * 1e3, result.Kx, label=r"$K_{x}^{IBS}$")
ax1.plot(result.time * 1e3, result.Ky, label=r"$K_{y}^{IBS}$")
ax1.plot(result.time * 1e3, result.Kz, label=r"$K_{z}^{IBS}$")
ax1.legend()

ax1.set_xlabel("Time [ms]")
ax0.set_ylabel(r"$\tilde{\varepsilon}_{x,y}$ [pm.rad]")
ax0b.set_ylabel(r"$\varepsilon_{\zeta}$ [m]")
ax1.set_ylabel(r"$K_{x,y,z}^{IBS}$ [$s^{-1}$]")
fig.align_ylabels((ax0, ax1))
plt.tight_layout()
plt.show()
