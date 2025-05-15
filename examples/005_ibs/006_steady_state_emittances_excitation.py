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
emittance_coupling_factor = 0.5  # for excitation this time

# One can provide specific values for starting emittances,
# but we need to ensure they respect the emittance coupling
# contraint we want to enforce
gemitt_x = 1.1 * tw.eq_gemitt_x  # larger horizontal emittance
gemitt_y = emittance_coupling_factor * gemitt_x  # enforce the constraint

# One can overwrite sigma_zeta / sigma_delta (larger
# values from potential well distortion for example)
overwrite_sigma_zeta = 1.2 * (tw.eq_gemitt_zeta * tw.bets0) ** 0.5  # larger sigma_zeta
overwrite_sigma_delta = 1.2 * (tw.eq_gemitt_zeta / tw.bets0) ** 0.5  # larger sigma_delta

# A specific time step or relative tolerance for convergence can also be provided.
result = tw.compute_equilibrium_emittances_from_sr_and_ibs(
    formalism="nagaitsev",  # can also be "bjorken-mtingwa"
    total_beam_intensity=bunch_intensity,
    gemitt_x=gemitt_x,  # provided explicitely
    gemitt_y=gemitt_y,  # provided explicitely
    overwrite_sigma_zeta=overwrite_sigma_zeta,  # will recompute gemitt_zeta
    overwrite_sigma_delta=overwrite_sigma_delta,  # will recompute gemitt_zeta
    emittance_coupling_factor=emittance_coupling_factor,
    emittance_constraint="excitation",
)

# The returned object is a Table
print(result)

# Table: 989 rows, 9 cols
# time                                  gemitt_x      nemitt_x      gemitt_y      nemitt_y   gemitt_zeta   nemitt_zeta    sigma_zeta   sigma_delta            Kx            Ky            Kz
# 1.2104374139405318e-06             1.07706e-10   5.26938e-07    5.3853e-11   2.63469e-07   4.05791e-06      0.069754    0.00413633   0.000981042       32.1477   -0.00949839       13.5932
# 9.306647341196751e-05              1.08146e-10   5.29091e-07    5.4073e-11   2.64546e-07   4.06402e-06      0.069859    0.00413944    0.00098178       32.1437   -0.00949715       13.5919
# 0.0001849225094099945              1.08574e-10   5.31185e-07    5.4287e-11   2.65592e-07   4.07004e-06     0.0699624    0.00414251   0.000982507        31.844   -0.00940374       13.4942
# ...
# 0.09057126193146717                1.22535e-10   5.99488e-07   6.12675e-11   2.99744e-07   4.70398e-06     0.0808596    0.00445345    0.00105626       21.8776   -0.00648841       9.10733
# 0.09066311796746519                1.22535e-10   5.99487e-07   6.12675e-11   2.99744e-07   4.70398e-06     0.0808597    0.00445345    0.00105626       21.8776   -0.00648841       9.10732
# 0.09075497400346322                1.22535e-10   5.99487e-07   6.12675e-11   2.99744e-07   4.70399e-06     0.0808598    0.00445345    0.00105626       21.8776   -0.00648841       9.10731


# The results from the table can easily be plotted . One notices
# the transverse coupling factor is respected at all steps.

#!end-doc-part
# fmt: off

fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, layout="constrained")

(l1,) = ax0.plot(result.time * 1e3, result.gemitt_x * 1e12, ls="-", label=r"$\tilde{\varepsilon}_x$")
(l2,) = ax0.plot(result.time * 1e3, result.gemitt_y * 1e12, ls="-", label=r"$\tilde{\varepsilon}_y$")
(l3,) = ax0.plot(result.time * 1e3, result.gemitt_y / emittance_coupling_factor * 1e12, ls=":", c="C1", label=r"$\tilde{\varepsilon}_y$ / factor")
ax0b = ax0.twinx()
(l4,) = ax0b.plot(result.time * 1e3, result.gemitt_zeta * 1e6, color="C2", label=r"$\varepsilon_z$")
ax0.legend(handles=[l1, l2, l3, l4], ncols=2)

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
