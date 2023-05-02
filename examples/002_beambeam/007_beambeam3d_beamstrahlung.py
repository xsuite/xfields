# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp
from matplotlib import pyplot as plt
from scipy import constants as cst
import json

context = xo.ContextCpu(omp_num_threads=0)

###########
# ttbar 2 #
###########
bunch_intensity     = 2.3e11  # [1]
energy              = 182.5  # [GeV]
p0c                 = 182.5e9  # [eV]
mass0               = .511e6  # [eV]
phi                 = 15e-3  # [rad] half xing
u_sr                = 9.2  # [GeV]
u_bs                = .0114  # [GeV]
k2_factor           = .4  # [1]
qx                  = .554  # [1] half arc
qy                  = .588  # [1]
qs                  = .0436  # [1]
physemit_x          = 1.46e-09  # [m]
physemit_y          = 2.9e-12  # [m]
beta_x              = 1  # [m]
beta_y              = .0016  # [m]
sigma_x             = np.sqrt(physemit_x*beta_x)  # [m]
sigma_px            = np.sqrt(physemit_x/beta_x)  # [m]
sigma_y             = np.sqrt(physemit_y*beta_y)  # [m]
sigma_py            = np.sqrt(physemit_y/beta_y)  # [m]
sigma_z             = .00194  # [m] sr
sigma_z_tot         = .00254  # [m] sr+bs
sigma_delta         = .0015  # [m]
sigma_delta_tot     = .00192  # [m]
beta_s              = sigma_z/sigma_delta  # [m]
physemit_s          = sigma_z*sigma_delta  # [m]
physemit_s_tot      = sigma_z_tot*sigma_delta_tot  # [m]
n_macroparticles_b1 = int(1e3)
n_macroparticles_b2 = int(1e6)

n_turns = 1000
n_slices = 100

#############
# particles #
#############

#e-
particles_b1 = xp.Particles(
            _context = context,
            q0        = -1,
            p0c       = p0c,
            mass0     = mass0,
            x         = sigma_x        *np.random.randn(n_macroparticles_b1),
            y         = sigma_y        *np.random.randn(n_macroparticles_b1),
            zeta      = sigma_z        *np.random.randn(n_macroparticles_b1),
            px        = sigma_px       *np.random.randn(n_macroparticles_b1),
            py        = sigma_py       *np.random.randn(n_macroparticles_b1),
            delta     = sigma_delta    *np.random.randn(n_macroparticles_b1),
            )

particles_b1.name = "b1"

########################
# half arc with synrad #
########################

alpha_x = 0
alpha_y = 0
gamma_x = (1.0+alpha_x**2)/beta_x
gamma_y = (1.0+alpha_y**2)/beta_y

damping_rate_s = u_sr/energy # Usr/E ~ 1e-3
damping_rate_x = damping_rate_s / 2
damping_rate_y = damping_rate_s / 2

beta_x_sext_left  = 3
beta_y_sext_left  = 5000
beta_x_sext_right = 3
beta_y_sext_right = 5000

alpha_x_sext_left  = 0
alpha_y_sext_left  = 0
alpha_x_sext_right = 0
alpha_y_sext_right = 0

#crab sextupoles
k2_left  = k2_factor / (2 * phi * beta_y * beta_y_sext_left ) * np.sqrt(beta_x / beta_x_sext_left )
k2_right = k2_factor / (2 * phi * beta_y * beta_y_sext_right) * np.sqrt(beta_x / beta_x_sext_right)

el_sextupole_left  = xt.Multipole(order=2, knl=[0, 0, k2_left], length=0.3)
el_sextupole_right = xt.Multipole(order=2, knl=[0, 0, -k2_right], length=0.3)

# from IP to right crab sextupole (sy2r.2)
el_arc_left_b1 = xt.LinearTransferMatrix(_context=context,
    Q_x = 0,  # 2pi phase advance so integer part is zero
    Q_y = 0.25,  # 2.5pi
    Q_s = 0,  # no dipole here so no synchrotron motion
    beta_x_0 = beta_x,
    beta_y_0 = beta_y,
    beta_x_1 = beta_x_sext_left,
    beta_y_1 = beta_y_sext_left,
    alpha_x_0 = 0,
    alpha_y_0 = 0,
    alpha_x_1 = alpha_x_sext_left,
    alpha_y_1 = alpha_y_sext_left,
    beta_s = beta_s,
)

# between 2 sextupoles
el_arc_mid_b1 = xt.LinearTransferMatrix(_context=context,
    Q_x =  qx,
    Q_y =  qy - 0.5, # subtract .25*2 phase advance from small arcs
    Q_s = -qs,
    beta_x_0 = beta_x_sext_left,
    beta_y_0 = beta_y_sext_left,
    beta_x_1 = beta_x_sext_right,
    beta_y_1 = beta_y_sext_right,
    alpha_x_0 = alpha_x_sext_left,
    alpha_y_0 = alpha_y_sext_left,
    alpha_x_1 = alpha_x_sext_right,
    alpha_y_1 = alpha_y_sext_right,
    beta_s = sigma_z/sigma_delta,
    damping_rate_x = damping_rate_x,
    damping_rate_y = damping_rate_y,
    damping_rate_s = damping_rate_s,
    equ_emit_x = physemit_x,
    equ_emit_y = physemit_y,
    equ_emit_s = physemit_s, # only here i need sigma_z delta SR
    energy_increment = u_bs*1e9, # U_BS for one IP
)

# from left crab sextupole to IP2 (sy2l.1)
el_arc_right_b1 = xt.LinearTransferMatrix(_context=context,
    Q_x = 0,  # 2pi phase advance so integer part is zero
    Q_y = 0.25,  # 2.5pi
    Q_s = 0,  # no dipole here so no synchrotron motion
    beta_x_0 = beta_x_sext_right,
    beta_y_0 = beta_y_sext_right,
    beta_x_1 = beta_x,
    beta_y_1 = beta_y,
    alpha_x_0 = alpha_x_sext_right,
    alpha_y_0 = alpha_y_sext_right,
    alpha_x_1 = 0,
    alpha_y_1 = 0,
    beta_s = beta_s,
)

# injection from initial distribution to right sextupole
el_inject_b1 = xt.LinearTransferMatrix(_context=context,
    beta_x_0 = beta_x,
    beta_y_0 = beta_y,
    beta_x_1 = beta_x_sext_right,
    beta_y_1 = beta_y_sext_right,
    alpha_x_0 = 0,
    alpha_y_0 = 0,
    alpha_x_1 = alpha_x_sext_right,
    alpha_y_1 = alpha_y_sext_right,
    beta_s = beta_s,
)

##########################################
# weakstrong beambeam with beamstrahlung #
##########################################

slicer = xf.TempSlicer(n_slices=n_slices, sigma_z=sigma_z_tot, mode="shatilov")

el_beambeam_b1 = xf.BeamBeamBiGaussian3D(
        _context=context,
        config_for_update = None,
        other_beam_q0=1,
        phi=phi,
        alpha=0,
        # decide between round or elliptical kick formula
        min_sigma_diff     = 1e-28,
        # slice intensity [num. real particles] n_slices inferred from length of this
        slices_other_beam_num_particles = slicer.bin_weights * bunch_intensity,
        # unboosted strong beam moments
        slices_other_beam_zeta_center = slicer.bin_centers,
        slices_other_beam_Sigma_11    = n_slices*[sigma_x**2],
        slices_other_beam_Sigma_22    = n_slices*[sigma_px**2],
        slices_other_beam_Sigma_33    = n_slices*[sigma_y**2],
        slices_other_beam_Sigma_44    = n_slices*[sigma_py**2],
        # only if BS on
        slices_other_beam_zeta_bin_width_star_beamstrahlung = slicer.bin_widths_beamstrahlung / np.cos(phi),  # boosted dz
        # has to be set
        slices_other_beam_Sigma_12    = n_slices*[0],
        slices_other_beam_Sigma_34    = n_slices*[0],
)

#############################
# track for 1000 half turns #
#############################

emit_x_arr = np.zeros(n_turns)
emit_y_arr = np.zeros_like(emit_x_arr)
emit_s_arr = np.zeros_like(emit_x_arr)
x_std_arr  = np.zeros_like(emit_x_arr)
y_std_arr  = np.zeros_like(emit_x_arr)
z_std_arr  = np.zeros_like(emit_x_arr)


monitor_emits  = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns, particle_id_range=(0,n_macroparticles_b1))
monitor_coords = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=n_turns, particle_id_range=(0,n_macroparticles_b1))

el_inject_b1.track(particles_b1)
line = xt.Line(elements = [monitor_emits,
                           el_sextupole_right,
                           el_arc_right_b1,
                           monitor_coords,
                           el_beambeam_b1,
                           el_arc_left_b1,
                           el_sextupole_left,
                           el_arc_mid_b1])
line.build_tracker()

########################
# Enable beamstrahlung #
########################

line.configure_radiation(model_beamstrahlung='quantum')

#########
# Track #
#########

record = line.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D,
    capacity={"beamstrahlungtable": int(1e5), "bhabhatable": int(0), "lumitable": int(0)})
line.track(particles_b1, num_turns=n_turns)
line.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

################################
# Change beamstrahlung to mean #
################################

el_beambeam_b1.slices_other_beam_sqrtSigma_11_beamstrahlung = n_slices*[sigma_x]
el_beambeam_b1.slices_other_beam_sqrtSigma_33_beamstrahlung = n_slices*[sigma_y]
el_beambeam_b1.slices_other_beam_sqrtSigma_55_beamstrahlung = slicer.bin_weights * sigma_z_tot
tracker.configure_radiation(model_beamstrahlung='mean')

#########
# Track #
#########

record_avg = tracker.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D, capacity={"beamstrahlungtable": int(1e5), "bhabhatable": int(0), "lumitable": int(0)})
tracker.track(particles_b1, num_turns=n_turns)
tracker.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

#########
# tests #
#########

# quantum and mean beamstrahlung comparison
photon_energy_mean = np.mean(sorted(set(record.beamstrahlungtable.photon_energy))[1:])
assert np.all(np.abs(record_avg.beamstrahlungtable.photon_energy - photon_energy_mean) / photon_energy_mean < 5e-2)

# average and maximum BS parameter
# https://www.researchgate.net/publication/2278298_Beam-Beam_Phenomena_In_Linear_Colliders
# page 20
r0 = cst.e**2/(4*np.pi*cst.epsilon_0*cst.m_e*cst.c**2) # - if pp

upsilon_max =   2 * r0**2 * energy/(mass0*1e-9) * bunch_intensity / (1/137*sigma_z_tot*(sigma_x + 1.85*sigma_y))
upsilon_avg = 5/6 * r0**2 * energy/(mass0*1e-9) * bunch_intensity / (1/137*sigma_z_tot*(sigma_x + sigma_y))
upsilon_avg_sim = np.mean(0.67 * record.beamstrahlungtable.photon_critical_energy / (record.beamstrahlungtable.primary_energy))
upsilon_max_sim = np.max(0.67 * record.beamstrahlungtable.photon_critical_energy / (record.beamstrahlungtable.primary_energy))

print("Y max. [1]:", upsilon_max)
print("Y avg. [1]:", upsilon_avg)
print("Y max. [1]:", upsilon_max_sim)
print("Y avg. [1]:", upsilon_avg_sim)
print("Y max. ratio [1]:", upsilon_max_sim / upsilon_max)
print("Y avg. ratio [1]:", upsilon_avg_sim / upsilon_avg)

assert np.abs(upsilon_max_sim / upsilon_max - 1) < 2e-2
assert np.abs(upsilon_avg_sim / upsilon_avg - 1) < 5e-2

#########
# plots #
#########

def stat_emittance_from_monitor(emits_dict, n_macroparticles, n_turns, alpha_x=0, alpha_y=0, beta_x=0, beta_y=0):
    """
    compute statistical emittances. First normalize coordinates by using (263) then (130) from
    https://arxiv.org/pdf/2107.02614.pdf
    """

    x     = np.reshape(emits_dict["x"],     (n_macroparticles, n_turns))
    px    = np.reshape(emits_dict["px"],    (n_macroparticles, n_turns))
    y     = np.reshape(emits_dict["y"],     (n_macroparticles, n_turns))
    py    = np.reshape(emits_dict["py"],    (n_macroparticles, n_turns))
    z     = np.reshape(emits_dict["zeta"],  (n_macroparticles, n_turns))
    delta = np.reshape(emits_dict["delta"], (n_macroparticles, n_turns))

    x_norm     = x / np.sqrt(beta_x)
    y_norm     = y / np.sqrt(beta_y)
    px_norm    = alpha_x / beta_x * x + beta_x * px
    py_norm    = alpha_y / beta_y * y + beta_y * py

    emit_x = np.sqrt(np.mean(( x_norm -  np.mean(x_norm, axis=0))**2, axis=0) *\
                     np.mean((px_norm - np.mean(px_norm, axis=0))**2, axis=0) -\
                     np.mean(( x_norm -  np.mean(x_norm, axis=0)) *\
                             (px_norm - np.mean(px_norm, axis=0)), axis=0)**2)

    emit_y = np.sqrt(np.mean(( y_norm -  np.mean(y_norm, axis=0))**2, axis=0) *\
                     np.mean((py_norm - np.mean(py_norm, axis=0))**2, axis=0) -\
                     np.mean(( y_norm -  np.mean(y_norm, axis=0)) *\
                             (py_norm - np.mean(py_norm, axis=0)), axis=0)**2)

    emit_s = np.sqrt(np.mean((    z - np.mean(    z, axis=0))**2, axis=0) *\
                     np.mean((delta - np.mean(delta, axis=0))**2, axis=0) -\
                     np.mean((    z - np.mean(    z, axis=0)) *\
                             (delta - np.mean(delta, axis=0)), axis=0)**2)

    return emit_x, emit_y, emit_s

# get emittances and RMS beam sizes
coords_dict = monitor_coords.to_dict()["data"]
emits_dict  = monitor_emits.to_dict()["data"]

emit_x_arr, emit_y_arr, emit_s_arr = stat_emittance_from_monitor(emits_dict, n_macroparticles_b1, n_turns,
                            alpha_x=alpha_x_sext_right,
                            alpha_y=alpha_y_sext_right,
                            beta_x=beta_x_sext_right,
                            beta_y=beta_y_sext_right)

x_std_arr = np.std(np.reshape(coords_dict["x"],    (n_macroparticles_b1, n_turns)), axis=0)
y_std_arr = np.std(np.reshape(coords_dict["y"],    (n_macroparticles_b1, n_turns)), axis=0)
z_std_arr = np.std(np.reshape(coords_dict["zeta"], (n_macroparticles_b1, n_turns)), axis=0)

# bunch size evolution
turns_arr = np.arange(n_turns)
w = 500

print("sigma_z blowup [1]:", np.mean(z_std_arr[-w:])/sigma_z_tot)

assert np.abs(np.mean(z_std_arr[-w:])/sigma_z_tot -1) < 5e-2

fig, ax = plt.subplots(3,2, figsize=(20,10))

ax[0,0].plot(turns_arr, x_std_arr, label="{:.2f}".format(np.mean(x_std_arr[-w:])/sigma_x))
ax[1,0].plot(turns_arr, y_std_arr, label="{:.2f}".format(np.mean(y_std_arr[-w:])/sigma_y))
ax[2,0].plot(turns_arr, z_std_arr, label="{:.2f}".format(np.mean(z_std_arr[-w:])/sigma_z_tot))

ax[0,1].plot(turns_arr, emit_x_arr, label="{:.2f}".format(np.mean(emit_x_arr[-w:])/physemit_x))
ax[1,1].plot(turns_arr, emit_y_arr, label="{:.2f}".format(np.mean(emit_y_arr[-w:])/physemit_y))
ax[2,1].plot(turns_arr, emit_s_arr, label="{:.2f}".format(np.mean(emit_s_arr[-w:])/physemit_s_tot))

ax[0,0].axhline(sigma_x, c="r")
ax[1,0].axhline(sigma_y, c="r")
ax[2,0].axhline(sigma_z_tot, c="r")

ax[0,1].axhline(physemit_x, c="r")
ax[1,1].axhline(physemit_y, c="r")
ax[2,1].axhline(physemit_s_tot, c="r")

ax[0,0].legend()
ax[1,0].legend()
ax[2,0].legend()
ax[0,1].legend()
ax[1,1].legend()
ax[2,1].legend()

plt.show()

# beamstrahlung photon spectrum
fig, ax = plt.subplots(1,1,figsize=(12,8))
ax.hist(record.beamstrahlungtable.photon_energy/1e9, bins=np.logspace(np.log10(1e-14), np.log10(1e1), 100), histtype="step");
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("E [GeV]")
ax.set_ylabel("Count [1]")
plt.show()

