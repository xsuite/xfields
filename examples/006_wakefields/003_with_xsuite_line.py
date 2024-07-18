import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.constants import c, e, m_p

from PyHEADTAIL.particles.slicing import UniformBinSlicer as PyHTUniformBinSlicer
from PyHEADTAIL.impedances.wakes import CircularResonator as PyHTCircularResonator

from PyHEADTAIL.impedances.wakes import WakeField as PyHTWakeField
from PyHEADTAIL.machines.synchrotron import Synchrotron as PyHTSynchrotron

import xfields as xf
import xtrack as xt
from xfields import WakeComponent

# Machine settings

n_turns_wake = 3
flatten = False

n_macroparticles = 100000  # per bunch
intensity = 2.3e11

alpha = 53.86**-2

p0 = 7000e9 * e / c
accQ_x = 0.31
accQ_y = 0.32
Q_s = 2.1e-3
chroma = 0

h_RF = 600
bunch_spacing_buckets = 5
n_bunches = 12
n_slices = 250
beta_x = 100
beta_y = 100

h_bunch = h_RF
circumference = 26658.883 / 35640 * h_RF

epsn_x = 2e-6
epsn_y = 2e-6
sigma_z = 0.09

machine = PyHTSynchrotron(
        optics_mode='smooth', n_segments=1, circumference=circumference,
        accQ_x=accQ_x, accQ_y=accQ_y,
        beta_x=beta_x, beta_y=beta_y, D_x=0, D_y=0,
        alpha_mom_compaction=alpha, longitudinal_mode='linear',
        h_RF=np.atleast_1d(h_RF), p0=p0,
        charge=e, mass=m_p, wrap_z=False, Q_s=Q_s)
transverse_map = machine.transverse_map.segment_maps[0]

# Filling scheme
filling_scheme = [i*bunch_spacing_buckets for i in range(n_bunches)]

# Initialise beam
bunches = machine.generate_6D_Gaussian_bunch(n_macroparticles, intensity,
                                             epsn_x, epsn_y, sigma_z=sigma_z,
                                             filling_scheme=filling_scheme,
                                             matched=False)

bucket_id_set = list(set(bunches.bucket_id))
bucket_id_set.sort()

bucket_length = machine.circumference / h_RF
z_all = -bunches.bucket_id * bucket_length + bunches.z

# apply a distortion to the bunch
amplitude = 1e-3
wavelength = 2
bunches.x = amplitude * np.sin(2 * np.pi * z_all / wavelength)
bunches.xp *= 0
for b_id in bucket_id_set:
    mask = bunches.bucket_id == b_id
    z_centroid = np.mean(bunches.z[mask])
    z_std = np.std(bunches.z[mask])
    # mask_tails = mask & (np.abs(bunches.z - z_centroid) > z_std)
    # bunches.x[mask_tails] = 0


# Initialise wakes
slicer = PyHTUniformBinSlicer(n_slices,
                              z_cuts=(-0.5*bucket_length, 0.5*bucket_length),
                              circumference=machine.circumference,
                              h_bunch=h_bunch)

# pipe radius [m]
b = 13.2e-3
# length of the pipe [m]
L = 100000.
# conductivity of the pipe 1/[Ohm m]
sigma = 1. / 7.88e-10

# wakes = CircularResistiveWall(b, L, sigma, b/c, beta_beam=machine.beta)
wakes = PyHTCircularResonator(R_shunt=135e6, frequency=1.97e9*0.6, Q=31000,
                              n_turns_wake=n_turns_wake)

R_s = wakes.R_shunt
Q = wakes.Q
f_r = wakes.frequency
omega_r = 2 * np.pi * f_r
alpha_t = omega_r / (2 * Q)
omega_bar = np.sqrt(omega_r**2 - alpha_t**2)
p0_SI = machine.p0

mpi_settings = False
# mpi_settings = 'memory_optimized'
# mpi_settings = 'linear_mpi_full_ring_fft'
wake_field = PyHTWakeField(slicer, wakes, mpi=mpi_settings)

plt.close('all')

fig0, (ax00, ax01) = plt.subplots(2, 1)
ax01.sharex(ax00)

x_at_wake_allbunches = []
xp_before_wake = []
xp_after_wake = []

n_turns = n_turns_wake
store_charge_per_mp = bunches.charge_per_mp
store_particlenumber_per_mp = bunches.particlenumber_per_mp

z_source_matrix_multiturn = np.zeros((n_bunches, n_slices, n_turns))
dipole_moment_matrix_multiturn = np.zeros((n_bunches, n_slices, n_turns))


particles = xt.Particles(
    mass0=xt.PROTON_MASS_EV,
    gamma0=bunches.gamma,
    x=bunches.x,
    px=bunches.xp,
    y=bunches.y,
    py=bunches.yp,
    zeta=z_all,
    delta=bunches.dp,
    weight=bunches.particlenumber_per_mp,
)


color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for i_turn in range(n_turns):

    bunches.clean_slices()

    x_before_wake = bunches.x.copy()
    xp_before_wake.append(bunches.xp.copy())

    wake_field.track(bunches)

    xp_after_wake.append(bunches.xp.copy())

    transverse_map.track(bunches)

# last turn on top
z_source_matrix_multiturn = z_source_matrix_multiturn[:, :, ::-1]
dipole_moment_matrix_multiturn = dipole_moment_matrix_multiturn[:, :, ::-1]

print(f'Circumference '
      f'occupancy {n_bunches * bunch_spacing_buckets/h_RF*100:.2f} %')


n_bunches_wake = 120  # Can be longer than filling scheme

wf = xf.ResonatorWake(
    r_shunt=wakes.R_shunt,
    q_factor=wakes.Q,
    frequency=wakes.frequency,
    source_moments=['num_particles', 'x'],
    kick='px',
    scale_kick=None,
    zeta_range=(-0.5*bucket_length, 0.5*bucket_length),
    num_slices=n_slices,
    bunch_spacing_zeta=bunch_spacing_buckets*bucket_length,
    num_slots=n_bunches_wake,
    num_turns=n_turns_wake,
    circumference=circumference,
    log_moments=['px'],
    _flatten=flatten
)

slicer_after_wf = xf.UniformBinSlicer(
    zeta_range=(-0.5*bucket_length, 0.5*bucket_length),
    num_slices=n_slices,
    num_bunches=n_bunches_wake,
    bunch_spacing_zeta=bunch_spacing_buckets*bucket_length,
    moments=['x', 'px'])

betatron_map = xt.LineSegmentMap(
    length=circumference, betx=beta_x, bety=beta_y,
    qx=accQ_x, qy=accQ_y,
    longitudinal_mode='frozen')

line = xt.Line(elements=[wf, slicer_after_wf, betatron_map],
               element_names=['wf', 'slicer_after_wf', 'betatron_map'])
particle_ref = xt.Particles(p0c=particles.p0c, mass0=particles.mass0,
                            q0=particles.q0)
line.build_tracker()

line.track(particles, num_turns=n_turns)

n_skip = 10

ax00.plot(wf.slicer.zeta_centers, wf.slicer.mean('x'), '.', color='b')
ax00.plot(z_all[::n_skip], x_before_wake[::n_skip], '.', color='r',
          markersize=1)

ax01.plot(wf.slicer.zeta_centers,
          slicer_after_wf.mean('px') - wf.slicer.mean('px'), '.', color='b')
ax01.plot(z_all[::n_skip],
          xp_after_wake[-1][::n_skip] - xp_before_wake[-1][::n_skip],
          '.', color='r', markersize=1)

plt.figure(200)
plt.plot(wf.z_wake.T, wf.G_aux.T * (-e**2 / (p0_SI * c)), alpha=0.5)

t0 = time.time()
line.track(particles)
t1 = time.time()
print(f"Xsuite turn time {(t1-t0)*1e3:.2f} ms")

t0 = time.time()
wake_field.track(bunches)
transverse_map.track(bunches)
bunches.get_slices(slicer,
                   statistics=['mean_x', 'mean_xp', 'mean_y', 'mean_yp'])
t1 = time.time()
print(f"PyHEADTAIL turn time {(t1-t0)*1e3} ms")

plt.show()
