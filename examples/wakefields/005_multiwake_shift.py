import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e, m_p

from PyHEADTAIL.particles.slicing import UniformBinSlicer as PyHTUniformBinSlicer

from PyHEADTAIL.machines.synchrotron import Synchrotron as PyHTSynchrotron

from wakefield import Wakefield, MultiWakefield, TempResonatorFunction

import xtrack as xt

# Machine settings
n_turns_wake = 1
flatten = False

alpha = 53.86**-2

p0 = 7000e9 * e / c
accQ_x = 60.31
accQ_y = 60.32
Q_s = 1.919e-3
chroma = 1

h_RF = 35640
bunch_spacing_buckets = 10
n_bunches = 1
n_slices = 100
n_macroparticles = 1*(n_slices - 1)  # per bunch
intensity = n_slices
beta_x = 70
beta_y = 70

h_bunch = h_RF
circumference = 26658.883 / 35640 * h_RF

machine = PyHTSynchrotron(
        optics_mode='smooth', n_segments=1, circumference=circumference,
        accQ_x=accQ_x, accQ_y=accQ_y, beta_x=beta_x, beta_y=beta_y,
        D_x=0, D_y=0, alpha_mom_compaction=alpha, longitudinal_mode='linear',
        h_RF=np.atleast_1d(h_RF), p0=p0,
        charge=e, mass=m_p, wrap_z=False, Q_s=Q_s,
        Qp_x=chroma, Qp_y=chroma
)
transverse_map = machine.transverse_map.segment_maps[0]

# Filling scheme
filling_scheme = [i*bunch_spacing_buckets for i in range(n_bunches)]
bucket_length = machine.circumference / h_RF

epsn_x = 2e-6
epsn_y = 2e-6
sigma_z = 0.07494811

# Initialise beam
bunches = machine.generate_6D_Gaussian_bunch_matched(
                                             n_macroparticles, intensity,
                                             epsn_x, epsn_y, sigma_z=sigma_z,
                                             filling_scheme=filling_scheme)

bucket_id_set = list(set(bunches.bucket_id))
bucket_id_set.sort()

zeta_range = (-0.5*bucket_length, 0.5*bucket_length)

slicer = PyHTUniformBinSlicer(n_slices, z_cuts=zeta_range,
                              circumference=machine.circumference,
                              h_bunch=h_bunch)

z_all = -bunches.bucket_id * bucket_length + bunches.z


# Initialise wakes
conductivity = 1. / 1.7e-8
pipe_radius = 10e-3
resistive_wall_length = circumference
dt_min = 0
Yokoya_X1 = 1
Yokoya_Y1 = 1
Yokoya_X2 = 0
Yokoya_Y2 = 0

gamma = bunches.gamma
momentumCompaction = alpha
eta = momentumCompaction - 1.0 / gamma ** 2
averageRadius = circumference / (2 * np.pi)
sigma_delta = Q_s * sigma_z / (averageRadius * eta)
beta_s = sigma_z / sigma_delta

n_bunches_wake = 120  # Can be longer than filling scheme

beta = np.sqrt(1-1/gamma**2)

wake_func = TempResonatorFunction(R_shunt=1e8, frequency=1e7, Q=1e3,
                                  beta=beta)

# apply a distortion to the bunch
amplitude = 1e-3
wavelength = np.pi/2*sigma_z*6
bunches.x *= 0
bunches.y *= 0
bunches.xp *= 0
bunches.yp *= 0

dz = (zeta_range[1] - zeta_range[0])/(n_slices-1)

bunches.z = np.concatenate([np.linspace(zeta_range[0]+dz/2, zeta_range[1]-dz/2,
                           n_macroparticles) + i*bunch_spacing_buckets*bucket_length for i in range(n_bunches)])

i_disp = -10

bunches.x[i_disp] += 1
bunches.y[i_disp] += 1

flag_plot = True

if flag_plot:
    plt.figure(74)
    plt.plot(bunches.z, bunches.x, 'x')
    plt.show()


particles = xt.Particles(
    mass0=xt.PROTON_MASS_EV,
    gamma0=bunches.gamma,
    x=bunches.x.copy(),
    px=bunches.xp.copy(),
    y=bunches.y.copy(),
    py=bunches.yp.copy(),
    zeta=bunches.z.copy(),
    delta=bunches.dp.copy(),
    weight=bunches.particlenumber_per_mp,
)

particle_ref = xt.Particles(p0c=particles.p0c, mass0=particles.mass0,
                            q0=particles.q0)

wfx = Wakefield(
    source_moments=['num_particles', 'x'],
    kick='px',
    scale_kick=None,
    function=wake_func
)

wfy = Wakefield(
    source_moments=['num_particles', 'y'],
    kick='py',
    scale_kick=None,
    function=wake_func
)

zeta_range_xf = zeta_range

wf = MultiWakefield(
    wakefields=[wfx, wfy],
    zeta_range=zeta_range_xf,
    num_slices=n_slices,  # per bunch
    bunch_spacing_zeta=bunch_spacing_buckets*bucket_length,
    num_bunches=n_bunches_wake,
    num_turns=n_turns_wake,
    circumference=circumference,
    log_moments=['px'],
    _flatten=flatten
)


line = xt.Line(elements=[wf],
               element_names=['wf'])

line.build_tracker()

n_skip = 1

line.track(particles, num_turns=1)

scale = -particles.q0**2 * e**2 / (particles.p0c * e)

if flag_plot:
    plt.figure(75)
    plt.plot(particles.zeta, particles.px, '.')
    plt.plot(particles.zeta, wfx.function(-particles.zeta[i_disp] +
                                          particles.zeta)*scale, 'x')
    plt.show()

    plt.figure(76)
    plt.plot(particles.zeta, particles.py, '.')
    plt.plot(particles.zeta, wfy.function(-particles.zeta[i_disp] +
                                          particles.zeta)*scale, 'x')
    plt.show()

mask = particles.zeta < particles.zeta[i_disp]
assert np.allclose(particles.px[mask],
                   (wfx.function(-particles.zeta[i_disp] +
                                particles.zeta) * scale)[mask],
                   rtol=1e-4, atol=0)
assert np.allclose(particles.py[mask],
                   (wfy.function(-particles.zeta[i_disp] +
                                particles.zeta) * scale)[mask],
                   rtol=1e-4, atol=0)