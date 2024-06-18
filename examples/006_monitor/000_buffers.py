import xfields as xf
import xtrack as xt
import xpart as xp
import xobjects as xo

import numpy as np
from scipy.constants import e, c, physical_constants

context = xo.ContextCpu(omp_num_threads=0)

n_macroparticles = int(1e5)
num_slices = n_macroparticles
zeta_range = (-1, 1)
dz = (zeta_range[1] - zeta_range[0])/num_slices

E0 = physical_constants['proton mass energy equivalent in MeV'][0]*1e6
E = 7000e9
p0 = E * e / c
gamma = E/E0
beta = np.sqrt(1-1/gamma**2)

intensity = 1e11
epsn_x = 2e-6
epsn_y = 2e-6
taub = 0.9e-9
sigma_z = taub*beta*c/4

zeta = np.linspace(zeta_range[0] + dz/2, zeta_range[1]-dz/2,
                   n_macroparticles)

monitor = xf.CollectiveMonitor(
    file_backend=None,
    monitor_bunches=True,
    monitor_slices=True,
    monitor_particles=False,
    n_steps=10,
    buffer_size=10,
    beta_gamma=beta*gamma,
    slicer_moments='all',
    zeta_range=zeta_range,  # These are [a, b] in the paper
    num_slices=num_slices,  # Per bunch, this is N_1 in the paper
    bunch_spacing_zeta=10,  # This is P in the paper
    num_slots=1
)

accQ_x = 60.275
accQ_y = 60.295
circumference = 26658.883
averageRadius = circumference / (2 * np.pi)
beta_x = averageRadius/accQ_x
beta_y = averageRadius/accQ_y
chroma = 10

alpha = 53.86**-2
h_RF = 35640
momentumCompaction = alpha
eta = momentumCompaction - 1.0 / gamma ** 2
V = 5e6
Q_s = np.sqrt((e*V*h_RF*eta)/(2*np.pi*beta*c*p0))
sigma_delta = Q_s * sigma_z / (averageRadius * eta)
beta_s = sigma_z / sigma_delta

betatron_map = xt.LineSegmentMap(
    length=circumference, betx=beta_x, bety=beta_y,
    qx=accQ_x, qy=accQ_y,
    longitudinal_mode='linear_fixed_qs',
    dqx=chroma, dqy=chroma,
    qs=Q_s, bets=beta_s
)

line = xt.Line(elements=[monitor, betatron_map],
               element_names=['monitor', 'betatron_map'])

line.particle_ref = xt.Particles(p0c=E)
line.build_tracker(_context=context)

particles = xp.generate_matched_gaussian_bunch(
         num_particles=n_macroparticles, total_intensity_particles=intensity,
         nemitt_x=epsn_x, nemitt_y=epsn_y, sigma_z=sigma_z,
         line=line, _context=context
)

line.track(particles)

print('a')


