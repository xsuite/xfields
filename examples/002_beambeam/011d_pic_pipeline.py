import numpy as np

import xfields as xf
import xtrack as xt
import xobjects as xo
# xo.context_default._kernels.clear()

from scipy.constants import e as qe
from scipy.constants import c as clight

constant_charge_slicing_gaussian = \
    xf.config_tools.beambeam_config_tools.config_tools.constant_charge_slicing_gaussian

# LHC-like parameter
mass0 = xt.PROTON_MASS_EV
p0c = 7e12
phi = 200e-6
betx = 0.15
bety = 0.2
sigma_z = 0.1
nemitt_x = 2e-6
nemitt_y = 1e-6
bunch_intensity = 2e10
num_slices = 101
slice_mode = 'constant_charge'

lntwiss = xt.Line(elements=[xt.Marker()])
lntwiss.particle_ref = xt.Particles(p0c=p0c, mass0=mass0)
twip = lntwiss.twiss(betx=betx, bety=bety)

cov = twip.get_beam_covariance(nemitt_x=nemitt_x, nemitt_y=nemitt_y)
sigma_x = cov.sigma_x[0]
sigma_y = cov.sigma_y[0]

num_particles = 1_000_000
bunch_b1 = lntwiss.build_particles(
    num_particles=num_particles,
    nemitt_x=nemitt_x, nemitt_y=nemitt_y,
    zeta=np.random.normal(size=num_particles) * sigma_z,
    x_norm=np.random.normal(size=num_particles),
    px_norm=np.random.normal(size=num_particles),
    y_norm=np.random.normal(size=num_particles),
    py_norm=np.random.normal(size=num_particles),
    W_matrix=twip.W_matrix[0],
    particle_on_co=twip.particle_on_co,
    weight = bunch_intensity / num_particles
)
n_test = 1000
p_test = lntwiss.build_particles(x=1.2 * sigma_x, y=1.3 * sigma_y,
                zeta=np.linspace(-2 * sigma_z, 2 * sigma_z, n_test),
                weight=0)
particles_b1 = xt.Particles.merge([p_test, bunch_b1])
particles_b2 = particles_b1.copy()

x_lim_grid = phi * 3 * sigma_z + 5 * sigma_x

pics = []
for ii in range(2):
    pics.append(xf.BeamBeamPIC3D(phi=phi, alpha=0,
        x_range=(-x_lim_grid, x_lim_grid), dx=0.1*sigma_x,
        y_range=(-7*sigma_y, 7*sigma_y), dy=0.1*sigma_y,
        z_range=(-2.5*sigma_z, 2.5*sigma_z), dz=0.2*sigma_z))

bbpic_b1 = pics[0]
bbpic_b2 = pics[1]

line_b1 = xt.Line(elements=[bbpic_b1])
line_b2 = xt.Line(elements=[bbpic_b2])

line_b1.build_tracker()
line_b2.build_tracker()

multitracker = xt.PipelineMultiTracker(
    branches=[xt.PipelineBranch(line=line_b1, particles=particles_b1),
              xt.PipelineBranch(line=line_b2, particles=particles_b2)],
    enable_debug_log=True, verbose=True)

multitracker.track(num_turns=1)