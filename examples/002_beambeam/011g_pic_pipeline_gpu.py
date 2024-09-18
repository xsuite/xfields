import numpy as np

import time
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
alpha = np.deg2rad(30)
betx = 0.15
bety = 0.2
sigma_z = 0.1
nemitt_x = 1.5e-6
nemitt_y = 2e-6
bunch_intensity = 2e10
num_slices = 101
slice_mode = 'constant_charge'

lntwiss = xt.Line(elements=[xt.Marker()])
lntwiss.particle_ref = xt.Particles(p0c=p0c, mass0=mass0)
twip = lntwiss.twiss(betx=betx, bety=bety)

cov = twip.get_beam_covariance(nemitt_x=nemitt_x, nemitt_y=nemitt_y)
sigma_x = cov.sigma_x[0]
sigma_y = cov.sigma_y[0]

num_particles = 10_000_000
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
p_test = lntwiss.build_particles(x=1.2 * sigma_x, y=1.2 * sigma_y,
                                 px=50e-6, py=50e-6,
                zeta=np.linspace(-2 * sigma_z, 2 * sigma_z, n_test),
                weight=0)
particles_b1 = xt.Particles.merge([p_test, bunch_b1])
particles_b2 = particles_b1.copy()

context_gpu = xo.ContextCupy()
particles_b1.move(_context=context_gpu)
particles_b2.move(_context=context_gpu)

x_lim_grid = phi * 3 * sigma_z + 5 * sigma_x
y_lim_grid = phi * 3 * sigma_z + 5 * sigma_y

pics = []
for ii in range(2):
    pics.append(xf.BeamBeamPIC3D(
        _context=context_gpu,
        phi={0: phi, 1: -phi}[ii],
        alpha={0: alpha, 1: -alpha}[ii],
        x_range=(-x_lim_grid, x_lim_grid), dx=0.1*sigma_x,
        y_range=(-y_lim_grid, y_lim_grid), dy=0.1*sigma_y,
        z_range=(-2.5*sigma_z, 2.5*sigma_z), dz=0.2*sigma_z))

bbpic_b1 = pics[0]
bbpic_b2 = pics[1]

print("==> Pipeline configuration")

# Pipeline configuration (some rationalization needed here!)
pipeline_manager = xt.PipelineManager()
pipeline_manager.add_particles('p_b1', rank=0)
pipeline_manager.add_particles('p_b2', rank=0)
pipeline_manager.add_element('IP1') # needs to be the same for the two lines (I guess...)
bbpic_b1.name = 'IP1'
bbpic_b2.name = 'IP1'
particles_b1.init_pipeline('p_b1')
particles_b2.init_pipeline('p_b2')
bbpic_b1.partner_name = 'p_b2'
bbpic_b2.partner_name = 'p_b1'
bbpic_b1.pipeline_manager = pipeline_manager
bbpic_b2.pipeline_manager = pipeline_manager

bbpic_b1._partner = bbpic_b2
bbpic_b2._partner = bbpic_b1

line_b1 = xt.Line(elements=[bbpic_b1])
line_b2 = xt.Line(elements=[bbpic_b2])

print("==> Build GPU trackers")

line_b1.build_tracker(_context=context_gpu)
line_b2.build_tracker(_context=context_gpu)

time_start = time.time()

multitracker = xt.PipelineMultiTracker(
    branches=[xt.PipelineBranch(line=line_b1, particles=particles_b1),
              xt.PipelineBranch(line=line_b2, particles=particles_b2)],
    enable_debug_log=True, verbose=True)

print("==> Multitracker track")

multitracker.track(num_turns=1)

print(f"@@@ Multitracker on GPU took {time.time() - time_start} s")
print("==> GPU tracking complete. Compare against hirata.")

# Compare against hirata
z_centroids, z_cuts, num_part_per_slice = constant_charge_slicing_gaussian(
                                bunch_intensity, sigma_z, num_slices)
z_centroids_from_tail = z_centroids[::-1]
bbg = xf.BeamBeamBiGaussian3D(
    phi=phi,
    alpha=alpha,
    other_beam_q0=1.,
    slices_other_beam_num_particles=num_part_per_slice,
    slices_other_beam_zeta_center=z_centroids_from_tail,
    slices_other_beam_Sigma_11=cov.Sigma11[0],
    slices_other_beam_Sigma_12=cov.Sigma12[0],
    slices_other_beam_Sigma_22=cov.Sigma22[0],
    slices_other_beam_Sigma_33=cov.Sigma33[0],
    slices_other_beam_Sigma_34=cov.Sigma34[0],
    slices_other_beam_Sigma_44=cov.Sigma44[0],
)
p_bbg = p_test.copy()
bbg.track(p_bbg)

particles_b1.move(_context=xo.ContextCpu())

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(4)
plt.plot(p_bbg.zeta, p_bbg.px, label='hirata')
plt.plot(particles_b1.zeta[:n_test], particles_b1.px[:n_test], label='pic')
plt.xlabel(r'$\zeta$ [m]')
plt.ylabel(r'$\Delta p_x$')
plt.legend()

plt.figure(5)
plt.plot(p_bbg.zeta, p_bbg.py, label='hirata')
plt.plot(particles_b1.zeta[:n_test], particles_b1.py[:n_test], label='pic')
plt.xlabel(r'$\zeta$ [m]')
plt.ylabel(r'$\Delta p_y$')
plt.legend()

plt.figure(6)
plt.plot(p_bbg.zeta, p_bbg.ptau, label='hirata')
plt.plot(particles_b1.zeta[:n_test], particles_b1.ptau[:n_test], label='pic')
plt.xlabel(r'$\zeta$ [m]')
plt.ylabel(r'$\Delta p_\tau$')
plt.legend()

plt.savefig('gpu.png')
