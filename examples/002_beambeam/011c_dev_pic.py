import numpy as np

import xfields as xf
import xtrack as xt

constant_charge_slicing_gaussian = \
    xf.config_tools.beambeam_config_tools.config_tools.constant_charge_slicing_gaussian

# LHC-like parameter
mass0 = xt.PROTON_MASS_EV
p0c = 7e12
phi = 200e-6
betx = 0.15
bety = 0.1
sigma_z = 0.1
nemitt_x = 2e-6
nemitt_y = 1e-6
bunch_intensity = 2e11
num_slices = 11
slice_mode = 'constant_charge'

lntwiss = xt.Line(elements=[xt.Marker()])
lntwiss.particle_ref = xt.Particles(p0c=p0c, mass0=mass0)
twip = lntwiss.twiss(betx=betx, bety=bety)

cov = twip.get_beam_covariance(nemitt_x=nemitt_x, nemitt_y=nemitt_y)
sigma_x = cov.sigma_x[0]
sigma_y = cov.sigma_y[0]

num_particles = 1_000_000
particles_b1 = lntwiss.build_particles(
    num_particles=num_particles,
    nemitt_x=nemitt_x, nemitt_y=nemitt_y,
    zeta=np.random.normal(size=num_particles) * sigma_z,
    x_norm=np.random.normal(size=num_particles),
    px_norm=np.random.normal(size=num_particles),
    y_norm=np.random.normal(size=num_particles),
    py_norm=np.random.normal(size=num_particles),
    W_matrix=twip.W_matrix[0],
    particle_on_co=twip.particle_on_co,
)
particles_b2 = particles_b1.copy()

x_lim_grid = phi * 3 * sigma_z + 3 * sigma_x

pics = []
for ii in range(2):
    pics.append(xf.BeamBeamPIC3D(phi=+200e-6, alpha=0,
        x_range=(-x_lim_grid, x_lim_grid), dx=0.05*sigma_x,
        y_range=(-5*sigma_y, 5*sigma_y), dy=0.05*sigma_y,
        z_range=(-3*sigma_z, 3*sigma_z), dz=0.05*sigma_z))

bbpic_b1 = pics[0]
bbpic_b2 = pics[1]

# Move particles to computation reference frame
bbpic_b1.change_ref_frame(particles_b1)
bbpic_b2.change_ref_frame(particles_b2)

# Compute charge density
bbpic_b1.fieldmap_self.update_from_particles(particles=particles_b1,
                                             update_phi=False)
bbpic_b2.fieldmap_self.update_from_particles(particles=particles_b2,
                                             update_phi=False)

# Exchange charge densities
bbpic_b1.fieldmap_other.update_rho(bbpic_b2.fieldmap_self.rho, reset=True)
bbpic_b2.fieldmap_other.update_rho(bbpic_b1.fieldmap_self.rho, reset=True)

# Compute potential
bbpic_b1.fieldmap_other.update_phi_from_rho()
bbpic_b2.fieldmap_other.update_phi_from_rho()


import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.pcolormesh(bbpic_b1.fieldmap_self.z_grid,
               bbpic_b1.fieldmap_self.x_grid,
               bbpic_b1.fieldmap_self.rho.sum(axis=1))
plt.colorbar()
plt.xlabel('z [m]')
plt.ylabel('x [m]')

plt.figure(2)
plt.pcolormesh(bbpic_b2.fieldmap_self.z_grid,
               bbpic_b2.fieldmap_self.x_grid,
               bbpic_b2.fieldmap_self.rho.sum(axis=1))
plt.colorbar()
plt.xlabel('z [m]')
plt.ylabel('x [m]')

plt.show()
