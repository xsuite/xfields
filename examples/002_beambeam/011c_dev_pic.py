import numpy as np

import xfields as xf
import xtrack as xt
import xobjects as xo

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

# Choose the time step to simulate
z_grid_b1 = bbpic_b1.fieldmap_self.z_grid[::-1] # earlier time first
z_grid_b2 = bbpic_b2.fieldmap_self.z_grid[::-1] # earlier time first

i_step = 5
z_step_b1 = z_grid_b1[i_step]
z_step_b2 = z_grid_b2[i_step]
xo.assert_allclose(z_step_b1, z_step_b2, rtol=0, atol=1e-15)

# Propagate transverse coordinates to the position at the time step
for pp in [particles_b1, particles_b2]:
    mask_alive = pp.state > 0
    gamma_gamma0 = (
        particles_b1.ptau[mask_alive] * particles_b1.beta0[mask_alive] + 1)
    pp.x[mask_alive] += pp.px[mask_alive] / gamma_gamma0  * z_step_b1

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


# Compute kick from the other beam
for pp, bbpic, z_step_self, z_step_other in zip([particles_b1, particles_b2],
                                                [bbpic_b1, bbpic_b2],
                                                [z_step_b1, z_step_b2],
                                                [z_step_b2, z_step_b1]
                                               ):
    # Compute coordinates in the reference system of the other beam
    beta_over_beta_other = 1 # Could be generalized with
                             # (beta_particle/beta_slice_other)
                             # One could for example store the beta of the
                             # closed orbit
    mask_alive = pp.state > 0
    z_other = (-beta_over_beta_other * pp.zeta[mask_alive]
               + z_step_other
               + beta_over_beta_other * z_step_self)
    x_other = -pp.x[mask_alive]
    y_other = pp.y[mask_alive]

    # Get fields in the reference system of the other beam
    dphi_dx, dphi_dy, dphi_dz= bbpic_b1.fieldmap_other.get_values_at_points(
        x=x_other, y=y_other, z=z_other,
        return_rho=False,
        return_phi=False,
        return_dphi_dx=True,
        return_dphi_dy=True,
        return_dphi_dz=True,
    )

    # Transform fields to self reference frame (dphi_dy is unchanged)
    dphi_dx *= -1
    dphi_dz *= -1





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

plt.figure(3)
plt.pcolormesh(bbpic_b1.fieldmap_other.z_grid,
               bbpic_b1.fieldmap_other.x_grid,
               bbpic_b1.fieldmap_other.dphi_dx[:, 30, :])

plt.show()
