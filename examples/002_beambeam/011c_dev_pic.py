import numpy as np

import xfields as xf
import xtrack as xt
import xobjects as xo
# xo.context_default._kernels.clear()

from scipy.constants import e as qe
from scipy.constants import c as clight

constant_charge_slicing_gaussian = \
    xf.config_tools.beambeam_config_tools.config_tools.constant_charge_slicing_gaussian

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("plane", type=str)
args = parser.parse_args()

plane = args.plane
assert plane in ['x', 'y', '45']

# LHC-like parameter
mass0 = xt.PROTON_MASS_EV
p0c = 7e12
phi = 200e-6
alpha = {'x': 0.0, 'y': np.pi/2, '45': np.pi/4}[plane]
betx = 0.15
bety = 0.15 #2
sigma_z = 0.1
nemitt_x = 2e-6
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
p_test = lntwiss.build_particles(x=1.2 * sigma_x, y=1.2 * sigma_y,
                                 px=50e-6, py=50e-6,
                zeta=np.linspace(-2 * sigma_z, 2 * sigma_z, n_test),
                weight=0)
particles_b1 = xt.Particles.merge([p_test, bunch_b1])
particles_b2 = particles_b1.copy()

particles_b1_init = particles_b1.copy()

particles_b1.name = 'p_b1'
particles_b2.name = 'p_b2'

x_lim_grid = phi * 3 * sigma_z + 5 * sigma_x
y_lim_grid = phi * 3 * sigma_z + 5 * sigma_y

pics = []
for ii in range(2):
    pics.append(xf.BeamBeamPIC3D(
        phi={0: phi, 1: -phi}[ii],
        alpha={0: alpha, 1: -alpha}[ii],
        x_range=(-x_lim_grid, x_lim_grid), dx=0.1*sigma_x,
        y_range=(-y_lim_grid, y_lim_grid), dy=0.1*sigma_y,
        z_range=(-2.5*sigma_z, 2.5*sigma_z), dz=0.2*sigma_z))

bbpic_b1 = pics[0]
bbpic_b2 = pics[1]

# Move particles to computation reference frame
bbpic_b1.change_ref_frame_bbpic(particles_b1)
bbpic_b2.change_ref_frame_bbpic(particles_b2)

z_grid_b1 = bbpic_b1.fieldmap_self.z_grid[::-1].copy() # earlier time first
z_grid_b2 = bbpic_b2.fieldmap_self.z_grid[::-1].copy() # earlier time first
assert len(z_grid_b1) == len(z_grid_b2)

progress = xt.progress_indicator.progress
dphi_dx_test_log = []
dphi_dz_test_log = []
dpz_test_tot = 0
for i_step in progress(range(len(z_grid_b1))):

    z_step_b1 = z_grid_b1[i_step]
    z_step_b2 = z_grid_b2[i_step]
    xo.assert_allclose(z_step_b1, z_step_b2, rtol=0, atol=1e-15) # we consider only this case
                                                                # that is tested, although
                                                                # the implementation should be
                                                                # more general

    # Propagate transverse coordinates to the position at the time step
    for pp, z_step_other in zip([particles_b1, particles_b2],
                                [z_step_b2, z_step_b1]):
        mask_alive = pp.state > 0
        gamma_gamma0 = (
            particles_b1.ptau[mask_alive] * particles_b1.beta0[mask_alive] + 1)
        pp.x[mask_alive] += (pp.px[mask_alive] / gamma_gamma0
                            * (pp.zeta[mask_alive] - z_step_other))
        pp.y[mask_alive] += (pp.py[mask_alive] / gamma_gamma0
                            * (pp.zeta[mask_alive] - z_step_other))

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

    # Compute and apply kick from the other beam
    mask_alive = pp.state > 0
    beta0_b1 = particles_b1.beta0[mask_alive][0]
    beta0_b2 = particles_b2.beta0[mask_alive][0]

    for pp, bbpic, z_step_self, z_step_other, beta0_other in zip(
                                                    [particles_b1, particles_b2],
                                                    [bbpic_b1, bbpic_b2],
                                                    [z_step_b1, z_step_b2],
                                                    [z_step_b2, z_step_b1],
                                                    [beta0_b2, beta0_b1]
                                                ):
        mask_alive = pp.state > 0
        # Assuming beta=1 for all particles for now
        z_other = (-pp.zeta[mask_alive] + z_step_other + z_step_self)
        x_other = -pp.x[mask_alive]
        y_other = pp.y[mask_alive]

        # Get fields in the reference system of the other beam
        dphi_dx, dphi_dy, dphi_dz= bbpic.fieldmap_other.get_values_at_points(
            x=x_other, y=y_other, z=z_other,
            return_rho=False,
            return_phi=False,
            return_dphi_dx=True,
            return_dphi_dy=True,
            return_dphi_dz=True,
        )
        dz = bbpic.fieldmap_other.dz

        # Transform fields to self reference frame (dphi_dy is unchanged)
        dphi_dx *= -1
        dphi_dz *= -1

        if pp is particles_b1:
            dphi_dx_test_log.append(dphi_dx[:n_test])
            dphi_dz_test_log.append(dphi_dz[:n_test])

        # Compute factor for the kick
        charge_mass_ratio = (pp.chi[mask_alive] * qe * pp.q0
                            / (pp.mass0 * qe /(clight * clight)))
        pp_beta0 = pp.beta0[mask_alive]
        factor = -(charge_mass_ratio
                / (pp.gamma0[mask_alive] * pp_beta0 * pp_beta0 * clight * clight)
                * (1 + beta0_other * pp_beta0))

        # Compute kick
        dpx = factor * dphi_dx * dz
        dpy = factor * dphi_dy * dz

        # Effect of the particle angle as in Hirata
        dpz = 0.5 *(
            dpx * (pp.px[mask_alive] + 0.5 * dpx)
          + dpy * (pp.py[mask_alive] + 0.5 * dpy))

        dpz_test_tot += dpz[:n_test]

        # Apply kick
        pp.px[mask_alive] += dpx
        pp.py[mask_alive] += dpy
        pp.delta[mask_alive] += dpz

    # Propagate transverse coordinates back to IP
    for pp, z_step_other in zip([particles_b1, particles_b2],
                                [z_step_b2, z_step_b1]):
        mask_alive = pp.state > 0
        gamma_gamma0 = (
            particles_b1.ptau[mask_alive] * particles_b1.beta0[mask_alive] + 1)
        pp.x[mask_alive] -= (pp.px[mask_alive] / gamma_gamma0
                            * (pp.zeta[mask_alive] - z_step_other))
        pp.y[mask_alive] -= (pp.py[mask_alive] / gamma_gamma0
                            * (pp.zeta[mask_alive] - z_step_other))

dphi_dx_test_log = np.array(dphi_dx_test_log)
dphi_dz_test_log = np.array(dphi_dz_test_log)

# Back to lab frame
bbpic_b1.change_back_ref_frame_and_subtract_dipolar_bbpic(particles_b1)
bbpic_b2.change_back_ref_frame_and_subtract_dipolar_bbpic(particles_b2)

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
p_bbg2 = p_test.copy()

bbg.track(p_bbg)
bbg._track_non_collective(p_bbg2)

bb2d = xf.BeamBeamBiGaussian2D(
    other_beam_q0=1,
    other_beam_beta0=1,
    other_beam_num_particles=bunch_intensity,
    other_beam_Sigma_11=sigma_x**2,
    other_beam_Sigma_33=sigma_y**2)
p_bb2d = p_test.copy()
bb2d.track(p_bb2d)

# Build a test python implementation (frankenstein)
p_fstn = p_test.copy()

import ducktrack as dtk
mathlib = dtk.mathlibs.MathlibDefault
gf = dtk.be_beamfields.gaussian_fields

ex, ey = gf.get_Ex_Ey_Gx_Gy_gauss(p_fstn.x, p_fstn.y,
                                 sigma_x=sigma_x, sigma_y=sigma_y,
                                 min_sigma_diff=1e-12, skip_Gs=True,
                                 mathlib=mathlib)

additional_factor = bunch_intensity * qe / (beta0_other + pp_beta0)[:n_test]
fm_factor = additional_factor * factor[:n_test]

dpx_fm = fm_factor * -ex
dpy_fm = fm_factor * -ey

dx = bbpic_b1.fieldmap_other.dx
dy = bbpic_b1.fieldmap_other.dy
z = bbpic_b1.fieldmap_other.z_grid
lam = bbpic_b1.fieldmap_other.rho.sum(axis=0).sum(axis=0) * dx * dy

ex_vs_z = lam * ex[0]

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
plt.colorbar()

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
plt.plot(p_bbg2.zeta, p_bbg2.ptau, label='hirata 2')
plt.plot(particles_b1.zeta[:n_test], particles_b1.ptau[:n_test], label='pic')
plt.xlabel(r'$\zeta$ [m]')
plt.ylabel(r'$\Delta p_\tau$')
plt.legend()

plt.show()
