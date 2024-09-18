import numpy as np

import xfields as xf
import xtrack as xt

constant_charge_slicing_gaussian = \
    xf.config_tools.beambeam_config_tools.config_tools.constant_charge_slicing_gaussian

# LHC-like parameter
mass0 = xt.PROTON_MASS_EV
p0c = 7e12
phi = 0#200e-6
betx = 0.15
bety = 0.2
sigma_z = 0.001
nemitt_x = 2e-6
nemitt_y = 1e-6
bunch_intensity = 2e10
num_slices = 11
slice_mode = 'constant_charge'

# # FCC-like parameters
# mass0 = xt.ELECTRON_MASS_EV
# p0c = 50e9
# phi = 15e-3
# betx = 0.15
# bety = 0.001
# sigma_z = 0.01
# nemitt_x = 270e-12
# nemitt_y = 1e-12
# bunch_intensity = 2e11
# num_slices = 100000 # needed to have motion of the particle within one slice
#                     # smaller that 1 sigma at the ip
# slice_mode = 'uniform'

lntwiss = xt.Line(elements=[xt.Marker()])
lntwiss.particle_ref = xt.Particles(p0c=p0c, mass0=mass0)
twip = lntwiss.twiss(betx=betx, bety=bety)
cov = twip.get_beam_covariance(nemitt_x=nemitt_x, nemitt_y=nemitt_y)

sigma_x = cov.sigma_x[0]
sigma_y = cov.sigma_y[0]


if slice_mode == 'constant_charge':
    sigma_z_limi = sigma_z / 2
    z_centroids, z_cuts, num_part_per_slice = constant_charge_slicing_gaussian(
                                    bunch_intensity, sigma_z_limi, num_slices)
    z_centroids_from_tail = z_centroids[::-1]
elif slice_mode == 'uniform':
    z_cuts = np.linspace(-3*sigma_z, 3*sigma_z, num_slices+1)
    z_centroids = 0.5 * (z_cuts[1:] + z_cuts[:-1])
    z_centroids_from_tail = z_centroids[::-1]
    num_part_per_slice = 0 * z_centroids
    from scipy.special import erf
    for ii in range(num_slices):
        num_part_per_slice[ii] = bunch_intensity / 2 * (erf(z_cuts[ii+1]/sigma_z) - erf(z_cuts[ii]/sigma_z))
else:
    raise ValueError


bbg = xf.BeamBeamBiGaussian3D(
    phi=phi,
    alpha=0,
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

particles = lntwiss.build_particles(x=1.2 * sigma_x, y=1.3 * sigma_y,
                zeta=np.linspace(-2 * sigma_z, 2 * sigma_z, 10000))

p1 = particles.copy()
bbg._track_non_collective(particles)
bbg.track(p1)

p_test = lntwiss.build_particles(x=1.2 * sigma_x, y=1.3 * sigma_y,
                zeta=[0.00553, 0.00557])
bbg._track_non_collective(p_test, i_slices=[1])

import matplotlib.pyplot as plt

plt.close('all')
plt.figure(1)
plt.plot(particles.zeta, particles.px)
plt.plot(particles.zeta, p1.px)

plt.grid()
# for zz in z_centroids:
#     plt.axvline(zz, color='r', alpha=0.1)

plt.show()