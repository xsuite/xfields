import xfields as xf
import xtrack as xt

constant_charge_slicing_gaussian = \
    xf.config_tools.beambeam_config_tools.config_tools.constant_charge_slicing_gaussian

lntwiss = xt.Line(elements=[xt.Marker()])
lntwiss.particle_ref = xt.Particles(p0c=7e12, mass0=xt.PROTON_MASS_EV)
twip = lntwiss.twiss(betx=0.15, bety=0.10)
cov = twip.get_beam_covariance(nemitt_x=2e-6, nemitt_y=1e-6)


sigma_z = 0.08
bunch_intensity = 2e11
num_slices = 101

sigma_z_limi = sigma_z / 2
z_centroids, z_cuts, num_part_per_slice = constant_charge_slicing_gaussian(
                                bunch_intensity, sigma_z_limi, num_slices)
z_centroids_from_tail = z_centroids[::-1]

bbg = xf.BeamBeamBiGaussian3D(
    phi=0.01,
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