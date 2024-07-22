import xfields as xf
import xtrack as xt

lntwiss = xt.Line(elements=[xt.Marker()])
lntwiss.particle_ref = xt.Particles(p0c=7e12, mass0=xt.PROTON_MASS_EV)
twip = lntwiss.twiss(betx=0.15, bety=0.10)
cov = twip.get_beam_covariance(nemitt_x=2e-6, nemitt_y=1e-6)

bbg = xf.BeamBeamBiGaussian3D(
    phi=0.01,
    alpha=0,
    other_beam_q0=1.,
    slices_other_beam_num_particles=None,
    slices_other_beam_Sigma_11=cov.Sigma_11[0],
    slices_other_beam_Sigma_12=cov.Sigma_12[0],
    slices_other_beam_Sigma_22=cov.Sigma_22[0],
    slices_other_beam_Sigma_33=cov.Sigma_33[0],
    slices_other_beam_Sigma_34=cov.Sigma_34[0],
    slices_other_beam_Sigma_44=cov.Sigma_44[0],
)