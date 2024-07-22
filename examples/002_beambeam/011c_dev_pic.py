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

bbpic_b1 = xf.BeamBeamPIC3D(phi=+200e-6, alpha=0)
bbpic_b2 = xf.BeamBeamPIC3D(phi=-200e-6, alpha=0)

bbpic_b1.change_ref_frame(particles_b1)
bbpic_b2.change_ref_frame(particles_b2)