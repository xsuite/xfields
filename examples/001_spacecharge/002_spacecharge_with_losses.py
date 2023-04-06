import numpy as np
from scipy.constants import e as qe

import xfields as xf
import xpart as xp

sc = xf.SpaceCharge3D(
    x_range=(-0.01, 0.01),
    y_range=(-0.01, 0.01),
    z_range=(-1, 1),
    nx=128,
    ny=128,
    nz=128,
    solver='FFTSolver2p5D'
)

particles = xp.Particles(
    x=[0.002, 0.003, 0.004],
    y=[0.002, 0.003, 0.004],
    zeta=[-0.5, 0., 0.05],
    mass0=xp.PROTON_MASS_EV,
    p0c=1e9,
    weight=[1., 1., 1.])

particles.state = np.array([1, 1, -20])

sc.track(particles)

dx = sc.fieldmap.dx
dy = sc.fieldmap.dy
dz = sc.fieldmap.dz

charges = np.sum(sc.fieldmap.rho[:]) * dx * dy * dz / qe