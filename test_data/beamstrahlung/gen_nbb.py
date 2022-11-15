import xobjects as xo
import numpy as np
import xpart as xp
import xfields as xf
from matplotlib import pyplot as plt
import json

# generate n_bb in the tests and examples
context = xo.ContextCpu(omp_num_threads=0)

bunch_intensity     = 2.3e11  # [1]
energy              = 182.5  # [GeV]
p0c                 = 182.5e9  # [eV]
mass0               = .511e6  # [eV]
physemit_x          = 1.46e-09  # [m]
physemit_y          = 2.9e-12  # [m]
beta_x              = 1  # [m]
beta_y              = .0016  # [m]
sigma_x             = np.sqrt(physemit_x*beta_x)  # [m]
sigma_px            = np.sqrt(physemit_x/beta_x)  # [m]
sigma_y             = np.sqrt(physemit_y*beta_y)  # [m]
sigma_py            = np.sqrt(physemit_y/beta_y)  # [m]
sigma_z_tot         = .00254  # [m] sr+bs
sigma_delta_tot     = .00192  # [m]
n_macroparticles_b2 = int(1e8)
n_slices = 100

particles_b2 = xp.Particles(
            _context = context, 
            q0        = 1,
            p0c       = p0c,
            mass0     = mass0,
            x         = sigma_x        *np.random.randn(n_macroparticles_b2),
            y         = sigma_y        *np.random.randn(n_macroparticles_b2),
            zeta      = sigma_z_tot    *np.random.randn(n_macroparticles_b2),
            px        = sigma_px       *np.random.randn(n_macroparticles_b2),
            py        = sigma_py       *np.random.randn(n_macroparticles_b2),
            delta     = sigma_delta_tot*np.random.randn(n_macroparticles_b2),
            )

bin_edges = sigma_z_tot*np.linspace(-3.0,3.0,n_slices+1)
slicer = xf.TempSlicer(bin_edges=bin_edges)
slice_idx = slicer.get_slice_indices(particles_b2)
counts = plt.hist(slice_idx, bins=n_slices+2)
n_bb = counts[0][1:-1] / n_macroparticles_b2 * bunch_intensity

with open("./gen_nbb.json", "w") as f:
    json.dump({"n_bb": n_bb}, f, cls=xo.JEncoder)
