import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e, m_p

import xtrack as xt
import xfields as xf

n_slots = 1
filling_scheme = np.ones(n_slots,dtype=int)
bunch_spacing = 25E-9*c
sigma_zeta = bunch_spacing/20

zeta_0 = np.linspace(-sigma_zeta,sigma_zeta,1000)

ioffset = np.argmin(np.abs(zeta_0))
wake_offset = zeta_0[ioffset]


print('Initialising particles')
particles = xt.Particles(p0c=7E12,zeta=zeta_0)
particles.x[ioffset] += 1.0

print('Initialising wake')
n_slices = 1000
n_turns_wake = 1
circumference = n_slots * bunch_spacing
wake_table_name = xf.general._pkg_root.joinpath('../test_data/HLLHC_wake.dat')
wake_file_columns = ['time', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y','dipole_xy','dipole_yx']
components = ['dipole_x']
wf = xf.MultiWakefield.from_table(wake_table_name,wake_file_columns, use_components = components,
    zeta_range=(-1.1*sigma_zeta,1.1*sigma_zeta),
    num_slices=n_slices,  # per bunch
    bunch_spacing_zeta=bunch_spacing,
    filling_scheme=filling_scheme,
    bunch_numbers = [0],
    num_turns=n_turns_wake,
    circumference=circumference,
)
wf.track(particles)
plt.show()


