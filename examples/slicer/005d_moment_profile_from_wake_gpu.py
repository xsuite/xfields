import xfields as xf
import xtrack as xt
import xobjects as xo
import xwakes as xw

import numpy as np
import matplotlib.pyplot as plt

import xobjects as xo
context = xo.ContextCupy(device=3)

# base coordinates
zeta_coord = np.random.normal(loc=5e-3, scale=5e-2,
                              size=100_000)
bunch_spacing_zeta = 10
num_bunches = 3

# n_bunches bunches with n_macroparticles each with different coordinates
particles = xt.Particles(
    zeta=np.concatenate([zeta_coord*(bid + 1) - bunch_spacing_zeta*bid
                         for bid in range(num_bunches)]),
    _context=context,
)
# dummy filling scheme
filling_scheme = np.ones(num_bunches, dtype=int)
bunch_selection = np.arange(num_bunches, dtype=int)

# wake from which the compressed profile is taken
wf = xw.WakeResonator(
    r=3e8,
    q=1e7,
    f_r=1e3,
    kind='dipolar_x',
)

wf.configure_for_tracking(
    zeta_range=(-1, 1),
    num_slices=20,
    bunch_spacing_zeta=bunch_spacing_zeta,
    filling_scheme=filling_scheme,
    bunch_selection=bunch_selection,
    num_turns=1,
    circumference=bunch_spacing_zeta*len(filling_scheme),
    _context=context,
)

wf.track(particles)

long_profile = wf._wake_tracker.moments_data.get_moment_profile(
    moment_name='num_particles', i_turn=0)

long_profile_x = long_profile[0].get()
long_profile_y = long_profile[1].get()

plt.figure(321)
plt.plot(long_profile_x, long_profile_y/np.sum(long_profile_y))
plt.xlabel('zeta [m]')
plt.ylabel('longitudinal profile [a.u.]')
plt.show()
