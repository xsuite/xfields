import xfields as xf
import xtrack as xt
import xobjects as xo
import xwakes as xw
from xfields.slicers.compressed_profile import CompressedProfile

import numpy as np
import matplotlib.pyplot as plt


# base coordinates
zeta_coord = np.random.normal(loc=5e-3, scale=5e-2,
                              size=100_000)
bunch_spacing_zeta = 10
num_bunches = 3

# n_bunches bunches with n_macroparticles each with different coordinates
particles = xt.Particles(
    zeta=np.concatenate([zeta_coord*(bid + 1) - bunch_spacing_zeta*bid
                         for bid in range(num_bunches)]),
)

# dummy filling scheme
filling_scheme = np.ones(num_bunches, dtype=int)
bunch_selection = np.arange(num_bunches, dtype=int)

zeta_range = (-1, 1)

# we create a compressed profile object and a slicer object. The slicer object
# is used to compute the moments of the beam at the slices, which are then
# inserted in the profile
moments = ['x', 'y', 'px', 'py', 'num_particles']

compressed_profile = CompressedProfile(
    moments=moments,
    zeta_range=zeta_range,
    num_slices=20,
    bunch_spacing_zeta=bunch_spacing_zeta,
    num_periods=num_bunches,
    num_turns=1,
    circumference=bunch_spacing_zeta*len(filling_scheme))


slicer = xf.UniformBinSlicer(
    zeta_range=zeta_range,
    num_slices=compressed_profile.num_slices,
    filling_scheme=filling_scheme,
    bunch_selection=bunch_selection,
    bunch_spacing_zeta=bunch_spacing_zeta,
    moments='all'
)

slicer.track(particles)

for i_bunch in range(num_bunches):
    dict_moments = {
        'num_particles': slicer.num_particles[i_bunch, :],
    }

    for moment in moments:
        if moment == 'num_particles':
            continue
        dict_moments[moment] = slicer.mean(moment)[i_bunch, :]

    compressed_profile.set_moments(i_turn=0, i_source=bunch_selection[i_bunch],
                                   moments=dict_moments)

long_profile = compressed_profile.get_moment_profile(
    moment_name='num_particles', i_turn=0)

plt.figure(321)
plt.plot(long_profile[0], long_profile[1]/np.sum(long_profile[1]))
plt.xlabel('zeta [m]')
plt.ylabel('longitudinal profile [a.u.]')
plt.show()
