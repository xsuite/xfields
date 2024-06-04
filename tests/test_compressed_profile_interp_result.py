import numpy as np
from scipy.constants import c

clight = c

from xfields.slicers import CompressedProfile
from xobjects.test_helpers import for_all_test_contexts
import xtrack as xt


def test_compressed_profile_interp_result():
    # Machine settings
    moments = ['result']
    zeta_range = (0, 1)
    num_slices = 10
    dz = (zeta_range[1] - zeta_range[0]) / num_slices
    num_bunches = 1
    num_turns = 3
    bunch_spacing_zeta = 10
    circumference = 100

    comp_prof = CompressedProfile(moments,
                                  zeta_range=zeta_range,
                                  num_slices=num_slices,
                                  bunch_spacing_zeta=bunch_spacing_zeta,
                                  num_periods=num_bunches,
                                  num_turns=num_turns,
                                  circumference=circumference
                                  )

    num_parts = 100

    interpolated_result = np.zeros(num_parts, dtype=float)
    i_slice_particles = np.linspace(0, num_slices - 1, num_parts, dtype=int)

    result_parts = np.zeros(num_parts)

    def func(z, i):
        return z + circumference * i

    for i_turn in range(num_turns):
        moments = {
            'result': func(np.linspace(zeta_range[0] + dz / 2,
                                       zeta_range[1] - dz / 2,
                                       num_slices), i_turn)
        }

        comp_prof.set_moments(moments=moments,
                              i_turn=i_turn, i_source=0)

        result_parts += func(i_slice_particles * dz + dz / 2, i_turn)

    particles = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        gamma0=np.ones(num_parts) * 1000,
        x=np.zeros(num_parts),
        px=np.zeros(num_parts),
        y=np.zeros(num_parts),
        py=np.zeros(num_parts),
        zeta=np.zeros(num_parts),
        delta=np.zeros(num_parts),
        weight=np.ones(num_parts),
    )

    comp_prof._interp_result(particles=particles,
                             data_shape_0=comp_prof.data.shape[0],
                             data_shape_1=comp_prof.data.shape[1],
                             data_shape_2=comp_prof.data.shape[2],
                             data=comp_prof.data,
                             i_bunch_particles=np.zeros(num_parts, dtype=int),
                             i_slice_particles=i_slice_particles,
                             out=interpolated_result
                             )

    assert np.allclose(result_parts, interpolated_result)
