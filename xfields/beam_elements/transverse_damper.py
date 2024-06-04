import numpy as np

import xfields as xf
from xfields.slicers.compressed_profile import CompressedProfile


class TransverseDamper:
    """
    A simple bunch-by-bunch transverse Damper implementation.

    Args:
        gain_x (float): the horizontal damper gain in 1/turns (corresponding to
        a damping rate of gain_x/2)
        gain_y (float): the vertical damper gain in 1/turns (corresponding to
        a damping rate of gain_x/2)
        num_bunches (float): the number of bunches in the beam
        filling_scheme (np.ndarray): an array of zeros and ones representing
            the filling scheme
        filled_slots (np.ndarray): an array indicating which slot each bunch
            occupies
        zeta_range (tuple): the range of zetas covered by the underlying slicer
        num_slices (int): the number of slices used by the underlying slicer,
        bunch_spacing_zeta (float): the bunch spacing in meters
        circumference (float): the machine circumference

    Returns:
        (TransverseDamper): A transverse damper beam element.
    """
    def __init__(self, gain_x, gain_y, num_bunches, filling_scheme,
                 filled_slots, zeta_range, num_slices, bunch_spacing_zeta,
                 circumference):
        self.gains = {
            'px': gain_x,
            'py': gain_y,
        }

        self.slicer = xf.UniformBinSlicer(
            filling_scheme=filling_scheme,
            bunch_numbers=filled_slots,
            zeta_range=zeta_range,
            num_slices=num_slices,
            bunch_spacing_zeta=bunch_spacing_zeta,
            moments=['px', 'py']
        )

        self.num_bunches = num_bunches

        self.moments_data = {}
        for moment in self.gains.keys():
            self.moments_data[moment] = CompressedProfile(
                moments=[moment],
                zeta_range=zeta_range,
                num_slices=num_slices,
                bunch_spacing_zeta=bunch_spacing_zeta,
                num_periods=num_bunches,
                num_turns=1,
                circumference=circumference
            )

    def track(self, particles, i_turn=0):
        i_slice_particles = particles.particle_id * 0 + -999
        i_bunch_particles = particles.particle_id * 0 + -9999

        self.slicer.slice(particles, i_slice_particles=i_slice_particles,
                          i_bunch_particles=i_bunch_particles)

        for moment in ['px', 'py']:

            slice_means = self.slicer.mean(moment)

            for i_bunch, bunch_number in enumerate(
                    self.slicer.bunch_numbers):

                nnz_slices = self.slicer.num_particles[i_bunch, :] > 0
                moments_bunch = {
                    moment: (np.ones_like(slice_means[i_bunch, :]) *
                             np.mean(slice_means[i_bunch, nnz_slices]))
                }

                self.moments_data[moment].set_moments(
                    moments=moments_bunch,
                    i_turn=0,
                    i_source=self.slicer.filled_slots[bunch_number])

            interpolated_result = particles.zeta * 0

            md = self.moments_data[moment]

            self.moments_data[moment]._interp_result(
                particles=particles,
                data_shape_0=md.data.shape[0],
                data_shape_1=md.data.shape[1],
                data_shape_2=md.data.shape[2],
                data=md.data,
                i_bunch_particles=i_bunch_particles,
                i_slice_particles=i_slice_particles,
                out=interpolated_result
            )

            getattr(particles, moment)[:] -= (self.gains[moment] *
                                              interpolated_result)
