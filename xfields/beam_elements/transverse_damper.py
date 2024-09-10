import numpy as np

import xpart as xp
import xfields as xf
import xtrack as xt
from xfields.slicers.compressed_profile import CompressedProfile


class TransverseDamper(xt.BeamElement):
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

    iscollective = True

    def __init__(self, gain_x, gain_y, filling_scheme, zeta_range, num_slices, bunch_spacing_zeta,
                 circumference, bunch_selection=None, **kwargs):
        self.gains = {
            'px': gain_x,
            'py': gain_y,
        }

        self.slicer = xf.UniformBinSlicer(
            filling_scheme=filling_scheme,
            bunch_selection=bunch_selection,
            zeta_range=zeta_range,
            num_slices=num_slices,
            bunch_spacing_zeta=bunch_spacing_zeta,
            moments=['px', 'py']
        )

        if filling_scheme is not None:
            i_last_bunch = np.where(filling_scheme)[0][-1]
            num_periods = i_last_bunch + 1
        else:
            num_periods = 1

        self.moments_data = {}
        for moment in self.gains.keys():
            self.moments_data[moment] = CompressedProfile(
                moments=[moment],
                zeta_range=zeta_range,
                num_slices=num_slices,
                bunch_spacing_zeta=bunch_spacing_zeta,
                num_periods=num_periods,
                num_turns=1,
                circumference=circumference
            )

    def _reconfigure_for_parallel(self, n_procs, my_rank):
        filled_slots = self.slicer.filled_slots
        scheme = np.zeros(np.max(filled_slots) + 1,
                        dtype=np.int64)
        scheme[filled_slots] = 1

        split_scheme = xp.matched_gaussian.split_scheme
        bunch_selection_rank = split_scheme(filling_scheme=scheme,
                                             n_chunk=int(n_procs))

        self.slicer = xf.UniformBinSlicer(
            filling_scheme=scheme,
            bunch_selection=bunch_selection_rank[my_rank],
            zeta_range=self.slicer.zeta_range,
            num_slices=self.slicer.num_slices,
            bunch_spacing_zeta=self.slicer.bunch_spacing_zeta,
            moments=['px', 'py']
        )

    def track(self, particles, i_turn=0):
        i_slice_particles = particles.particle_id * 0 + -999
        i_slot_particles = particles.particle_id * 0 + -9999

        self.slicer.slice(particles, i_slice_particles=i_slice_particles,
                          i_slot_particles=i_slot_particles)

        for moment in ['px', 'py']:

            slice_means = self.slicer.mean(moment)

            for i_bunch, bunch_number in enumerate(
                    self.slicer.bunch_selection):

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
                i_slot_particles=i_slot_particles,
                i_slice_particles=i_slice_particles,
                out=interpolated_result
            )

            getattr(particles, moment)[:] -= (self.gains[moment] *
                                              interpolated_result)
