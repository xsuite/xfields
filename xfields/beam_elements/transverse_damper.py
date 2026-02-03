import numpy as np

import xpart as xp
import xfields as xf
import xtrack as xt
from xfields.slicers.compressed_profile import CompressedProfile


class TransverseDamper(xt.BeamElement):
    """
    A simple bunch-by-bunch transverse Damper implementation.

    Parameters
    ----------
    gain_x : float
        the horizontal damper gain in 1/turns (corresponding to a damping
        rate of gain_x/2).
    gain_y : float
        the vertical damper gain in 1/turns (corresponding to a damping rate
        of gain_x/2).
    zeta_range : tuple
        the range of zetas covered by the underlying slicer.
    num_slices : int
        the number of slices per bunch used by the underlying slicer.
    filling_scheme : np.ndarray, optional
        an array of zeros and ones representing the filling scheme. Only
        needed for multi-bunch tracking.
    bunch_selection : np.ndarray, optional
        an array indicating which slot each bunch occupies in the filling
        scheme. Only needed for multi-bunch tracking
    bunch_spacing_zeta : float, optional
        the bunch spacing in meters. Only needed for multi-bunch tracking
    circumference : float, optional
        the machine circumference. Only needed for multi-bunch tracking.

    """

    iscollective = True

    def __init__(self, gain_x, gain_y, zeta_range, num_slices,
                 circumference=None, bunch_spacing_zeta=None, filling_scheme=None,
                 bunch_selection=None, **kwargs):
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
            particles_moment = getattr(particles, moment)[:]
            slice_means = self.slicer.mean(moment)

            for i_bunch, bunch_number in enumerate(
                                self.slicer.bunch_selection):
                slot_mask = i_slot_particles == bunch_number
                slot_slices = np.unique(i_slice_particles[slot_mask])

                if len(self.slicer.bunch_selection) == 1:
                    slot_mean = np.mean(slice_means[slot_slices])
                else:
                    slot_mean = np.mean(slice_means[i_bunch, slot_slices])

                slot_mean = np.mean(particles_moment[slot_mask])

                particles_moment[slot_mask] -= (self.gains[moment] *
                                                slot_mean)
