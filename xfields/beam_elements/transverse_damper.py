import numpy as np

import xpart as xp
import xfields as xf
import xtrack as xt
from xtrack._filling_pattern import _FillingPattern
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
    filling_pattern : np.ndarray, optional
        an array of zeros and ones representing the filling pattern. Only
        needed for multi-bunch tracking.
    filling_scheme : np.ndarray, optional
        Compatibility alias for ``filling_pattern``.
    filled_slots : np.ndarray, optional
        Sparse list of filled physical slots. Mutually exclusive with the
        dense filling inputs.
    num_slots : int, optional
        Total number of slots associated with ``filled_slots``.
    bunch_selection : np.ndarray, optional
        an array indicating which slot each bunch occupies in the filling
        scheme. Only needed for multi-bunch tracking
    bunch_spacing_zeta : float, optional
        the bunch spacing in meters. Only needed for multi-bunch tracking
    circumference : float, optional
        the machine circumference. Only needed for multi-bunch tracking.

    """

    def __init__(self, gain_x, gain_y, zeta_range, num_slices,
                 circumference=None, bunch_spacing_zeta=None,
                 filling_pattern=None, bunch_selection=None,
                 filling_scheme=None, filled_slots=None, num_slots=None,
                 **kwargs):
        filling = _FillingPattern.from_inputs(
            filling_pattern=filling_pattern,
            filled_slots=filled_slots,
            filling_scheme=filling_scheme,
            num_slots=num_slots)
        filling_pattern = (
            None if filling is None else filling.filling_pattern)
        self.gains = {
            'px': gain_x,
            'py': gain_y,
        }
        
        self.iscollective = True
        
        self.xoinitialize(**kwargs)

        self.slicer = xf.UniformBinSlicer(
            filling_pattern=filling_pattern,
            bunch_selection=bunch_selection,
            zeta_range=zeta_range,
            num_slices=num_slices,
            bunch_spacing_zeta=bunch_spacing_zeta,
            moments=['px', 'py'],
            _context=self._context
        )

        if filling_pattern is not None:
            i_last_bunch = np.where(filling_pattern)[0][-1]
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
        scheme = self.slicer.filling_pattern

        bunch_selection_rank = xp.split_filling_pattern(
            filling_pattern=scheme, n_chunk=int(n_procs))

        self.slicer = xf.UniformBinSlicer(
            filling_pattern=scheme,
            bunch_selection=bunch_selection_rank[my_rank],
            zeta_range=self.slicer.zeta_range,
            num_slices=self.slicer.num_slices,
            bunch_spacing_zeta=self.slicer.bunch_spacing_zeta,
            moments=['px', 'py'],
            _context=self._context
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
