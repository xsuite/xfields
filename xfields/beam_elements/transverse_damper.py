import numpy as np

import xpart as xp
import xfields as xf
import xtrack as xt
from xfields.slicers.compressed_profile import CompressedProfile


class TransverseDamper(xt.BeamElement):
    """
    A simple bunch-by-bunch transverse Damper implementation
    featuring a single kicker (only px and py are updated).

    Parameters
    ----------
    gain_x : complex
        the horizontal damper gain in 1/turns (corresponding to a damping
        rate of Re(gain_x)/2 and a tune shift of Im(gain_x)*betx/(4pi)).
    gain_y : complex
        the vertical damper gain in 1/turns (corresponding to a damping rate
        of Re(gain_y)/2 and a tune shift of Im(gain_y)*bety/(4pi)).
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

    def __init__(self, gain_x, gain_y, zeta_range, num_slices,
                 circumference=None, bunch_spacing_zeta=None, filling_scheme=None,
                 bunch_selection=None, **kwargs):

        self.gains = {'px':gain_x,
                      'py':gain_y}
        
        source_moments = []
        if np.real(gain_x) != 0:
            source_moments.append('px')
        if np.imag(gain_x) != 0:
            source_moments.append('x')
        if np.real(gain_y) != 0:
            source_moments.append('py')
        if np.imag(gain_y) != 0:
            source_moments.append('y')
        
        self.iscollective = True
        
        self.xoinitialize(**kwargs)

        self.slicer = xf.UniformBinSlicer(
            filling_scheme=filling_scheme,
            bunch_selection=bunch_selection,
            zeta_range=zeta_range,
            num_slices=num_slices,
            bunch_spacing_zeta=bunch_spacing_zeta,
            moments=source_moments,
            _context=self._context
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
            moments=self.slicer.moments,
            _context=self._context
        )
        
    def get_slot_mean(self, slice_means,slot_slices,i_bunch):
        if len(self.slicer.bunch_selection) == 1:
            return np.mean(slice_means[slot_slices])
        else:
            return np.mean(slice_means[i_bunch, slot_slices])

    def track(self, particles, i_turn=0):
        i_slice_particles = particles.particle_id * 0 + -999
        i_slot_particles = particles.particle_id * 0 + -9999

        self.slicer.slice(particles, i_slice_particles=i_slice_particles,
                          i_slot_particles=i_slot_particles)

        for moment in self.gains.keys():
            position = moment[1]
            particles_moment = getattr(particles, moment)[:]
            do_resistive = moment in self.slicer.moments
            do_reactive = position in self.slicer.moments

            for i_bunch, bunch_number in enumerate(
                                self.slicer.bunch_selection):
                slot_mask = i_slot_particles == bunch_number
                slot_slices = np.unique(i_slice_particles[slot_mask])

                if do_resistive:
                    particles_moment[slot_mask] -= (np.real(self.gains[moment]) *
                                                self.get_slot_mean(self.slicer.mean(moment),slot_slices,i_bunch))
                if do_reactive:
                    particles_moment[slot_mask] -= (np.imag(self.gains[moment]) *
                                                self.get_slot_mean(self.slicer.mean(position),slot_slices,i_bunch))
                                                
