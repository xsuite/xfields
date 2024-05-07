import numpy as np

from .sliced_element import SlicedElement


class SlicedMonitor(SlicedElement):
    """
    Base class for elements with a slicer.

    Parameters
    ----------
    slicer_moments: List
        Moments for the slicer
    zeta_range : Tuple
        Zeta range for each bunch used in the underlying slicer.
    num_slices : int
        Number of slices per bunch used in the underlying slicer.
    bunch_spacing_zeta : float
        Bunch spacing in meters.
    num_slots : int
        Number of filled slots.
    filling_scheme: np.ndarray
        List of zeros and ones representing the filling scheme. The length
        of the array is equal to the number of slots in the machine and each
        element of the array holds a one if the slot is filled or a zero
        otherwise.
    bunch_numbers: np.ndarray
        List of the bunches indicating which slots from the filling scheme are
        used (not all the bunches are used when using multi-processing)
    _flatten: bool
        Use flattened wakes
    """

    def __init__(self,
                 file_backend,
                 slicer_moments=None,
                 zeta_range=None,  # These are [a, b] in the paper
                 num_slices=None,  # Per bunch, this is N_1 in the paper
                 bunch_spacing_zeta=None,  # This is P in the paper
                 num_slots=None,
                 filling_scheme=None,
                 bunch_numbers=None,
                 _flatten=False):

        self.file_backend = file_backend

        super().__init__(
            slicer_moments=slicer_moments,
            zeta_range=zeta_range,  # These are [a, b] in the paper
            num_slices=num_slices,  # Per bunch, this is N_1 in the paper
            bunch_spacing_zeta=bunch_spacing_zeta,  # This is P in the paper
            num_slots=num_slots,
            filling_scheme=filling_scheme,
            bunch_numbers=bunch_numbers,
            with_compressed_profile=False
        )

        self.i_turn = 0

    def track(self, particles, _slice_result=None, _other_bunch_slicers=None):
        super().track(particles=particles,
                      _slice_result=_slice_result,
                      _other_bunch_slicers=_other_bunch_slicers
                      )

        # dump beam data
        self.file_backend.dump_data(self.i_turn, self.slicer, particles)

        self.i_turn += 1


class BaseFileBackend:
    def __init__(self, base_filename, monitor_beam, beam_monitor_stride,
                 monitor_bunches, bunch_monitor_stride,
                 monitor_particles, particle_monitor_stride):

        self.base_filename = base_filename
        self.monitor_beam = monitor_beam
        self.beam_monitor_stride = beam_monitor_stride
        self.monitor_bunches = monitor_bunches
        self.bunch_monitor_stride = bunch_monitor_stride
        self.monitor_particles = monitor_particles
        self.particle_monitor_stride = particle_monitor_stride

    def init_files(self):
        raise RuntimeError('File backend must implement init_files')

    def dump_data(self, i_turn, slicer, particles):
        if self.monitor_beam and i_turn % self.beam_monitor_stride == 0:
            self.dump_beam_data(slicer)

        if self.monitor_beam and i_turn % self.beam_monitor_stride == 0:
            self.dump_beam_data(slicer)

    def dump_beam_data(self, slicer):
        raise RuntimeError('File backend must implement dump_beam_data')

    def dump_bunch_data(self, slicer):
        raise RuntimeError('File backend must implement dump_bunch_data')

    def dump_particle_data(self, slicer, particles):
        raise RuntimeError('File backend must implement dump_bunch_data')