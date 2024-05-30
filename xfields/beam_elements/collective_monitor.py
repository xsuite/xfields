import numpy as np

from .sliced_element import SlicedElement
import openpmd_api as io

COORDS = ['x', 'px', 'y', 'py', 'zeta', 'delta']
SECOND_MOMENTS = {}
for c1 in COORDS:
    for c2 in COORDS:
        if c1 + '_' + c2 in SECOND_MOMENTS or c2 + '_' + c1 in SECOND_MOMENTS:
            continue
        SECOND_MOMENTS[c1 + '_' + c2] = (c1, c2)


class CollectiveMonitor(SlicedElement):
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

    _stats_to_store = [
                'mean_x', 'mean_px', 'mean_y', 'mean_py', 'mean_zeta',
                'mean_delta', 'sigma_x', 'sigma_y', 'sigma_zeta',
                'sigma_px', 'sigma_py', 'sigma_delta',
                'epsn_x', 'epsn_y', 'epsn_zeta', 'num_particles']

    def __init__(self,
                 base_file_name,
                 monitor_bunches,
                 monitor_slices,
                 monitor_particles,
                 n_steps,
                 beta_gamma,
                 slicer_moments='all',
                 zeta_range=None,  # These are [a, b] in the paper
                 num_slices=None,  # Per bunch, this is N_1 in the paper
                 bunch_spacing_zeta=None,  # This is P in the paper
                 num_slots=None,
                 filling_scheme=None,
                 bunch_numbers=None,
                 stats_to_store=None,
                 output_extension=None,
                 _flatten=False):

        if stats_to_store is not None:
            for stat in stats_to_store:
                assert stat in self._stats_to_store
            self.stats_to_store = stats_to_store
        else:
            self.stats_to_store = self._stats_to_store

        self.base_file_name = base_file_name
        self.bunch_buffer = None
        self.slice_buffer = None
        self.particle_buffer = None
        self.n_steps = n_steps
        self.beta_gamma = beta_gamma
        self.monitor_bunches = monitor_bunches
        self.monitor_slices = monitor_slices
        self.monitor_particles = monitor_particles

        self.output_extension = output_extension

        if slicer_moments == 'all':
            slicer_moments = COORDS + list(SECOND_MOMENTS.keys())

        self.pipeline_manager = None

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

        if self.monitor_bunches:
            self.bunch_series = io.Series(
                self.base_file_name + '_bunches.h5',
                io.Access.create)

        if self.monitor_slices:
            self.slice_series = io.Series(
                self.base_file_name + '_slices.h5',
                io.Access.create)

        if self.monitor_particles:
            self.particle_series = io.Series(
                self.base_file_name + '_particles.h5',
                io.Access.create)

        self.i_turn = 0

    '''
    def _init_bunch_buffer(self):
        buf = {}
        for bid in self.slicer.bunch_numbers:
            buf[bid] = {}
            for stats in self.stats_to_store:
                buf[bid][stats] = np.zeros(self.n_steps)

        return buf

    def _init_slice_buffer(self):
        buf = {}
        for bid in self.slicer.bunch_numbers:
            buf[bid] = {}
            for stats in self.stats_to_store:
                buf[bid][stats] = np.zeros((self.slicer.num_slices,
                                            self.n_steps))

        return buf
    '''

    def track(self, particles, _slice_result=None, _other_bunch_slicers=None):
        super().track(particles=particles,
                      _slice_result=_slice_result,
                      _other_bunch_slicers=_other_bunch_slicers
                      )

        #if self.monitor_bunches:
        #    if self.bunch_buffer is None:
        #        self.bunch_buffer = self._init_bunch_buffer()
        #
        #    self._update_bunch_buffer(particles)

        #if self.monitor_slices:
        #    if self.slice_buffer is None:
        #        self.slice_buffer = self._init_slice_buffer()
        #
        #    self._update_slice_buffer()

        if self.monitor_bunches:
            self._update_bunch_series(particles)

        # self.file_backend.dump_data(self.i_turn, self.slicer, particles)

        self.i_turn += 1

    def _update_bunch_series(self, particles):
        i_bunch_particles = self._slice_result['i_bunch_particles']

        bunch_it = self.bunch_series.iterations[self.i_turn]
        bunches = bunch_it.particles['bunches']

        for i_bunch, bid in enumerate(self.slicer.bunch_numbers):
            bunch_mask = (i_bunch_particles == bid)
            for stat in self.stats_to_store:
                if stat == 'num_particles':
                    val = np.sum(self.slicer.num_particles[i_bunch, :])
                else:
                    mom = getattr(particles, stat.split('_')[-1])
                    if stat.startswith('mean'):
                        val = np.mean(mom[bunch_mask])
                    elif stat.startswith('sigma'):
                        val = np.std(mom[bunch_mask])
                    elif stat.startswith('epsn'):
                        if stat.split('_')[-1] != 'zeta':
                            mom_p_str = 'p' + stat.split('_')[-1]
                        else:
                            mom_p_str = 'delta'
                        mom_p = getattr(particles, mom_p_str)
                        val = (np.sqrt(np.linalg.det(np.cov(mom[bunch_mask],
                                                            mom_p[bunch_mask]))) *
                               self.beta_gamma)
                    elif stat == 'num_particles':
                        val = np.sum(self.slicer.num_particles[i_bunch, :])
                    else:
                        raise ValueError('Unknown statistics f{stat}')

                self.bunch_buffer[bid][stat][write_pos] = val

    def _update_bunch_buffer(self, particles):
        i_bunch_particles = self._slice_result['i_bunch_particles']

        write_pos = self.i_turn % self.buffer_size
        for i_bunch, bid in enumerate(self.slicer.bunch_numbers):
            bunch_mask = (i_bunch_particles == bid)
            for stat in self.stats_to_store:
                if stat == 'num_particles':
                    val = np.sum(self.slicer.num_particles[i_bunch, :])
                else:
                    mom = getattr(particles, stat.split('_')[-1])
                    if stat.startswith('mean'):
                        val = np.mean(mom[bunch_mask])
                    elif stat.startswith('sigma'):
                        val = np.std(mom[bunch_mask])
                    elif stat.startswith('epsn'):
                        if stat.split('_')[-1] != 'zeta':
                            mom_p_str = 'p' + stat.split('_')[-1]
                        else:
                            mom_p_str = 'delta'
                        mom_p = getattr(particles, mom_p_str)
                        val = (np.sqrt(np.linalg.det(np.cov(mom[bunch_mask],
                                                            mom_p[bunch_mask]))) *
                               self.beta_gamma)
                    elif stat == 'num_particles':
                        val = np.sum(self.slicer.num_particles[i_bunch, :])
                    else:
                        raise ValueError('Unknown statistics f{stat}')

                self.bunch_buffer[bid][stat][write_pos] = val

    def _update_slice_buffer(self):
        print('bla')
        write_pos = self.i_turn % self.buffer_size
        for i_bunch, bid in enumerate(self.slicer.bunch_numbers):
            for stat in self.stats_to_store:
                mom_str = stat.split('_')[-1]
                if stat.startswith('mean'):
                    val = self.slicer.mean(mom_str)[i_bunch, :]
                elif stat.startswith('sigma'):
                    val = self.slicer.std(mom_str)[i_bunch, :]
                elif stat.startswith('epsn'):
                    if mom_str != 'zeta':
                        mom_p_str = 'p' + stat.split('_')[-1]
                    else:
                        mom_p_str = 'delta'

                    val = (np.sqrt(self.slicer.var(mom_str)[i_bunch, :] *
                           self.slicer.var(mom_p_str)[i_bunch, :] -
                           self.slicer.cov(mom_str, mom_p_str)[i_bunch, :]) *
                           self.beta_gamma)
                elif stat == 'num_particles':
                    val = self.slicer.num_particles[i_bunch, :]
                else:
                    raise ValueError('Unknown statistics f{stat}')

                self.slice_buffer[bid][stat][:, write_pos] = val


class BaseFileBackend:
    def __init__(self, base_filename,
                 #monitor_bunches, bunch_monitor_stride,
                 #monitor_slices, slice_monitor_stride,
                 #monitor_particles, particle_monitor_stride,
                 parameters_dict=None, bunch_ids=None):

        self.base_filename = base_filename
        #self.monitor_bunches = monitor_bunches
        #self.bunch_monitor_stride = bunch_monitor_stride
        #self.monitor_slices = monitor_slices
        #self.slice_monitor_stride = slice_monitor_stride
        #self.monitor_particles = monitor_particles
        #self.particle_monitor_stride = particle_monitor_stride
        self.parameters_dict = parameters_dict

        if bunch_ids is not None:
            self.bunch_ids = bunch_ids
        else:
            self.bunch_ids = np.array([0])

        self.bunch_buffer = None
        self.slice_buffer = None
        self.particle_buffer = None

    '''
    def init_files(self, stats_to_store, n_steps):
        if self.monitor_bunches:
            self.init_bunch_monitor_file(stats_to_store, n_steps)

        if self.monitor_slices:
            self.init_slice_monitor_file()

        if self.monitor_particles:
            self.init_particle_monitor_file()
    '''

    def init_bunch_monitor_file(self, stats_to_store, n_steps):
        raise RuntimeError('File backend must implement '
                           'init_bunch_monitor_file')

    def init_slice_monitor_file(self):
        raise RuntimeError('File backend must implement '
                           'init_slice_monitor_file')

    def init_particle_monitor_file(self):
        raise RuntimeError('File backend must implement '
                           'init_particle_monitor_file')

    def dump_data(self, i_turn, slicer, particles):
        if self.monitor_bunches and i_turn % self.bunch_monitor_stride == 0:
            self.dump_bunch_data(i_turn, particles)

        if self.monitor_slices and i_turn % self.slice_monitor_stride == 0:
            self.dump_slice_data(i_turn, slicer)

        if self.monitor_particles and i_turn % self.particle_monitor_stride == 0:
            self.dump_particle_data(i_turn, slicer, particles)

    def dump_bunch_data(self, i_turn, slicer):
        raise RuntimeError('File backend must implement dump_bunch_data')

    def dump_slice_data(self, i_turn, slicer):
        raise RuntimeError('File backend must implement dump_slice_data')

    def dump_particle_data(self, i_turn, slicer, particles):
        raise RuntimeError('File backend must implement dump_bunch_data')

'''
class HDF5BackEnd(BaseFileBackend):
    def __init__(self, base_filename,
                 monitor_bunches, bunch_monitor_stride,
                 monitor_slices, slice_monitor_stride,
                 monitor_particles, particle_monitor_stride,
                 parameters_dict=None, bunch_ids=None, use_mpi=False):

        self.use_mpi = use_mpi

        super().__init__(base_filename=base_filename,
                         monitor_bunches=monitor_bunches,
                         bunch_monitor_stride=bunch_monitor_stride,
                         monitor_slices=monitor_slices,
                         slice_monitor_stride=slice_monitor_stride,
                         monitor_particles=monitor_particles,
                         particle_monitor_stride=particle_monitor_stride,
                         parameters_dict=parameters_dict,
                         bunch_ids=bunch_ids)

    def init_bunch_monitor_file(self, stats_to_store, n_steps):
        """
        Initialize HDF5 file and create its basic structure (groups and
        datasets). One group is created for bunch-specific data. One dataset for
        each of the quantities defined in self.stats_to_store is generated.
        If specified by the user, write the contents of the parameters_dict as
        metadata (attributes) to the file. Maximum file compression is activated
        only if not using MPI.
        """
        if self.base_filename is not None:
            filename = f'{self.base_filename}_bunchmonitor.h5'
        else:
            filename = f'bunchmonitor.h5'

        if self.use_mpi:
            h5file = hp.File(filename, 'w', driver='mpio',
                             comm=MPI.COMM_WORLD)
            kwargs_gr = {}
        else:
            h5file = hp.File(filename, 'w')
            kwargs_gr = {'compression': 'gzip', 'compression_opts': 9}

        if self.parameters_dict:
            for key in self.parameters_dict:
                h5file.attrs[key] = self.parameters_dict[key]

        h5group = h5file.create_group('Bunches')
        for bid in self.bunch_ids:
            gr = h5group.create_group(repr(bid))
            for stats in sorted(stats_to_store):
                gr.create_dataset(stats, shape=(n_steps,),
                                  **kwargs_gr)

        h5file.close()

    def dump_bunch_data(self, i_turn, particles):
        pass

    def _write_data_to_buffer(self, i_turn, particles):
        """ Store the data in the self.buffer dictionary before writing
        them to file. The buffer is implemented as a shift register. To
        find the slice_set-specific data, a slice_set, defined by the
        slicing configuration self.slicer must be requested from the
        bunch (instance of the Particles class), including all the
        statistics that are to be saved. """

        # Handle the different statistics quantities, which can
        # either be methods (like mean(), ...) or simply attributes
        # (macroparticlenumber or n_macroparticles_per_slice) of the bunch
        # or slice_set resp.

        # bunch-specific data.
        pass
'''