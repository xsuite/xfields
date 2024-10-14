import numpy as np

from .element_with_slicer import ElementWithSlicer
import json
import os
import xfields as xf
import xpart as xp

COORDS = ['x', 'px', 'y', 'py', 'zeta', 'delta']
SECOND_MOMENTS = {}
for c1 in COORDS:
    for c2 in COORDS:
        if c1 + '_' + c2 in SECOND_MOMENTS or c2 + '_' + c1 in SECOND_MOMENTS:
            continue
        SECOND_MOMENTS[c1 + '_' + c2] = (c1, c2)


class CollectiveMonitor(ElementWithSlicer):
    """
    A class to monitor the collective motion of the beam. The monitor can save
    bunch-by-bunch, slice-by-slice and particle-by-particle data. For the
    particle-by-particle data, a mask can be used to select the particles to
    be monitored. The monitor collects a buffer of `flush_data_every` turns
    before saving the data to disk, in order to reduce the number of I/O and
    avoid storing too much data in memory.

    Parameters
    ----------
    monitor_bunches : Bool
        If True, the bunch-by-bunch data is monitored
    monitor_slices : Bool
        If True, the slice-by-slice data is monitored
    monitor_particles : Bool
        If True, the particle-by-particle data is monitored
    base_file_name : str
        Base file name for the output files. If it is not specified, the data
        will not be saved to disk.
    particle_monitor_mask : np.ndarray
        Mask identifying the particles to be monitored. If later on we try to
        monitor a number of particles different than the length of the mask, an
        error will be raised.
    flush_data_every: int
        Number of turns after which the data is saved to disk
    stats_to_store : list
        List of the statistics to store for the bunch-by-bunch and
        slice-by-slice data
    stats_to_store_particles : list
        List of the statistics to store for the particle-by-particle data
    backend: str
        Backend used to save the data. For the moment only 'hdf5' and 'json'
        are supported, with 'hdf5' being the default
    zeta_range : Tuple
        Zeta range for each bunch used in the underlying slicer.
    num_slices : int
        Number of slices per bunch used in the underlying slicer. It should be
        specified if the slice-by-slice data is monitored.
    bunch_spacing_zeta : float
        Bunch spacing in meters.
    filling_scheme: np.ndarray
        List of zeros and ones representing the filling scheme. The length
        of the array is equal to the number of slots in the machine and each
        element of the array holds a one if the slot is filled or a zero
        otherwise.
    bunch_selection: np.ndarray
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

    _stats_to_store_particles = [
        'x', 'px', 'y', 'py', 'zeta', 'delta', 'weight', 'state']

    # Mapping from the stats to store in the monitor to the moments to store in
    # the slicer
    stat_to_slicer_moments = {
        'mean_x': ['x'],
        'mean_px': ['px'],
        'mean_y': ['y'],
        'mean_py': ['py'],
        'mean_zeta': ['zeta'],
        'mean_delta': ['delta'],
        'sigma_x': ['x', 'x_x'],
        'sigma_y': ['y', 'y_y'],
        'sigma_zeta': ['zeta', 'zeta_zeta'],
        'sigma_px': ['px', 'px_px'],
        'sigma_py': ['py', 'py_py'],
        'sigma_delta': ['delta', 'delta_delta'],
        'epsn_x': ['x', 'px', 'x_x', 'px_px', 'x_px'],
        'epsn_y': ['y', 'py', 'y_y', 'py_py', 'y_py'],
        'epsn_zeta': ['zeta', 'delta', 'delta_delta', 'zeta_zeta',
                      'zeta_delta'],
        'num_particles': ['num_particles']
    }

    def __init__(self,
                 monitor_bunches,
                 monitor_slices,
                 monitor_particles,
                 base_file_name=None,
                 particle_monitor_mask=None,
                 flush_data_every=1,
                 zeta_range=None,
                 num_slices=1,
                 bunch_spacing_zeta=None,
                 filling_scheme=None,
                 bunch_selection=None,
                 stats_to_store=None,
                 stats_to_store_particles=None,
                 backend='hdf5',
                 _flatten=False):

        slicer_moments = []
        if stats_to_store is not None:
            for stat in stats_to_store:
                assert stat in self._stats_to_store
            self.stats_to_store = stats_to_store
        else:
            self.stats_to_store = self._stats_to_store

        if stats_to_store_particles is not None:
            for stat in stats_to_store_particles:
                assert stat in self._stats_to_store_particles
            self.stats_to_store_particles = stats_to_store_particles
        else:
            self.stats_to_store_particles = self._stats_to_store_particles

        for stat in self.stats_to_store:
            # For each stat we store, we add the corresponding moment
            if stat == 'num_particles':
                slicer_moments.append(stat)
            else:
                slicer_moments.extend(self.stat_to_slicer_moments[stat])

        if backend == 'hdf5':
            self.extension = 'h5'
            self.flush_buffer_to_file_func = flush_buffer_to_file_hdf5
        elif backend == 'json':
            self.extension = 'json'
            self.flush_buffer_to_file_func = flush_buffer_to_file_json
        else:
            raise ValueError('Only hdf5 and json backends are supported.')

        self.base_file_name = base_file_name
        self.bunch_buffer = None
        self.slice_buffer = None
        self.particle_buffer = None
        self.monitor_bunches = monitor_bunches
        self.monitor_slices = monitor_slices
        self.monitor_particles = monitor_particles
        self.flush_data_every = flush_data_every
        self.particle_monitor_mask = particle_monitor_mask

        if base_file_name is not None:
            self._bunches_filename = (self.base_file_name + '_bunches.' +
                                    self.extension)

            if os.path.exists(self._bunches_filename) and monitor_bunches:
                os.remove(self._bunches_filename)

            self._slices_filename = (self.base_file_name + '_slices.' +
                                    self.extension)

            if os.path.exists(self._slices_filename) and monitor_slices:
                os.remove(self._slices_filename)

            self._particles_filename = (self.base_file_name + '_particles.' +
                                        self.extension)

            if os.path.exists(self._particles_filename) and monitor_particles:
                os.remove(self._particles_filename)

        self.pipeline_manager = None

        self.i_turn = 0

        super().__init__(
            slicer_moments=slicer_moments,
            zeta_range=zeta_range,  # These are [a, b] in the paper
            num_slices=num_slices,  # Per bunch, this is N_1 in the paper
            bunch_spacing_zeta=bunch_spacing_zeta,  # This is P in the paper
            filling_scheme=filling_scheme,
            bunch_selection=bunch_selection,
            with_compressed_profile=False
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
            moments=self.slicer.moments
        )

    def track(self, particles, _slice_result=None, _other_bunch_slicers=None):
        super().track(particles=particles,
                      _slice_result=_slice_result,
                      _other_bunch_slicers=_other_bunch_slicers
                      )

        self.i_turn += 1

        if self.monitor_bunches:
            if self.bunch_buffer is None:
                self.bunch_buffer = self._init_bunch_buffer()

            self._update_bunch_buffer(particles)

            if (self.i_turn % self.flush_data_every == 0 and
                    self.base_file_name is not None):
                self.flush_buffer_to_file_func(
                    self.bunch_buffer, self._bunches_filename)
                self.bunch_buffer = self._init_bunch_buffer()

        if self.monitor_slices:
            if self.slice_buffer is None:
                self.slice_buffer = self._init_slice_buffer()

            self._update_slice_buffer(particles)

            if (self.i_turn % self.flush_data_every == 0 and
                    self.base_file_name is not None):
                self.flush_buffer_to_file_func(
                    self.slice_buffer, self._slices_filename)
                self.slice_buffer = self._init_slice_buffer()

        if self.monitor_particles:
            if self.particle_buffer is None:
                self.particle_buffer = self._init_particle_buffer()

            self._update_particle_buffer(particles)

            if (self.i_turn % self.flush_data_every == 0 and
                    self.base_file_name is not None):
                self.flush_buffer_to_file_func(
                    self.particle_buffer, self._particles_filename)
                self.particle_buffer = self._init_particle_buffer()

    def _init_bunch_buffer(self):
        buf = {}
        for bid in self.slicer.bunch_selection:
            # Convert to int to avoid json serialization issues
            buf[int(bid)] = {}
            for stats in self.stats_to_store:
                buf[int(bid)][stats] = np.zeros(self.flush_data_every)

        return buf

    def _init_slice_buffer(self):
        buf = {}
        for bid in self.slicer.bunch_selection:
            # Convert to int to avoid json serialization issues
            buf[int(bid)] = {}
            for stats in self.stats_to_store:
                buf[int(bid)][stats] = np.zeros((self.flush_data_every,
                                                 self.slicer.num_slices))

        return buf

    def _init_particle_buffer(self):
        num_particles = np.sum(self.particle_monitor_mask)
        buf = {}

        for stat in self.stats_to_store_particles:
            buf[stat] = np.zeros((self.flush_data_every, num_particles))

        return buf

    def _update_bunch_buffer(self, particles):
        i_bunch_particles = self._slice_result['i_slot_particles']

        for i_bunch, bid in enumerate(self.slicer.bunch_selection):
            bunch_mask = (i_bunch_particles == bid)
            beta = np.mean(particles.beta0[bunch_mask])
            gamma = 1 / np.sqrt(1 - beta**2)
            beta_gamma = beta * gamma
            for stat in self.stats_to_store:
                if stat == 'num_particles':
                    if len(self.slicer.bunch_selection) > 1:
                        val = np.sum(self.slicer.num_particles[i_bunch, :])
                    else:
                        val = np.sum(self.slicer.num_particles)
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
                                                            mom_p[bunch_mask])))
                               * beta_gamma)
                    elif stat == 'num_particles':
                        val = np.sum(self.slicer.num_particles[i_bunch, :])
                    else:
                        raise ValueError('Unknown statistics f{stat}')

                self.bunch_buffer[bid][stat][self.i_turn %
                                             self.flush_data_every] = val

    def _update_slice_buffer(self, particles):
        i_bunch_particles = self._slice_result['i_slot_particles']

        for i_bunch, bid in enumerate(self.slicer.bunch_selection):
            bunch_mask = (i_bunch_particles == bid)
            # we use the bunch beta_gamma to calculate the emittance, while
            # we should use the slice beta_gamma
            beta = np.mean(particles.beta0[bunch_mask])
            gamma = 1 / np.sqrt(1 - beta**2)
            beta_gamma = beta * gamma
            for stat in self.stats_to_store:
                mom_str = stat.split('_')[-1]
                if stat.startswith('mean'):
                    if len(self.slicer.bunch_selection) > 1:
                        val = self.slicer.mean(mom_str)[i_bunch, :]
                    else:
                        val = self.slicer.mean(mom_str)
                elif stat.startswith('sigma'):
                    if len(self.slicer.bunch_selection) > 1:
                        val = self.slicer.std(mom_str)[i_bunch, :]
                    else:
                        val = self.slicer.std(mom_str)
                elif stat.startswith('epsn'):
                    if mom_str != 'zeta':
                        mom_p_str = 'p' + stat.split('_')[-1]
                    else:
                        mom_p_str = 'delta'

                    if len(self.slicer.bunch_selection) > 1:
                        val = (np.sqrt(self.slicer.var(mom_str)[i_bunch, :] *
                                    self.slicer.var(mom_p_str)[i_bunch, :] -
                                    self.slicer.cov(mom_str, mom_p_str)[i_bunch,
                                    :]) *
                            beta_gamma)
                    else:
                        val = (np.sqrt(self.slicer.var(mom_str) *
                                    self.slicer.var(mom_p_str) -
                                    self.slicer.cov(mom_str, mom_p_str)) *
                            beta_gamma)
                elif stat == 'num_particles':
                    if len(self.slicer.bunch_selection) > 1:
                        val = self.slicer.num_particles[i_bunch, :]
                    else:
                        val = self.slicer.num_particles
                else:
                    raise ValueError('Unknown statistics f{stat}')

                self.slice_buffer[bid][stat][self.i_turn %
                                             self.flush_data_every, :] = val

    def _update_particle_buffer(self, particles):
        for stat in self.stats_to_store_particles:
            if (self.particle_monitor_mask is not None
                and len(self.particle_monitor_mask) != len(particles.x)):
                raise ValueError('The length of the particle monitor mask is '
                                 'different from the number of particles being tracked')
            val = getattr(particles, stat)[self.particle_monitor_mask]
            self.particle_buffer[stat][self.i_turn %
                                       self.flush_data_every, :] = val


def flush_buffer_to_file_hdf5(buffer, filename):
    try:
        import h5py
    except ImportError:
        raise ImportError('h5py is required to save the data in hdf5 format')
    # Iif the file does not exist, create it and write the buffer
    if not os.path.exists(filename):
        with h5py.File(filename, 'w') as f:
            for key in buffer:
                if type(buffer[key]) is not dict:
                    if len(buffer[key].shape) == 1:
                        f.create_dataset(str(key),
                                         data=buffer[key],
                                         chunks=True, maxshape=(None,))
                    elif len(buffer[key].shape) == 2:
                        f.create_dataset(str(key),
                                         data=buffer[key],
                                         chunks=True,
                                         maxshape=(None, buffer[key].shape[1]))
                    else:
                        raise ValueError('The shape of the buffer is not '
                                         'supported')
                else:
                    # this is for the bunch or slice buffer
                    for subkey in buffer[key]:
                        if len(buffer[key][subkey].shape) == 1:
                            f.create_dataset(str(key) + '/' + str(subkey),
                                             data=buffer[key][subkey],
                                             chunks=True, maxshape=(None,))
                        elif len(buffer[key][subkey].shape) == 2:
                            f.create_dataset(str(key) + '/' + str(subkey),
                                             data=buffer[key][subkey],
                                             chunks=True,
                                             maxshape=(None,
                                                       buffer[key][subkey].shape[1]))
                        else:
                            raise ValueError('The shape of the buffer is not '
                                             'supported')

    # If the file exists, append the buffer
    else:
        with h5py.File(filename, 'a') as f:
            for key in buffer:
                if type(buffer[key]) is not dict:
                    # this is for the particle buffer
                    f[str(key)].resize((f[str(key)].shape[0] +
                                        buffer[key].shape[0]), axis=0)
                    f[str(key)][-buffer[key].shape[0]:] = buffer[key]
                else:
                    # this is for the bunch or slice buffer
                    for subkey in buffer[key]:
                        f[str(key) + '/' + str(subkey)].resize(
                            (f[str(key) + '/' + str(subkey)].shape[0] +
                             buffer[key][subkey].shape[0]), axis=0)
                        val = buffer[key][subkey]
                        f[str(key) + '/' +
                          str(subkey)][-buffer[key][subkey].shape[0]:] = val


def flush_buffer_to_file_json(buffer, filename):
    # To prepare the buffer for json, we need to convert the numpy arrays to
    # lists
    buffer_lists = {}
    for key in buffer:
        if type(buffer[key]) is not dict:
            # this is for the particle buffer
            buffer_lists[str(key)] = buffer[key].tolist()
        else:
            # this is for the bunch or slice buffer
            buffer_lists[str(key)] = {}
            for subkey in buffer[key]:
                buffer_lists[str(key)][subkey] = buffer[key][subkey].tolist()

    # Now we can write the buffer to the file
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            json.dump(buffer_lists, f, indent=2)
    else:
        with open(filename, 'r') as f:
            old_buffer = json.load(f)
        for key in buffer_lists:
            if type(buffer_lists[key]) is not dict:
                # this is for the particle buffer
                old_buffer[key] = np.concatenate((old_buffer[key], buffer_lists[key]), axis=0).tolist()
            else:
                # this is for the bunch or slice buffer
                for subkey in buffer_lists[key]:
                    old_buffer[key][subkey] = np.concatenate(
                        (old_buffer[key][subkey], buffer_lists[key][subkey]),
                        axis=0).tolist()

        with open(filename, 'w') as f:
            json.dump(old_buffer, f, indent=2)
