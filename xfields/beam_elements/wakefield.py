import numpy as np

from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.interpolate import interp1d

import xtrack as xt
import xfields as xf
from xfields.slicers.compressed_profile import CompressedProfile

from scipy.signal import convolve
from matplotlib import pyplot as plt

class MultiWakefield:

    def __init__(self, wakefields,
                zeta_range=None, # These are [a, b] in the paper
                num_slices=None, # Per bunch, this is N_1 in the paper
                bunch_spacing_zeta=None, # This is P in the paper
                num_slots=None,
                filling_scheme=None,
                bunch_numbers=None,
                num_turns=1,
                circumference=None,
                log_moments=None,
                _flatten=False):

        self.wakefields = wakefields
        
        if filling_scheme is None and bunch_numbers is None:
            if num_slots is None:
                    num_slots = 1
            filling_scheme = np.ones(num_slots,dtype=np.int64)
            bunch_numbers = np.arange(num_slots,dtype=np.int64)
        else:
            assert num_slots is None and filling_scheme is not None and bunch_numbers is not None
            num_slots = len(bunch_numbers)

        all_slicer_moments = []
        for wf in self.wakefields:
            assert wf.moments_data is None
            wf._initialize_moments_and_conv_data(
                    zeta_range=zeta_range, # These are [a, b] in the paper
                    num_slices=num_slices, # Per bunch, this is N_1 in the paper
                    bunch_spacing_zeta=bunch_spacing_zeta, # This is P in the paper
                    filling_scheme=filling_scheme,
                    bunch_numbers=bunch_numbers,
                    circumference=circumference,
                    log_moments=log_moments,
                    _flatten=_flatten)
            all_slicer_moments += wf.slicer.moments

        all_slicer_moments = list(set(all_slicer_moments))

        self.slicer = xf.UniformBinSlicer(
            zeta_range=zeta_range,
            num_slices=num_slices,
            filling_scheme=filling_scheme,
            bunch_numbers=bunch_numbers,
            bunch_spacing_zeta=bunch_spacing_zeta,
            moments=all_slicer_moments
            )
            
        self.pipeline_manager = None
    
    @classmethod
    def from_table(cls, wake_file, wake_file_columns,
                use_components=None, beta0 = 1.0, **kwargs):
        """ Load data from the wake_file and store them in a dictionary
        self.wake_table. Keys are the names specified by the user in
        wake_file_columns and describe the names of the wake field
        components (e.g. dipole_x or dipole_yx). The dict values are
        given by the corresponding data read from the table. The
        nomenclature of the wake components must be strictly obeyed.
        Valid names for wake components are:

        'constant_x', 'constant_y', 'dipole_x', 'dipole_y', 'dipole_xy',
        'dipole_yx', 'quadrupole_x', 'quadrupole_y', 'quadrupole_xy',
        'quadrupole_yx', 'longitudinal'.

        The order of wake_file_columns is relevant and must correspond
        to the one in the wake_file. There is no way to check this here
        and it is in the responsibility of the user to ensure it is
        correct. Two checks made here are whether the length of
        wake_file_columns corresponds to the number of columns in the
        wake_file and whether a column 'time' is specified.

        The units and signs of the wake table data are assumed to follow
        the HEADTAIL conventions, i.e.
          time: [ns]
          transverse wake components: [V/pC/mm]
          longitudinal wake component: [V/pC].
        """

        valid_wake_components = ['constant_x', 'constant_y', 'dipole_x', 'dipole_y', 'dipole_xy',
            'dipole_yx', 'quadrupole_x', 'quadrupole_y', 'quadrupole_xy',
            'quadrupole_yx', 'longitudinal']
        
        wake_data = np.loadtxt(wake_file)
        if len(wake_file_columns) != wake_data.shape[1]:
            raise ValueError("Length of wake_file_columns list does not" +
                             " correspond to the number of columns in the" +
                             " specified wake_file. \n")
        if 'time' not in wake_file_columns:
            raise ValueError("No wake_file_column with name 'time' has" +
                             " been specified. \n")
        itime = wake_file_columns.index('time')
        wake_distance = -1E-9 * wake_data[:,itime] * beta0 * clight
        wakefields = []
        for iwake_component,wake_component in enumerate(wake_file_columns):
            if iwake_component != itime and (use_components is None or wake_component in use_components):
                assert wake_component in valid_wake_components
                scale_kick = None
                source_moments = ['num_particles']
                if wake_component == 'longitudinal':
                    source_moments.append('zeta')
                    kick = 'delta'
                else:
                    tokens = wake_component.split('_')
                    coord_target = tokens[1][0]
                    if len(tokens[1]) == 2:
                        coord_source = tokens[1][1]
                    else:
                        coord_source = coord_target
                    kick = 'p'+coord_target
                    if tokens[0] == 'dipole':
                        source_moments.append(coord_source)
                    elif tokens[0] == 'quadrupole':
                        scale_kick = coord_source
                wake_strength = -1E15 * wake_data[:,iwake_component]
                wakefield = xf.Wakefield(
                    source_moments=source_moments,
                    kick=kick,
                    scale_kick=scale_kick,
                    function=interp1d(wake_distance, wake_strength,bounds_error=False,fill_value=0.0)
                )
                wakefields.append(wakefield)
        return MultiWakefield(wakefields,**kwargs)
        
    def init_pipeline(self,pipeline_manager,element_name,partners_names):
        self.pipeline_manager = pipeline_manager
        for wf in self.wakefields:
            assert wf.pipeline_manager == None
        self.partners_names = partners_names
        self.name = element_name
        
        self._send_buffer = self.slicer._to_npbuffer()
        self._send_buffer_length = np.zeros(1,dtype=int)
        self._send_buffer_length[0] = len(self._send_buffer)

        self._recv_buffer = np.zeros_like(self._send_buffer)
        self._recv_buffer_length_buffer = np.zeros(1,dtype=int)
        
    def _slicer_to_buffer(self,slicer):
        self._send_buffer = slicer._to_npbuffer()
        self._send_buffer_length[0] = len(slicer._to_npbuffer())
        
    def _ensure_recv_buffer_size(self):
        if self._recv_buffer_length_buffer[0] != len(self._recv_buffer):
            self._recv_buffer = np.zeros(self._recv_buffer_length_buffer[0]
                                          ,dtype=self._recv_buffer.dtype)
        
    def _slice_set_from_buffer(self):
        return xf.UniformBinSlicer._from_npbuffer(self._recv_buffer)
    
    def _slice_and_store(self,particles):
        self.i_slice_particles = particles.particle_id * 0 + -999
        self.i_bunch_particles = particles.particle_id * 0 + -9999
        self.slicer.slice(particles, i_slice_particles=self.i_slice_particles,
                        i_bunch_particles=self.i_bunch_particles)
                        
    def track(self, particles):
        _slice_result = None
        if self.pipeline_manager is None:
            self._slice_and_store(particles)
            other_bunches_slicers = None
        else:
            other_bunches_slicers = []
            is_ready_to_send = True
            for i_partner,partner_name in enumerate(self.partners_names):
                if not self.pipeline_manager.is_ready_to_send(self.name,particles.name,partner_name,particles.at_turn[0],internal_tag=0):
                    is_ready_to_send = False
                    break
            if is_ready_to_send:
                self._slice_and_store(particles)
                self._slicer_to_buffer(self.slicer)
                for i_partner,partner_name in enumerate(self.partners_names):
                    self.pipeline_manager.send_message(self._send_buffer_length,element_name=self.name,sender_name=particles.name,reciever_name=partner_name,turn=particles.at_turn[0],internal_tag=0)
                    self.pipeline_manager.send_message(self._send_buffer,element_name=self.name,sender_name=particles.name,reciever_name=partner_name,turn=particles.at_turn[0],internal_tag=1)
            for i_partner,partner_name in enumerate(self.partners_names):
                if not self.pipeline_manager.is_ready_to_recieve(self.name,partner_name,particles.name,internal_tag=0):
                    return xt.PipelineStatus(on_hold=True)
        if self.pipeline_manager is not None:
            for i_partner,partner_name in enumerate(self.partners_names):
                self.pipeline_manager.recieve_message(self._recv_buffer_length_buffer,self.name,partner_name,particles.name,internal_tag=0)
                self._ensure_recv_buffer_size()
                self.pipeline_manager.recieve_message(self._recv_buffer,self.name,partner_name,particles.name,internal_tag=1)
                other_bunches_slicers.append(self._slice_set_from_buffer())

        _slice_result = {'i_slice_particles': self.i_slice_particles,
                        'i_bunch_particles': self.i_bunch_particles,
                        'slicer': self.slicer}

        for wf in self.wakefields:
            wf.track(particles, _slice_result=_slice_result,_other_bunches_slicers=other_bunches_slicers)


class Wakefield:

    def __init__(self,
                source_moments,
                kick,
                scale_kick,
                function,
                zeta_range=None, # These are [a, b] in the paper
                num_slices=None, # Per bunch, this is N_1 in the paper
                bunch_spacing_zeta=None, # This is P in the paper
                num_slots=None,
                filling_scheme=None,
                bunch_numbers=None,
                num_turns=1,
                circumference=None,
                log_moments=None,
                _flatten=False):

        self._flatten = _flatten

        assert isinstance(source_moments, (list, tuple))
        assert isinstance(log_moments, (list, tuple)) or log_moments is None

        self.kick = kick
        self.scale_kick = scale_kick
        self.source_moments = source_moments
        self.function = function
        self.moments_data = None

        if filling_scheme is None and bunch_numbers is None:
            if num_slots is None:
                    num_slots = 1
            filling_scheme = np.ones(num_slots,dtype=np.int64)
            bunch_numbers = np.arange(num_slots,dtype=np.int64)
        else:
            assert num_slots is None and filling_scheme is not None and bunch_numbers is not None
            num_slots = len(bunch_numbers)

        if zeta_range is not None:
            self._initialize_moments_and_conv_data(
                    zeta_range=zeta_range, # These are [a, b] in the paper
                    num_slices=num_slices, # Per bunch, this is N_1 in the paper
                    bunch_spacing_zeta=bunch_spacing_zeta, # This is P in the paper
                    filling_scheme=filling_scheme,
                    bunch_numbers=bunch_numbers,
                    num_turns=num_turns,
                    circumference=circumference,
                    log_moments=log_moments,
                    _flatten=_flatten)
                    
        self.pipeline_manager = None
        
    def init_pipeline(self,pipeline_manager,element_name,partners_names):
        self.pipeline_manager = pipeline_manager
        self.partners_names = partners_names
        self.name = element_name
        
        self._send_buffer = self.slicer._to_npbuffer()
        self._send_buffer_length = np.zeros(1,dtype=int)
        self._send_buffer_length[0] = len(self._send_buffer)

        self._recv_buffer = np.zeros_like(self._send_buffer)
        self._recv_buffer_length_buffer = np.zeros(1,dtype=int)
                            
    def _slicer_to_buffer(self,slicer):
        self._send_buffer = slicer._to_npbuffer()
        self._send_buffer_length[0] = len(slicer._to_npbuffer())
        
    def _ensure_recv_buffer_size(self):
        if self._recv_buffer_length_buffer[0] != len(self._recv_buffer):
            self._recv_buffer = np.zeros(self._recv_buffer_length_buffer[0]
                                          ,dtype=self._recv_buffer.dtype)
        
    def _slice_set_from_buffer(self):
        return xf.UniformBinSlicer._from_npbuffer(self._recv_buffer)
    
    def _initialize_moments_and_conv_data(self,
                zeta_range=None, # These are [a, b] in the paper
                num_slices=None, # Per bunch, this is N_1 in the paper
                bunch_spacing_zeta=None, # This is P in the paper
                filling_scheme=None,
                bunch_numbers=None,
                num_turns=1,
                circumference=None,
                log_moments=None,
                _flatten=False):

        slicer_moments = self.source_moments.copy()
        if log_moments is not None:
            slicer_moments += log_moments
        slicer_moments = list(set(slicer_moments))
        if 'num_particles' in slicer_moments:
            slicer_moments.remove('num_particles')

        self.slicer = xf.UniformBinSlicer(
            zeta_range=zeta_range,
            num_slices=num_slices,
            filling_scheme=filling_scheme,
            bunch_numbers=bunch_numbers,
            bunch_spacing_zeta=bunch_spacing_zeta,
            moments=slicer_moments
            )

        self.moments_data = CompressedProfile(
                moments=self.source_moments + ['result'],
                zeta_range=zeta_range,
                num_slices=num_slices,
                bunch_spacing_zeta=bunch_spacing_zeta,
                num_periods=len(filling_scheme),
                num_turns=num_turns,
                circumference=circumference)

        if not _flatten:

            self._N_aux = self.moments_data._N_aux
            self._M_aux = self.moments_data._M_aux
            self._N_S = self.moments_data._N_S
            self._N_T = self._N_S
            self._BB = 1 # B in the paper
                        # (for now we assume that B=0 is the first bunch in time
                        # and the last one in zeta)
            self._AA = self._BB - self._N_S
            self._CC = self._AA
            self._DD = self._BB

            # Build wake matrix
            self.z_wake = _build_z_wake(self._z_a, self._z_b, self.num_turns,
                        self._N_aux, self._M_aux,
                        self.circumference, self.dz, self._AA, self._BB, self._CC,
                        self._DD, self._z_P)

            self.G_aux = self.function(self.z_wake)

            phase_term = np.exp(1j * 2 * np.pi * np.arange(self._M_aux//2 + 1) # only positive frequencies because we are using rfft
                            * ((self._N_S - 1)* self._N_aux + self._N_1)
                            / self._M_aux)

        else:
            self._N_S_flatten = self.moments_data._N_S * self.num_turns
            self._N_T_flatten = self.moments_data._N_S
            self._N_aux = self.moments_data._N_aux
            self._M_aux_flatten = ((self._N_S_flatten + self._N_T_flatten - 1)
                                   * self._N_aux)
            self._BB_flatten = 1 # B in the paper
            # (for now we assume that B=0 is the first bunch in time
            # and the last one in zeta)
            self._AA_flatten = self._BB_flatten - self._N_S_flatten
            self._CC_flatten = self._AA_flatten
            self._DD_flatten = self._AA_flatten + self._N_T_flatten

            # Build wake matrix
            self.z_wake = _build_z_wake(self._z_a, self._z_b,
                        1, # num_turns
                        self._N_aux, self._M_aux_flatten,
                        0, # circumference, does not matter since we are doing one pass
                        self.dz,
                        self._AA_flatten, self._BB_flatten,
                        self._CC_flatten, self._DD_flatten, self._z_P)
            self.G_aux = self.function(self.z_wake)

            phase_term = np.exp(1j * 2 * np.pi
                            * np.arange(self._M_aux_flatten//2 + 1) # only positive frequencies because we are using rfft
                            * ((self._N_S_flatten - 1) * self._N_aux + self._N_1)
                            / self._M_aux_flatten)

        self._G_hat_dephased = phase_term * np.fft.rfft(self.G_aux, axis=1)
        self._G_aux_shifted = np.fft.irfft(self._G_hat_dephased, axis=1)

    def _slice_and_store(self,particles,_slice_result):
        if _slice_result is not None:
            self.i_slice_particles = _slice_result['i_slice_particles']
            self.i_bunch_particles = _slice_result['i_bunch_particles']
            self.slicer = _slice_result['slicer']
        else:
            # Measure slice moments and get slice indeces
            self.i_slice_particles = particles.particle_id * 0 + -999
            self.i_bunch_particles = particles.particle_id * 0 + -9999
            self.slicer.slice(particles, i_slice_particles=self.i_slice_particles,
                            i_bunch_particles=self.i_bunch_particles)
        
    def _add_slicer_moments_to_moments_data(self,slicer):
        # Set slice moments for fast convolution
        means = {}
        for mm in self.moments_data.moments_names:
            if mm == 'num_particles' or mm == 'result':
                continue
            means[mm] = slicer.mean(mm)

        for i_bunch_in_slicer,bunch_number in enumerate(slicer.bunch_numbers):
            moments_bunch = {}
            for nn in means.keys():
                moments_bunch[nn] = means[nn][i_bunch_in_slicer, :]
            moments_bunch['num_particles'] = (
                slicer.num_particles[i_bunch_in_slicer, :])
            self.moments_data.set_moments(moments=moments_bunch,
                                        i_turn=0, i_source=slicer.filled_slots[bunch_number])

    def _update_moments_for_new_turn(self,particles, _slice_result = None):
        # Trash oldest turn
        self.moments_data.data[:, 1:, :] = self.moments_data.data[:, :-1, :]
        self.moments_data.data[:, 0, :] = 0

        self._slice_and_store(particles,_slice_result)
        self._add_slicer_moments_to_moments_data(self.slicer)
        
    def track(self, particles, _slice_result=None,_other_bunches_slicers=None):
        if self.moments_data is None:
            raise ValueError('moments_data is None. '
                             'Please initialize it before tracking.')
        
        if self.pipeline_manager is None:
            self._update_moments_for_new_turn(particles, _slice_result)
        else:
            is_ready_to_send = True
            for i_partner,partner_name in enumerate(self.partners_names):
                if not self.pipeline_manager.is_ready_to_send(self.name,particles.name,partner_name,particles.at_turn[0],internal_tag=0):
                    is_ready_to_send = False
                    break
            if is_ready_to_send:
                self._update_moments_for_new_turn(particles, _slice_result)
                self._slicer_to_buffer(self.slicer)
                for i_partner,partner_name in enumerate(self.partners_names):
                    self.pipeline_manager.send_message(self._send_buffer_length,element_name=self.name,sender_name=particles.name,reciever_name=partner_name,turn=particles.at_turn[0],internal_tag=0)
                    self.pipeline_manager.send_message(self._send_buffer,element_name=self.name,sender_name=particles.name,reciever_name=partner_name,turn=particles.at_turn[0],internal_tag=1)
            for i_partner,partner_name in enumerate(self.partners_names):
                if not self.pipeline_manager.is_ready_to_recieve(self.name,partner_name,particles.name,internal_tag=0):
                    return xt.PipelineStatus(on_hold=True)
        if self.pipeline_manager is not None:
            for i_partner,partner_name in enumerate(self.partners_names):
                self.pipeline_manager.recieve_message(self._recv_buffer_length_buffer,self.name,partner_name,particles.name,internal_tag=0)
                self._ensure_recv_buffer_size()
                self.pipeline_manager.recieve_message(self._recv_buffer,self.name,partner_name,particles.name,internal_tag=1)
                other_bunch_slicer = self._slice_set_from_buffer()
                self._add_slicer_moments_to_moments_data(other_bunch_slicer)
        
        if _other_bunches_slicers is not None:
            for other_bunch_slicer in _other_bunches_slicers:
                self._add_slicer_moments_to_moments_data(other_bunch_slicer)

        # Compute convolution
        self._compute_convolution(moment_names=self.source_moments)
        # Apply kicks
        interpolated_result = particles.zeta * 0
        assert self.moments_data.moments_names[-1] == 'result'
        md = self.moments_data
        self.moments_data._interp_result(particles=particles,
                    data_shape_0=md.data.shape[0],
                    data_shape_1=md.data.shape[1],
                    data_shape_2=md.data.shape[2],
                    data=md.data,
                    i_bunch_particles=self.i_bunch_particles,
                    i_slice_particles=self.i_slice_particles,
                    out=interpolated_result)
        # interpolated result will be zero for lost particles (so nothing to do for them)
        scaling_constant = -particles.q0**2 * qe**2 / (particles.p0c * qe)

        if self.scale_kick is not None:
            scaling_constant *= getattr(particles, self.scale_kick)

        getattr(particles, self.kick)[:] += scaling_constant * interpolated_result # remember to handle lost particles!!!

    def _compute_convolution(self, moment_names):

        if isinstance(moment_names, str):
            moment_names = [moment_names]

        rho_aux = np.ones(shape=self.moments_data['result'].shape,
                        dtype=np.float64)

        for nn in moment_names:
            rho_aux *= self.moments_data[nn]

        if not self._flatten:
            rho_hat = np.fft.rfft(rho_aux, axis=1)
            res = np.fft.irfft(rho_hat * self._G_hat_dephased, axis=1)
        else:
            rho_aux_flatten = np.zeros((1, self._M_aux_flatten), dtype=np.float64)
            _N_aux_turn = self.moments_data._N_S * self._N_aux
            for tt in range(self.num_turns):
                rho_aux_flatten[
                    0, tt * _N_aux_turn: (tt + 1) * _N_aux_turn] = \
                        rho_aux[tt, :_N_aux_turn]

            rho_hat_flatten = np.fft.rfft(rho_aux_flatten, axis=1)
            res_flatten = np.fft.irfft(
                rho_hat_flatten * self._G_hat_dephased, axis=1).real
            self._res_flatten_fft = res_flatten # for debugging

            # # The following is faster in some cases, we might go back to it in the future
            # res_flatten = fftconvolve(rho_aux_flatten, self._G_aux_shifted, mode='full')
            # self._res_flatten_full = res_flatten # for debugging
            # res_flatten = res_flatten[:, -len(rho_aux_flatten[0, :])+1:]

            res = rho_aux * 0
            res[0, :_N_aux_turn] = res_flatten[0, :_N_aux_turn] # Here we cannot separate the effect of the different turns
                                                                # We put everything in one turn and leave the rest to be zero

            self._res_flatten = res_flatten # for debugging
            self._rho_flatten = rho_aux_flatten # for debugging

        self.moments_data['result'] = res.real

    # Parameters from CompressedProfile
    @property
    def _N_1(self):
        return self.moments_data._N_1

    @property
    def _N_2(self):
        return self.moments_data._N_2


    @property
    def _z_a(self):
        return self.moments_data._z_a

    @property
    def _z_b(self):
        return self.moments_data._z_b

    @property
    def z_period(self):
        return self.moments_data.z_period

    @property
    def _z_P(self):
        return self.moments_data._z_P

    @property
    def circumference(self):
        return self.moments_data.circumference

    @property
    def dz(self):
        return self.moments_data.dz

    @property
    def num_slices(self):
        return self.moments_data.num_slices

    @property
    def num_periods(self):
        return self.moments_data.num_periods

    @property
    def num_turns(self):
        return self.moments_data.num_turns


    def set_moments(self, i_source, i_turn, moments):

        """
        Set the moments for a given source and turn.

        Parameters
        ----------
        i_source : int
            The source index, 0 <= i_source < self.num_periods
        i_turn : int
            The turn index, 0 <= i_turn < self.num_turns
        moments : dict
            A dictionary of the form {moment_name: moment_value}

        """

        self.moments_data.set_moments(i_source, i_turn, moments)

    def get_moment_profile(self, moment_name, i_turn):

        '''
        Get the moment profile for a given turn.

        Parameters
        ----------
        moment_name : str
            The name of the moment to get
        i_turn : int
            The turn index, 0 <= i_turn < self.num_turns

        Returns
        -------
        z_out : np.ndarray
            The z positions within the moment profile
        moment_out : np.ndarray
            The moment profile
        '''

        z_out, moment_out = self.moments_data.get_moment_profile(
                moment_name, i_turn)

        return z_out, moment_out


class TempResonatorFunction:
    def __init__(self, R_shunt, frequency, Q):
        self.R_shunt = R_shunt
        self.frequency = frequency
        self.Q = Q

    def __call__(self, z):
        R_s = self.R_shunt
        Q = self.Q
        f_r = self.frequency
        omega_r = 2 * np.pi * f_r
        alpha_t = omega_r / (2 * Q)
        omega_bar = np.sqrt(omega_r**2 - alpha_t**2)

        res = (z < 0) * (R_s * omega_r**2 / (Q * omega_bar)
               * np.exp(alpha_t * z / clight)
                * np.sin(omega_bar * z / clight))# Wake definition
        return res

def _build_z_wake(z_a, z_b, num_turns, N_aux, M_aux, circumference, dz,
                 AA, BB, CC, DD, z_P):
    z_c = z_a # For wakefield, z_c = z_a
    z_d = z_b # For wakefield, z_d = z_b
    z_wake = np.zeros((num_turns, M_aux))
    for tt in range(num_turns):
        z_a_turn = z_a + tt * circumference
        z_b_turn = z_b + tt * circumference
        temp_z = np.arange(
            z_c - z_b_turn, z_d - z_a_turn + dz/10, dz)[:-1]
        for ii, ll in enumerate(range(
                            CC - BB + 1, DD - AA)):
            z_wake[tt, ii*N_aux:(ii+1)*N_aux] = temp_z + ll * z_P
    return z_wake
