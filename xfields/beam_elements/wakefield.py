import numpy as np
import pandas as pd

from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.interpolate import interp1d

import xobjects as xo
import xfields as xf
from .element_with_slicer import ElementWithSlicer


class Wakefield(ElementWithSlicer):
    """
    An object handling many WakeField instances as a single beam element.

    Parameters
    ----------
    components : xfields.WakeField
        List of wake fields.
    zeta_range : Tuple
        Zeta range for each bunch used in the underlying slicer.
    num_slices : int
        Number of slices per bunch used in the underlying slicer.
    bunch_spacing_zeta : float
        Bunch spacing in meters.
    filling_scheme: np.ndarray
        List of zeros and ones representing the filling scheme. The length
        of the array is equal to the number of slots in the machine and each
        element of the array holds a one if the slot is filled or a zero
        otherwise.
    bunch_numbers: np.ndarray
        List of the bunches indicating which slots from the filling scheme are
        used (not all the bunches are used when using multi-processing)
    num_turns : int
        Number of turns which are consiered for the multi-turn wake.
    circumference: float
        Machine length in meters.
    log_moments: list
        List of moments logged in the slicer.
    _flatten: bool
        Use flattened wakes
    """

    def __init__(self, components,
                 zeta_range=None,  # These are [a, b] in the paper
                 num_slices=None,  # Per bunch, this is N_1 in the paper
                 bunch_spacing_zeta=None,  # This is P in the paper
                 filling_scheme=None,
                 bunch_numbers=None,
                 num_turns=1,
                 circumference=None,
                 log_moments=None,
                 _flatten=False):

        self.components = components
        self.pipeline_manager = None

        all_slicer_moments = []
        for cc in self.components:
            assert cc.moments_data is None
            all_slicer_moments += cc.source_moments

        self.all_slicer_moments = list(set(all_slicer_moments))

        super().__init__(
            slicer_moments=all_slicer_moments,
            log_moments=log_moments,
            zeta_range=zeta_range,  # These are [a, b] in the paper
            num_slices=num_slices,  # Per bunch, this is N_1 in the paper
            bunch_spacing_zeta=bunch_spacing_zeta,  # This is P in the paper
            filling_scheme=filling_scheme,
            bunch_numbers=bunch_numbers,
            num_turns=num_turns,
            circumference=circumference,
            _flatten=False,
            with_compressed_profile=True
        )

        self._initialize_moments(
            zeta_range=zeta_range,  # These are [a, b] in the paper
            num_slices=num_slices,  # Per bunch, this is N_1 in the paper
            bunch_spacing_zeta=bunch_spacing_zeta,  # This is P in the paper
            filling_scheme=filling_scheme,
            num_turns=num_turns,
            circumference=circumference)

        for cc in self.components:
            cc._initialize_conv_data(_flatten=_flatten,
                                     moments_data=self.moments_data)

        all_slicer_moments = list(set(all_slicer_moments))

    @classmethod
    def from_table(cls, df, use_components, beta0=1.0,
                   **kwargs):
        valid_wake_components = ['constant_x', 'constant_y', 'dipole_x',
                                 'dipole_y', 'dipole_xy', 'dipole_yx',
                                 'quadrupole_x', 'quadrupole_y',
                                 'quadrupole_xy', 'quadrupole_yx',
                                 'longitudinal']

        if 'time' not in list(df.keys()):
            raise ValueError("No wake_file_column with name 'time' has" +
                             " been specified. \n")

        if use_components is not None:
            for component in use_components:
                assert component in valid_wake_components

        wake_distance = df['time'] * beta0 * clight
        components = []
        for component in list(df.keys()):
            if component != 'time' and (use_components is None or
                                        component in use_components):
                assert component in valid_wake_components
                scale_kick = None
                source_moments = ['num_particles']
                if component == 'longitudinal':
                    kick = 'delta'
                else:
                    tokens = component.split('_')
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
                wake_strength = df[component]
                wakefield = xf.WakeComponent(
                    source_moments=source_moments,
                    kick=kick,
                    scale_kick=scale_kick,
                    function=interp1d(wake_distance, wake_strength,
                                      bounds_error=False, fill_value=0.0)
                )
                components.append(wakefield)
        return cls(components, **kwargs)

    @staticmethod
    def table_from_headtail_file(wake_file, wake_file_columns, use_components=None,
                                 flag_pyht_units=False):
        valid_wake_components = ['constant_x', 'constant_y', 'dipole_x',
                                 'dipole_y', 'dipole_xy', 'dipole_yx',
                                 'quadrupole_x', 'quadrupole_y',
                                 'quadrupole_xy', 'quadrupole_yx',
                                 'longitudinal']

        wake_data = np.loadtxt(wake_file)
        if len(wake_file_columns) != wake_data.shape[1]:
            raise ValueError("Length of wake_file_columns list does not" +
                             " correspond to the number of columns in the" +
                             " specified wake_file. \n")
        if 'time' not in wake_file_columns:
            raise ValueError("No wake_file_column with name 'time' has" +
                             " been specified. \n")

        if use_components is not None:
            for component in use_components:
                assert component in valid_wake_components
                assert component in wake_file_columns

        dict_components = {}

        conversion_factor_time = -1E-9 if flag_pyht_units else 1

        itime = wake_file_columns.index('time')
        dict_components['time'] = conversion_factor_time * wake_data[:, itime]

        for i_component, component in enumerate(wake_file_columns):
            if component != 'time' and (use_components is None or
                                        component in use_components):
                assert component in valid_wake_components
                if component == 'longitudinal':
                    conversion_factor = -1E12 if flag_pyht_units else 1
                else:
                    conversion_factor = -1E15 if flag_pyht_units else 1

                dict_components[component] = (wake_data[:, i_component] *
                                              conversion_factor)

        df = pd.DataFrame(data=dict_components)

        return df

    def init_pipeline(self, pipeline_manager, element_name, partners_names):

        super().init_pipeline(pipeline_manager=pipeline_manager,
                              element_name=element_name,
                              partners_names=partners_names)

    def track(self, particles):

        # Use common slicer from parent class to measure all moments
        super().track(particles)

        for wf in self.components:
            wf.track(particles,
                     i_bunch_particles=self.i_bunch_particles,
                     i_slice_particles=self.i_slice_particles,
                     moments_data=self.moments_data)

    @property
    def zeta_range(self):
        return (self.slicer.zeta_centers[0] - self.slicer.dzeta / 2,
                self.slicer.zeta_centers[-1] + self.slicer.dzeta / 2)

    @property
    def num_slices(self):
       return self.slicer.num_slices

    @property
    def bunch_spacing_zeta(self):
        return self.slicer.bunch_spacing_zeta

    @property
    def filling_scheme(self):
        assert len(self.slicer.filled_slots) == 1, (
            'Only single bunch mode is supported for now')
        assert self.slicer.filled_slots[0] == 0, (
            'Only single bunch mode is supported for now')
        return None

    @property
    def bunch_numbers(self):
        return self.slicer.bunch_numbers

    @property
    def num_turns(self):
        return self.moments_data.num_turns

    @property
    def circumference(self):
        return self.moments_data.circumference

    def __add__(self, other):

        if other == 0:
            return self

        assert isinstance(other, Wakefield)

        new_components = self.components + other.components

        # Check consistency
        xo.assert_allclose(self.zeta_range, other.zeta_range, atol=1e-12, rtol=0)
        xo.assert_allclose(self.num_slices, other.num_slices, atol=0, rtol=0)
        if self.bunch_spacing_zeta is None:
            assert other.bunch_spacing_zeta is None, (
                'Bunch spacing zeta is not consistent')
        else:
            xo.assert_allclose(self.bunch_spacing_zeta, other.bunch_spacing_zeta, atol=1e-12, rtol=0)
        if self.filling_scheme is None:
            assert other.filling_scheme is None, (
                'Filling scheme is not consistent')
        else:
            xo.assert_allclose(self.filling_scheme, other.filling_scheme, atol=0, rtol=0)
        xo.assert_allclose(self.bunch_numbers, other.bunch_numbers, atol=0, rtol=0)
        xo.assert_allclose(self.num_turns, other.num_turns, atol=0, rtol=0)
        xo.assert_allclose(self.circumference, other.circumference, atol=0, rtol=0)

        return Wakefield(
                 components=new_components,
                 zeta_range=self.zeta_range,
                 num_slices=self.num_slices,
                 bunch_spacing_zeta=self.bunch_spacing_zeta,
                 filling_scheme=self.filling_scheme,
                 bunch_numbers=(self.bunch_numbers if self.filling_scheme else None),
                 num_turns=self.num_turns,
                 circumference=self.circumference
        )

    def __radd__(self, other):
        return self.__add__(other)

class WakeComponent:
    """
    A beam element modelling a wakefield kick

    Parameters
    ----------
    source_moments: list
        List of moment which are used as the wake source (e.g. for an x-dipolar
        wake it is ['num_particles', 'x'], while for a constant or
        quadrupolar wake it is only ['num_particles']).
    kick : str
        Moment to which the kick is applied (e.g. it is 'px' for an x wake, it
        is 'py' for a y wake and it is 'delta' for a longitudinal wake).
    scale_kick:
        Moment by which the wake kick is scaled (e.g. it is None for a constant,
        or dipolar, while it is 'x' for a x-quadrupolar wake).
    log_moments: list
        List of moments logged in the slicer.
    _flatten: bool
        Use flattened wakes
    """

    def __init__(self,
                 source_moments,
                 kick,
                 scale_kick,
                 function,
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

    def _initialize_conv_data(self, _flatten=False, moments_data=None):
        assert moments_data is not None
        if not _flatten:
            self._N_aux = moments_data._N_aux
            self._M_aux = moments_data._M_aux
            self._N_1 = moments_data._N_1
            self._N_S = moments_data._N_S
            self._N_T = moments_data._N_S
            self._BB = 1  # B in the paper
            # (for now we assume that B=0 is the first bunch in time and the
            # last one in zeta)
            self._AA = self._BB - self._N_S
            self._CC = self._AA
            self._DD = self._BB

            # Build wake matrix
            self.z_wake = _build_z_wake(moments_data._z_a, moments_data._z_b,
                                        moments_data.num_turns,
                                        moments_data._N_aux, moments_data._M_aux,
                                        moments_data.circumference,
                                        moments_data.dz, self._AA,
                                        self._BB, self._CC, self._DD,
                                        moments_data._z_P)

            self.G_aux = self.function(self.z_wake)

            # only positive frequencies because we are using rfft
            phase_term = np.exp(
                1j * 2 * np.pi * np.arange(self._M_aux//2 + 1) *
                ((self._N_S - 1) * self._N_aux + self._N_1) / self._M_aux)

        else:
            raise NotImplementedError('Flattened wakes are not implemented yet')
            self._N_S_flatten = moments_data._N_S * moments_data.num_turns
            self._N_T_flatten = moments_data._N_S
            self._N_aux = moments_data._N_aux
            self._M_aux_flatten = ((self._N_S_flatten + self._N_T_flatten - 1)
                                   * self._N_aux)
            self._BB_flatten = 1  # B in the paper
            # (for now we assume that B=0 is the first bunch in time
            # and the last one in zeta)
            self._AA_flatten = self._BB_flatten - self._N_S_flatten
            self._CC_flatten = self._AA_flatten
            self._DD_flatten = self._AA_flatten + self._N_T_flatten

            # Build wake matrix
            self.z_wake = _build_z_wake(
                moments_data._z_a,
                moments_data._z_b,
                1,  # num_turns
                self._N_aux, self._M_aux_flatten,
                0,  # circumference, does not matter since we are doing one pass
                moments_data.dz,
                self._AA_flatten, self._BB_flatten,
                self._CC_flatten, self._DD_flatten, moments_data._z_P)

            self.G_aux = self.function(self.z_wake)

            # only positive frequencies because we are using rfft
            phase_term = np.exp(
                1j * 2 * np.pi * np.arange(self._M_aux_flatten//2 + 1) *
                ((self._N_S_flatten - 1)
                 * self._N_aux + self._N_1) / self._M_aux_flatten)

        self._G_hat_dephased = phase_term * np.fft.rfft(self.G_aux, axis=1)
        self._G_aux_shifted = np.fft.irfft(self._G_hat_dephased, axis=1)

    def track(self, particles, i_bunch_particles, i_slice_particles,
              moments_data):

        # Compute convolution
        self._compute_convolution(moment_names=self.source_moments,
                                  moments_data=moments_data)
        # Apply kicks
        interpolated_result = particles.zeta * 0
        assert moments_data.moments_names[-1] == 'result'
        md = moments_data
        moments_data._interp_result(
            particles=particles,
            data_shape_0=md.data.shape[0],
            data_shape_1=md.data.shape[1],
            data_shape_2=md.data.shape[2],
            data=md.data,
            i_bunch_particles=i_bunch_particles,
            i_slice_particles=i_slice_particles,
            out=interpolated_result)
        # interpolated result will be zero for lost particles (so nothing to
        # do for them)
        scaling_constant = -particles.q0**2 * qe**2 / (particles.p0c * qe)

        if self.scale_kick is not None:
            scaling_constant *= getattr(particles, self.scale_kick)

        # remember to handle lost particles!!!
        getattr(particles, self.kick)[:] += (scaling_constant *
                                             interpolated_result)

    def _compute_convolution(self, moment_names, moments_data):

        if isinstance(moment_names, str):
            moment_names = [moment_names]

        rho_aux = np.ones(shape=moments_data['result'].shape,
                          dtype=np.float64)

        for nn in moment_names:
            rho_aux *= moments_data[nn]

        if not self._flatten:
            rho_hat = np.fft.rfft(rho_aux, axis=1)
            res = np.fft.irfft(rho_hat * self._G_hat_dephased, axis=1)
        else:
            rho_aux_flatten = np.zeros((1, self._M_aux_flatten),
                                       dtype=np.float64)
            _N_aux_turn = moments_data._N_S * self._N_aux
            for tt in range(self.num_turns):
                rho_aux_flatten[
                    0, tt * _N_aux_turn: (tt + 1) * _N_aux_turn] = \
                        rho_aux[tt, :_N_aux_turn]

            rho_hat_flatten = np.fft.rfft(rho_aux_flatten, axis=1)
            res_flatten = np.fft.irfft(
                rho_hat_flatten * self._G_hat_dephased, axis=1).real
            self._res_flatten_fft = res_flatten  # for debugging

            # The following is faster in some cases, we might go back to it in
            # the future
            # res_flatten = fftconvolve(rho_aux_flatten, self._G_aux_shifted,
            # mode='full')
            # self._res_flatten_full = res_flatten # for debugging
            # res_flatten = res_flatten[:, -len(rho_aux_flatten[0, :])+1:]

            res = rho_aux * 0
            # Here we cannot separate the effect of the different turns
            # We put everything in one turn and leave the rest to be zero
            res[0, :_N_aux_turn] = res_flatten[0, :_N_aux_turn]

            self._res_flatten = res_flatten  # for debugging
            self._rho_flatten = rho_aux_flatten  # for debugging

        moments_data['result'] = res.real


class ResonatorWake(WakeComponent):
    """
    A resonator wake. On top of the following parameters it takes the same
    parameters as WakeField.
    Changing r_shunt, q_factor, and frequency after initialization is forbidded
    because the wake-associated quantities are computed upon initialization and
    changing the parameters would not update them.

    Parameters
    ----------
    r_shunt: float
        Resonator shunt impedance
    frequency: float
        Resonator frequency
    q_factor: float
        Resonator quality factor
    beta: float
        Lorentz factor of the beam

    Returns
    -------
    A resonator Wakefield
    """

    def __init__(self, r_shunt, frequency, q_factor, beta=1, ** kwargs):

        assert 'function' not in kwargs

        self._r_shunt = r_shunt
        self._frequency = frequency
        self._q_factor = q_factor
        self.beta = beta

        if kwargs['kick'] == 'delta':
            function = self._longitudinal_resonator_function
        else:
            function = self._transverse_resonator_function

        super().__init__(function=function, **kwargs)

    @property
    def r_shunt(self):
        return self._r_shunt

    @r_shunt.setter
    def r_shunt(self, value):
        if hasattr(self, 'r_shunt'):
            raise AttributeError('r_shunt cannot be changed after '
                                 'initialization')
        self._r_shunt = value

    @property
    def q_factor(self):
        return self._q_factor

    @q_factor.setter
    def q_factor(self, value):
        if hasattr(self, 'q_factor'):
            raise AttributeError('q_factor cannot be changed after '
                                 'initialization')
        self._q_factor = value

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        if hasattr(self, 'frequency'):
            raise AttributeError('frequency cannot be changed after '
                                 'initialization')
        self._frequency = value

    def _transverse_resonator_function(self, z):
        omega_r = 2 * np.pi * self.frequency
        alpha_t = omega_r / (2 * self.q_factor)
        omega_bar = np.sqrt(omega_r ** 2 - alpha_t ** 2)

        dt = self.beta*clight

        res = (z < 0) * (self.r_shunt *
                         omega_r ** 2 / (self.q_factor * omega_bar) *
                         np.exp(alpha_t * z / dt) *
                         np.sin(omega_bar * z / dt))  # Wake definition
        return res

    def _longitudinal_resonator_function(self, z):
        omega_r = 2 * np.pi * self.frequency
        alpha_t = omega_r / (2 * self.q_factor)
        omega_bar = np.sqrt(np.abs(omega_r ** 2 - alpha_t ** 2))

        dt = self.beta*clight

        res = (z < 0) * (-self.r_shunt * alpha_t *
                         np.exp(alpha_t * z / dt) *
                         (np.cos(omega_bar * z / dt) +
                          alpha_t / omega_bar * np.sin(omega_bar * z / dt)))

        return res


def _build_z_wake(z_a, z_b, num_turns, n_aux, m_aux, circumference, dz,
                  aa, bb, cc, dd, z_p):
    z_c = z_a  # For wakefield, z_c = z_a
    z_d = z_b  # For wakefield, z_d = z_b
    z_wake = np.zeros((num_turns, m_aux))
    for tt in range(num_turns):
        z_a_turn = z_a + tt * circumference
        z_b_turn = z_b + tt * circumference
        temp_z = np.arange(
            z_c - z_b_turn, z_d - z_a_turn + dz/10, dz)[:-1]

        if z_p is None: # single bunch mode
            assert dd - aa - (cc - bb + 1) == 1
            z_p = 0

        for ii, ll in enumerate(range(
                cc - bb + 1, dd - aa)):
            z_wake[tt, ii * n_aux:(ii + 1) * n_aux] = temp_z + ll * z_p
    return z_wake
