import numpy as np
from scipy.constants import e as qe

from typing import Tuple


class _ConvData:

    def __init__(self, component, wakefield=None, _flatten=False, log_moments=None):

        self._flatten = _flatten
        self.component = component
        self.wakefield = wakefield

        source_exponents = component.source_exponents

        assert isinstance(log_moments, (list, tuple)) or log_moments is None

        source_moments = ['num_particles']
        if source_exponents[0] != 0:
            source_moments.append('x')
        if source_exponents[1] != 0:
            source_moments.append('y')


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

            self.G_aux = self.component.function(self.z_wake)

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
        self._compute_convolution(moment_names=self.component.source_moments,
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

        if self.component.test_exponents[0] != 0:
            scaling_constant *= particles.x**self.component.test_exponents[0]
        if self.component.test_exponents[1] != 0:
            scaling_constant *= particles.y**self.component.test_exponents[1]

        # remember to handle lost particles!!!
        getattr(particles, self.component.kick)[:] += (scaling_constant *
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

def _build_z_wake(z_a, z_b, num_turns, n_aux, m_aux, circumference, dz,
                  aa, bb, cc, dd, z_p):

    if num_turns == 1 or num_turns is None:
        circumference = 0.

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