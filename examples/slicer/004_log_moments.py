import xtrack as xt
import xpart as xp
import xfields as xf

import numpy as np

line = xt.Line.from_json(
    '../../../xtrack/test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.build_tracker()

line.vars['vrf400'] = 16

tw = line.twiss()


p = xp.generate_matched_gaussian_bunch(
    line=line,
    num_particles=int(1e3),
    nemitt_x=2e-6,
    nemitt_y=2.5e-6,
    sigma_z=0.07)


class LogBunchMoments:
    def __init__(self, twiss_parameters=True, transverse_coupling=True):
        self.slicer = xf.UniformBinSlicer(zeta_range=(-999, +999), num_slices=1)
        self.dummy_line = xt.Line(elements=[xt.Drift(length=1e-12)])
        self.transverse_coupling = transverse_coupling
        self.twiss_parameters = twiss_parameters

    def __call__(self, line, particles):

        # Measure moments
        slicer = self.slicer
        slicer.slice(particles)

        # Build covariance matrix
        cov_matrix = np.zeros((6, 6))
        for ii, vii in enumerate(['x', 'px', 'y', 'py', 'zeta', 'delta']):
            for jj, vjj in enumerate(['x', 'px', 'y', 'py', 'zeta', 'delta']):
                if ii <= jj:
                    cov_matrix[ii, jj] = slicer.cov(vii, vjj)[0, 0]
                else:
                    cov_matrix[ii, jj] = cov_matrix[jj, ii]
        Sig =  cov_matrix

        if not self.transverse_coupling:
            Sig[0:2, 2:4] = 0
            Sig[2:4, 0:2] = 0

        # The matrix Sig * S has the same eigenvector of the R matrix (see Wolski's book)
        S = xt.linear_normal_form.S

        eival, eivec = np.linalg.eig(Sig @ S)

        # Keep only one from each complex conjugate pair
        eival_list = []
        eivec_list = []
        for ii in range(6):
            if ii == 0:
                eival_list.append(eival[ii])
                eivec_list.append(eivec[:, ii])
                continue
            found_conj = False
            for jj in range(len(eival_list)):
                if np.allclose(eival[ii], np.conj(eival_list[jj]), rtol=0, atol=1e-14):
                    found_conj = True
                    break
            if not found_conj:
                eival_list.append(eival[ii])
                eivec_list.append(eivec[:, ii])

        assert len(eival_list) == 3

        # Find longitudinal mode
        norm = np.linalg.norm
        i_long = 0
        if norm(eivec_list[1][5:6]) > norm(eivec_list[i_long][5:6]):
            i_long = 1
        if norm(eivec_list[2][5:6]) > norm(eivec_list[i_long][5:6]):
            i_long = 2

        eival_zeta = eival_list[i_long]
        eivec_zeta = eivec_list[i_long]

        # Find vertical mode
        eival_list.pop(i_long)
        eivec_list.pop(i_long)
        if norm(eivec_list[0][3:4]) > norm(eivec_list[1][3:4]):
            i_vert = 0
        else:
            i_vert = 1

        eival_y = eival_list[i_vert]
        eivec_y = eivec_list[i_vert]

        # Find horizontal mode
        eival_list.pop(i_vert)
        eivec_list.pop(i_vert)
        eival_x = eival_list[0]
        eivec_x = eivec_list[0]

        # Emittances are the eigenvalues of S*R (see Wolski's book)
        gamma0 = line.particle_ref.gamma0[0]
        beta0 = line.particle_ref.beta0[0]
        nemitt_x = eival_x.imag * gamma0 * beta0
        nemitt_y = eival_y.imag * gamma0 * beta0
        nemitt_zeta = eival_zeta.imag * gamma0 * beta0

        if self.twiss_parameters:
            # I build a dummy stable R matrix with the same eigenvectors
            dummy_lam = np.diag([
                np.exp(-1j*np.pi/3), np.exp(+1j*np.pi/3),
                np.exp(-1j*np.pi/4), np.exp(+1j*np.pi/4),
                np.exp(-1j*np.pi/5), np.exp(+1j*np.pi/5),
            ])
            dummy_R = eivec @ dummy_lam @ np.linalg.inv(eivec)

            # Feed to twiss method to get optics parameters
            tw_from_sigmas = self.dummy_line.twiss(
                                    particle_on_co=line.particle_ref.copy(),
                                    R_matrix=dummy_R,
                                    compute_chromatic_properties=False)

        # build output
        out = dict(nemitt_x=nemitt_x, nemitt_y=nemitt_y, nemitt_zeta=nemitt_zeta)
        if self.twiss_parameters:
            for kk in tw_from_sigmas._col_names:
                out[kk] = tw_from_sigmas[kk, 0]

        out['mean_x'] = slicer.mean('x')[0, 0]
        out['mean_px'] = slicer.mean('px')[0, 0]
        out['mean_y'] = slicer.mean('y')[0, 0]
        out['mean_py'] = slicer.mean('py')[0, 0]
        out['mean_zeta'] = slicer.mean('zeta')[0, 0]
        out['mean_delta'] = slicer.mean('delta')[0, 0]

        out['std_x'] = slicer.std('x')[0, 0]
        out['std_px'] = slicer.std('px')[0, 0]
        out['std_y'] = slicer.std('y')[0, 0]
        out['std_py'] = slicer.std('py')[0, 0]
        out['std_zeta'] = slicer.std('zeta')[0, 0]
        out['std_delta'] = slicer.mean('delta')[0, 0]

        self._store = list(out.keys())

        return out

line.enable_time_dependent_vars = True
logmom = LogBunchMoments(transverse_coupling=True, twiss_parameters=True)
line.track(p, num_turns=10, log=[logmom, 't_turn_s'])
