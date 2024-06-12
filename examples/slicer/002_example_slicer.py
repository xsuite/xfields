import xtrack as xt
import xpart as xp
import xfields as xf

import numpy as np

line = xt.Line.from_json(
    '../../../xtrack/test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.build_tracker()

line.vars['vrf400'] = 16

p = xp.generate_matched_gaussian_bunch(
    line=line,
    num_particles=int(1e5),
    nemitt_x=2e-6,
    nemitt_y=2.5e-6,
    sigma_z=0.07)

slicer = xf.UniformBinSlicer(zeta_range=(-999, +999), num_slices=1)

slicer.slice(p)

cov_matrix = np.zeros((6, 6))

for ii, vii in enumerate(['x', 'px', 'y', 'py', 'zeta', 'delta']):
    for jj, vjj in enumerate(['x', 'px', 'y', 'py', 'zeta', 'delta']):
        if ii <= jj:
            cov_matrix[ii, jj] = slicer.cov(vii, vjj)[0, 0]
        else:
            cov_matrix[ii, jj] = cov_matrix[jj, ii]

assert np.allclose(np.sqrt(cov_matrix[0, 0]), np.std(p.x), atol=0, rtol=1e-6)

tw = line.twiss()
Sig =  cov_matrix

S = xt.linear_normal_form.S

eival, eivec = np.linalg.eig(Sig @ S)

nex = eival[0].imag * tw.gamma0 * tw.beta0
ney = eival[2].imag * tw.gamma0 * tw.beta0
nez = eival[4].imag * tw.gamma0 * tw.beta0

print(f'{nex=} {ney=} {nez=}')

dummy_lam = np.diag([
    np.exp(-1j*np.pi/3), np.exp(+1j*np.pi/3),
    np.exp(-1j*np.pi/4), np.exp(+1j*np.pi/4),
    np.exp(-1j*np.pi/5), np.exp(+1j*np.pi/5),
])

dummy_R = eivec @ dummy_lam @ np.linalg.inv(eivec)

dummy_line = xt.Line(elements=[xt.Drift(length=1e-12)])
p_dummy = line.build_particles(x=0)

tw_from_sigmas = dummy_line.twiss(
                        particle_on_co=p_dummy,
                        R_matrix=dummy_R,
                        compute_chromatic_properties=False)

print('betx/bety')
print(f'betx (from line)    = {tw.betx[0]}')
print(f'betx (from sigmas)  = {tw_from_sigmas.betx[0]}')
print(f'bety (from line)    = {tw.bety[0]}')
print(f'bety (from sigmas)  = {tw_from_sigmas.bety[0]}')
print()
print('alfx/alfy')
print(f'alfx (from line)    = {tw.alfx[0]}')
print(f'alfx (from sigmas)  = {tw_from_sigmas.alfx[0]}')
print(f'alfy (from line)    = {tw.alfy[0]}')
print(f'alfy (from sigmas)  = {tw_from_sigmas.alfy[0]}')
print()
print('dx/dy')
print(f'dx (from line)      = {tw.dx[0]}')
print(f'dx (from sigmas)    = {tw_from_sigmas.dx[0]}')
print(f'dy (from line)      = {tw.dy[0]}')
print(f'dy (from sigmas)    = {tw_from_sigmas.dy[0]}')
print()
print('coupled betas')
print(f'betx2 (from line)   = {tw.betx2[0]}')
print(f'betx2 (from sigmas) = {tw_from_sigmas.betx2[0]}')
print(f'bety2 (from line)   = {tw.bety2[0]}')
print(f'bety2 (from sigmas) = {tw_from_sigmas.bety2[0]}')


# - make it work with one slice
# - populate a Sigma matrix
# - p.get_sigma_matrix()
# - p.get_statistical_