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

# - make it work with one slice
# - populate a Sigma matrix
# - p.get_sigma_matrix()
# - p.get_statistical_