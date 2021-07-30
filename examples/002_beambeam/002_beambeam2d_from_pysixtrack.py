import numpy as np
from scipy.constants import e as qe

import xfields as xf
import xtrack as xt

import xline

# TODO: change q0 from Coulomb to elementary charges


bb_pyst = xline.elements.BeamBeam4D(
        charge=1e14,
        sigma_x=2e-3,
        sigma_y=3e-3,
        beta_r=0.7,
        x_bb=1e-3,
        y_bb=-1.3e-3,
        d_px=2e-6,
        d_py=-2e-6
        )

bb = xf.BeamBeamBiGaussian2D.from_xline(bb_pyst)

pyst_part = xline.Particles(
        p0c=6500e9,
        x=-1.23e-3,
        px = 5e-6,
        y = 2e-3,
        py = 2.7e-6,
        sigma = 3.,
        delta = 2e-4)

part= xt.Particles(**pyst_part.to_dict())

bb.track(part)
print('------------------------')

bb_pyst.track(pyst_part)

for cc in 'x px y py zeta delta'.split():
    val_test = getattr(part, cc)[0]
    val_ref = getattr(pyst_part, cc)
    print('\n')
    print(f'xline: {cc} = {val_ref:.12e}')
    print(f'xsuite:     {cc} = {val_test:.12e}')
    assert np.isclose(val_test, val_ref, rtol=1e-12, atol=1e-12)

