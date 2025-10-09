import numpy as np
import pytest

import xtrack as xt
import xpart as xp
import xfields as xf

from xobjects.test_helpers import for_all_test_contexts
from xfields.config_tools.spacecharge_config_tools import replace_spacecharge_with_PIC


@pytest.mark.parametrize(
    "solver, user_apply, expected",
    [
        ("FFTSolver2p5D", None, False),
        ("FFTSolver3D",   None, True),
        ("FFTSolver2p5D", True,  True),
        ("FFTSolver3D",   False, False),
    ]
)
@for_all_test_contexts
def test_replace_spacecharge_with_PIC_apply_z_kick_logic(solver, user_apply, expected, test_context):
    bunch_intensity = 2.5e11
    sigma_x = 3e-3
    sigma_y = 2e-3
    sigma_z = 0.30
    p0c = 25.92e9

    lprof = xf.LongitudinalProfileQGaussian(
        _context=test_context,
        number_of_particles=bunch_intensity,
        sigma_z=sigma_z,
        z0=0.0,
        q_parameter=1.0
    )

    sc1 = xf.SpaceChargeBiGaussian(
        _context=test_context,
        update_on_track=False,
        length=1.0,
        longitudinal_profile=lprof,
        mean_x=0.0, mean_y=0.0,
        sigma_x=sigma_x, sigma_y=sigma_y,
        min_sigma_diff=1e-12,
    )
    sc2 = xf.SpaceChargeBiGaussian(
        _context=test_context,
        update_on_track=False,
        length=2.0,
        longitudinal_profile=lprof,
        mean_x=0.0, mean_y=0.0,
        sigma_x=sigma_x, sigma_y=sigma_y,
        min_sigma_diff=1e-12,
    )

    line = xt.Line(elements=[
        xt.Drift(length=0.1),
        sc1,
        xt.Drift(length=0.2),
        sc2,
        xt.Drift(length=0.1),
    ])

    line.build_tracker(_context=test_context, compile=False)
    line.particle_ref = xp.Particles(_context=test_context, p0c=p0c)

    n_sigmas_range_pic_x = 4.0
    n_sigmas_range_pic_y = 4.0
    nx_grid = 16
    ny_grid = 16
    nz_grid = 16
    n_lims_x = 3
    n_lims_y = 3
    z_range = (-3*sigma_z, 3*sigma_z)

    kwargs = dict(
        line=line,
        n_sigmas_range_pic_x=n_sigmas_range_pic_x,
        n_sigmas_range_pic_y=n_sigmas_range_pic_y,
        nx_grid=nx_grid, ny_grid=ny_grid, nz_grid=nz_grid,
        n_lims_x=n_lims_x, n_lims_y=n_lims_y,
        z_range=z_range,
        solver=solver,
    )
    if user_apply is not None:
        kwargs["apply_z_kick"] = user_apply

    pic_collection, all_pics = replace_spacecharge_with_PIC(**kwargs)

    assert pic_collection.apply_z_kick == expected, (
        f"PICCollection.apply_z_kick mismatch for solver={solver}, "
        f"user_apply={user_apply}; expected {expected}"
    )
