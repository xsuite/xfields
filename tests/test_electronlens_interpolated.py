# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
import xpart as xp
import xtrack as xt
import xfields as xf

from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_electronlens_interpolated(test_context):
    outer_radius = 3.e-3
    inner_radius = 1.5e-3

    x_center = 0.
    y_center = 0.

    x_range = (-0.5e-2, 0.5e-2)
    y_range = (-0.55e-2, 0.55e-2)
    nx = 101
    ny = 151
    x_grid = np.linspace(x_range[0], x_range[1], nx)
    y_grid = np.linspace(y_range[0], y_range[1], ny)
    dx=x_grid[1] - x_grid[0]
    dy=y_grid[1] - y_grid[0]


    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    rho = np.zeros_like(X)
    rho[:] = 0
    R = np.sqrt( (X - x_center)**2 + (Y - y_center)**2)
    rho[ (R > inner_radius) & (R < outer_radius) ] = 1.
    #rho[ X < 0 ] = 0
    norm_rho = np.sum(rho[:,:])*dx*dy
    rho[:] /= norm_rho

    elens = xf.ElectronLensInterpolated(current=1, length=1, voltage=15e3,
                                        x_grid=x_grid, y_grid=y_grid, rho=rho)

    elens_ideal = xt.Elens(current=1, elens_length=1, voltage=15e3,
                                        inner_radius=inner_radius, outer_radius=outer_radius)

    npart = 1000
    x_init = np.linspace(-0.42e-2, 0.42e-2, npart)
    y_init = np.linspace(-0.42e-2, 0.42e-2, npart)
    X_init, Y_init = np.meshgrid(x_init, y_init)
    X_init = X_init.flatten()
    Y_init = Y_init.flatten()

    part = xp.Particles(x=X_init[:], y=Y_init[:],
                        zeta=[0], p0c=450e9
                       )

    part_ideal = xp.Particles(x=X_init[:], y=Y_init[:],
                        zeta=[0], p0c=450e9
                       )

    elens.track(part)
    elens_ideal.track(part_ideal)

    sort_mask = np.argsort(part.particle_id)
    sort_ideal_mask = np.argsort(part_ideal.particle_id)
    print(np.max(np.abs(part.px[sort_mask] - part_ideal.px[sort_ideal_mask])))
    print(np.max(np.abs(part.py[sort_mask] - part_ideal.py[sort_ideal_mask])))
    assert np.allclose(part.px[sort_mask], part_ideal.px[sort_ideal_mask],
                       atol=1.e-8, rtol=1.e-15)
    assert np.allclose(part.py[sort_mask], part_ideal.py[sort_ideal_mask],
                       atol=1.e-8, rtol=1.e-15)
    assert np.all(part.delta == 0.)
    assert np.all(part.ptau == 0.)
