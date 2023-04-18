# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
from numpy.random import default_rng
import xobjects as xo
import xpart as xp
import xfields as xf

from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_tricubic_interpolation(test_context):
    scale = 0.05
    ff = lambda x, y, z: sum([ scale * x**i * y**j * z**k
        for i in range(4) for j in range(4) for k in range(4)])
    dfdx = lambda x, y, z: sum([ i * scale * x**(i-1) * y**j * z**k
        for i in range(1,4) for j in range(4) for k in range(4)])
    dfdy = lambda x, y, z: sum([ j * scale * x**i * y**(j-1) * z**k
        for i in range(4) for j in range(1,4) for k in range(4)])
    dfdz = lambda x, y, z: sum([ k * scale * x**i * y**j * z**(k-1)
        for i in range(4) for j in range(4) for k in range(1,4)])
    dfdxy = lambda x, y, z: sum([ i * j * scale * x**(i-1) * y**(j-1) * z**k
        for i in range(1,4) for j in range(1,4) for k in range(4)])
    dfdxz = lambda x, y, z: sum([ i * k * scale * x**(i-1) * y**j * z**(k-1)
        for i in range(1,4) for j in range(4) for k in range(1,4)])
    dfdyz = lambda x, y, z: sum([ j * k * scale * x**i * y**(j-1) * z**(k-1)
        for i in range(4) for j in range(1,4) for k in range(1,4)])
    dfdxyz = lambda x, y, z: sum([ i * j * k * scale * x**(i-1) * y**(j-1) * z**(k-1)
        for i in range(1,4) for j in range(1,4) for k in range(1,4)])

    NN=21
    x_grid = np.linspace(-0.5, 0.5, NN)
    y_grid = np.linspace(-0.5, 0.5, NN)
    z_grid = np.linspace(-0.5, 0.5, NN)

    fieldmap = xf.TriCubicInterpolatedFieldMap(_context=test_context,
            x_grid=x_grid, y_grid=y_grid, z_grid=z_grid)
    ecloud = xf.ElectronCloud(length=1, fieldmap=fieldmap,
                              _buffer=fieldmap._buffer)

    x0 = fieldmap.x_grid[0]
    y0 = fieldmap.y_grid[0]
    z0 = fieldmap.z_grid[0]
    dx = fieldmap.dx
    dy = fieldmap.dy
    dz = fieldmap.dz
    nx = fieldmap.nx
    ny = fieldmap.ny
    for ix in range(NN):
        for iy in range(NN):
            for iz in range(NN):
                index = 0 + 8 * ix + 8 * nx * iy + 8 * nx * ny * iz
                fieldmap._phi_taylor[index + 0] = ff(x_grid[ix], y_grid[iy], z_grid[iz])
                fieldmap._phi_taylor[index + 1] = dfdx(x_grid[ix], y_grid[iy], z_grid[iz]) * dx
                fieldmap._phi_taylor[index + 2] = dfdy(x_grid[ix], y_grid[iy], z_grid[iz]) * dy
                fieldmap._phi_taylor[index + 3] = dfdz(x_grid[ix], y_grid[iy], z_grid[iz]) * dz
                fieldmap._phi_taylor[index + 4] = dfdxy(x_grid[ix], y_grid[iy], z_grid[iz]) * dx * dy
                fieldmap._phi_taylor[index + 5] = dfdxz(x_grid[ix], y_grid[iy], z_grid[iz]) * dx * dz
                fieldmap._phi_taylor[index + 6] = dfdyz(x_grid[ix], y_grid[iy], z_grid[iz]) * dy * dz
                fieldmap._phi_taylor[index + 7] = dfdxyz(x_grid[ix], y_grid[iy], z_grid[iz]) * dx * dy * dz

    n_parts = 1000
    rng = default_rng(12345)
    x_test = rng.random(n_parts) * 1.2 - 0.6
    y_test = rng.random(n_parts) * 1.2 - 0.6
    tau_test = rng.random(n_parts) * 1.2 - 0.6


    p0c = 450e9
    testp0 = xp.Particles(p0c=p0c)
    beta0 = testp0.beta0
    part = xp.Particles(_context=test_context, x=x_test, y=y_test,
                        zeta=beta0*tau_test, p0c=p0c)
    ecloud.track(part)

    part.move(_context=xo.ContextCpu())
    mask_p = part.state != -11
    true_px = np.array([-dfdx(xx, yy, zz) for xx, yy, zz in zip(part.x[mask_p], part.y[mask_p],
                                                                part.zeta[mask_p] / part.beta0[mask_p])])
    true_py = np.array([-dfdy(xx, yy, zz) for xx, yy, zz in zip(part.x[mask_p], part.y[mask_p],
                                                                part.zeta[mask_p] / part.beta0[mask_p])])
    true_ptau = np.array([-dfdz(xx, yy, zz) for xx, yy, zz in zip(part.x[mask_p], part.y[mask_p],
                                                                part.zeta[mask_p] / part.beta0[mask_p])])

    # print(true_px[:5])
    # print(part.ptau[:5])
    # print(part.state[:5])
    # print(f"px kick diff.: {np.mean(part.px[mask_p]-true_px):.2e} +- {np.std(part.px[mask_p] - true_px):.2e}")
    # print(f"py kick diff.: {np.mean(part.py[mask_p]-true_py):.2e} +- {np.std(part.py[mask_p] - true_py):.2e}")
    # print(f"ptau kick diff.: {np.mean(part.ptau[mask_p]-true_ptau):.2e} +- {np.std(part.ptau[mask_p] - true_ptau):.2e}")
    # print(np.allclose(part.px[mask_p], true_px, atol=1.e-13, rtol=1.e-13))
    # print(np.allclose(part.py[mask_p], true_py, atol=1.e-13, rtol=1.e-13))
    # print(np.allclose(part.ptau[mask_p], true_ptau, atol=1.e-13, rtol=1.e-13))
    # print(np.max(np.abs(part.px[mask_p]- true_px)))
    # print(np.max(np.abs(part.py[mask_p]- true_py)))
    # print(np.max(np.abs(part.ptau[mask_p]- true_ptau)))

    assert np.allclose(part.px[mask_p], true_px, atol=1.e-13, rtol=1.e-13)
    assert np.allclose(part.py[mask_p], true_py, atol=1.e-13, rtol=1.e-13)
    assert np.allclose(part.ptau[mask_p], true_ptau, atol=1.e-13, rtol=1.e-13)
