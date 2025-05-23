# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
import json
import pathlib
from numpy.random import default_rng

import xobjects as xo
import xpart as xp
import xfields as xf
import xtrack as xt

from xobjects.test_helpers import for_all_test_contexts

XTRACK_TEST_DATA = pathlib.Path(__file__).parent.parent.parent / "xtrack" / "test_data/"



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
    zeta_test = rng.random(n_parts) * 1.2 - 0.6


    p0c = 450e9
    testp0 = xp.Particles(p0c=p0c)
    part = xp.Particles(_context=test_context, x=x_test, y=y_test,
                        zeta=zeta_test, p0c=p0c)
    ecloud.track(part)

    part.move(_context=xo.ContextCpu())
    mask_p = part.state != -11
    true_px = np.array([-dfdx(xx, yy, zz) for xx, yy, zz in zip(part.x[mask_p], part.y[mask_p],
                                                                part.zeta[mask_p])])
    true_py = np.array([-dfdy(xx, yy, zz) for xx, yy, zz in zip(part.x[mask_p], part.y[mask_p],
                                                                part.zeta[mask_p])])
    true_pzeta = np.array([-dfdz(xx, yy, zz) for xx, yy, zz in zip(part.x[mask_p], part.y[mask_p],
                                                                part.zeta[mask_p])])

    xo.assert_allclose(part.px[mask_p], true_px, atol=1.e-13, rtol=1.e-13)
    xo.assert_allclose(part.py[mask_p], true_py, atol=1.e-13, rtol=1.e-13)
    xo.assert_allclose(part.pzeta[mask_p], true_pzeta, atol=1.e-13, rtol=1.e-13)

@for_all_test_contexts
def test_electroncloud_config(test_context):
    fname_line = XTRACK_TEST_DATA/"lhc_no_bb/line_and_particle.json"

    with open(fname_line, 'r') as fid:
        input_data = json.load(fid)
    line = xt.Line.from_dict(input_data['line'])
    line.particle_ref = xp.Particles(p0c=input_data['particle']['p0c'])

    xfields_test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

    pinch_filenames = {'mqf': xfields_test_data_folder/"pyecloud_pinch/refined_Pinch_MTI1.0_MLI1.0_DTO2.0_DLO1.0.h5",
                       'mqd': xfields_test_data_folder/"pyecloud_pinch/refined_Pinch_MTI1.0_MLI1.0_DTO2.0_DLO1.0.h5"}

    zeta_max = 0.1
    ecloud_info = json.load(open(xfields_test_data_folder/"pyecloud_pinch/eclouds.json", "r"))

    reduced_ecloud_info = {
        'mqf': {key: ecloud_info['mqf'][key] for key in list(ecloud_info['mqf'].keys())[:5]},
        'mqd': {key: ecloud_info['mqd'][key] for key in list(ecloud_info['mqd'].keys())[:5]}
    }

    twiss_without_ecloud, twiss_with_ecloud = xf.full_electroncloud_setup(line=line,
                ecloud_info=reduced_ecloud_info, filenames=pinch_filenames,
                context=test_context, zeta_max=zeta_max)

    xo.assert_allclose(twiss_with_ecloud['x'],twiss_without_ecloud['x'], atol=3e-12, rtol=0)
    xo.assert_allclose(twiss_with_ecloud['y'],twiss_without_ecloud['y'], atol=3e-12, rtol=0)

    tt = line.get_table()
    assert np.all(tt.rows[tt.element_type == 'ElectronCloud'].name == np.array(
       ['ecloud.mqd.12.0', 'ecloud.mqf.12.0', 'ecloud.mqd.12.1',
        'ecloud.mqf.12.1', 'ecloud.mqd.12.10', 'ecloud.mqf.12.10',
        'ecloud.mqd.12.11', 'ecloud.mqf.12.11', 'ecloud.mqd.12.12',
        'ecloud.mqf.12.12']))
