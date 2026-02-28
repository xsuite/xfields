# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
import pytest

from xobjects.test_helpers import for_all_test_contexts, fix_random_seed

import xpart as xp
import xfields as xf


@for_all_test_contexts
@fix_random_seed(12345)
def test_fftsolver3d_refresh_geometry_matches_reinit(test_context):

    from xfields.solvers.fftsolvers import FFTSolver3D

    nx, ny, nz = 8, 6, 4

    dx0, dy0, dz0 = 1.0e-3, 1.5e-3, 2.0e-3
    dx1, dy1, dz1 = 0.7e-3, 1.1e-3, 1.3e-3

    # Original solver with initial geometry
    solver = FFTSolver3D(
        dx=dx0, dy=dy0, dz=dz0,
        nx=nx, ny=ny, nz=nz,
        context=test_context,
    )

    # Target grids
    x_grid1 = np.arange(nx, dtype=np.float64) * dx1
    y_grid1 = np.arange(ny, dtype=np.float64) * dy1
    z_grid1 = np.arange(nz, dtype=np.float64) * dz1

    # Refresh geometry in-place
    solver.refresh_geometry(x_grid1, y_grid1, z_grid1)

    assert np.isclose(solver.dx, dx1)
    assert np.isclose(solver.dy, dy1)
    assert np.isclose(solver.dz, dz1)

    # Reference solver built directly at the target geometry
    solver_ref = FFTSolver3D(
        dx=dx1, dy=dy1, dz=dz1,
        nx=nx, ny=ny, nz=nz,
        context=test_context,
    )

    rho = np.random.normal(size=(nx, ny, nz))

    phi_refresh = solver.solve(rho)
    phi_ref = solver_ref.solve(rho)

    p2np = test_context.nparray_from_context_array
    phi_refresh_np = p2np(phi_refresh)
    phi_ref_np = p2np(phi_ref)

    denom = np.max(np.abs(phi_ref_np))
    assert denom > 0.0

    rel_err = np.max(np.abs(phi_refresh_np - phi_ref_np)) / denom
    assert rel_err < 1e-10


@for_all_test_contexts
@fix_random_seed(54321)
def test_fftsolver2p5d_refresh_geometry_matches_reinit(test_context):

    from xfields.solvers.fftsolvers import FFTSolver2p5D

    nx, ny, nz = 8, 6, 4

    dx0, dy0, dz0 = 1.0e-3, 1.5e-3, 2.0e-3
    dx1, dy1, dz1 = 0.7e-3, 1.1e-3, 1.3e-3

    solver = FFTSolver2p5D(
        dx=dx0, dy=dy0, dz=dz0,
        nx=nx, ny=ny, nz=nz,
        context=test_context,
    )

    x_grid1 = np.arange(nx, dtype=np.float64) * dx1
    y_grid1 = np.arange(ny, dtype=np.float64) * dy1
    z_grid1 = np.arange(nz, dtype=np.float64) * dz1

    solver.refresh_geometry(x_grid1, y_grid1, z_grid1)

    assert np.isclose(solver.dx, dx1)
    assert np.isclose(solver.dy, dy1)
    assert np.isclose(solver.dz, dz1)

    solver_ref = FFTSolver2p5D(
        dx=dx1, dy=dy1, dz=dz1,
        nx=nx, ny=ny, nz=nz,
        context=test_context,
    )

    rho = np.random.normal(size=(nx, ny, nz))

    phi_refresh = solver.solve(rho)
    phi_ref = solver_ref.solve(rho)

    p2np = test_context.nparray_from_context_array
    phi_refresh_np = p2np(phi_refresh)
    phi_ref_np = p2np(phi_ref)

    denom = np.max(np.abs(phi_ref_np))
    assert denom > 0.0

    rel_err = np.max(np.abs(phi_refresh_np - phi_ref_np)) / denom
    assert rel_err < 1e-10


@for_all_test_contexts
@fix_random_seed(24680)
def test_trilinear_fieldmap_retile_xyz_equivalent_to_new_3d(test_context):

    nx, ny, nz = 8, 6, 4

    x_range0 = (-1.0, 1.0)
    y_range0 = (-1.0, 1.0)
    z_range0 = (-1.0, 1.0)

    x_range1 = (-0.8, 1.2)
    y_range1 = (-1.1, 0.9)
    z_range1 = (-0.5, 1.5)

    # Fieldmap that will be re-tiled
    fm = xf.TriLinearInterpolatedFieldMap(
        _context=test_context,
        x_range=x_range0,
        y_range=y_range0,
        z_range=z_range0,
        nx=nx, ny=ny, nz=nz,
        solver="FFTSolver3D",
    )

    # Reference fieldmap built directly with the target geometry
    fm_ref = xf.TriLinearInterpolatedFieldMap(
        _context=test_context,
        x_range=x_range1,
        y_range=y_range1,
        z_range=z_range1,
        nx=nx, ny=ny, nz=nz,
        solver="FFTSolver3D",
    )

    rho = np.random.normal(size=(nx, ny, nz))

    fm_ref.update_rho(rho, force=True)
    fm_ref.update_phi_from_rho()

    # Apply geometry change to fm and then set the same rho
    fm.retile_xyz(
        xmin=x_range1[0], xmax=x_range1[1],
        ymin=y_range1[0], ymax=y_range1[1],
        zmin=z_range1[0], zmax=z_range1[1],
        zero_fields=True,
    )

    # Geometry should match
    assert np.isclose(fm.x_grid[0], x_range1[0])
    assert np.isclose(fm.x_grid[-1], x_range1[1])
    assert np.isclose(fm.y_grid[0], y_range1[0])
    assert np.isclose(fm.y_grid[-1], y_range1[1])
    assert np.isclose(fm.z_grid[0], z_range1[0])
    assert np.isclose(fm.z_grid[-1], z_range1[1])
    assert fm.nx == nx
    assert fm.ny == ny
    assert fm.nz == nz

    fm.update_rho(rho, force=True)
    fm.update_phi_from_rho()

    p2np = test_context.nparray_from_context_array
    phi_retiled = p2np(fm.phi)
    phi_ref = p2np(fm_ref.phi)

    denom = np.max(np.abs(phi_ref))
    assert denom > 0.0

    rel_err = np.max(np.abs(phi_retiled - phi_ref)) / denom
    assert rel_err < 1e-10


@for_all_test_contexts
@fix_random_seed(24681)
def test_trilinear_fieldmap_retile_xy_equivalent_to_new_2p5d(test_context):

    nx, ny, nz = 8, 6, 4

    x_range0 = (-1.0, 1.0)
    y_range0 = (-1.0, 1.0)
    z_range = (-1.0, 1.0)

    x_range1 = (-0.8, 1.2)
    y_range1 = (-1.1, 0.9)

    fm = xf.TriLinearInterpolatedFieldMap(
        _context=test_context,
        x_range=x_range0,
        y_range=y_range0,
        z_range=z_range,
        nx=nx, ny=ny, nz=nz,
        solver="FFTSolver2p5D",
    )

    # Reference fieldmap built directly with the target transverse mesh
    fm_ref = xf.TriLinearInterpolatedFieldMap(
        _context=test_context,
        x_range=x_range1,
        y_range=y_range1,
        z_range=z_range,
        nx=nx, ny=ny, nz=nz,
        solver="FFTSolver2p5D",
    )

    rho = np.random.normal(size=(nx, ny, nz))

    fm_ref.update_rho(rho, force=True)
    fm_ref.update_phi_from_rho()

    # Apply transverse geometry change to fm and then set the same rho
    fm.retile_xy(
        xmin=x_range1[0], xmax=x_range1[1],
        ymin=y_range1[0], ymax=y_range1[1],
        zero_fields=True,
    )

    # Transverse geometry should match (z unchanged)
    assert np.isclose(fm.x_grid[0], x_range1[0])
    assert np.isclose(fm.x_grid[-1], x_range1[1])
    assert np.isclose(fm.y_grid[0], y_range1[0])
    assert np.isclose(fm.y_grid[-1], y_range1[1])
    assert np.isclose(fm.z_grid[0], z_range[0])
    assert np.isclose(fm.z_grid[-1], z_range[1])
    assert fm.nx == nx
    assert fm.ny == ny
    assert fm.nz == nz

    fm.update_rho(rho, force=True)
    fm.update_phi_from_rho()

    p2np = test_context.nparray_from_context_array
    phi_retiled = p2np(fm.phi)
    phi_ref = p2np(fm_ref.phi)

    denom = np.max(np.abs(phi_ref))
    assert denom > 0.0

    rel_err = np.max(np.abs(phi_retiled - phi_ref)) / denom
    assert rel_err < 1e-10


@for_all_test_contexts
@fix_random_seed(24683)
def test_trilinear_fieldmap_retile_xyz_equivalent_to_new_2p5d(test_context):

    nx, ny, nz = 8, 6, 4

    x_range0 = (-1.0, 1.0)
    y_range0 = (-1.0, 1.0)
    z_range0 = (-1.0, 1.0)

    x_range1 = (-0.8, 1.2)
    y_range1 = (-1.1, 0.9)
    z_range1 = (-0.5, 1.5)

    # Fieldmap that will be re-tiled
    fm = xf.TriLinearInterpolatedFieldMap(
        _context=test_context,
        x_range=x_range0,
        y_range=y_range0,
        z_range=z_range0,
        nx=nx, ny=ny, nz=nz,
        solver="FFTSolver2p5D",
    )

    # Reference fieldmap built directly with the target geometry
    fm_ref = xf.TriLinearInterpolatedFieldMap(
        _context=test_context,
        x_range=x_range1,
        y_range=y_range1,
        z_range=z_range1,
        nx=nx, ny=ny, nz=nz,
        solver="FFTSolver2p5D",
    )

    rho = np.random.normal(size=(nx, ny, nz))

    fm_ref.update_rho(rho, force=True)
    fm_ref.update_phi_from_rho()

    # Apply full geometry change to fm and then set the same rho
    fm.retile_xyz(
        xmin=x_range1[0], xmax=x_range1[1],
        ymin=y_range1[0], ymax=y_range1[1],
        zmin=z_range1[0], zmax=z_range1[1],
        zero_fields=True,
    )

    # Geometry should match
    assert np.isclose(fm.x_grid[0], x_range1[0])
    assert np.isclose(fm.x_grid[-1], x_range1[1])
    assert np.isclose(fm.y_grid[0], y_range1[0])
    assert np.isclose(fm.y_grid[-1], y_range1[1])
    assert np.isclose(fm.z_grid[0], z_range1[0])
    assert np.isclose(fm.z_grid[-1], z_range1[1])
    assert fm.nx == nx
    assert fm.ny == ny
    assert fm.nz == nz

    fm.update_rho(rho, force=True)
    fm.update_phi_from_rho()

    p2np = test_context.nparray_from_context_array
    phi_retiled = p2np(fm.phi)
    phi_ref = p2np(fm_ref.phi)

    denom = np.max(np.abs(phi_ref))
    assert denom > 0.0

    rel_err = np.max(np.abs(phi_retiled - phi_ref)) / denom
    assert rel_err < 1e-10


@pytest.mark.parametrize(
    "nx, ny, nz",
    [
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
    ],
)
@pytest.mark.parametrize(  # stress test with many particles
    "num_particles",
    [
        1_000,
        5_000,
        50_000,
        500_000,
        5_000_000,
    ],
)
@for_all_test_contexts
@fix_random_seed(13579)
def test_spacecharge3d_set_xyz_mesh_equivalent_to_new(
    test_context,
    nx,
    ny,
    nz,
    num_particles
):

    sigma_x = 3e-3
    sigma_y = 2e-3
    sigma_z = 3e-1
    p0c = 5.0e9

    p_gen = xp.Particles(
        p0c=p0c,
        x=np.random.normal(0.0, sigma_x, num_particles),
        y=np.random.normal(0.0, sigma_y, num_particles),
        zeta=np.random.normal(0.0, sigma_z, num_particles),
    )

    particles_ret = xp.Particles(_context=test_context, **p_gen.to_dict())
    particles_ref = xp.Particles(_context=test_context, **p_gen.to_dict())

    x_lim0 = 5.0 * sigma_x
    y_lim0 = 5.0 * sigma_y
    z_lim0 = 5.0 * sigma_z

    x_range0 = (-x_lim0, x_lim0)
    y_range0 = (-y_lim0, y_lim0)
    z_range0 = (-z_lim0, z_lim0)

    # Target mesh
    x_lim1 = 3.0 * sigma_x
    y_lim1 = 3.0 * sigma_y
    z_lim1 = 3.0 * sigma_z

    x_range1 = (-x_lim1, x_lim1)
    y_range1 = (-y_lim1, y_lim1)
    z_range1 = (-z_lim1, z_lim1)

    gamma0 = float(p_gen.gamma0[0])

    # SpaceCharge3D element that will be re-meshed
    sc_ret = xf.SpaceCharge3D(
        _context=test_context,
        length=1.0,
        update_on_track=True,
        apply_z_kick=True,
        x_range=x_range0,
        y_range=y_range0,
        z_range=z_range0,
        nx=nx, ny=ny, nz=nz,
        solver="FFTSolver3D",
        gamma0=gamma0,
    )

    # Apply mesh change
    sc_ret.set_xyz_mesh(
        x_range=x_range1,
        y_range=y_range1,
        z_range=z_range1,
        zero_fields=True,
    )

    # Reference SpaceCharge3D built directly with the target mesh
    sc_ref = xf.SpaceCharge3D(
        _context=test_context,
        length=1.0,
        update_on_track=True,
        apply_z_kick=True,
        x_range=x_range1,
        y_range=y_range1,
        z_range=z_range1,
        nx=nx, ny=ny, nz=nz,
        solver="FFTSolver3D",
        gamma0=gamma0,
    )

    sc_ret.track(particles_ret)
    sc_ref.track(particles_ref)

    p2np = test_context.nparray_from_context_array

    for coord in ("x", "px", "y", "py", "zeta", "pzeta"):
        arr_ret = p2np(getattr(particles_ret, coord))
        arr_ref = p2np(getattr(particles_ref, coord))
        denom = np.max(np.abs(arr_ref))
        assert denom > 0.0
        rel_err = np.max(np.abs(arr_ret - arr_ref)) / denom
        assert rel_err < 1e-8


@pytest.mark.parametrize(
    "nx, ny, nz",
    [
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
    ],
)
@for_all_test_contexts
@fix_random_seed(24682)
def test_spacecharge2p5d_set_xyz_mesh_equivalent_to_new(
    test_context,
    nx,
    ny,
    nz
):

    num_particles = 1000
    sigma_x = 3e-3
    sigma_y = 2e-3
    sigma_z = 3e-1
    p0c = 5.0e9

    p_gen = xp.Particles(
        p0c=p0c,
        x=np.random.normal(0.0, sigma_x, num_particles),
        y=np.random.normal(0.0, sigma_y, num_particles),
        zeta=np.random.normal(0.0, sigma_z, num_particles),
    )

    particles_ret = xp.Particles(_context=test_context, **p_gen.to_dict())
    particles_ref = xp.Particles(_context=test_context, **p_gen.to_dict())

    x_lim0 = 5.0 * sigma_x
    y_lim0 = 5.0 * sigma_y
    z_lim0 = 5.0 * sigma_z

    x_range0 = (-x_lim0, x_lim0)
    y_range0 = (-y_lim0, y_lim0)
    z_range0 = (-z_lim0, z_lim0)

    # Target mesh
    x_lim1 = 3.0 * sigma_x
    y_lim1 = 3.0 * sigma_y
    z_lim1 = 3.0 * sigma_z

    x_range1 = (-x_lim1, x_lim1)
    y_range1 = (-y_lim1, y_lim1)
    z_range1 = (-z_lim1, z_lim1)

    # SpaceCharge3D element that will be re-meshed with 2.5D solver
    sc_ret = xf.SpaceCharge3D(
        _context=test_context,
        length=1.0,
        update_on_track=True,
        apply_z_kick=False,
        x_range=x_range0,
        y_range=y_range0,
        z_range=z_range0,
        nx=nx, ny=ny, nz=nz,
        solver="FFTSolver2p5D",
    )

    # Apply full mesh change
    sc_ret.set_xyz_mesh(
        x_range=x_range1,
        y_range=y_range1,
        z_range=z_range1,
        zero_fields=True,
    )

    # Reference SpaceCharge3D built directly with the target mesh
    sc_ref = xf.SpaceCharge3D(
        _context=test_context,
        length=1.0,
        update_on_track=True,
        apply_z_kick=False,
        x_range=x_range1,
        y_range=y_range1,
        z_range=z_range1,
        nx=nx, ny=ny, nz=nz,
        solver="FFTSolver2p5D",
    )

    sc_ret.track(particles_ret)
    sc_ref.track(particles_ref)

    p2np = test_context.nparray_from_context_array

    for coord in ("x", "px", "y", "py"):
        arr_ret = p2np(getattr(particles_ret, coord))
        arr_ref = p2np(getattr(particles_ref, coord))
        denom = np.max(np.abs(arr_ref))
        assert denom > 0.0
        rel_err = np.max(np.abs(arr_ret - arr_ref)) / denom
        assert rel_err < 1e-8


@pytest.mark.parametrize(
    "nx, ny, nz",
    [
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
    ],
)
@for_all_test_contexts
@fix_random_seed(24681)
def test_spacecharge2p5d_set_xy_mesh_equivalent_to_new(
    test_context,
    nx,
    ny,
    nz
):

    num_particles = 1000
    sigma_x = 3e-3
    sigma_y = 2e-3
    sigma_z = 3e-1
    p0c = 5.0e9

    p_gen = xp.Particles(
        p0c=p0c,
        x=np.random.normal(0.0, sigma_x, num_particles),
        y=np.random.normal(0.0, sigma_y, num_particles),
        zeta=np.random.normal(0.0, sigma_z, num_particles),
    )

    particles_ret = xp.Particles(_context=test_context, **p_gen.to_dict())
    particles_ref = xp.Particles(_context=test_context, **p_gen.to_dict())

    x_lim0 = 5.0 * sigma_x
    y_lim0 = 5.0 * sigma_y
    z_lim0 = 5.0 * sigma_z

    x_range0 = (-x_lim0, x_lim0)
    y_range0 = (-y_lim0, y_lim0)
    z_range0 = (-z_lim0, z_lim0)

    # Target mesh
    x_lim1 = 3.0 * sigma_x
    y_lim1 = 3.0 * sigma_y

    x_range1 = (-x_lim1, x_lim1)
    y_range1 = (-y_lim1, y_lim1)

    # SpaceCharge3D element that will be re-meshed
    sc_ret = xf.SpaceCharge3D(
        _context=test_context,
        length=1.0,
        update_on_track=True,
        apply_z_kick=False,
        x_range=x_range0,
        y_range=y_range0,
        z_range=z_range0,
        nx=nx, ny=ny, nz=nz,
        solver="FFTSolver2p5D",
    )

    # Apply mesh change
    sc_ret.set_xy_mesh(
        x_range=x_range1,
        y_range=y_range1,
        zero_fields=True,
    )

    # Reference SpaceCharge3D built directly with the target mesh
    sc_ref = xf.SpaceCharge3D(
        _context=test_context,
        length=1.0,
        update_on_track=True,
        apply_z_kick=False,
        x_range=x_range1,
        y_range=y_range1,
        # only x/y mesh is retiled.
        z_range=z_range0,
        nx=nx, ny=ny, nz=nz,
        solver="FFTSolver2p5D",
    )

    sc_ret.track(particles_ret)
    sc_ref.track(particles_ref)

    p2np = test_context.nparray_from_context_array

    for coord in ("x", "px", "y", "py"):
        arr_ret = p2np(getattr(particles_ret, coord))
        arr_ref = p2np(getattr(particles_ref, coord))
        denom = np.max(np.abs(arr_ref))
        assert denom > 0.0
        rel_err = np.max(np.abs(arr_ret - arr_ref)) / denom
        assert rel_err < 1e-8
