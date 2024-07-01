import xfields as xf
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

import json
import h5py

import numpy as np


@for_all_test_contexts
def test_bunch_monitor_hdf5(test_context):
    n_macroparticles = int(1e6)
    num_slices = 10
    zeta_range = (-1, 1)

    energy = 7e12

    offs_x = 1e-3
    offs_px = 2e-3
    offs_y = 3e-3
    offs_py = 4e-3
    offs_zeta = 5e-3
    offs_delta = 6e-3

    sigma_x = 1e-3
    sigma_px = 2e-3
    sigma_y = 3e-3
    sigma_py = 4e-3
    sigma_zeta = 5e-3
    sigma_delta = 6e-3

    # base coordinates
    x_coord = sigma_x * np.random.random(n_macroparticles) + offs_x
    px_coord = sigma_px * np.random.random(n_macroparticles) + offs_px
    y_coord = sigma_y * np.random.random(n_macroparticles) + offs_y
    py_coord = sigma_py * np.random.random(n_macroparticles) + offs_py
    zeta_coord = sigma_zeta * np.random.random(n_macroparticles) + offs_zeta
    delta_coord = sigma_delta * np.random.random(n_macroparticles) + offs_delta

    bunch_spacing_zeta = 10

    n_bunches = 3

    # n_bunches bunches with n_macroparticles each with different coordinates
    particles = xt.Particles(
        _context=test_context, p0c=energy,
        x=np.concatenate([x_coord * (bid + 1) for bid in range(n_bunches)]),
        px=np.concatenate([px_coord * (bid + 1) for bid in range(n_bunches)]),
        y=np.concatenate([y_coord * (bid + 1) for bid in range(n_bunches)]),
        py=np.concatenate([py_coord * (bid + 1) for bid in range(n_bunches)]),
        zeta=np.concatenate([zeta_coord * (bid + 1) - bunch_spacing_zeta * bid
                             for bid in range(n_bunches)]),
        delta=np.concatenate(
            [delta_coord * (bid + 1) for bid in range(n_bunches)]),
    )

    beta = particles.beta0[0]
    gamma = np.sqrt(1 / (1 - beta ** 2))

    # dummy filling scheme
    filling_scheme = np.ones(n_bunches, dtype=int)
    bunch_numbers = np.arange(n_bunches, dtype=int)

    flush_data_every = 10

    monitor = xf.CollectiveMonitor(
        base_file_name='test_monitor',
        backend='hdf5',
        monitor_bunches=True,
        monitor_slices=False,
        monitor_particles=False,
        flush_data_every=flush_data_every,
        zeta_range=zeta_range,
        num_slices=num_slices,
        filling_scheme=filling_scheme,
        bunch_numbers=bunch_numbers,
        bunch_spacing_zeta=bunch_spacing_zeta,
    )

    # track for twice flush_data_every turns so that we test the reshaping
    for _ in range(2 * flush_data_every):
        monitor.track(particles)

    with h5py.File(monitor._bunches_filename, 'r') as h5file:
        for bid in bunch_numbers:
            bunch = h5file[str(bid)]
            print(bunch['mean_x'][:], np.mean(x_coord) * (bid + 1))
            assert np.allclose(bunch['mean_x'][:],
                               np.mean(x_coord) * (bid + 1))
            assert np.allclose(bunch['mean_px'][:],
                               np.mean(px_coord) * (bid + 1))
            assert np.allclose(bunch['mean_y'][:],
                               np.mean(y_coord) * (bid + 1))
            assert np.allclose(bunch['mean_py'][:],
                               np.mean(py_coord) * (bid + 1))
            assert np.allclose(bunch['mean_zeta'][:],
                               (np.mean(zeta_coord * (bid + 1) -
                                        bunch_spacing_zeta * bid)))
            assert np.allclose(bunch['mean_delta'][:],
                               np.mean(delta_coord) * (bid + 1))

            assert np.allclose(bunch['sigma_x'][:],
                               np.std(x_coord) * (bid + 1))
            assert np.allclose(bunch['sigma_px'][:],
                               np.std(px_coord) * (bid + 1))
            assert np.allclose(bunch['sigma_y'][:],
                               np.std(y_coord) * (bid + 1))
            assert np.allclose(bunch['sigma_py'][:],
                               np.std(py_coord) * (bid + 1))
            assert np.allclose(bunch['sigma_zeta'][:],
                               np.std(zeta_coord) * (bid + 1))
            assert np.allclose(bunch['sigma_delta'][:],
                               np.std(delta_coord) * (bid + 1))

            epsn_x = np.sqrt(
                np.linalg.det(np.cov(x_coord, px_coord))) * beta * gamma
            epsn_y = np.sqrt(
                np.linalg.det(np.cov(y_coord, py_coord))) * beta * gamma
            epsn_zeta = np.sqrt(
                np.linalg.det(np.cov(zeta_coord, delta_coord))) * beta * gamma

            assert np.allclose(bunch['epsn_x'][:],
                               epsn_x * (bid + 1) ** 2)
            assert np.allclose(bunch['epsn_y'][:],
                               epsn_y * (bid + 1) ** 2)
            assert np.allclose(bunch['epsn_zeta'][:],
                               epsn_zeta * (bid + 1) ** 2)

            assert np.allclose(bunch['num_particles'][:], n_macroparticles)


@for_all_test_contexts
def test_bunch_monitor_json(test_context):
    n_macroparticles = int(1e6)
    num_slices = 10
    zeta_range = (-1, 1)

    energy = 7e12

    offs_x = 1e-3
    offs_px = 2e-3
    offs_y = 3e-3
    offs_py = 4e-3
    offs_zeta = 5e-3
    offs_delta = 6e-3

    sigma_x = 1e-3
    sigma_px = 2e-3
    sigma_y = 3e-3
    sigma_py = 4e-3
    sigma_zeta = 5e-3
    sigma_delta = 6e-3

    # base coordinates
    x_coord = sigma_x * np.random.random(n_macroparticles) + offs_x
    px_coord = sigma_px * np.random.random(n_macroparticles) + offs_px
    y_coord = sigma_y * np.random.random(n_macroparticles) + offs_y
    py_coord = sigma_py * np.random.random(n_macroparticles) + offs_py
    zeta_coord = sigma_zeta * np.random.random(n_macroparticles) + offs_zeta
    delta_coord = sigma_delta * np.random.random(n_macroparticles) + offs_delta

    bunch_spacing_zeta = 10

    n_bunches = 3

    # n_bunches bunches with n_macroparticles each with different coordinates
    particles = xt.Particles(
        _context=test_context, p0c=energy,
        x=np.concatenate([x_coord * (bid + 1) for bid in range(n_bunches)]),
        px=np.concatenate([px_coord * (bid + 1) for bid in range(n_bunches)]),
        y=np.concatenate([y_coord * (bid + 1) for bid in range(n_bunches)]),
        py=np.concatenate([py_coord * (bid + 1) for bid in range(n_bunches)]),
        zeta=np.concatenate([zeta_coord * (bid + 1) - bunch_spacing_zeta * bid
                             for bid in range(n_bunches)]),
        delta=np.concatenate(
            [delta_coord * (bid + 1) for bid in range(n_bunches)]),
    )

    beta = particles.beta0[0]
    gamma = np.sqrt(1 / (1 - beta ** 2))

    # dummy filling scheme
    filling_scheme = np.ones(n_bunches, dtype=int)
    bunch_numbers = np.arange(n_bunches, dtype=int)

    n_turns = 10

    monitor = xf.CollectiveMonitor(
        base_file_name='test_monitor',
        backend='json',
        monitor_bunches=True,
        monitor_slices=False,
        monitor_particles=False,
        flush_data_every=n_turns,
        zeta_range=zeta_range,
        num_slices=num_slices,
        filling_scheme=filling_scheme,
        bunch_numbers=bunch_numbers,
        bunch_spacing_zeta=bunch_spacing_zeta,
    )

    # track for twice flush_data_every turns so that we test the reshaping
    for _ in range(2 * n_turns):
        monitor.track(particles)

    with open(monitor._bunches_filename, 'r') as f:
        bunches_dict = json.load(f)

    for bid in bunch_numbers:
        bunch = bunches_dict[str(bid)]
        assert np.allclose(bunch['mean_x'][:], (np.mean(x_coord) *
                                                (bid + 1)))
        assert np.allclose(bunch['mean_px'][:], (np.mean(px_coord) *
                                                 (bid + 1)))
        assert np.allclose(bunch['mean_y'][:], (np.mean(y_coord) *
                                                (bid + 1)))
        assert np.allclose(bunch['mean_py'][:], (np.mean(py_coord) *
                                                 (bid + 1)))
        assert np.allclose(bunch['mean_zeta'][:],
                           (np.mean(zeta_coord * (bid + 1) -
                                    bunch_spacing_zeta * bid)))
        assert np.allclose(bunch['mean_delta'][:],
                           np.mean(delta_coord) * (bid + 1))

        assert np.allclose(bunch['sigma_x'][:], (np.std(x_coord) *
                                                 (bid + 1)))
        assert np.allclose(bunch['sigma_px'][:], (np.std(px_coord) *
                                                  (bid + 1)))
        assert np.allclose(bunch['sigma_y'][:], (np.std(y_coord) *
                                                 (bid + 1)))
        assert np.allclose(bunch['sigma_py'][:], (np.std(py_coord) *
                                                  (bid + 1)))
        assert np.allclose(bunch['sigma_zeta'][:],
                           (np.std(zeta_coord) *
                            (bid + 1)))
        assert np.allclose(bunch['sigma_delta'][:],
                           (np.std(delta_coord) * (bid + 1)))

        epsn_x = np.sqrt(
            np.linalg.det(np.cov(x_coord, px_coord))) * beta * gamma
        epsn_y = np.sqrt(
            np.linalg.det(np.cov(y_coord, py_coord))) * beta * gamma
        epsn_zeta = np.sqrt(np.linalg.det(np.cov(zeta_coord,
                                                 delta_coord))) * beta * gamma

        assert np.allclose(bunch['epsn_x'][:], epsn_x * (bid + 1) ** 2)
        assert np.allclose(bunch['epsn_y'][:], epsn_y * (bid + 1) ** 2)
        assert np.allclose(bunch['epsn_zeta'][:], epsn_zeta * (bid + 1) ** 2)

        assert np.allclose(bunch['num_particles'][:], n_macroparticles)


@for_all_test_contexts
def test_slice_monitor_hdf5(test_context):
    n_macroparticles = int(1e6)
    num_slices = 10
    zeta_range = (0, 1)

    energy = 7e12

    offs_x = 1
    offs_px = 2
    offs_y = 3
    offs_py = 4
    offs_delta = 5

    sigma_x = 6
    sigma_px = 7
    sigma_y = 8
    sigma_py = 9
    sigma_delta = 10

    # dummy filling scheme
    n_bunches = 3
    filling_scheme = np.ones(n_bunches, dtype=int)
    bunch_numbers = np.arange(n_bunches, dtype=int)
    bunch_spacing_zeta = 10

    flush_data_every = 10

    monitor = xf.CollectiveMonitor(
        base_file_name='test_monitor',
        backend='hdf5',
        monitor_bunches=False,
        monitor_slices=True,
        monitor_particles=False,
        flush_data_every=flush_data_every,
        zeta_range=zeta_range,
        num_slices=num_slices,
        bunch_spacing_zeta=bunch_spacing_zeta,
        filling_scheme=filling_scheme,
        bunch_numbers=bunch_numbers
    )

    particles = xt.Particles(
        _context=test_context,
        p0c=energy,
        x=np.zeros(n_macroparticles),
    )

    beta = particles.beta0[0]
    gamma = np.sqrt(1 / (1 - beta ** 2))

    n_slice = 5

    dzeta = monitor.slicer.dzeta
    zeta_coord_base = np.random.random(n_macroparticles)

    x_coord_bunch = []
    px_coord_bunch = []
    y_coord_bunch = []
    py_coord_bunch = []
    zeta_coord_bunch = []
    delta_coord_bunch = []
    n_macroparticles_slice_bunch = []

    for bid in range(n_bunches):
        zeta_coord = zeta_coord_base * (bid + 1) - bid * bunch_spacing_zeta

        bin_min = monitor.slicer.zeta_centers[bid, n_slice] - dzeta / 2
        bin_max = monitor.slicer.zeta_centers[bid, n_slice] + dzeta / 2

        slice_mask = np.logical_and(zeta_coord < bin_max, zeta_coord >= bin_min)

        n_macroparticles_slice = np.sum(slice_mask)

        x_coord = (sigma_x * np.random.random(
            n_macroparticles_slice) + offs_x) * (bid + 1)
        px_coord = (sigma_px * np.random.random(
            n_macroparticles_slice) + offs_px) * (bid + 1)
        y_coord = (sigma_y * np.random.random(
            n_macroparticles_slice) + offs_y) * (bid + 1)
        py_coord = (sigma_py * np.random.random(
            n_macroparticles_slice) + offs_py) * (bid + 1)
        delta_coord = (sigma_delta * np.random.random(n_macroparticles_slice) +
                       offs_delta) * (bid + 1)

        x_coord_bunch.append(x_coord)
        px_coord_bunch.append(px_coord)
        y_coord_bunch.append(y_coord)
        py_coord_bunch.append(py_coord)
        delta_coord_bunch.append(delta_coord)
        zeta_coord_bunch.append(zeta_coord[slice_mask])
        n_macroparticles_slice_bunch.append(n_macroparticles_slice)

        particles.x[slice_mask] = x_coord
        particles.px[slice_mask] = px_coord
        particles.y[slice_mask] = y_coord
        particles.py[slice_mask] = py_coord
        particles.zeta[slice_mask] = zeta_coord[slice_mask]
        particles.delta[slice_mask] = delta_coord

    # track for twice flush_data_every turns so that we test the reshaping
    for _ in range(flush_data_every * 2):
        monitor.track(particles)

    with h5py.File(monitor._slices_filename, 'r') as h5file:
        for bid in bunch_numbers:
            slice_data = h5file[str(bid)]
            print(x_coord_bunch[bid])
            assert np.allclose(slice_data['mean_x'][:, n_slice],
                               np.mean(x_coord_bunch[bid]))
            assert np.allclose(slice_data['mean_px'][:, n_slice],
                               np.mean(px_coord_bunch[bid]))
            assert np.allclose(slice_data['mean_y'][:, n_slice],
                               np.mean(y_coord_bunch[bid]))
            assert np.allclose(slice_data['mean_py'][:, n_slice],
                               np.mean(py_coord_bunch[bid]))
            assert np.allclose(slice_data['mean_zeta'][:, n_slice],
                               np.mean(zeta_coord_bunch[bid]))
            assert np.allclose(slice_data['mean_delta'][:, n_slice],
                               np.mean(delta_coord_bunch[bid]))

            assert np.allclose(slice_data['sigma_x'][:, n_slice],
                               np.std(x_coord_bunch[bid]))
            assert np.allclose(slice_data['sigma_px'][:, n_slice],
                               np.std(px_coord_bunch[bid]))
            assert np.allclose(slice_data['sigma_y'][:, n_slice],
                               np.std(y_coord_bunch[bid]))
            assert np.allclose(slice_data['sigma_py'][:, n_slice],
                               np.std(py_coord_bunch[bid]))
            assert np.allclose(slice_data['sigma_zeta'][:, n_slice],
                               np.std(zeta_coord_bunch[bid]))
            assert np.allclose(slice_data['sigma_delta'][:, n_slice],
                               np.std(delta_coord_bunch[bid]))

            epsn_x = (np.sqrt(np.var(x_coord_bunch[bid]) *
                              np.var(px_coord_bunch[bid]) -
                              np.cov(x_coord_bunch[bid], px_coord_bunch[bid])[
                                  0, 1])
                      * beta * gamma)
            epsn_y = (np.sqrt(np.var(y_coord_bunch[bid]) *
                              np.var(py_coord_bunch[bid]) -
                              np.cov(y_coord_bunch[bid], py_coord_bunch[bid])[
                                  0, 1])
                      * beta * gamma)
            epsn_zeta = (np.sqrt(np.var(zeta_coord_bunch[bid]) *
                                 np.var(delta_coord_bunch[bid]) -
                                 np.cov(zeta_coord_bunch[bid],
                                        delta_coord_bunch[bid])[
                                     0, 1]) * beta * gamma)

            assert np.allclose(slice_data['epsn_x'][:, n_slice], epsn_x)
            assert np.allclose(slice_data['epsn_y'][:, n_slice], epsn_y)
            assert np.allclose(slice_data['epsn_zeta'][:, n_slice], epsn_zeta)

            assert np.allclose(slice_data['num_particles'][:, n_slice],
                               n_macroparticles_slice_bunch[bid])


@for_all_test_contexts
def test_slice_monitor_json(test_context):
    n_macroparticles = int(1e6)
    num_slices = 10
    zeta_range = (0, 1)

    energy = 7e12

    offs_x = 1
    offs_px = 2
    offs_y = 3
    offs_py = 4
    offs_delta = 5

    sigma_x = 6
    sigma_px = 7
    sigma_y = 8
    sigma_py = 9
    sigma_delta = 10

    # dummy filling scheme
    n_bunches = 3
    filling_scheme = np.ones(n_bunches, dtype=int)
    bunch_numbers = np.arange(n_bunches, dtype=int)
    bunch_spacing_zeta = 10

    flush_data_every = 10

    monitor = xf.CollectiveMonitor(
        base_file_name='test_monitor',
        backend='json',
        monitor_bunches=False,
        monitor_slices=True,
        monitor_particles=False,
        flush_data_every=flush_data_every,
        zeta_range=zeta_range,
        num_slices=num_slices,
        bunch_spacing_zeta=bunch_spacing_zeta,
        filling_scheme=filling_scheme,
        bunch_numbers=bunch_numbers
    )

    particles = xt.Particles(
        _context=test_context,
        p0c=energy,
        x=np.zeros(n_macroparticles),
    )

    beta = particles.beta0[0]
    gamma = np.sqrt(1 / (1 - beta ** 2))

    n_slice = 5

    dzeta = monitor.slicer.dzeta
    zeta_coord_base = np.random.random(n_macroparticles)

    x_coord_bunch = []
    px_coord_bunch = []
    y_coord_bunch = []
    py_coord_bunch = []
    zeta_coord_bunch = []
    delta_coord_bunch = []
    n_macroparticles_slice_bunch = []

    for bid in range(n_bunches):
        zeta_coord = zeta_coord_base * (bid + 1) - bid * bunch_spacing_zeta

        bin_min = monitor.slicer.zeta_centers[bid, n_slice] - dzeta / 2
        bin_max = monitor.slicer.zeta_centers[bid, n_slice] + dzeta / 2

        slice_mask = np.logical_and(zeta_coord < bin_max, zeta_coord >= bin_min)

        n_macroparticles_slice = np.sum(slice_mask)

        x_coord = (sigma_x * np.random.random(
            n_macroparticles_slice) + offs_x) * (bid + 1)
        px_coord = (sigma_px * np.random.random(
            n_macroparticles_slice) + offs_px) * (bid + 1)
        y_coord = (sigma_y * np.random.random(
            n_macroparticles_slice) + offs_y) * (bid + 1)
        py_coord = (sigma_py * np.random.random(
            n_macroparticles_slice) + offs_py) * (bid + 1)
        delta_coord = (sigma_delta * np.random.random(n_macroparticles_slice) +
                       offs_delta) * (bid + 1)

        x_coord_bunch.append(x_coord)
        px_coord_bunch.append(px_coord)
        y_coord_bunch.append(y_coord)
        py_coord_bunch.append(py_coord)
        delta_coord_bunch.append(delta_coord)
        zeta_coord_bunch.append(zeta_coord[slice_mask])
        n_macroparticles_slice_bunch.append(n_macroparticles_slice)

        particles.x[slice_mask] = x_coord
        particles.px[slice_mask] = px_coord
        particles.y[slice_mask] = y_coord
        particles.py[slice_mask] = py_coord
        particles.zeta[slice_mask] = zeta_coord[slice_mask]
        particles.delta[slice_mask] = delta_coord

    # track for twice flush_data_every turns so that we test the reshaping
    for _ in range(2 * flush_data_every):
        monitor.track(particles)

    with open(monitor._slices_filename, 'r') as f:
        bunches_data = json.load(f)

    for bid in bunch_numbers:
        slice_data = bunches_data[str(bid)]

        for turn in range(2 * flush_data_every):
            assert np.isclose(slice_data['mean_x'][turn][n_slice],
                              np.mean(x_coord_bunch[bid]))
            assert np.isclose(slice_data['mean_px'][turn][n_slice],
                              np.mean(px_coord_bunch[bid]))
            assert np.isclose(slice_data['mean_y'][turn][n_slice],
                              np.mean(y_coord_bunch[bid]))
            assert np.isclose(slice_data['mean_py'][turn][n_slice],
                              np.mean(py_coord_bunch[bid]))
            assert np.isclose(slice_data['mean_zeta'][turn][n_slice],
                              np.mean(zeta_coord_bunch[bid]))
            assert np.isclose(slice_data['mean_delta'][turn][n_slice],
                              np.mean(delta_coord_bunch[bid]))

            assert np.isclose(slice_data['sigma_x'][turn][n_slice],
                              np.std(x_coord_bunch[bid]))
            assert np.isclose(slice_data['sigma_px'][turn][n_slice],
                              np.std(px_coord_bunch[bid]))
            assert np.isclose(slice_data['sigma_y'][turn][n_slice],
                              np.std(y_coord_bunch[bid]))
            assert np.isclose(slice_data['sigma_py'][turn][n_slice],
                              np.std(py_coord_bunch[bid]))
            assert np.isclose(slice_data['sigma_zeta'][turn][n_slice],
                              np.std(zeta_coord_bunch[bid]))
            assert np.isclose(slice_data['sigma_delta'][turn][n_slice],
                              np.std(delta_coord_bunch[bid]))

            epsn_x = (np.sqrt(np.var(x_coord_bunch[bid]) *
                              np.var(px_coord_bunch[bid]) -
                              np.cov(x_coord_bunch[bid], px_coord_bunch[bid])[
                                  0, 1])
                      * beta * gamma)
            epsn_y = (np.sqrt(np.var(y_coord_bunch[bid]) *
                              np.var(py_coord_bunch[bid]) -
                              np.cov(y_coord_bunch[bid], py_coord_bunch[bid])[
                                  0, 1])
                      * beta * gamma)
            epsn_zeta = (np.sqrt(np.var(zeta_coord_bunch[bid]) *
                                 np.var(delta_coord_bunch[bid]) -
                                 np.cov(zeta_coord_bunch[bid],
                                        delta_coord_bunch[bid])[
                                     0, 1]) * beta * gamma)

            assert np.isclose(slice_data['epsn_x'][turn][n_slice], epsn_x)
            assert np.isclose(slice_data['epsn_y'][turn][n_slice], epsn_y)
            assert np.isclose(slice_data['epsn_zeta'][turn][n_slice], epsn_zeta)

            assert np.isclose(slice_data['num_particles'][turn][n_slice],
                              n_macroparticles_slice_bunch[bid])


@for_all_test_contexts
def test_particle_monitor_hdf5(test_context):
    n_macroparticles = int(1e2)
    num_slices = 10
    zeta_range = (-1, 1)

    particles = xt.Particles(
        _context=test_context, p0c=7e12,
        x=np.random.random(n_macroparticles),
        px=np.random.random(n_macroparticles),
        y=np.random.random(n_macroparticles),
        py=np.random.random(n_macroparticles),
        zeta=np.random.random(n_macroparticles),
        delta=np.random.random(n_macroparticles),
    )

    # dummy particle mask
    particle_monitor_mask = particles.x > 0

    flush_data_every = 10

    monitor = xf.CollectiveMonitor(
        base_file_name='test_monitor',
        backend='hdf5',
        monitor_bunches=False,
        monitor_slices=False,
        monitor_particles=True,
        flush_data_every=flush_data_every,
        zeta_range=zeta_range,
        num_slices=num_slices,
        particle_monitor_mask=particle_monitor_mask,
    )

    # track for twice flush_data_every turns so that we test the reshaping
    for _ in range(2*flush_data_every):
        monitor.track(particles)

    with h5py.File(monitor._particles_filename, 'r') as h5file:
        for stat in monitor._stats_to_store_particles:
            saved_data = h5file[stat]
            for turn in range(2 * flush_data_every):
                assert np.allclose(getattr(particles,
                                           stat)[particle_monitor_mask],
                                   saved_data[turn, :])


@for_all_test_contexts
def test_particle_monitor_json(test_context):
    n_macroparticles = int(1e2)
    num_slices = 10
    zeta_range = (-1, 1)

    particles = xt.Particles(
        _context=test_context, p0c=7e12,
        x=np.random.random(n_macroparticles),
        px=np.random.random(n_macroparticles),
        y=np.random.random(n_macroparticles),
        py=np.random.random(n_macroparticles),
        zeta=np.random.random(n_macroparticles),
        delta=np.random.random(n_macroparticles),
    )

    # dummy particle mask
    particle_monitor_mask = particles.x > 0

    flush_data_every = 10

    monitor = xf.CollectiveMonitor(
        base_file_name='test_monitor',
        backend='json',
        monitor_bunches=False,
        monitor_slices=False,
        monitor_particles=True,
        flush_data_every=flush_data_every,
        zeta_range=zeta_range,
        num_slices=num_slices,
        particle_monitor_mask=particle_monitor_mask,
    )

    # track for twice flush_data_every turns so that we test the reshaping
    for _ in range(2*flush_data_every):
        monitor.track(particles)

    with open(monitor._particles_filename, 'r') as f:
        buffer = json.load(f)

    for stat in monitor._stats_to_store_particles:
        saved_data = buffer[stat]
        for turn in range(2*flush_data_every):
            assert np.allclose(getattr(particles, stat)[particle_monitor_mask],
                               saved_data[turn][:])
