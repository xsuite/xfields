# mpiexec -n 3 pytetst test_slicer_with_filling_scheme_mpi.py

import xfields as xf
import xtrack as xt
import numpy as np
import xobjects as xo
import xwakes as xw
from mpi4py import MPI

import xo.test_helpers.for_all_test_contexts

@for_all_test_contexts
def test_wakes_with_filling_scheme_mpi(test_context):
    num_turns = 3

    fact_y = -2

    wake_ref = (xw.WakeResonator(kind='dipole_x', r=1e9, q=5, f_r=20e6) +
                xw.WakeResonator(kind='dipole_y', r=fact_y*1e9, q=5, f_r=20e6))
    wake_mpi = (xw.WakeResonator(kind='dipole_x', r=1e9, q=5, f_r=20e6) +
                xw.WakeResonator(kind='dipole_y', r=fact_y*1e9, q=5, f_r=20e6))

    for ww in [wake_ref, wake_mpi]:
        ww.configure_for_tracking(
            zeta_range=(-1, 1),
            num_slices=10,
            filling_scheme=[1, 0, 1, 1, 1],
            bunch_spacing_zeta=5,
            num_turns=num_turns,
            circumference=100)

    zeta = np.linspace(-25, 25, 1000000)
    particles_ref = xt.Particles(p0c=7000e9,
                                 zeta=zeta,
                                  _context=test_context)

    # different weight for the different bunches
    zeta = particles_ref.zeta
    mask_bunch0 = (zeta > -1) & (zeta < 1)
    particles_ref.weight[mask_bunch0] = 1
    mask_bunch1 = (zeta > -6) & (zeta < -4)
    particles_ref.weight[mask_bunch1] = 2
    mask_bunch2 = (zeta > -11) & (zeta < -9)
    particles_ref.weight[mask_bunch2] = 3
    mask_bunch3 = (zeta > -16) & (zeta < -14)
    particles_ref.weight[mask_bunch3] = 4
    mask_bunch4 = (zeta > -21) & (zeta < -19)
    particles_ref.weight[mask_bunch4] = 5

    particles_ref.x += 2e-3
    particles_ref.y += 2e-3

    particles_mpi = particles_ref.copy()

    line_mpi = xt.Line(elements=[wake_mpi])
    line_mpi.build_tracker()

    xw.config_pipeline_for_wakes(particles=particles_mpi, line=line_mpi,
                                communicator=MPI.COMM_WORLD)


    assert wake_mpi._wake_tracker.pipeline_manager is not None
    assert wake_ref._wake_tracker.pipeline_manager is None

    comm = wake_mpi._wake_tracker.pipeline_manager._communicator
    assert comm is MPI.COMM_WORLD

    n_proc = comm.Get_size()
    assert n_proc == 3

    my_rank = comm.Get_rank()
    assert my_rank in [0, 1, 2]

    expected_bunch_numbers = {
        0: [0, 1],
        1: [2],
        2: [3]
    }[my_rank]

    expected_partner_names = {
        0: ['particles1', 'particles2'],
        1: ['particles0', 'particles2'],
        2: ['particles0', 'particles1']
    }[my_rank]

    slicer_mpi = wake_mpi._wake_tracker.slicer
    slice_ref = wake_ref._wake_tracker.slicer

    assert (slicer_mpi.bunch_numbers
            == np.array(expected_bunch_numbers)).all()
    assert (slicer_mpi.num_bunches
            == len(expected_bunch_numbers))
    assert (np.array(wake_mpi._wake_tracker.partner_names)
            == expected_partner_names).all()

    assert (slicer_mpi.filled_slots == [0, 2, 3, 4]).all()


    line_mpi.track(particles_mpi)
    wake_ref.track(particles_ref)

    expected_num_particles = {
        0: np.array([
            [ 4000.,  4000.,  4000.,  4000.,  4000.,  4000.,  4000.,  4000.,  4000.,  4000.],
            [12000., 12000., 12000., 12000., 12000., 12000., 12000., 12000., 12000., 12000.]],
            dtype=np.float64),
        1: np.array(
            [16000., 16000., 16000., 16000., 16000., 16000., 16000., 16000., 16000., 16000.],
            dtype=np.float64),
        2: np.array(
            [20000., 20000., 20000., 20000., 20000., 20000., 20000., 20000., 20000., 20000.],
            dtype=np.float64),
    }

    assert slicer_mpi.num_particles.shape == expected_num_particles[my_rank].shape
    xo.assert_allclose(slicer_mpi.num_particles, expected_num_particles[my_rank],
                    rtol=1e-5, atol=1e-5)


    moments_data_mpi = wake_mpi._wake_tracker.moments_data
    moments_data_ref = wake_ref._wake_tracker.moments_data

    z_prof_mpi, prof_mpi = moments_data_mpi.get_moment_profile('num_particles', i_turn=0)
    z_prof_ref, prof_ref = moments_data_ref.get_moment_profile('num_particles', i_turn=0)

    assert z_prof_mpi.shape == z_prof_ref.shape
    assert prof_mpi.shape == prof_ref.shape
    xo.assert_allclose(z_prof_mpi, z_prof_ref, rtol=0, atol=1e-14)
    xo.assert_allclose(prof_mpi, prof_ref, rtol=0, atol=1e-14)

    for i_turn in range(1, num_turns):
        particles_mpi.weight *= 2
        particles_ref.weight *= 2
        line_mpi.track(particles_mpi)
        wake_ref.track(particles_ref)

    for i_check in range(1, num_turns):
        z_prof_turn_mpi,  prof_turn_mpi = moments_data_mpi.get_moment_profile('num_particles', i_turn=i_check)
        z_prof_turn_ref,  prof_turn_ref = moments_data_ref.get_moment_profile('num_particles', i_turn=i_check)
        xo.assert_allclose(z_prof_turn_mpi, z_prof_turn_ref, rtol=0, atol=1e-12)
        xo.assert_allclose(prof_turn_mpi, prof_turn_ref, rtol=0, atol=1e-12)
        xo.assert_allclose(prof_turn_mpi, (2**(num_turns-1-i_check)) * prof_mpi, rtol=0, atol=1e-12)

    conv_data_mpi_dict = wake_mpi.components[0].components[0]._conv_data.__dict__
    conv_data_ref_dict = wake_mpi.components[0].components[0]._conv_data.__dict__

    for conv_data_mpi_key in conv_data_mpi_dict:
        assert conv_data_mpi_key in conv_data_ref_dict
        if conv_data_mpi_key == 'component' or conv_data_mpi_key == 'waketracker':
            continue
        xo.assert_allclose(conv_data_mpi_dict[conv_data_mpi_key],
                        conv_data_ref_dict[conv_data_mpi_key],
                        rtol=0, atol=1e-12)

    for conv_data_ref_key in conv_data_ref_dict:
        assert conv_data_ref_key in conv_data_mpi_dict

    xo.assert_allclose(particles_mpi.px, particles_mpi.py/fact_y, rtol=0, atol=1e-12)
