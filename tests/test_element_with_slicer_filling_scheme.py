import xfields as xf
import xtrack as xt
import numpy as np
import xobjects as xo
import pytest


from xfields.beam_elements.element_with_slicer import ElementWithSlicer

@pytest.mark.parametrize('buffer_round_trip', [True, False])
@pytest.mark.parametrize('num_turns', [1, 2, 3])
def test_element_with_slicer_filling_scheme(buffer_round_trip, num_turns):
    ele = ElementWithSlicer(
        zeta_range=(-1, 1),
        num_slices=10,
        filling_scheme=[1, 0, 1, 1],
        bunch_spacing_zeta=5,
        num_turns=num_turns,
        circumference=100,
        with_compressed_profile=True)
    slicer = ele.slicer

    ele1 = ElementWithSlicer(
        zeta_range=(-1, 1),
        num_slices=10,
        filling_scheme=[1, 0, 1, 1],
        bunch_spacing_zeta=5,
        num_turns=num_turns,
        circumference=100,
        bunch_selection=[1, 2],
        with_compressed_profile=True)
    slicer1 = ele1.slicer

    ele2 = ElementWithSlicer(
        zeta_range=(-1, 1),
        num_slices=10,
        filling_scheme=[1, 0, 1, 1],
        bunch_spacing_zeta=5,
        num_turns=num_turns,
        circumference=100,
        bunch_selection=[0],
        with_compressed_profile=True)
    slicer2 = ele2.slicer

    zeta = np.linspace(-20, 20, 1000000)
    particles = xt.Particles(p0c=7000e9,
                            zeta=zeta)

    # different weight for the different bunches
    zeta = particles.zeta
    mask_bunch0 = (zeta > -1) & (zeta < 1)
    particles.weight[mask_bunch0] = 1
    mask_bunch1 = (zeta > -6) & (zeta < -4)
    particles.weight[mask_bunch1] = 2
    mask_bunch2 = (zeta > -11) & (zeta < -9)
    particles.weight[mask_bunch2] = 3
    mask_bunch3 = (zeta > -16) & (zeta < -14)
    particles.weight[mask_bunch3] = 4

    # line density
    num_particles_bunch0 = np.sum(particles.weight[mask_bunch0])
    xo.assert_allclose(num_particles_bunch0, 50000.0, rtol=1e-5, atol=0)

    ele.track(particles)
    ele1.track(particles)
    ele2.track(particles)

    if buffer_round_trip:
        slicer = xf.UniformBinSlicer._from_npbuffer(slicer._to_npbuffer())
        slicer1 = xf.UniformBinSlicer._from_npbuffer(slicer1._to_npbuffer())
        slicer2 = xf.UniformBinSlicer._from_npbuffer(slicer2._to_npbuffer())

    assert (slicer.filled_slots == np.array([0, 2, 3])).all()
    assert (slicer1.filled_slots == np.array([0, 2, 3])).all()
    assert (slicer2.filled_slots == np.array([0, 2, 3])).all()
    assert (slicer.bunch_selection == np.array([0, 1, 2])).all()
    assert (slicer1.bunch_selection == np.array([1, 2])).all()
    assert (slicer2.bunch_selection == np.array([0])).all()
    assert slicer.num_bunches == 3
    assert slicer1.num_bunches == 2
    assert slicer2.num_bunches == 1

    xo.assert_allclose(slicer.zeta_centers,
        np.array([[ -0.9,  -0.7,  -0.5,  -0.3,  -0.1,   0.1,   0.3,   0.5,   0.7,   0.9],
                [-10.9, -10.7, -10.5, -10.3, -10.1,  -9.9,  -9.7,  -9.5,  -9.3,  -9.1],
                [-15.9, -15.7, -15.5, -15.3, -15.1, -14.9, -14.7, -14.5, -14.3, -14.1]]),
        rtol=0, atol=1e-12)

    xo.assert_allclose(slicer1.zeta_centers,
        np.array([[-10.9, -10.7, -10.5, -10.3, -10.1,  -9.9,  -9.7,  -9.5,  -9.3,  -9.1],
                [-15.9, -15.7, -15.5, -15.3, -15.1, -14.9, -14.7, -14.5, -14.3, -14.1]]),
        rtol=0, atol=1e-12)

    xo.assert_allclose(slicer2.zeta_centers,
        np.array([[-0.9,  -0.7,  -0.5,  -0.3,  -0.1,   0.1,   0.3,   0.5,   0.7,   0.9]]),
        rtol=0, atol=1e-12)

    xo.assert_allclose(slicer.zeta_range[0], -1, rtol=0, atol=1e-12)
    xo.assert_allclose(slicer1.zeta_range[0], -1, rtol=0, atol=1e-12)
    xo.assert_allclose(slicer2.zeta_range[0], -1, rtol=0, atol=1e-12)
    xo.assert_allclose(slicer.zeta_range[1], 1, rtol=0, atol=1e-12)
    xo.assert_allclose(slicer1.zeta_range[1], 1, rtol=0, atol=1e-12)
    xo.assert_allclose(slicer2.zeta_range[1], 1, rtol=0, atol=1e-12)

    xo.assert_allclose(slicer.num_particles,
        np.array([
            [ 5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.],
            [15000., 15000., 15000., 15000., 15000., 15000., 15000., 15000., 15000., 15000.],
            [20000., 20000., 20000., 20000., 20000., 20000., 20000., 20000., 20000., 20000.]]),
        rtol=0, atol=1e-12)

    xo.assert_allclose(slicer1.num_particles,
        np.array([
            [15000., 15000., 15000., 15000., 15000., 15000., 15000., 15000., 15000., 15000.],
            [20000., 20000., 20000., 20000., 20000., 20000., 20000., 20000., 20000., 20000.]]),
        rtol=0, atol=1e-12)

    xo.assert_allclose(slicer2.num_particles,
        np.array([
            [ 5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.],
            ]),
        rtol=0, atol=1e-12)

    z_prof, prof = ele.moments_data.get_moment_profile('num_particles', i_turn=0)
    xo.assert_allclose(z_prof,
        np.array([-15.9, -15.7, -15.5, -15.3, -15.1, -14.9, -14.7, -14.5, -14.3,
                -14.1, -10.9, -10.7, -10.5, -10.3, -10.1,  -9.9,  -9.7,  -9.5,
                -9.3,  -9.1,  -5.9,  -5.7,  -5.5,  -5.3,  -5.1,  -4.9,  -4.7,
                -4.5,  -4.3,  -4.1,  -0.9,  -0.7,  -0.5,  -0.3,  -0.1,   0.1,
                    0.3,   0.5,   0.7,   0.9]), rtol=0, atol=1e-12)
    z_prof1, prof1 = ele1.moments_data.get_moment_profile('num_particles', i_turn=0)
    z_prof2, prof2 = ele2.moments_data.get_moment_profile('num_particles', i_turn=0)

    xo.assert_allclose(z_prof, z_prof1, rtol=0, atol=1e-12)
    xo.assert_allclose(z_prof, z_prof2, rtol=0, atol=1e-12)

    xo.assert_allclose(prof,
        np.array([20000., 20000., 20000., 20000., 20000., 20000., 20000., 20000.,
                20000., 20000., 15000., 15000., 15000., 15000., 15000., 15000.,
                15000., 15000., 15000., 15000.,     0.,     0.,     0.,     0.,
                    0.,     0.,     0.,     0.,     0.,     0.,  5000.,  5000.,
                5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.,  5000.]),
        rtol=0, atol=1e-12)

    xo.assert_allclose(prof1,
        np.array([20000., 20000., 20000., 20000., 20000., 20000., 20000., 20000.,
                20000., 20000., 15000., 15000., 15000., 15000., 15000., 15000.,
                15000., 15000., 15000., 15000.,     0.,     0.,     0.,     0.,
                    0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.,
                    0.,     0.,     0.,     0.,     0.,     0.,     0.,     0.]),
        rtol=0, atol=1e-12)

    xo.assert_allclose(prof2,
        np.array([   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
                    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
                    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
                    0.,    0.,    0., 5000., 5000., 5000., 5000., 5000., 5000.,
                5000., 5000., 5000., 5000.]),
        rtol=0, atol=1e-12)

    ele1._add_slicer_moments_to_moments_data(ele2.slicer)
    ele2._add_slicer_moments_to_moments_data(ele1.slicer)

    z_prof1_sum, prof1_sum = ele1.moments_data.get_moment_profile('num_particles', i_turn=0)
    z_prof2_sum, prof2_sum = ele2.moments_data.get_moment_profile('num_particles', i_turn=0)

    xo.assert_allclose(z_prof1_sum, z_prof, rtol=0, atol=1e-12)
    xo.assert_allclose(z_prof2_sum, z_prof, rtol=0, atol=1e-12)
    xo.assert_allclose(prof1_sum, prof, rtol=0, atol=1e-12)
    xo.assert_allclose(prof2_sum, prof, rtol=0, atol=1e-12)

    for i_turn in range(1, num_turns):
        particles.weight *= 2
        ele.track(particles)
        ele1.track(particles)
        ele2.track(particles)
        ele1._add_slicer_moments_to_moments_data(ele2.slicer)
        ele2._add_slicer_moments_to_moments_data(ele1.slicer)

    for i_check in range(1, num_turns):
        z_prof_turn,  prof_turn = ele.moments_data.get_moment_profile('num_particles', i_turn=i_check)
        z_prof1_turn, prof1_turn = ele1.moments_data.get_moment_profile('num_particles', i_turn=i_check)
        z_prof2_turn, prof2_turn = ele2.moments_data.get_moment_profile('num_particles', i_turn=i_check)
        xo.assert_allclose(z_prof_turn, z_prof, rtol=0, atol=1e-12)
        xo.assert_allclose(z_prof1_turn, z_prof, rtol=0, atol=1e-12)
        xo.assert_allclose(z_prof2_turn, z_prof, rtol=0, atol=1e-12)

        xo.assert_allclose(prof_turn, (2**(num_turns-1-i_check)) * prof, rtol=0, atol=1e-12)

