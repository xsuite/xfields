import numpy as np
import pytest

import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp

from xobjects.test_helpers import for_all_test_contexts

import ducktrack as dtk


@for_all_test_contexts(excluding="ContextPyopencl")
def test_digitize(test_context):
    print(repr(test_context))

    slicer = xf.TempSlicer(
        _context=test_context, n_slices=3, sigma_z=1, mode="unicharge"
    )
    z = np.sort(np.hstack([10, slicer.bin_centers, slicer.bin_edges, -10]))

    particles = xp.Particles(
        _context=test_context,
        zeta=z,
    )

    # get slice indices using kernel, bins sorted in decreasing order
    slice_indices = slicer.get_slice_indices(particles)

    # get slice indices using python
    np_digitize = (
        test_context.nplike_lib.array(
            [np.digitize(i, slicer.bin_edges, right=True) for i in z]
        )
        - 1
    )

    # x in ]xmin, xmax]
    # slice id 0 is the head of the bunch
    # slice id -1 is ahead of first bin, n_slices is behind last bin or at last bin lower edge
    assert np.all(slice_indices == np_digitize), "Slice indices do not match!"


@for_all_test_contexts(excluding="ContextPyopencl")
def test_compute_moments_1(test_context):
    print(repr(test_context))

    ###########
    # ttbar 2 #
    ###########
    p0c = 182.5e9  # [eV]
    mass0 = 0.511e6  # [eV]
    physemit_x = 1.46e-09  # [m]
    physemit_y = 2.9e-12  # [m]
    beta_x = 1  # [m]
    beta_y = 0.0016  # [m]
    sigma_x = np.sqrt(physemit_x * beta_x)  # [m]
    sigma_px = np.sqrt(physemit_x / beta_x)  # [m]
    sigma_y = np.sqrt(physemit_y * beta_y)  # [m]
    sigma_py = np.sqrt(physemit_y / beta_y)  # [m]
    sigma_z_tot = 0.00254  # [m] sr+bs
    sigma_delta_tot = 0.00192  # [m]
    n_macroparticles = int(1e6)

    threshold_num_macroparticles = 20
    n_slices_list = [1, 2, 5, 10, 100]

    # on GPU check for multiple grid settings
    if isinstance(test_context, xo.ContextCupy):
        default_blocksize_list = [1, 256, 1024]
    else:
        default_blocksize_list = [0]

    for default_blocksize in default_blocksize_list:
        if isinstance(test_context, xo.ContextCupy):
            test_context.default_block_size = default_blocksize
        for n_slices in n_slices_list:
            print(f"[test.py] blocksize: {default_blocksize}, n_slices: {n_slices}")

            #############
            # particles #
            #############

            # e-
            part_range = np.linspace(
                -5 * sigma_z_tot, 5 * sigma_z_tot, n_macroparticles
            )
            particles_b0 = xp.Particles(
                _context=test_context,
                q0=-1,
                p0c=p0c,
                mass0=mass0,
                x=part_range,
                px=part_range,
                y=part_range,
                py=part_range,
                zeta=part_range,
                delta=part_range,
            )

            # this is where the kernel gets built and the shared mem. and blocksize gets updated on GPU
            slicer = xf.TempSlicer(
                _context=test_context,
                n_slices=n_slices,
                sigma_z=sigma_z_tot,
                mode="unicharge",
            )

            particles_b1 = particles_b0.copy()
            particles_b2 = particles_b0.copy()
            particles_b2.state[
                : int(n_macroparticles / 4)
            ] = 0  # set 1/4 of the particles to lost (reduce if more slices)

            # np.cumsum[-1] =/= np.sum due to different order of summation
            # use np.isclose instead of ==; np.sum does pariwise sum which orders values differently thus causing a numerical error
            # https://stackoverflow.com/questions/69610452/why-does-the-last-entry-of-numpy-cumsum-not-necessarily-equal-numpy-sum
            # check if the mean and std of the alive particles in each slice agrees with Xfields compute_moments
            for particles in [particles_b1, particles_b2]:
                # compute slice moments: lost particles are labeled with state=0 and their slice idx will be set to -1
                slice_moments_xfields = slicer.compute_moments(
                    particles, threshold_num_macroparticles=threshold_num_macroparticles
                )

                # check if all lost particles have slice idx = -1
                assert np.all(
                    particles.slice[particles.state == 0] == -1
                ), "Not all lost particles have slice -1!"

                slice_moments = test_context.zeros(
                    n_slices * (1 + 6 + 10), dtype=np.float64
                )  # count (1) + moments (16)

                # compute moments here in python
                for i_slice in range(n_slices):
                    mask = (
                        particles.slice == i_slice
                    )  # dead particles are all in slice -1
                    num_macroparticles_slice = mask.sum()  # sums up True as 1

                    if num_macroparticles_slice < threshold_num_macroparticles:
                        slice_moments[i_slice] = 0  # n macroparts
                        slice_moments[n_slices + i_slice] = 0  # <x>
                        slice_moments[2 * n_slices + i_slice] = 0  # <px>
                        slice_moments[3 * n_slices + i_slice] = 0  # <y>
                        slice_moments[4 * n_slices + i_slice] = 0  # <py>
                        slice_moments[5 * n_slices + i_slice] = 0  # <z>
                        slice_moments[6 * n_slices + i_slice] = 0  # <delta>

                        slice_moments[7 * n_slices + i_slice] = 0  # Sigma_11
                        slice_moments[8 * n_slices + i_slice] = 0  # Sigma_12
                        slice_moments[9 * n_slices + i_slice] = 0  # Sigma_13
                        slice_moments[10 * n_slices + i_slice] = 0  # Sigma_14
                        slice_moments[11 * n_slices + i_slice] = 0  # Sigma_22
                        slice_moments[12 * n_slices + i_slice] = 0  # Sigma_23
                        slice_moments[13 * n_slices + i_slice] = 0  # Sigma_24
                        slice_moments[14 * n_slices + i_slice] = 0  # Sigma_33
                        slice_moments[15 * n_slices + i_slice] = 0  # Sigma_34
                        slice_moments[16 * n_slices + i_slice] = 0  # Sigma_44
                    else:
                        slice_moments[
                            i_slice
                        ] = num_macroparticles_slice  # n macroparts
                        slice_moments[n_slices + i_slice] = (
                            particles.x[mask].sum() / num_macroparticles_slice
                        )  # <x>
                        slice_moments[2 * n_slices + i_slice] = (
                            particles.px[mask].sum() / num_macroparticles_slice
                        )  # <px>
                        slice_moments[3 * n_slices + i_slice] = (
                            particles.y[mask].sum() / num_macroparticles_slice
                        )  # <y>
                        slice_moments[4 * n_slices + i_slice] = (
                            particles.py[mask].sum() / num_macroparticles_slice
                        )  # <py>
                        slice_moments[5 * n_slices + i_slice] = (
                            particles.zeta[mask].sum() / num_macroparticles_slice
                        )  # <z>
                        slice_moments[6 * n_slices + i_slice] = (
                            particles.delta[mask].sum() / num_macroparticles_slice
                        )  # <delta>

                        x_diff = particles.x[mask] - slice_moments[n_slices + i_slice]
                        px_diff = (
                            particles.px[mask] - slice_moments[2 * n_slices + i_slice]
                        )
                        y_diff = (
                            particles.y[mask] - slice_moments[3 * n_slices + i_slice]
                        )
                        py_diff = (
                            particles.py[mask] - slice_moments[4 * n_slices + i_slice]
                        )

                        slice_moments[7 * n_slices + i_slice] = (
                            x_diff**2
                        ).sum() / num_macroparticles_slice  # Sigma_11
                        slice_moments[8 * n_slices + i_slice] = (
                            x_diff * px_diff
                        ).sum() / num_macroparticles_slice  # Sigma_12
                        slice_moments[9 * n_slices + i_slice] = (
                            x_diff * y_diff
                        ).sum() / num_macroparticles_slice  # Sigma_13
                        slice_moments[10 * n_slices + i_slice] = (
                            x_diff * py_diff
                        ).sum() / num_macroparticles_slice  # Sigma_14
                        slice_moments[11 * n_slices + i_slice] = (
                            px_diff**2
                        ).sum() / num_macroparticles_slice  # Sigma_22
                        slice_moments[12 * n_slices + i_slice] = (
                            px_diff * y_diff
                        ).sum() / num_macroparticles_slice  # Sigma_23
                        slice_moments[13 * n_slices + i_slice] = (
                            px_diff * py_diff
                        ).sum() / num_macroparticles_slice  # Sigma_24
                        slice_moments[14 * n_slices + i_slice] = (
                            y_diff**2
                        ).sum() / num_macroparticles_slice  # Sigma_33
                        slice_moments[15 * n_slices + i_slice] = (
                            y_diff * py_diff
                        ).sum() / num_macroparticles_slice  # Sigma_34
                        slice_moments[16 * n_slices + i_slice] = (
                            py_diff**2
                        ).sum() / num_macroparticles_slice  # Sigma_44

                # check for each moment
                assert np.allclose(
                    slice_moments, slice_moments_xfields, atol=1e-16
                ), f"Xfields moment computation (n_slices={n_slices}, n_macroparticles={n_macroparticles}, blocksize(on GPU only)={default_blocksize}) is wrong!"


@for_all_test_contexts(excluding="ContextPyopencl")
def test_compute_moments_2(test_context):
    print(repr(test_context))

    ###########
    # ttbar 2 #
    ###########
    p0c = 182.5e9  # [eV]
    mass0 = 0.511e6  # [eV]
    physemit_x = 1.46e-09  # [m]
    physemit_y = 2.9e-12  # [m]
    beta_x = 1  # [m]
    beta_y = 0.0016  # [m]
    sigma_x = np.sqrt(physemit_x * beta_x)  # [m]
    sigma_px = np.sqrt(physemit_x / beta_x)  # [m]
    sigma_y = np.sqrt(physemit_y * beta_y)  # [m]
    sigma_py = np.sqrt(physemit_y / beta_y)  # [m]
    sigma_z_tot = 0.00254  # [m] sr+bs
    sigma_delta_tot = 0.00192  # [m]
    n_macroparticles = int(1e6)

    n_slices = 2
    threshold_num_macroparticles = 20

    #############
    # particles #
    #############

    # e-
    part_range = np.linspace(-5 * sigma_z_tot, 5 * sigma_z_tot, n_macroparticles)

    particles_b0 = xp.Particles(
        _context=test_context,
        q0=-1,
        p0c=p0c,
        mass0=mass0,
        x=part_range,
        zeta=part_range,
    )

    binning_list = ["unibin", "unicharge", "shatilov"]

    # on GPU check for multiple grid settings
    default_blocksize_list = [0]
    if isinstance(test_context, xo.ContextCupy):
        default_blocksize_list = [1, 256, 1024]

    for default_blocksize in default_blocksize_list:
        if isinstance(test_context, xo.ContextCupy):
            test_context.default_block_size = default_blocksize
            print(f"[test.py] default_blocksize: {test_context.default_block_size}")

        for binning in binning_list:
            print(f"[test.py] binning: {binning}")

            particles = particles_b0.copy()

            # unibin extends [-5sigma, 5sigma], unicharge and shatilov extend ]xmin, xmax]
            slicer = xf.TempSlicer(
                _context=test_context,
                n_slices=n_slices,
                sigma_z=sigma_z_tot,
                mode=binning,
            )

            # compute number of particles in bunch head slice
            slice_moments_xfields = slicer.compute_moments(
                particles, threshold_num_macroparticles=threshold_num_macroparticles
            )
            x_center_before = slice_moments_xfields[n_slices : 2 * n_slices]
            print(
                f"[test.py] x_center_before: {x_center_before}, slice indices: {particles.slice}"
            )

            # lose a particle in the head of the bunch (slice 0) during tracking, slice moment is messed up
            lost_idx = -1  # index 0 is in slice idx n_slices so out of bins
            particles.x[lost_idx] = 1e34

            slice_moments_xfields = slicer.compute_moments(
                particles, threshold_num_macroparticles=threshold_num_macroparticles
            )
            x_center = slice_moments_xfields[n_slices : 2 * n_slices]
            print(f"[test.py] x_center: {x_center}, slice indices: {particles.slice}")

            # now the first slice mean should be huge
            assert x_center[0] > 1e20, f"Mismatch in {binning} binning!"

            # update particle status to dead, recompute slices
            particles.state[lost_idx] = 0
            slice_moments_xfields = slicer.compute_moments(
                particles, threshold_num_macroparticles=threshold_num_macroparticles
            )
            x_center_after = slice_moments_xfields[n_slices : 2 * n_slices]
            print(
                f"[test.py] x_center_after: {x_center_after}, slice indices: {particles.slice}"
            )

            # using 1e6 particles, the diff in means if I skip one element should be order of 1e-6
            assert (
                np.abs((x_center_after[0] - x_center_before[0]) / x_center_before[0])
                < 1e-5
            ), f"Mismatch in {binning} binning!"


def sigma_configurations():
    print("decoupled round beam")
    (
        Sig_11_0,
        Sig_12_0,
        Sig_13_0,
        Sig_14_0,
        Sig_22_0,
        Sig_23_0,
        Sig_24_0,
        Sig_33_0,
        Sig_34_0,
        Sig_44_0,
    ) = (20e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20e-6, 0.0, 0.0)
    yield (
        Sig_11_0,
        Sig_12_0,
        Sig_13_0,
        Sig_14_0,
        Sig_22_0,
        Sig_23_0,
        Sig_24_0,
        Sig_33_0,
        Sig_34_0,
        Sig_44_0,
    )

    print("decoupled tall beam")
    (
        Sig_11_0,
        Sig_12_0,
        Sig_13_0,
        Sig_14_0,
        Sig_22_0,
        Sig_23_0,
        Sig_24_0,
        Sig_33_0,
        Sig_34_0,
        Sig_44_0,
    ) = (20e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 40e-6, 0.0, 0.0)
    yield (
        Sig_11_0,
        Sig_12_0,
        Sig_13_0,
        Sig_14_0,
        Sig_22_0,
        Sig_23_0,
        Sig_24_0,
        Sig_33_0,
        Sig_34_0,
        Sig_44_0,
    )

    print("decoupled fat beam")
    (
        Sig_11_0,
        Sig_12_0,
        Sig_13_0,
        Sig_14_0,
        Sig_22_0,
        Sig_23_0,
        Sig_24_0,
        Sig_33_0,
        Sig_34_0,
        Sig_44_0,
    ) = (40e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20e-6, 0.0, 0.0)
    yield (
        Sig_11_0,
        Sig_12_0,
        Sig_13_0,
        Sig_14_0,
        Sig_22_0,
        Sig_23_0,
        Sig_24_0,
        Sig_33_0,
        Sig_34_0,
        Sig_44_0,
    )

    print("coupled beam")
    (
        Sig_11_0,
        Sig_12_0,
        Sig_13_0,
        Sig_14_0,
        Sig_22_0,
        Sig_23_0,
        Sig_24_0,
        Sig_33_0,
        Sig_34_0,
        Sig_44_0,
    ) = (
        8.4282060230000004e-06,
        1.8590458800000001e-07,
        -3.5512334410000001e-06,
        -3.8254462239999997e-08,
        4.101510281e-09,
        -7.5517657920000006e-08,
        -8.1134615060000002e-10,
        1.031446898e-05,
        1.177863077e-07,
        1.3458251810000001e-09,
    )
    yield (
        Sig_11_0,
        Sig_12_0,
        Sig_13_0,
        Sig_14_0,
        Sig_22_0,
        Sig_23_0,
        Sig_24_0,
        Sig_33_0,
        Sig_34_0,
        Sig_44_0,
    )


for_all_sigma_configurations = pytest.mark.parametrize(
    "Sig_11_0, Sig_12_0, Sig_13_0, Sig_14_0, Sig_22_0, Sig_23_0, Sig_24_0,"
    "Sig_33_0, Sig_34_0, Sig_44_0",
    list(sigma_configurations()),
    ids=[
        "decoupled round beam",
        "decoupled tall beam",
        "decoupled fat beam",
        "coupled beam",
    ],
)


@for_all_test_contexts
@for_all_sigma_configurations
def test_beambeam3d(
    test_context,
    Sig_11_0,
    Sig_12_0,
    Sig_13_0,
    Sig_14_0,
    Sig_22_0,
    Sig_23_0,
    Sig_24_0,
    Sig_33_0,
    Sig_34_0,
    Sig_44_0,
):
    # crossing plane
    alpha = 0.7

    # crossing angle
    phi = 0.8

    # separations
    x_bb_co = 5e-3
    y_bb_co = -4e-3
    charge_slices = np.array([1e16, 2e16, 5e16])
    z_slices = np.array([-6.0, 0.2, 5.5])

    x_co = 2e-3
    px_co = 1e-6
    y_co = -3e-3
    py_co = -2e-6
    zeta_co = 0.01
    delta_co = 1.2e-3

    d_x = 1.5e-3
    d_px = 1.6e-6
    d_y = -1.7e-3
    d_py = -1.8e-6
    d_zeta = 0.019
    d_delta = 3e-4

    Sig_11_0 = Sig_11_0 + np.zeros_like(charge_slices)
    Sig_12_0 = Sig_12_0 + np.zeros_like(charge_slices)
    Sig_13_0 = Sig_13_0 + np.zeros_like(charge_slices)
    Sig_14_0 = Sig_14_0 + np.zeros_like(charge_slices)
    Sig_22_0 = Sig_22_0 + np.zeros_like(charge_slices)
    Sig_23_0 = Sig_23_0 + np.zeros_like(charge_slices)
    Sig_24_0 = Sig_24_0 + np.zeros_like(charge_slices)
    Sig_33_0 = Sig_33_0 + np.zeros_like(charge_slices)
    Sig_34_0 = Sig_34_0 + np.zeros_like(charge_slices)
    Sig_44_0 = Sig_44_0 + np.zeros_like(charge_slices)

    # I modify one slice to check that properties are working correctly
    Sig_11_0[1] *= 1000
    Sig_12_0[1] *= 1000
    Sig_13_0[1] *= 1000
    Sig_14_0[1] *= 1000
    Sig_22_0[1] *= 1000
    Sig_23_0[1] *= 1000
    Sig_24_0[1] *= 1000
    Sig_33_0[1] *= 1000
    Sig_34_0[1] *= 1000
    Sig_44_0[1] *= 1000

    print("------------------------")
    print(locals())

    bb_dtk = dtk.elements.BeamBeam6D(
        phi=phi,
        alpha=alpha,
        x_bb_co=x_bb_co,
        y_bb_co=y_bb_co,
        charge_slices=charge_slices,
        zeta_slices=z_slices,
        sigma_11=Sig_11_0[0],
        sigma_12=Sig_12_0[0],
        sigma_13=Sig_13_0[0],
        sigma_14=Sig_14_0[0],
        sigma_22=Sig_22_0[0],
        sigma_23=Sig_23_0[0],
        sigma_24=Sig_24_0[0],
        sigma_33=Sig_33_0[0],
        sigma_34=Sig_34_0[0],
        sigma_44=Sig_44_0[0],
        x_co=x_co,
        px_co=px_co,
        y_co=y_co,
        py_co=py_co,
        zeta_co=zeta_co,
        delta_co=delta_co,
        d_x=d_x,
        d_px=d_px,
        d_y=d_y,
        d_py=d_py,
        d_zeta=d_zeta,
        d_delta=d_delta,
    )

    bb = xf.BeamBeamBiGaussian3D(
        _context=test_context,
        phi=phi,
        alpha=alpha,
        other_beam_q0=1,
        slices_other_beam_num_particles=charge_slices[::-1],
        slices_other_beam_zeta_center=z_slices[::-1],
        slices_other_beam_Sigma_11=Sig_11_0,
        slices_other_beam_Sigma_12=Sig_12_0,
        slices_other_beam_Sigma_13=Sig_13_0,
        slices_other_beam_Sigma_14=Sig_14_0,
        slices_other_beam_Sigma_22=Sig_22_0,
        slices_other_beam_Sigma_23=Sig_23_0,
        slices_other_beam_Sigma_24=Sig_24_0,
        slices_other_beam_Sigma_33=Sig_33_0,
        slices_other_beam_Sigma_34=Sig_34_0,
        slices_other_beam_Sigma_44=Sig_44_0,
        ref_shift_x=x_co,
        ref_shift_px=px_co,
        ref_shift_y=y_co,
        ref_shift_py=py_co,
        ref_shift_zeta=zeta_co,
        ref_shift_pzeta=delta_co,
        other_beam_shift_x=x_bb_co,
        other_beam_shift_y=y_bb_co,
        post_subtract_x=d_x,
        post_subtract_px=d_px,
        post_subtract_y=d_y,
        post_subtract_py=d_py,
        post_subtract_zeta=d_zeta,
        post_subtract_pzeta=d_delta,
    )

    ctx2np = bb._context.nparray_from_context_array  # Patch for Pyopencl
    bb.slices_other_beam_Sigma_11[1] = ctx2np(bb.slices_other_beam_Sigma_11)[0]
    bb.slices_other_beam_Sigma_12[1] = ctx2np(bb.slices_other_beam_Sigma_12)[0]
    bb.slices_other_beam_Sigma_13[1] = ctx2np(bb.slices_other_beam_Sigma_13)[0]
    bb.slices_other_beam_Sigma_14[1] = ctx2np(bb.slices_other_beam_Sigma_14)[0]
    bb.slices_other_beam_Sigma_22[1] = ctx2np(bb.slices_other_beam_Sigma_22)[0]
    bb.slices_other_beam_Sigma_23[1] = ctx2np(bb.slices_other_beam_Sigma_23)[0]
    bb.slices_other_beam_Sigma_24[1] = ctx2np(bb.slices_other_beam_Sigma_24)[0]
    bb.slices_other_beam_Sigma_33[1] = ctx2np(bb.slices_other_beam_Sigma_33)[0]
    bb.slices_other_beam_Sigma_34[1] = ctx2np(bb.slices_other_beam_Sigma_34)[0]
    bb.slices_other_beam_Sigma_44[1] = ctx2np(bb.slices_other_beam_Sigma_44)[0]

    dtk_part = dtk.TestParticles(
        p0c=6500e9, x=-1.23e-3, px=50e-3, y=2e-3, py=27e-3, sigma=3.0, delta=2e-4
    )

    part = xp.Particles(_context=test_context, **dtk_part.to_dict())

    part.name = "beam1_bunch1"

    ret = bb.track(part)

    bb_dtk.track(dtk_part)

    part.move(_context=xo.context_default)
    for cc in "x px y py zeta delta".split():
        val_test = getattr(part, cc)[0]
        val_ref = getattr(dtk_part, cc)
        print("")
        print(f"ducktrack: {cc} = {val_ref:.12e}")
        print(f"xsuite:    {cc} = {val_test:.12e}")
        assert np.isclose(val_test, val_ref, rtol=0, atol=5e-12)
    part.move(_context=test_context)

    # Scaling down bb:
    bb.scale_strength = 0
    part_before_tracking = part.copy()
    bb.track(part)

    part_before_tracking.move(_context=xo.context_default)
    part.move(_context=xo.context_default)
    for cc in "x px y py zeta delta".split():
        val_test = getattr(part, cc)[0]
        val_ref = getattr(part_before_tracking, cc)[0]
        print("")
        print(f"before: {cc} = {val_ref:.12e}")
        print(f"after bb off:    {cc} = {val_test:.12e}")
        assert np.allclose(val_test, val_ref, rtol=0, atol=1e-14)


@for_all_test_contexts
@for_all_sigma_configurations
def test_beambeam3d_single_slice(
    test_context,
    Sig_11_0,
    Sig_12_0,
    Sig_13_0,
    Sig_14_0,
    Sig_22_0,
    Sig_23_0,
    Sig_24_0,
    Sig_33_0,
    Sig_34_0,
    Sig_44_0,
):
    # crossing plane
    alpha = 0.7

    # crossing angle
    phi = 0.8

    # separations
    x_bb_co = 5e-3
    y_bb_co = -4e-3
    charge_slices = [1e16]
    z_slices = [-6.0]

    x_co = 2e-3
    px_co = 1e-6
    y_co = -3e-3
    py_co = -2e-6
    zeta_co = 0.01
    delta_co = 1.2e-3

    d_x = 1.5e-3
    d_px = 1.6e-6
    d_y = -1.7e-3
    d_py = -1.8e-6
    d_zeta = 0.019
    d_delta = 3e-4

    Sig_11_0 = Sig_11_0 + np.zeros_like(charge_slices)
    Sig_12_0 = Sig_12_0 + np.zeros_like(charge_slices)
    Sig_13_0 = Sig_13_0 + np.zeros_like(charge_slices)
    Sig_14_0 = Sig_14_0 + np.zeros_like(charge_slices)
    Sig_22_0 = Sig_22_0 + np.zeros_like(charge_slices)
    Sig_23_0 = Sig_23_0 + np.zeros_like(charge_slices)
    Sig_24_0 = Sig_24_0 + np.zeros_like(charge_slices)
    Sig_33_0 = Sig_33_0 + np.zeros_like(charge_slices)
    Sig_34_0 = Sig_34_0 + np.zeros_like(charge_slices)
    Sig_44_0 = Sig_44_0 + np.zeros_like(charge_slices)

    print("------------------------")
    print(locals())

    bb_dtk = dtk.elements.BeamBeam6D(
        phi=phi,
        alpha=alpha,
        x_bb_co=x_bb_co,
        y_bb_co=y_bb_co,
        charge_slices=charge_slices,
        zeta_slices=z_slices,
        sigma_11=Sig_11_0[0],
        sigma_12=Sig_12_0[0],
        sigma_13=Sig_13_0[0],
        sigma_14=Sig_14_0[0],
        sigma_22=Sig_22_0[0],
        sigma_23=Sig_23_0[0],
        sigma_24=Sig_24_0[0],
        sigma_33=Sig_33_0[0],
        sigma_34=Sig_34_0[0],
        sigma_44=Sig_44_0[0],
        x_co=x_co,
        px_co=px_co,
        y_co=y_co,
        py_co=py_co,
        zeta_co=zeta_co,
        delta_co=delta_co,
        d_x=d_x,
        d_px=d_px,
        d_y=d_y,
        d_py=d_py,
        d_zeta=d_zeta,
        d_delta=d_delta,
    )

    bb = xf.BeamBeamBiGaussian3D(
        _context=test_context,
        phi=phi,
        alpha=alpha,
        other_beam_q0=1,
        slices_other_beam_num_particles=charge_slices[::-1],
        slices_other_beam_zeta_center=z_slices[::-1],
        slices_other_beam_Sigma_11=Sig_11_0,
        slices_other_beam_Sigma_12=Sig_12_0,
        slices_other_beam_Sigma_13=Sig_13_0,
        slices_other_beam_Sigma_14=Sig_14_0,
        slices_other_beam_Sigma_22=Sig_22_0,
        slices_other_beam_Sigma_23=Sig_23_0,
        slices_other_beam_Sigma_24=Sig_24_0,
        slices_other_beam_Sigma_33=Sig_33_0,
        slices_other_beam_Sigma_34=Sig_34_0,
        slices_other_beam_Sigma_44=Sig_44_0,
        ref_shift_x=x_co,
        ref_shift_px=px_co,
        ref_shift_y=y_co,
        ref_shift_py=py_co,
        ref_shift_zeta=zeta_co,
        ref_shift_pzeta=delta_co,
        other_beam_shift_x=x_bb_co,
        other_beam_shift_y=y_bb_co,
        post_subtract_x=d_x,
        post_subtract_px=d_px,
        post_subtract_y=d_y,
        post_subtract_py=d_py,
        post_subtract_zeta=d_zeta,
        post_subtract_pzeta=d_delta,
    )

    dtk_part = dtk.TestParticles(
        p0c=6500e9, x=-1.23e-3, px=50e-3, y=2e-3, py=27e-3, sigma=3.0, delta=2e-4
    )

    part = xp.Particles(_context=test_context, **dtk_part.to_dict())

    part.name = "beam1_bunch1"

    ret = bb.track(part)

    bb_dtk.track(dtk_part)

    part.move(_context=xo.context_default)
    for cc in "x px y py zeta delta".split():
        val_test = getattr(part, cc)[0]
        val_ref = getattr(dtk_part, cc)
        print("")
        print(f"ducktrack: {cc} = {val_ref:.12e}")
        print(f"xsuite:    {cc} = {val_test:.12e}")
        assert np.isclose(val_test, val_ref, rtol=0, atol=5e-12)
    part.move(_context=test_context)

    # Scaling down bb:
    bb.scale_strength = 0
    part_before_tracking = part.copy()
    bb.track(part)

    part_before_tracking.move(_context=xo.context_default)
    part.move(_context=xo.context_default)
    for cc in "x px y py zeta delta".split():
        val_test = getattr(part, cc)[0]
        val_ref = getattr(part_before_tracking, cc)[0]
        print("")
        print(f"before: {cc} = {val_ref:.12e}")
        print(f"after bb off:    {cc} = {val_test:.12e}")
        assert np.allclose(val_test, val_ref, rtol=0, atol=1e-14)
