import numpy as np
import pytest

import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp

from xobjects.test_helpers import for_all_test_contexts

import ducktrack as dtk

@for_all_test_contexts
def test_digitize(test_context):

    print(repr(test_context))

    if isinstance(test_context, xo.ContextPyopencl):
        pytest.skip("Not implemented for OpenCL")
        return

    if isinstance(test_context, xo.ContextCupy):
        pytest.skip("Not implemented for cupy")
        return

    slicer = xf.TempSlicer(_context=test_context, n_slices=3, sigma_z=1, mode="unicharge")
    z = np.sort(np.hstack([10, slicer.bin_centers, slicer.bin_edges, -10]))

    particles = xp.Particles(_context=test_context, zeta=z)

    # get slice indices using kernel, bins sorted in decreasing order
    slice_indices = slicer.get_slice_indices(particles)

    # get slice indices using python
    np_digitize = test_context.nplike_lib.array(
        [np.digitize(i, slicer.bin_edges, right=True) for i in z]) - 1

    # x in ]xmin, xmax]
    # slice id 0 is the head of the bunch
    # slice id -1 is ahead of first bin, n_slices is behind last bin or at last bin lower edge
    assert np.all(slice_indices == np_digitize), "Slice indices do not match!"

@for_all_test_contexts
def test_compute_moments_1(test_context):

    if isinstance(test_context, xo.ContextPyopencl):
        pytest.skip("Not implemented for OpenCL")
        return

    if isinstance(test_context, xo.ContextCupy):
        pytest.skip("Not implemented for cupy")
        return

    ###########
    # ttbar 2 #
    ###########
    p0c                 = 182.5e9  # [eV]
    mass0               = .511e6  # [eV]
    physemit_x          = 1.46e-09  # [m]
    physemit_y          = 2.9e-12  # [m]
    beta_x              = 1  # [m]
    beta_y              = .0016  # [m]
    sigma_x             = np.sqrt(physemit_x*beta_x)  # [m]
    sigma_px            = np.sqrt(physemit_x/beta_x)  # [m]
    sigma_y             = np.sqrt(physemit_y*beta_y)  # [m]
    sigma_py            = np.sqrt(physemit_y/beta_y)  # [m]
    sigma_z_tot         = .00254  # [m] sr+bs
    sigma_delta_tot     = .00192  # [m]
    n_macroparticles = int(1e6)

    threshold_num_macroparticles=20
    n_slices_list = [1, 100]

    # on GPU check for multiple grid settings
    if isinstance(test_context, xo.ContextCupy):
        default_blocksize_list = [1, 256, 1024]
    else:
        default_blocksize_list = [0]

    for default_blocksize in default_blocksize_list:

        for n_slices in n_slices_list:

            print(f"[test.py] n_slices: {n_slices}")

            if isinstance(test_context, xo.ContextCupy):
                test_context.default_shared_mem_size_bytes = n_slices * 17 * 8

            #############
            # particles #
            #############

            #e-
            part_range = np.linspace(-5*sigma_z_tot,5*sigma_z_tot,n_macroparticles)

            particles_b0 = xp.Particles(
                        _context = test_context,
                        q0        = -1,
                        p0c       = p0c,
                        mass0     = mass0,
                        x         = part_range,
                        px        = part_range,
                        y         = part_range,
                        py        = part_range,
                        zeta      = part_range,
                        delta     = part_range,
                        )

            slicer = xf.TempSlicer(
                _context=test_context, n_slices=n_slices, sigma_z=sigma_z_tot,
                mode="unicharge")

            particles_b1 = particles_b0.copy()
            particles_b2 = particles_b0.copy()
            particles_b2.state[:int(n_macroparticles/4)] = 0  # set 1/4 of the particles to lost (reduce if more slices)

            # np.cumsum[-1] =/= np.sum due to different order of summation
            # use np.isclose instead of ==; np.sum does pariwise sum which orders values differently thus causing a numerical error
            # https://stackoverflow.com/questions/69610452/why-does-the-last-entry-of-numpy-cumsum-not-necessarily-equal-numpy-sum
            # check if the mean and std of the alive particles in each slice agrees with Xfields compute_moments
            for particles in [particles_b1, particles_b2]:

                # compute slice moments: lost particles are labeled with state=0 and their slice idx will be set to -1
                slice_moments_xfields = slicer.compute_moments(particles, threshold_num_macroparticles=threshold_num_macroparticles)

                # check if all lost particles have slice idx = -1
                assert np.all(particles.slice[particles.state == 0] == -1), "Not all lost particles have slice -1!"

                slice_moments = test_context.zeros(n_slices*(1+6+10),dtype=np.float64)  # count (1) + moments (16)

                for i_slice in range(n_slices):
                    mask = (particles.slice == i_slice)  # dead particles are all in slice -1
                    num_macroparticles_slice = mask.sum()  # sums up True as 1

                    if num_macroparticles_slice < threshold_num_macroparticles:
                        slice_moments[i_slice]              = 0  # n macroparts
                        slice_moments[n_slices+i_slice]   = 0  # <x>
                        slice_moments[2*n_slices+i_slice] = 0  # <px>
                        slice_moments[3*n_slices+i_slice] = 0  # <y>
                        slice_moments[4*n_slices+i_slice] = 0  # <py>
                        slice_moments[5*n_slices+i_slice] = 0  # <z>
                        slice_moments[6*n_slices+i_slice] = 0  # <delta>

                        slice_moments[7*n_slices+i_slice]  = 0  # Sigma_11
                        slice_moments[8*n_slices+i_slice]  = 0  # Sigma_12
                        slice_moments[9*n_slices+i_slice]  = 0  # Sigma_13
                        slice_moments[10*n_slices+i_slice] = 0  # Sigma_14
                        slice_moments[11*n_slices+i_slice] = 0  # Sigma_22
                        slice_moments[12*n_slices+i_slice] = 0  # Sigma_23
                        slice_moments[13*n_slices+i_slice] = 0  # Sigma_24
                        slice_moments[14*n_slices+i_slice] = 0  # Sigma_33
                        slice_moments[15*n_slices+i_slice] = 0  # Sigma_34
                        slice_moments[16*n_slices+i_slice] = 0  # Sigma_44
                    else:
                        slice_moments[i_slice]            = num_macroparticles_slice # n macroparts
                        slice_moments[n_slices+i_slice]   = particles.    x[mask].sum() / num_macroparticles_slice # <x>
                        slice_moments[2*n_slices+i_slice] = particles.   px[mask].sum() / num_macroparticles_slice # <px>
                        slice_moments[3*n_slices+i_slice] = particles.    y[mask].sum() / num_macroparticles_slice # <y>
                        slice_moments[4*n_slices+i_slice] = particles.   py[mask].sum() / num_macroparticles_slice # <py>
                        slice_moments[5*n_slices+i_slice] = particles. zeta[mask].sum() / num_macroparticles_slice # <z>
                        slice_moments[6*n_slices+i_slice] = particles.delta[mask].sum() / num_macroparticles_slice # <delta>

                        x_diff  = particles. x[mask] - slice_moments[n_slices+i_slice]
                        px_diff = particles.px[mask] - slice_moments[2*n_slices+i_slice]
                        y_diff  = particles. y[mask] - slice_moments[3*n_slices+i_slice]
                        py_diff = particles.py[mask] - slice_moments[4*n_slices+i_slice]

                        slice_moments[7*n_slices+i_slice]  = ( x_diff**2       ).sum() / num_macroparticles_slice  # Sigma_11
                        slice_moments[8*n_slices+i_slice]  = ( x_diff * px_diff).sum() / num_macroparticles_slice  # Sigma_12
                        slice_moments[9*n_slices+i_slice]  = ( x_diff *  y_diff).sum() / num_macroparticles_slice  # Sigma_13
                        slice_moments[10*n_slices+i_slice] = ( x_diff * py_diff).sum() / num_macroparticles_slice  # Sigma_14
                        slice_moments[11*n_slices+i_slice] = (px_diff**2       ).sum() / num_macroparticles_slice  # Sigma_22
                        slice_moments[12*n_slices+i_slice] = (px_diff *  y_diff).sum() / num_macroparticles_slice  # Sigma_23
                        slice_moments[13*n_slices+i_slice] = (px_diff * py_diff).sum() / num_macroparticles_slice  # Sigma_24
                        slice_moments[14*n_slices+i_slice] = ( y_diff**2       ).sum() / num_macroparticles_slice  # Sigma_33
                        slice_moments[15*n_slices+i_slice] = ( y_diff * py_diff).sum() / num_macroparticles_slice  # Sigma_34
                        slice_moments[16*n_slices+i_slice] = (py_diff**2       ).sum() / num_macroparticles_slice  # Sigma_44

                # check for each moment
                assert np.allclose(
                    slice_moments, slice_moments_xfields, atol=1e-16),(
                    f"Xfields moment computation (n_slices={n_slices}, "
                    f"n_macroparticles={n_macroparticles}, "
                    "blocksize(on GPU only)={default_blocksize}) is wrong!")

@for_all_test_contexts
def test_compute_moments_2(test_context):

    print(repr(test_context))

    if isinstance(test_context, xo.ContextPyopencl):
        pytest.skip("Not implemented for OpenCL")

    if isinstance(test_context, xo.ContextCupy):
        pytest.skip("Not implemented for cupy")

    ###########
    # ttbar 2 #
    ###########
    p0c                 = 182.5e9  # [eV]
    mass0               = .511e6  # [eV]
    physemit_x          = 1.46e-09  # [m]
    physemit_y          = 2.9e-12  # [m]
    beta_x              = 1  # [m]
    beta_y              = .0016  # [m]
    sigma_x             = np.sqrt(physemit_x*beta_x)  # [m]
    sigma_px            = np.sqrt(physemit_x/beta_x)  # [m]
    sigma_y             = np.sqrt(physemit_y*beta_y)  # [m]
    sigma_py            = np.sqrt(physemit_y/beta_y)  # [m]
    sigma_z_tot         = .00254  # [m] sr+bs
    sigma_delta_tot     = .00192  # [m]
    n_macroparticles = int(1e6)

    n_slices = 2
    threshold_num_macroparticles=20

    #############
    # particles #
    #############

    #e-
    part_range = np.linspace(-5*sigma_z_tot,5*sigma_z_tot,n_macroparticles)

    particles_b0 = xp.Particles(
                _context = test_context,
                q0        = -1,
                p0c       = p0c,
                mass0     = mass0,
                x         = part_range,
                zeta      = part_range,
                )

    binning_list = ["unibin", "unicharge", "shatilov"]

    # on GPU check for multiple grid settings
    default_blocksize_list = [0]
    if isinstance(test_context, xo.ContextCupy):
        test_context.default_shared_mem_size_bytes=n_slices*17*8
        default_blocksize_list = [1, 256, 1024]

    for default_blocksize in default_blocksize_list:
        if isinstance(test_context, xo.ContextCupy):
            test_context.default_block_size=default_blocksize
            print(f"[test.py] default_blocksize: {test_context.default_block_size}")

        for binning in binning_list:
            print(f"[test.py] binning: {binning}")

            particles = particles_b0.copy()

            # unibin extends [-5sigma, 5sigma], unicharge and shatilov extend ]xmin, xmax]
            slicer = xf.TempSlicer(_context=test_context, n_slices=n_slices, sigma_z=sigma_z_tot, mode=binning)

            # compute number of particles in bunch head slice
            slice_moments_xfields = slicer.compute_moments(particles, threshold_num_macroparticles=threshold_num_macroparticles)
            x_center_before  = slice_moments_xfields[n_slices:2*n_slices]
            print(f"[test.py] x_center_before: {x_center_before}, slice indices: {particles.slice}")

            # lose a particle in the head of the bunch (slice 0) during tracking, slice moment is messed up
            lost_idx = -1  # index 0 is in slice idx n_slices so out of bins
            particles.x[lost_idx] = 1e34

            slice_moments_xfields = slicer.compute_moments(particles, threshold_num_macroparticles=threshold_num_macroparticles)
            x_center              = slice_moments_xfields[   n_slices:2*n_slices]
            print(f"[test.py] x_center: {x_center}, slice indices: {particles.slice}")

            # now the first slice mean should be huge
            assert x_center[0] > 1e20, f"Mismatch in {binning} binning!"

            # update particle status to dead, recompute slices
            particles.state[lost_idx] = 0
            slice_moments_xfields     = slicer.compute_moments(particles, threshold_num_macroparticles=threshold_num_macroparticles)
            x_center_after            = slice_moments_xfields[n_slices:2*n_slices]
            print(f"[test.py] x_center_after: {x_center_after}, slice indices: {particles.slice}")

            # using 1e6 particles, the diff in means if I skip one element should be order of 1e-6
            assert np.abs((x_center_after[0]-x_center_before[0])/x_center_before[0]) < 1e-5, f"Mismatch in {binning} binning!"

