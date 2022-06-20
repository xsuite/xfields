# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
from scipy.special import wofz as wofz_scipy
import xobjects as xo
from xobjects.context import available
from xfields.general import _pkg_root


def test_cerrf_q1():

    ctx = xo.ContextCpu(omp_num_threads=2)

    xx = np.logspace(-8, +8, 51, dtype=np.float64)
    yy = np.logspace(-8, +8, 51, dtype=np.float64)

    n_re = len(xx)
    n_im = len(yy)
    n_z = len(yy) * len(xx)

    re_absc = np.arange(n_z, dtype=np.float64).reshape(n_im, n_re)
    im_absc = np.arange(n_z, dtype=np.float64).reshape(n_im, n_re)
    wz_cmp_re = np.arange(n_z, dtype=np.float64).reshape(n_im, n_re)
    wz_cmp_im = np.arange(n_z, dtype=np.float64).reshape(n_im, n_re)

    for jj, y in enumerate(yy):
        re_absc[jj, :] = xx[:]

    for ii, x in enumerate(xx):
        im_absc[:, ii] = yy[:]

    # Using scipy's wofz implemenation of the Faddeeva method. This is
    # (at the time of this writing in 2021) based on the MIT ab-initio
    # implementation using a combination of Algorithm 680 for large |z| and
    # Algorithm 916 for the remainder fo C. It claims a relative accuracy of
    # 1e-13 across the whole of C and is thus suitable to check the accuracy
    # of the cerrf_q1 implementation which has a target accuracy of 10^{-10}
    # in the *absolute* error.

    for jj, y in enumerate(yy):
        for ii, x in enumerate(xx):
            z = x + 1.0j * y
            wz = wofz_scipy(z)
            wz_cmp_re[jj, ii] = wz.real
            wz_cmp_im[jj, ii] = wz.imag

    src_code = """
    /*gpukern*/ void eval_cerrf_q1(
        const int n,
        /*gpuglmem*/ double const* /*restrict*/ re,
        /*gpuglmem*/ double const* /*restrict*/ im,
        /*gpuglmem*/ double* /*restrict*/ wz_re,
        /*gpuglmem*/ double* /*restrict*/ wz_im )
    {
        int tid = 0;
        for( ; tid < n ; ++tid ) { //autovectorized

            if( tid < n )
            {
                double const x = re[ tid ];
                double const y = im[ tid ];
                double wz_x, wz_y;

                cerrf_q1( x, y, &wz_x, &wz_y );

                wz_re[ tid ] = wz_x;
                wz_im[ tid ] = wz_y;
            }
        }
    }
    """

    kernel_descriptions = {
        "eval_cerrf_q1": xo.Kernel(
            args=[
                xo.Arg(xo.Int32, name="n"),
                xo.Arg(xo.Float64, name="re", const=True, pointer=True),
                xo.Arg(xo.Float64, name="im", const=True, pointer=True),
                xo.Arg(xo.Float64, name="wz_re", pointer=True),
                xo.Arg(xo.Float64, name="wz_im", pointer=True),
            ],
            n_threads="n",
        ),
    }

    headers = [
        _pkg_root.joinpath("headers/constants.h"),
        _pkg_root.joinpath("headers/sincos.h"),
        _pkg_root.joinpath("headers/power_n.h"),
        _pkg_root.joinpath("fieldmaps/bigaussian_src/complex_error_function.h"),
    ]

    wz_re = np.arange(n_z, dtype=np.float64)
    wz_im = np.arange(n_z, dtype=np.float64)

    re_absc_dev = ctx.nparray_to_context_array(re_absc.reshape(n_z))
    im_absc_dev = ctx.nparray_to_context_array(im_absc.reshape(n_z))
    wz_re_dev = ctx.nparray_to_context_array(wz_re)
    wz_im_dev = ctx.nparray_to_context_array(wz_im)

    ctx.add_kernels(
        sources=[src_code], kernels=kernel_descriptions, extra_headers=headers
    )

    ctx.kernels.eval_cerrf_q1(
        n=n_z, re=re_absc_dev, im=im_absc_dev, wz_re=wz_re_dev, wz_im=wz_im_dev
    )

    wz_re = ctx.nparray_from_context_array(wz_re_dev).reshape(n_im, n_re)
    wz_im = ctx.nparray_from_context_array(wz_im_dev).reshape(n_im, n_re)

    d_abs_re = np.fabs(wz_re - wz_cmp_re)
    d_abs_im = np.fabs(wz_im - wz_cmp_im)

    # NOTE: target accuracy of cerrf_q1 is 0.5e-10 but the algorithm does
    #       not converge to within target accuracy for all arguments in C,
    #       especially close to the real axis. We therfore require that
    #       d_abs_re.max(), d_abs_im.max() < 0.5e-9

    assert d_abs_re.max() < 0.5e-9
    assert d_abs_im.max() < 0.5e-9


def test_cerrf_all_quadrants():
    x0 = 5.33
    y0 = 4.29
    num_args = 10000

    if xo.ContextCpu not in available:
        return

    ctx = xo.ContextCpu(omp_num_threads=2)

    re_max = np.float64(np.sqrt(2.0) * x0)
    im_max = np.float64(np.sqrt(2.0) * y0)

    # Extending the sampled area symmetrically into Q3 and Q4 would
    # get the zeros of w(z) into the fold which are located close to the
    # first medians of these quadrants at Im(z) = \pm Re(z) for Re(z) > 1.99146
    #
    # This would lead to a degradation in the accuracy by at least an order
    # of magnitude due to cancellation effects and could distort the test ->
    # By excluding anything with an imaginary part < -1.95, this should be on
    # the safe side.

    np.random.seed(20210811)

    im_min = np.float64(-1.95)
    re_min = -re_max

    re_absc = np.random.uniform(re_min, re_max, num_args)
    im_absc = np.random.uniform(im_min, im_max, num_args)
    wz_cmp_re = np.arange(num_args, dtype=np.float64)
    wz_cmp_im = np.arange(num_args, dtype=np.float64)

    # Create comparison data for veryfing the correctness of cerrf().
    # Cf. the comments about scipy's wofz implementation in test_cerrf_q1()
    # for details!

    for ii, (x, y) in enumerate(zip(re_absc, im_absc)):
        wz = wofz_scipy(x + 1.0j * y)
        wz_cmp_re[ii] = wz.real
        wz_cmp_im[ii] = wz.imag

    src_code = """
    /*gpukern*/ void eval_cerrf_all_quadrants(
        const int n,
        /*gpuglmem*/ double const* /*restrict*/ re,
        /*gpuglmem*/ double const* /*restrict*/ im,
        /*gpuglmem*/ double* /*restrict*/ wz_re,
        /*gpuglmem*/ double* /*restrict*/ wz_im )
    {
        int tid = 0;
        for( ; tid < n ; ++tid ) { //autovectorized

            if( tid < n )
            {
                double const x = re[ tid ];
                double const y = im[ tid ];
                double wz_x, wz_y;

                cerrf( x, y, &wz_x, &wz_y );

                wz_re[ tid ] = wz_x;
                wz_im[ tid ] = wz_y;
            }
        }
    }
    """

    kernel_descriptions = {
        "eval_cerrf_all_quadrants": xo.Kernel(
            args=[
                xo.Arg(xo.Int32, name="n"),
                xo.Arg(xo.Float64, name="re", const=True, pointer=True),
                xo.Arg(xo.Float64, name="im", const=True, pointer=True),
                xo.Arg(xo.Float64, name="wz_re", pointer=True),
                xo.Arg(xo.Float64, name="wz_im", pointer=True),
            ],
            n_threads="n",
        ),
    }

    headers = [
        _pkg_root.joinpath("headers/constants.h"),
        _pkg_root.joinpath("headers/sincos.h"),
        _pkg_root.joinpath("headers/power_n.h"),
        _pkg_root.joinpath("fieldmaps/bigaussian_src/complex_error_function.h"),
    ]

    wz_re = np.arange(num_args, dtype=np.float64)
    wz_im = np.arange(num_args, dtype=np.float64)

    re_absc_dev = ctx.nparray_to_context_array(re_absc)
    im_absc_dev = ctx.nparray_to_context_array(im_absc)
    wz_re_dev = ctx.nparray_to_context_array(wz_re)
    wz_im_dev = ctx.nparray_to_context_array(wz_im)

    ctx.add_kernels(
        sources=[src_code], kernels=kernel_descriptions, extra_headers=headers
    )

    ctx.kernels.eval_cerrf_all_quadrants(
        n=num_args,
        re=re_absc_dev,
        im=im_absc_dev,
        wz_re=wz_re_dev,
        wz_im=wz_im_dev,
    )

    wz_re = ctx.nparray_from_context_array(wz_re_dev)
    wz_im = ctx.nparray_from_context_array(wz_im_dev)

    d_abs_re = np.fabs(wz_re - wz_cmp_re)
    d_abs_im = np.fabs(wz_im - wz_cmp_im)

    assert d_abs_re.max() < 0.5e-9
    assert d_abs_im.max() < 0.5e-9
