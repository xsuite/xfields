# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
from scipy.special import wofz as wofz_scipy
import pytest
import xobjects as xo
from xobjects.context import available
from xfields.general import _pkg_root
from xobjects.test_helpers import for_all_test_contexts


@pytest.fixture
def faddeeva_calculator():
    source = '''
        /*gpukern*/ void FaddeevaCalculator_compute(FaddeevaCalculatorData data) {
            int64_t len = FaddeevaCalculatorData_len_z_re(data);

            for (int64_t ii = 0; ii < len; ii++) {  //vectorize_over ii len
                double z_re = FaddeevaCalculatorData_get_z_re(data, ii);
                double z_im = FaddeevaCalculatorData_get_z_im(data, ii);
                double w_re, w_im;

                faddeeva_w(z_re, z_im, &w_re, &w_im);

                FaddeevaCalculatorData_set_w_re(data, ii, w_re);
                FaddeevaCalculatorData_set_w_im(data, ii, w_im);
            } //end_vectorize
        }
    '''

    class FaddeevaCalculator(xo.HybridClass):
        _xofields = {
            'z_re': xo.Float64[:],
            'z_im': xo.Float64[:],
            'w_re': xo.Float64[:],
            'w_im': xo.Float64[:],
        }

        _extra_c_sources = [
            _pkg_root.joinpath("headers/constants.h"),
            _pkg_root.joinpath("headers/sincos.h"),
            _pkg_root.joinpath("headers/power_n.h"),
            _pkg_root.joinpath("fieldmaps/bigaussian_src/faddeeva.h"),
            source,
        ]

        _kernels = {
            'FaddeevaCalculator_compute': xo.Kernel(
                args=[
                    xo.Arg(xo.ThisClass, name='data'),
                ],
            )
        }

        def __init__(self, z, **kwargs):
            z = np.array(z)

            self.xoinitialize(
                z_re=z.real.copy(),
                z_im=z.imag.copy(),
                w_re=len(z),
                w_im=len(z),
                **kwargs,
            )

        @property
        def w(self):
            re = self._context.nparray_from_context_array(self.w_re)
            im = self._context.nparray_from_context_array(self.w_im)
            return re + 1j * im

        def compute(self):
            self._xobject.compile_kernels(only_if_needed=True)
            kernel = self._context.kernels.FaddeevaCalculator_compute
            kernel.set_n_threads(len(self.z_re))
            kernel(data=self)

    return FaddeevaCalculator


@for_all_test_contexts
def test_faddeeva_w_q1(faddeeva_calculator, test_context):
    FaddeevaCalculator = faddeeva_calculator

    # Generate the test grid
    xx = np.concatenate(([0], np.logspace(-8, +8, 51))).astype(np.float64)
    yy = np.concatenate(([0], np.logspace(-8, +8, 51))).astype(np.float64)

    n_re = len(xx)
    n_im = len(yy)
    n_z = len(yy) * len(xx)

    re_absc, im_absc = np.meshgrid(xx, yy)

    # Calculate the values based on the grid
    z = (re_absc + 1j * im_absc).reshape(n_re * n_im)
    calculator = FaddeevaCalculator(z=z, _context=test_context)
    calculator.compute()

    # Using scipy's wofz implemenation of the Faddeeva method. This is
    # (at the time of this writing in 2021) based on the MIT ab-initio
    # implementation using a combination of Algorithm 680 for large |z| and
    # Algorithm 916 for the remainder fo C. It claims a relative accuracy of
    # 1e-13 across the whole of C and is thus suitable to check the accuracy
    # of the faddeeva_w_q1 implementation which has a target accuracy of 10^{-10}
    # in the *absolute* error.
    wz_cmp = wofz_scipy(re_absc + 1.0j * im_absc)

    wz_re = calculator.w.real.reshape(n_im, n_re)
    wz_im = calculator.w.imag.reshape(n_im, n_re)

    d_abs_re = np.fabs(wz_re - wz_cmp.real)
    d_abs_im = np.fabs(wz_im - wz_cmp.imag)

    # NOTE: target accuracy of faddeeva_w_q1 is 0.5e-10 but the algorithm does
    #       not converge to within target accuracy for all arguments in C,
    #       especially close to the real axis. We therfore require that
    #       d_abs_re.max(), d_abs_im.max() < 0.5e-9

    assert d_abs_re.max() < 0.5e-9
    assert d_abs_im.max() < 0.5e-9


@for_all_test_contexts
def test_faddeeva_w_all_quadrants(faddeeva_calculator, test_context):
    FaddeevaCalculator = faddeeva_calculator

    x0 = 5.33
    y0 = 4.29
    num_args = 10000

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
    z = re_absc + 1j * im_absc

    # Calculate the values based on the grid
    calculator = FaddeevaCalculator(z=z, _context=test_context)
    calculator.compute()

    # Create comparison data for veryfing the correctness of faddeeva_w().
    # Cf. the comments about scipy's wofz implementation in test_faddeeva_w_q1()
    # for details!
    wz_cmp = wofz_scipy(z)

    difference = calculator.w - wz_cmp
    d_abs_re = np.fabs(difference.real)
    d_abs_im = np.fabs(difference.imag)

    assert d_abs_re.max() < 0.5e-9
    assert d_abs_im.max() < 0.5e-9
