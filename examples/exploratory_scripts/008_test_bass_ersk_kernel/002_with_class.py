import time

import numpy as np

import xobjects as xo
from xfields.fieldmaps import BiGaussianFieldMap


from pysixtrack.be_beamfields.gaussian_fields import get_Ex_Ey_Gx_Gy_gauss
from pysixtrack.mathlibs import MathlibDefault

ctx = xo.ContextCpu()
ctx = xo.ContextCpu(omp_num_threads=4)
#ctx = xo.ContextCupy()
#ctx = xo.ContextPyopencl()

print(ctx)

x_test = 0.25 * np.random.randn(1000)
y_test = 0.25 * np.random.randn(1000)

n_part_time = 1000000
x_time = 0.25 * np.random.randn(n_part_time)
y_time = 0.25 * np.random.randn(n_part_time)

fmap = BiGaussianFieldMap(
        context=ctx,
        sigma_x=1., # to be updated later
        sigma_y=1., # to be updated later
        )

for sigma_x, sigma_y in ((0.2, 0.3), (0.3, 0.2), (0.2, 0.2)):

    print(f'{sigma_x=} {sigma_y=}')

    fmap.sigma_x = sigma_x
    fmap.sigma_y = sigma_y

    x_dev = ctx.nparray_to_context_array(x_test)
    y_dev = ctx.nparray_to_context_array(y_test)
    Ex_dev = 0 * x_dev
    Ey_dev = 0 * y_dev

    dphi_dx_dev, dphi_dy_dev = fmap.get_values_at_points(x_dev, y_dev,
            return_dphi_dx=True, return_dphi_dy=True)

    Ex = ctx.nparray_from_context_array(-dphi_dx_dev)
    Ey = ctx.nparray_from_context_array(-dphi_dy_dev)

    Ex_pst, Ey_pst, Gx_pst, Gy_pst = get_Ex_Ey_Gx_Gy_gauss(x_test, y_test,
            sigma_x, sigma_y, min_sigma_diff=1e-5, skip_Gs=False,
            mathlib=MathlibDefault())

    assert np.allclose(Ex, Ex_pst)
    assert np.allclose(Ey, Ey_pst)

    # Time
    x_dev = ctx.nparray_to_context_array(x_time)
    y_dev = ctx.nparray_to_context_array(y_time)
    Ex_dev = 0 * x_dev
    Ey_dev = 0 * y_dev
    Gx_dev = 0 * x_dev
    Gy_dev = 0 * y_dev

    n_rep = 3

    for _ in range(n_rep):
        t1 = time.time()
        ctx.kernels.get_Ex_Ey_Gx_Gy_gauss(
            n_points=len(x_dev),
            x_ptr=x_dev,
            y_ptr=y_dev,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            min_sigma_diff=1e-5,
            skip_Gs=0,
            Ex_ptr=Ex_dev,
            Ey_ptr=Ey_dev,
            Gx_ptr=Gx_dev,
            Gy_ptr=Gy_dev,
            )
        ctx.synchronize()
        t2 = time.time()
        print(f'Time: {(t2-t1)*1e3:.2f} ms')
