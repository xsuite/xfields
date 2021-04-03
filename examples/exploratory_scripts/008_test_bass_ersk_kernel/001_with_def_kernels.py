import time

import numpy as np

import xobjects as xo
from xfields.contexts import add_default_kernels

from pysixtrack.be_beamfields.gaussian_fields import get_Ex_Ey_Gx_Gy_gauss
from pysixtrack.mathlibs import MathlibDefault

ctx = xo.ContextCpu()
#ctx = xo.ContextCupy()
#ctx = xo.ContextPyopencl()

print(ctx)

add_default_kernels(ctx)

x_test = 0.25 * np.random.randn(1000)
y_test = 0.25 * np.random.randn(1000)

n_part_time = 1000000
x_time = 0.25 * np.random.randn(n_part_time)
y_time = 0.25 * np.random.randn(n_part_time)

sigma_x=0.2
sigma_y=0.3

for sigma_x, sigma_y in ((0.2, 0.3), (0.3, 0.2), (0.2, 0.2)):

    print(f'{sigma_x=} {sigma_y=}')

    x_dev = ctx.nparray_to_context_array(x_test)
    y_dev = ctx.nparray_to_context_array(y_test)
    Ex_dev = 0 * x_dev
    Ey_dev = 0 * y_dev
    Gx_dev = 0 * x_dev
    Gy_dev = 0 * y_dev

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

    Ex = ctx.nparray_from_context_array(Ex_dev)
    Ey = ctx.nparray_from_context_array(Ey_dev)
    Gx = ctx.nparray_from_context_array(Gx_dev)
    Gy = ctx.nparray_from_context_array(Gy_dev)

    Ex_pst, Ey_pst, Gx_pst, Gy_pst = get_Ex_Ey_Gx_Gy_gauss(x_test, y_test,
            sigma_x, sigma_y, min_sigma_diff=1e-5, skip_Gs=False,
            mathlib=MathlibDefault())

    assert np.allclose(Ex, Ex_pst)
    assert np.allclose(Ey, Ey_pst)
    assert np.allclose(Gx, Gx_pst)
    assert np.allclose(Gy, Gy_pst)

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
        print(f'Time): {(t2-t1)*1e3:.2f} ms')
