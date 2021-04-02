import numpy as np

import xobjects as xo
from xfields.contexts import add_default_kernels

from pysixtrack.be_beamfields.gaussian_fields import get_Ex_Ey_Gx_Gy_gauss
from pysixtrack.mathlibs import MathlibDefault

ctx = xo.ContextCpu()
ctx = xo.ContextCupy()
ctx = xo.ContextPyopencl()

print(ctx)

add_default_kernels(ctx)

x = np.linspace(-1, 1, 1000)
y = 2*np.linspace(-1, 1, 1000)

sigma_x=0.2
sigma_y=0.3

x_dev = ctx.nparray_to_context_array(x)
y_dev = ctx.nparray_to_context_array(y)
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
    Gx_ptr=Gy_dev,
    Gy_ptr=Gx_dev,
    )

Ex = ctx.nparray_from_context_array(Ex_dev)
Ey = ctx.nparray_from_context_array(Ey_dev)
Gx = ctx.nparray_from_context_array(Gx_dev)
Gy = ctx.nparray_from_context_array(Gy_dev)

Ex_pst, Ey_pst, Gx_pst, Gy_pst = get_Ex_Ey_Gx_Gy_gauss(x, y, sigma_x, sigma_y,
        min_sigma_diff=1e-5, skip_Gs=False, mathlib=MathlibDefault())

assert np.allclose(Ex, Ex_pst)
assert np.allclose(Ey, Ey_pst)
assert np.allclose(Gx, Gx_pst)
assert np.allclose(Gy, Gy_pst)
