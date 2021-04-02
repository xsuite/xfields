import numpy as np

import xobjects as xo

ctx = xo.ContextCpu()
ctx = xo.ContextCupy()
ctx = xo.ContextPyopencl()

print(ctx)

kernel_descriptions = {'get_Ex_Ey_Gx_Gy_gauss':{
    'args':(
    (('scalar', np.int32  ), 'n_points'),
    (('array',  np.float64), 'x_ptr'),
    (('array',  np.float64), 'y_ptr'),
    (('scalar', np.float64), 'sigma_x'),
    (('scalar', np.float64), 'sigma_y'),
    (('scalar', np.float64), 'min_sigma_diff'),
    (('scalar', np.int32  ), 'skip_Gs'),
    (('array',  np.float64), 'Ex_ptr'),
    (('array',  np.float64), 'Ey_ptr'),
    (('array',  np.float64), 'Gx_ptr'),
    (('array',  np.float64), 'Gy_ptr'),
        ),
    'num_threads_from_arg': 'n_points'
    },}

ctx.add_kernels(src_files=[
            '../../../xfields/src/complex_error_function.h',
            '../../../xfields/src/constants.h',
            '../../../xfields/src/fields_bigaussian.h',
            ],
            kernel_descriptions=kernel_descriptions,
            save_src_as='_test.c')

x = np.linspace(-1, 1, 1000)
y = 2*np.linspace(-1, 1, 1000)

x_dev = ctx.nparray_to_context_array(x)
y_dev = ctx.nparray_to_context_array(y)
Ex_dev = 0 * x_dev
Ey_dev = 0 * y_dev

ctx.kernels.get_Ex_Ey_Gx_Gy_gauss(
    n_points=len(x_dev),
    x_ptr=x_dev,
    y_ptr=y_dev,
    sigma_x=0.2,
    sigma_y=0.3,
    min_sigma_diff=1e-5,
    skip_Gs=1,
    Ex_ptr=Ex_dev,
    Ey_ptr=Ey_dev,
    Gx_ptr=Ey_dev,#unused
    Gy_ptr=Ex_dev,#unused
    )

Ex = ctx.nparray_from_context_array(Ex_dev)
Ey = ctx.nparray_from_context_array(Ey_dev)

import matplotlib.pyplot as plt
plt.close('all')
plt.plot(x, Ex)
plt.plot(y, Ey)
plt.show()
