import numpy as np

import xobjects as xo
from xobjects.context import available


for (CTX, kwargs) in zip(
        (xo.ContextCpu, xo.ContextPyopencl, xo.ContextCupy),
        ({'omp_num_threads': 2}, {}, {})):

    if CTX not in available:
        continue

    print(f"Test {CTX}")
    ctx = CTX(**kwargs)

    src_code='''
    __device__ //only_for_context cuda
    void myfun(double x, double y,
        double* z){
        z[0] = x * y;
        }

    /*gpukern*/
    void my_mul(const int n,
        /*gpuglmem*/ const double* x1,
        /*gpuglmem*/ const double* x2,
        /*gpuglmem*/       double* y) {
        int tid = 0 //vectorize_over tid n
        double z;
        if (tid < n){
            myfun(x1[tid], x2[tid], &z);
            y[tid] = z;
            }
        //end_vectorize
        }
    '''
    kernel_descriptions = {'my_mul':{
        'args':(
            (('scalar', np.int32),   'n',),
            (('array',  np.float64), 'x1',),
            (('array',  np.float64), 'x2',),
            (('array',  np.float64), 'y',),
            ),
        'num_threads_from_arg': 'n'
        },}

    # Import kernel in context
    ctx.add_kernels(src_code=src_code,
            kernel_descriptions=kernel_descriptions,
            save_src_as=None)

    x1_host = np.array([1.,2.,3.], dtype=np.float64)
    x2_host = np.array([7.,8.,9.], dtype=np.float64)

    x1_dev = ctx.nparray_to_context_array(x1_host)
    x2_dev = ctx.nparray_to_context_array(x2_host)
    y_dev = ctx.zeros(shape=x1_host.shape, dtype=x1_host.dtype)

    ctx.kernels.my_mul(n=len(x1_host), x1=x1_dev, x2=x2_dev, y=y_dev)

    y_host = ctx.nparray_from_context_array(y_dev)

    assert np.allclose(y_host, x1_host*x2_host)

