import numpy as np

import xobjects as xo
from xobjects.context import available


for (CTX, kwargs) in zip(
    (xo.ContextCpu, xo.ContextPyopencl, xo.ContextCupy),
    ({"omp_num_threads": 2}, {}, {}),
):

    if CTX not in available:
        continue

    print(f"Test {CTX}")
    ctx = CTX(**kwargs)

    src_code = """
    /*gpufun*/
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
    """
    kernel_descriptions = {
        "my_mul": xo.Kernel(
            args=[
                xo.Arg(xo.Int32, name="n"),
                xo.Arg(xo.Float64, name="x1", const=True, pointer=True),
                xo.Arg(xo.Float64, name="x2", const=True, pointer=True),
                xo.Arg(xo.Float64, name="y", pointer=True),
            ],
            n_threads="n",
        ),
    }

    # Import kernel in context
    ctx.add_kernels(sources=[src_code], kernels=kernel_descriptions)

    x1_host = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x2_host = np.array([7.0, 8.0, 9.0], dtype=np.float64)

    x1_dev = ctx.nparray_to_context_array(x1_host)
    x2_dev = ctx.nparray_to_context_array(x2_host)
    y_dev = ctx.zeros(shape=x1_host.shape, dtype=x1_host.dtype)

    ctx.kernels.my_mul(n=len(x1_host), x1=x1_dev, x2=x2_dev, y=y_dev)

    y_host = ctx.nparray_from_context_array(y_dev)

    assert np.allclose(y_host, x1_host * x2_host)
