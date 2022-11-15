import xobjects as xo
from xfields import _pkg_root

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz

mode = 'special_y_0'
mode = 'standard'

ctx = xo.ContextCpu(omp_num_threads=0)

src_code = """
    /*gpukern*/ void eval_faddeeva_w_q1(
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

                faddeeva_w( x, y, &wz_x, &wz_y );

                wz_re[ tid ] = wz_x;
                wz_im[ tid ] = wz_y;
            }
        }
    }
    """

kernel_descriptions = {
    "eval_faddeeva_w_q1": xo.Kernel(
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
    _pkg_root.joinpath("fieldmaps/bigaussian_src/faddeeva.h"),
]

assert mode in ['special_y_0', 'standard']
if mode == "special_y_0":
    headers = ["#define FADDEEVA_SPECIAL_Y_0"] + headers

ctx.add_kernels(
    sources=headers + [src_code], kernels=kernel_descriptions)

# Plot on the real axis

z_re = np.linspace(-10, 10, 101)
z_im = 0 * z_re

wz_re = 0 * z_re
wz_im = 0 * z_re

assert len(z_re) == len(z_im)
ctx.kernels.eval_faddeeva_w_q1(
    n=len(z_re), re=z_re, im=z_im, wz_re=wz_re, wz_im=wz_im
)

plt.close("all")
plt.suptitle(f'mode = {mode}')
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(z_re, wz_re, 'b.')
plt.plot(z_re, wofz(z_re + 1j * z_im).real, 'b')
plt.plot(z_re, wz_im, 'r.')
plt.plot(z_re, wofz(z_re + 1j * z_im).imag, 'r')
plt.title('y = 0')
plt.subplot(2,1,2)
plt.plot(z_re, wz_re - wofz(z_re + 1j * z_im).real, 'b')
plt.plot(z_re, wz_im - wofz(z_re + 1j * z_im).imag, 'r')
plt.xlabel('x')


# Plot on assigned x value

x_val = 1.0

z_im = np.arange(-1, 10, 0.1)
z_im[np.abs(z_im) < 1e-10] = 0
z_re = 0 * z_im + x_val

wz_re = 0 * z_re
wz_im = 0 * z_re

assert len(z_re) == len(z_im)
ctx.kernels.eval_faddeeva_w_q1(
    n=len(z_re), re=z_re, im=z_im, wz_re=wz_re, wz_im=wz_im
)

plt.figure(2)
plt.suptitle(f'mode = {mode}')
plt.subplot(2,1,1)
plt.plot(z_im, wz_re, 'b.')
plt.plot(z_im, wofz(z_re + 1j * z_im).real, 'b')
plt.plot(z_im, wz_im, 'r.')
plt.plot(z_im, wofz(z_re + 1j * z_im).imag, 'r')
plt.title(f'x = {x_val}')
plt.subplot(2,1,2)
plt.plot(z_im, wz_re - wofz(z_re + 1j * z_im).real, 'b')
plt.plot(z_im, wz_im - wofz(z_re + 1j * z_im).imag, 'r')
plt.xlabel('y')

#Time it
n_test = int(1e6)

z_im = np.random.uniform(size=n_test)
z_re = np.random.uniform(size=n_test)

res_re = 0 * z_re
res_im = 0 * z_re
t1 = time.time()
ctx.kernels.eval_faddeeva_w_q1(
    n=len(z_re), re=z_re, im=z_im, wz_re=res_re, wz_im=res_im
)
t2 = time.time()
print(f"Time: {(t2 - t1)/n_test:e} s/point")

plt.show()