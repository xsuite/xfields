import os
import time
import numpy as np
import cffi

ffi_interface = cffi.FFI()

src = r'''
#include <math.h>
#include <omp.h>

double sinsin(double x){
    return sin(x);
}

void mysin(double* x, int n){
   int ii;
   #pragma omp parallel for private(ii)
   for (ii=0; ii<n; ii++){
      x[ii] = sinsin(x[ii]);
   }
}
'''

ffi_interface.cdef("void mysin(double*, int);")
ffi_interface.cdef("void omp_set_num_threads(int);")

ffi_interface.set_source("_example", src,
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],)

ffi_interface.compile(verbose=False)

from _example import ffi, lib

x_test = np.linspace(0, 2*np.pi, 200000000)
x_cffi = ffi_interface.cast('double *', ffi_interface.from_buffer(x_test))

N_test = 1
for n_threads in [1, 48]:
    lib.omp_set_num_threads(n_threads)
    t1 = time.time()
    for _ in range(N_test):
        lib.mysin(x_cffi, len(x_test))
    t2 = time.time()
    print(f'Time ({n_threads=}): {1e3*(t2-t1)/N_test:.2f} ms')
