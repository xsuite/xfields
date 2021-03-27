import numpy as np
import cffi

ffi_interface = cffi.FFI()

src = r'''
#include <math.h>
void mysin(double* x, int n){
   int ii;
   for (ii=0; ii<n; ii++){
      x[ii] = sin(x[ii]);
   }
}
'''

ffi_interface.cdef("void mysin(double*, int);")

ffi_interface.set_source("_example", src)

ffi_interface.compile(verbose=True)

from _example import ffi, lib

x_test = np.linspace(0, 2*np.pi, 1000)
x_cffi = ffi_interface.cast('double *', ffi_interface.from_buffer(x_test))

lib.mysin(x_cffi, len(x_test))
