import xobjects as xo
import time
import numpy as np

size_x = 512
size_y = 512
size_z = 100

context = xo.ContextCupy()
#context = xo.ContextPyopencl()
#context = xo.ContextCpu()
print(context)

for order in ('F', 'C'):
    a = context.zeros((size_x, size_y, size_z),
            dtype=np.complex128, order=order)
    b = context.zeros((size_x, size_y, size_z),
            dtype=np.complex128, order=order)
    print(f'\n\n{order=}')
    for _ in range(5):
        t1 = time.time()
        a[:,:,:] *= b
        context.synchronize()
        t2 = time.time()
        print(f'Time "a[:,:,:] *= b":     {(t2-t1)*1000:.2f} ms')

        t1 = time.time()
        a.T[:,:,:] *= b.T
        context.synchronize()
        t2 = time.time()
        print(f'Time "a.T[:,:,:] *= b.T": {(t2-t1)*1000:.2f} ms')

