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
    fftplan = context.plan_FFT(a, axes=(0,1))
    fftplanT = context.plan_FFT(a.T, axes=(2,1))
    fftplan2 = context.plan_FFT(a, axes=(1,0))
    fftplan2T = context.plan_FFT(a.T, axes=(1,2))
    print(f'\n\n{order=}')
    for _ in range(5):
        t1 = time.time()
        fftplan.transform(a)
        context.synchronize()
        t2 = time.time()
        print(f'Time FFT: {(t2-t1)*1000:.2f} ms')

        t1 = time.time()
        fftplanT.transform(a.T)
        context.synchronize()
        t2 = time.time()
        print(f'Time FFT.T: {(t2-t1)*1000:.2f} ms')

        t1 = time.time()
        fftplan2.transform(a)
        context.synchronize()
        t2 = time.time()
        print(f'Time FFT2: {(t2-t1)*1000:.2f} ms')

        t1 = time.time()
        fftplan2T.transform(a.T)
        context.synchronize()
        t2 = time.time()
        print(f'Time FFT2.T: {(t2-t1)*1000:.2f} ms')
