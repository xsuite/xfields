import time

import numpy as np

import xobjects as xo
from xfields.contexts import add_default_kernels

from pysixtrack.be_beamfields.gaussian_fields import get_Ex_Ey_Gx_Gy_gauss
from pysixtrack.mathlibs import MathlibDefault

ctx = xo.ContextCpu()
ctx = xo.ContextCpu(omp_num_threads=4)
#ctx = xo.ContextCupy()
#ctx = xo.ContextPyopencl()

print(ctx)


kernel_descriptions = {'q_gaussian_profile':{
    'args':(
    (('scalar', np.int32  ), 'n'),
    (('array',  np.float64), 'z'),
    (('scalar', np.float64), 'z0'),
    (('scalar', np.float64), 'z_min'),
    (('scalar', np.float64), 'z_max'),
    (('scalar', np.float64), 'beta'),
    (('scalar', np.float64), 'q'),
    (('scalar', np.float64), 'q_tol'),
    (('scalar', np.float64), 'factor'),
    (('array',  np.float64), 'res'),
        ),
    'num_threads_from_arg': 'n'
    },}

ctx.add_kernels(src_files=['../../../xfields/src/qgaussian.h'],
                kernel_descriptions=kernel_descriptions)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
for qq in [0.95, 1., 1.05]:
    z = np.linspace(-2., 2., 1000)
    res = 0*z
    ctx.kernels.q_gaussian_profile(
            n=len(z),
            z=z,
            z0=0.5,
            z_min=-0.8,
            z_max=1.9,
            beta=1./2./0.5**2,
            q=qq,
            q_tol=1e-10,
            factor=1,
            res=res)
    plt.plot(z, res, label=f'q={qq}')

plt.legend(loc='best')
plt.show()
