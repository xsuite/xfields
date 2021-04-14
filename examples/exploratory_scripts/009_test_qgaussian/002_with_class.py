import time

import numpy as np
import matplotlib.pyplot as plt

import xobjects as xo


ctx = xo.ContextCpu()
ctx = xo.ContextCpu(omp_num_threads=4)
ctx = xo.ContextCupy()
#ctx = xo.ContextPyopencl()

print(ctx)

from xfields import LongitudinalProfileQGaussian

plt.close('all')
plt.figure()

z0 = 0.1
sigma_z = 0.2
npart = 1e11

for qq in [0, 0.5, 0.95, 1.05, 1.3]:
    lprofile = LongitudinalProfileQGaussian(
            context=ctx,
            number_of_particles=npart,
            sigma_z=sigma_z,
            z0=z0,
            q_parameter=qq)

    z = np.linspace(-10., 10., 10000)
    z_dev = ctx.nparray_to_context_array(z)
    lden_dev = lprofile.line_density(z_dev)
    lden = ctx.nparray_from_context_array(lden_dev)

    area = np.trapz(lden, z)
    z_mean = np.trapz(lden*z/area, z)
    z_std = np.sqrt(np.trapz(lden*(z-z_mean)**2/area, z))
    assert np.isclose(area, npart)
    assert np.isclose(z_mean, z0)
    assert np.isclose(z_std, sigma_z)

    plt.plot(z, lden, label=f'q={lprofile.q_parameter}')
plt.legend(loc='best')
plt.show()
