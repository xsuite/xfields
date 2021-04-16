import xobjects as xo
import time
import numpy as np

size_x = 512
size_y = 512
size_z = 100

context = xo.ContextCupy()

a = context.zeros((size_x, size_y, size_z), dtype=np.complex128, order='F')
b = context.zeros((size_x, size_y, size_z), dtype=np.complex128, order='F')


#sigma_x = 3e-3
#sigma_y = 2e-3
#sigma_z = 30e-2
#
#x_lim = 5.*sigma_x
#y_lim = 5.*sigma_y
#z_lim = 5.*sigma_z
#
#from xfields import SpaceCharge3D
#
#spcharge = SpaceCharge3D(
#        length=1, update_on_track=True, apply_z_kick=False,
#        x_range=(-x_lim, x_lim),
#        y_range=(-y_lim, y_lim),
#        z_range=(-z_lim, z_lim),
#        nx=256, ny=256, nz=100,
#        solver='FFTSolver2p5D',
#        context=context)
#
#a[:,:,:] = spcharge.fieldmap.solver._workspace_dev
#b[:,:,:] = spcharge.fieldmap.solver._gint_rep_transf_dev
#
#a = spcharge.fieldmap.solver._workspace_dev.T
#b = spcharge.fieldmap.solver._gint_rep_transf_dev.T

for _ in range(5):
    t1 = time.time()
    a[:,:,:] = a*b
    context.synchronize()
    t2 = time.time()
    print(f'{(t2-t1)*1000} ms')
