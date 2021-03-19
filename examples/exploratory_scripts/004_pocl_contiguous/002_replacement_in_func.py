import numpy as np
import pyopencl as cl

from xfields.contexts import XfPyopenclContext

context = XfPyopenclContext()
ctx = context.pyopencl_context
queue = context.command_queue

cla = context.nplike_lib

a_cont = cla.to_device(queue=context.command_queue,
        ary=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]], order='F',
            dtype=np.float64))

a = a_cont[1:, 1:]

b_cont = a_cont * 10
b = b_cont[1:, 1:]

a[:, :] = b
b[:, :] = 10

b[1:, 2:] = 20

# Try complex 
c_cont = cla.to_device(queue=context.command_queue,
             ary=1j*np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]], order='F',
             dtype=np.complex128))

c = c_cont[1:, 1:]
c[:, :] = a
