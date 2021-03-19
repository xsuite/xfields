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

prg = cl.Program(ctx, """
    __kernel void replace(__global double *buffer,
                      double value, int pos)
    {
      int gid = get_global_id(0);
      buffer[pos] = value;
    }
    """).build()

repknl = prg.replace

event = repknl(queue, (1,), None, a.base_data, np.float64(100.), np.int32(a.offset//8))

print('\n\n')
print(f'{a_cont.offset=}')
print(f'{a.offset=}')
