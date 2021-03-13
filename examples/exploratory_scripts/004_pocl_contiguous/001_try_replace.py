import numpy as np
import pyopencl as cl

from xfields.platforms import XfPoclPlatform

platform = XfPoclPlatform()
ctx = platform.pocl_context
queue = platform.command_queue

cla = platform.nplike_lib

a_cont = cla.to_device(queue=platform.command_queue,
        ary=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]], order='F',
            dtype=np.float64))
a = a_cont[1:, 1:]

b_cont = a_cont * 10
b = b_cont[1:, 1:]

prg = cl.Program(ctx, """
    __kernel void copy_array_fcont(
        __global const int*   shape,
                 const int    itemsize,
        __global const char*  buffer_src,
        __global const int*   strides_src,
                 const int    offset_src,
        __global       char*  buffer_dest,
        __global const int*   strides_dest,
                 const int    offset_dest
                      )
    {
      int gid = get_global_id(0);
      int ibyte;

      for (ibyte=0; ibyte<itemsize; ibyte++){
        buffer_dest[offset_dest + ibyte] = buffer_src[offset_src + ibyte];
        }
    }
    """).build()

knl_copy_array_fcont = prg.copy_array_fcont

assert a.shape == b.shape
shape = cla.to_device(queue, np.array(a.shape, dtype=np.int32))
itemzisize = np.int32(8)
buffer_src = a.base_data
strides_src = cla.to_device(queue, np.array(a.strides, dtype=np.int32))
offset_src = np.int32(a.offset)
buffer_dest = b.base_data
strides_dest = cla.to_device(queue, np.array(b.strides, dtype=np.int32))
offset_dest = np.int32(b.offset)

event = knl_copy_array_fcont(queue, (1,), None,
        # args:
        shape.data, itemzisize, buffer_src,strides_src.data, offset_src,
        buffer_dest, strides_dest.data, offset_dest)
prrrr

print('\n\n')
print(f'{a_cont.offset=}')
print(f'{a.offset=}')
