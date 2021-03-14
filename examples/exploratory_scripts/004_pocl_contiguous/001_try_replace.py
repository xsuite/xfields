import numpy as np
import pyopencl as cl

from xfields.platforms import XfPoclPlatform

platform = XfPoclPlatform()
ctx = platform.pocl_context
queue = platform.command_queue

cla = platform.nplike_lib


prg = cl.Program(ctx, """
    __kernel void copy_array_fcont(
                 const int    fcont, // bool not accepted
                 const int    ndim,
                 const int    nelem,
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
      int ibyte, idim, this_shape, slice_size, this_index, flat_index;
      int pos_src, pos_dest, this_stride_src, this_stride_dest;

      slice_size = nelem;
      flat_index = gid;
      pos_src = offset_src;
      pos_dest = offset_dest;
      for (idim=0; idim<ndim; idim++){
        if (fcont){
           this_shape = shape[ndim-idim-1];              // for f contiguous
           this_stride_src = strides_src[ndim-idim-1];   // for f contiguous
           this_stride_dest = strides_dest[ndim-idim-1]; // for f contiguous
           }
        else {
           this_shape = shape[idim];              // for c contiguous
           this_stride_src = strides_src[idim];   // for c contiguous
           this_stride_dest = strides_dest[idim]; // for c contiguous
        }


        slice_size = slice_size/this_shape;
        this_index = flat_index/slice_size;
        flat_index = flat_index - this_index*slice_size;

        pos_src = pos_src + this_index * this_stride_src;
        pos_dest = pos_dest + this_index * this_stride_dest;

      }

      for (ibyte=0; ibyte<itemsize; ibyte++){
        buffer_dest[pos_dest + ibyte] = buffer_src[pos_src + ibyte];
        }
    }
    """).build()

knl_copy_array_fcont = prg.copy_array_fcont

def _infer_fccont(arr):
    if arr.strides[0]<arr.strides[-1]:
        return 'F'
    else:
        return 'C'

def copy_non_cont(src, dest):

    assert src.shape == dest.shape
    assert src.dtype == dest.dtype

    if src.strides[0] != src.strides[-1]: # check is needed for 1d arrays
        assert _infer_fccont(src) == _infer_fccont(dest)

    fcontiguous = 0
    if _infer_fccont(dest) == 'F':
        fcontiguous = 1
    fcont = np.int32(fcontiguous)
    shape = cla.to_device(queue, np.array(src.shape, dtype=np.int32))
    ndim = np.int32(len(shape))
    nelem = np.int32(np.prod(src.shape))
    itemzisize = np.int32(src.dtype.itemsize)
    buffer_src = src.base_data
    strides_src = cla.to_device(src.queue,
            np.array(src.strides, dtype=np.int32))
    offset_src = np.int32(src.offset)
    buffer_dest = dest.base_data
    strides_dest = cla.to_device(src.queue,
            np.array(dest.strides, dtype=np.int32))
    offset_dest = np.int32(dest.offset)

    event = knl_copy_array_fcont(src.queue, (nelem,), None,
            # args:
            fcont, ndim,  nelem, shape.data, itemzisize,
            buffer_src, strides_src.data, offset_src,
            buffer_dest, strides_dest.data, offset_dest)
    event.wait()


def mysetitem(self, *args, **kwargs):
    try:
        self._old_setitem(*args, **kwargs)
    except (NotImplementedError, ValueError):
        dest = self[args[0]]
        src = args[1]
        if np.isscalar(src):
            src = dest._cont_zeros_like_me() + src
        copy_non_cont(src, dest)

def mycopy(self):
    res = self._cont_zeros_like_me()
    copy_non_cont(self, res)
    return res

def myget(self):
    try:
        return self._old_get()
    except AssertionError:
        return self.copy().get()

def _cont_zeros_like_me(self):
     res = cla.zeros(self.queue, shape=self.shape, dtype=self.dtype,
                order=_infer_fccont(self))
     return res

if not hasattr(cla.Array, '_old_setitem'):

    cla.Array._cont_zeros_like_me = _cont_zeros_like_me

    cla.Array._old_copy = cla.Array.copy
    cla.Array.copy = mycopy

    cla.Array._old_setitem = cla.Array.__setitem__
    cla.Array.__setitem__ = mysetitem

    cla.Array._old_get = cla.Array.get
    cla.Array.get = myget


a_cont = cla.to_device(queue=platform.command_queue,
        ary=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]], order='F',
            dtype=np.float64))

a = a_cont[1:, 1:]

b_cont = a_cont * 10
b = b_cont[1:, 1:]

a[:, :] = b
b[:, :] = 10

b[1:, 2:] = 20

# # Try complex 
# c_cont = cla.to_device(queue=platform.command_queue,
#             ary=1j*np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]], order='F',
#             dtype=np.complex128))
# 
# c = c_cont[1:, 1:]
# c[:, :] = b
