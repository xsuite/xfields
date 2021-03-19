
from .default_kernels import cpu_default_kernels
from .default_kernels import cupy_default_kernels
from .default_kernels import pyopencl_default_kernels

from xobjects.context import ContextCpu, ContextCupy, ContextPyopencl

def add_default_kernels(context):

    if isinstance(context, ContextCpu):
        default_kernels = cpu_default_kernels
    elif isinstance(context, ContextCupy):
        default_kernels = cupy_default_kernels
    elif isinstance(context, ContextPyopencl):
        default_kernels = pyopencl_default_kernels
    else:
        raise TypeError(f'Unknown context type: {repr(type(context))}')

    for kk in default_kernels['kernel_descriptions'].keys():
        if kk not in context.kernels.keys():
            context.add_kernels(
                src_files=default_kernels['src_files'],
                kernel_descriptions=default_kernels['kernel_descriptions'])
            break

    for kk in default_kernels['kernel_descriptions'].keys():
        assert kk in context.kernels.keys()


