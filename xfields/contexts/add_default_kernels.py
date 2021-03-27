
from .default_kernels import default_kernels

from xobjects.context import ContextCpu, ContextCupy, ContextPyopencl

def add_default_kernels(context):

    for kk in default_kernels['kernel_descriptions'].keys():
        if kk not in context.kernels.keys():
            context.add_kernels(
                src_files=default_kernels['src_files'],
                kernel_descriptions=default_kernels['kernel_descriptions'])
            break

    for kk in default_kernels['kernel_descriptions'].keys():
        assert kk in context.kernels.keys()


