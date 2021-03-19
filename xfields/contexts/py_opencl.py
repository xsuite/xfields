import numpy as np

from xobjects.context import ContextPyopencl

from .default_kernels import pyopencl_default_kernels


class XfPyopenclContext(ContextPyopencl):

    """
    Creates a Pyopencl Context object, that allows performing the computations
    on GPUs and CPUs through PyOpenCL.

    Args:
        device (str or Device): The device (CPU or GPU) for the simulation.
        default_kernels (bool): If ``True``, the Xfields defult kernels are
            automatically imported.
        patch_pyopencl_array (bool): If ``True``, the PyOpecCL class is patched to
            allow some operations with non-contiguous arrays.

    Returns:
        XfPyopenclContext: context object.

    """

    def __init__(self, device='0.0', default_kernels=True, patch_pyopencl_array=True):

        super().__init__(device=device, patch_pyopencl_array=patch_pyopencl_array)

        if default_kernels:
            self.add_kernels(src_files=pyopencl_default_kernels['src_files'],
                    kernel_descriptions=pyopencl_default_kernels['kernel_descriptions'])

    def nparray_to_context_mem(self, arr):
        return self.nparray_to_context_array(arr)

    def nparray_from_context_mem(self, arr):
        return self.nparray_from_context_array(arr)

