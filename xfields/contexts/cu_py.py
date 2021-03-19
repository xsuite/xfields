import numpy as np

from xobjects.context import ContextCupy

from .default_kernels import cupy_default_kernels

class XfCupyContext(ContextCupy):

    """

    Creates a Cupy Context object, that allows performing the computations
    on nVidia GPUs.

    Args:
        default_kernels (bool): If ``True``, the Xfields defult kernels are
            automatically imported.
        default_block_size (int):  CUDA thread size that is used by default
            for kernel execution in case a block size is not specified
            directly in the kernel object. The default value is 256.
    Returns:
        XfCupyContext: context object.

    """

    def __init__(self, default_kernels=True, default_block_size=256):

        super().__init__(default_block_size=default_block_size)

        if default_kernels:
            self.add_kernels(src_files=cupy_default_kernels['src_files'],
                    kernel_descriptions=cupy_default_kernels['kernel_descriptions'])

