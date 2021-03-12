import numpy as np

try:
    import cupy
    from cupyx.scipy import fftpack as cufftp
except ImportError:
    print('WARNING: cupy is not installed, this platform will not be available')
    from .platnotavail import ModuleNotAvailable
    cupy = ModuleNotAvailable(message=('cupy is not installed. '
                            'this platform is not available!'))
    cufftp = cupy

from .default_kernels import cupy_default_kernels

class MinimalDotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr)

class XfCupyPlatform(object):

    """Creates a Cupy Platform object, that allows performing the computations
    on nVidia GPUs.

    Args:
        default_kernels (bool): If ``True``, the Xfields defult kernels are 
            automatically imported.
        default_block_size (int):  CUDA thread size that is used by default
            for kernel execution in case a block size is not specified 
            directly in the kernel object. The default value is 256.

    Returns:
        XfCupyPlatform: platform object.

    """

    def __init__(self, default_kernels=True, default_block_size=256):

        self.default_block_size = default_block_size
        self.kernels = MinimalDotDict()

        if default_kernels:
            self.add_kernels(src_files=cupy_default_kernels['src_files'],
                    kernel_descriptions=cupy_default_kernels['kernel_descriptions'])

    @property
    def nplike_lib(self):
        """
        Module containing all the numpy features supported by cupy.
       
        Example:

        .. code-block:: python

            platform =  XfCupyPlatform()
            nplike = platform.nplike_lib
            
            # This returns an array of zeros on the computing device (GPU):
            a = nplike.zeros((10,10), dtype=nplike.float64) 
            
        """
        return cupy

    def nparray_to_platform_mem(self, arr):
         """Copies a numpy array to the device memory.
        Args:
            arr (np.ndarray): Array to be transferred
        Returns:
            cupy.ndarray: the same array copied to the device.
        """
        dev_arr = cupy.array(arr)
        return dev_arr

    def nparray_from_platform_mem(self, dev_arr):
        """Copies a numpy array to the device memory.

        Args:
            arr (np.ndarray): Array to be transferred

        Returns:
            cupy.ndarray: the same array copied to the device.

        """
        
        return dev_arr.get()

    def plan_FFT(self, data, axes, ):
        """Creates an array on the current device.

        This function currently does not support the ``subok`` option.

        Args:
            obj: :class:`cupy.ndarray` object or any other object that can be
                passed to :func:`numpy.array`.
            dtype: Data type specifier.
            copy (bool): If ``False``, this function returns ``obj`` if possible.
                Otherwise this function always returns a new array.
            order ({'C', 'F', 'A', 'K'}): Row-major (C-style) or column-major
                (Fortran-style) order.
                When ``order`` is ``'A'``, it uses ``'F'`` if ``a`` is column-major
                and uses ``'C'`` otherwise.
                And when ``order`` is ``'K'``, it keeps strides as closely as
                possible.
                If ``obj`` is :class:`numpy.ndarray`, the function returns ``'C'``
                or ``'F'`` order array.
            subok (bool): If ``True``, then sub-classes will be passed-through,
                otherwise the returned array will be forced to be a base-class
                array (default).
            ndmin (int): Minimum number of dimensions. Ones are inserted to the
                head of the shape if needed.

        Returns:
            cupy.ndarray: An array on the current device.

        .. note::
        This method currently does not support ``subok`` argument.

        .. seealso:: :func:`numpy.array`

        """
        return XfCupyFFT(self, data, axes)

    def add_kernels(self, src_code='', src_files=[], kernel_descriptions={}):

        src_content = 'extern "C"{'
        for ff in src_files:
            with open(ff, 'r') as fid:
                src_content += ('\n\n' + fid.read())
        src_content += "}"

        module = cupy.RawModule(code=src_content)

        ker_names = kernel_descriptions.keys()
        for nn in ker_names:
            kk = module.get_function(nn)
            aa = kernel_descriptions[nn]['args']
            nt_from = kernel_descriptions[nn]['num_threads_from_arg']
            aa_types, aa_names = zip(*aa)
            self.kernels[nn] = XfCupyKernel(cupy_kernel=kk,
                arg_names=aa_names, arg_types=aa_types,
                num_threads_from_arg=nt_from,
                block_size=self.default_block_size)

class XfCupyKernel(object):

    def __init__(self, cupy_kernel, arg_names, arg_types,
                 num_threads_from_arg, block_size):

        assert (len(arg_names) == len(arg_types))
        assert num_threads_from_arg in arg_names

        self.cupy_kernel = cupy_kernel
        self.arg_names = arg_names
        self.arg_types = arg_types
        self.num_threads_from_arg = num_threads_from_arg
        self.block_size = block_size

    @property
    def num_args(self):
        return len(self.arg_names)

    def __call__(self, **kwargs):
        assert len(kwargs.keys()) == self.num_args
        arg_list = []
        for nn, tt in zip(self.arg_names, self.arg_types):
            vv = kwargs[nn]
            if tt[0] == 'scalar':
                assert np.isscalar(vv)
                arg_list.append(tt[1](vv))
            elif tt[0] == 'array':
                assert isinstance(vv, cupy.ndarray)
                arg_list.append(vv.data)
            else:
                raise ValueError(f'Type {tt} not recognized')

        n_threads = kwargs[self.num_threads_from_arg]
        grid_size = int(np.ceil(n_threads/self.block_size))
        self.cupy_kernel((grid_size, ), (self.block_size, ), arg_list)


class XfCupyFFT(object):
    def __init__(self, platform, data, axes):

        self.platform = platform
        self.axes = axes

        assert len(data.shape) > max(axes)

        from cupyx.scipy import fftpack as cufftp
        self._fftplan = cufftp.get_fft_plan(
                data, axes=self.axes, value_type='C2C')

    def transform(self, data):
        data[:] = cufftp.fftn(data, axes=self.axes, plan=self._fftplan)[:]
        """The transform is done inplace"""


    def itransform(self, data):
        """The transform is done inplace"""
        data[:] = cufftp.ifftn(data, axes=self.axes, plan=self._fftplan)[:]

