import numpy as np

from .base import XfBasePlatform

try:
    import pyopencl as cl
    import pyopencl.array as cla
except ImportError:
    print('WARNING: pyopencl is not installed, this platform will not be available')
    from .platnotavail import ModuleNotAvailable
    cl = ModuleNotAvailable(message=('pyopencl is not installed. '
                            'this platform is not available!'))
    cla = cl

from .default_kernels import pocl_default_kernels

class MinimalDotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr)

class XfPoclPlatform(XfBasePlatform):

    """ 

        Creates a Pocl Platform object, that allows performing the computations
        on GPUs and CPUs through PyOpenCL.

    Args:
        default_kernels (bool): If ``True``, the Xfields defult kernels are
            automatically imported.
        pocl_context: A PyOpenCL context can be optionally provided. Otherwise
            a context is created by the platform.
        command_queue: A PyOpenCL command que can be optionally provided. 
            Otherwise a queue is created by the platform.
        patch_pocl_array (bool): If ``True``, the PyOpecCL class is patched to
            allow some operations with non-contiguous arrays.

    Returns:
        XfPoclPlatform: platform object.

    """

    def __init__(self, pocl_context=None, command_queue=None, default_kernels=True,
                 patch_pocl_array=True):

        if pocl_context is None:
            pocl_context = cl.create_some_context()

        if command_queue is None:
            command_queue = cl.CommandQueue(pocl_context)

        assert command_queue.context == pocl_context

        self.pocl_context = pocl_context
        self.command_queue = command_queue
        self.kernels = MinimalDotDict()

        if patch_pocl_array:
            from ._patch_pocl_array import _patch_pocl_array
            _patch_pocl_array(cl, cla, pocl_context)

        if default_kernels:
            self.add_kernels(src_files=pocl_default_kernels['src_files'],
                    kernel_descriptions=pocl_default_kernels['kernel_descriptions'])

    @property
    def nplike_lib(self):
        """
        Module containing all the numpy features supported by PyOpenCL (optionally
        with patches to operate with non-contiguous arrays).

        Example:

        .. code-block:: python

            platform =  XfPoclPlatform()
            nplike = platform.nplike_lib

            # This returns an array of zeros on the computing device (GPU):
            a = nplike.zeros((10,10), dtype=nplike.float64

        """
        return cla

    def synchronize(self):
        """
        Ensures that all computations submitted to the platform are completed.
        No action is performed by this function in the Pocl platform. The method
        is provided so that the Pocl platform has an identical API to the Cupy one.
        """
        pass

    def zeros(self, *args, **kwargs):
        """
        Allocates an array of zeros on the device. The function has the same
         interface of numpy.zeros"""
        return self.nplike_lib.zeros(queue=self.command_queue, *args, **kwargs)


    def nparray_to_platform_mem(self, arr):
        """Copies a numpy array to the device memory.
        Args:
            arr (numpy.ndarray): Array to be transferred

        Returns:
            pyopencl.array.Array:The same array copied to the device.

        """
        dev_arr = cla.to_device(self.command_queue, arr)
        return dev_arr

    def nparray_from_platform_mem(self, dev_arr):
        """Copies an array to the device to a numpy array.

        Args:
            dev_arr (pyopencl.array.Array): Array to be transferred.
        Returns:
            numpy.ndarray: The same data copied to a numpy array.

        """
        return dev_arr.get()

    def plan_FFT(self, data, axes, wait_on_call=True):
        """Generates an FFT plan object to be executed on the platform.

        Args:
            data (pyopencl.array.Array): Array having type and shape for which
                the FFT needs to be planned.
            axes (sequence of ints): Axes along which the FFT needs to be
                performed.
        Returns:
            XfPoclFFT: FFT plan for the required array shape, type and axes.

        Example:

        .. code-block:: python

            plan = platform.plan_FFT(data, axes=(0,1))

            data2 = 2*data

            # Forward tranform (in place)
            plan.transform(data2)

            # Inverse tranform (in place)
            plan.itransform(data2)
        """
        return XfPoclFFT(self, data, axes, wait_on_call)

    def add_kernels(self, src_code='', src_files=[], kernel_descriptions={}):

        """Adds user-defined kernels to to the platform. The kernel source
           code is provided as a string and/or in source files and must contain
           the kernel names defined in the kernel descriptions.

        Args:
            src_code (str): String with the kernel source code. Default: empty
                string.
            src_files (list of strings): paths to files containing the
                source code. Default: empty list.
            kernel_descriptions (dict): Dictionary with the kernel descriptions
                in the form given by the following examples. The decriptions
                define the kernel names, the type and name of the arguments
                and identifies one input argument that defines the number of
                threads to be launched.

        Example:

        .. code-block:: python

            src_code = r'''
            __global__
            void my_mul(const int n, const float* x1,
                        const float* x2, float* y) {
                int tid = blockDim.x * blockIdx.x + threadIdx.x;
                if (tid < n){
                    y[tid] = x1[tid] * x2[tid];
                    }
                }
            '''
            kernel_descriptions = {'my_mul':{
                args':(
                    (('scalar', np.int32),   'n',),
                    (('array',  np.float64), 'x1',),
                    (('array',  np.float64), 'x2',),
                    )
                'num_threads_from_arg': 'nparticles'
                },}

            # Import kernel in platform
            platform.add_kernels(src_code, kernel_descriptions)

            # With a1 and a2 being arrays on the platform, the kernel
            # can be called as follows:
            platform.kernels.my_mul(n=len(a1), x1=a1, x2=a2)
        """

        src_content = src_code
        for ff in src_files:
            with open(ff, 'r') as fid:
                src_content += ('\n\n' + fid.read())

        prg = cl.Program(self.pocl_context, src_content).build()

        ker_names = kernel_descriptions.keys()
        for nn in ker_names:
            kk = getattr(prg, nn)
            aa = kernel_descriptions[nn]['args']
            nt_from = kernel_descriptions[nn]['num_threads_from_arg']
            aa_types, aa_names = zip(*aa)
            self.kernels[nn] = XfPoclKernel(pocl_kernel=kk,
                arg_names=aa_names, arg_types=aa_types,
                num_threads_from_arg=nt_from,
                command_queue=self.command_queue)

class XfPoclKernel(object):

    def __init__(self, pocl_kernel, arg_names, arg_types,
                 num_threads_from_arg, command_queue,
                 wait_on_call=True):

        assert (len(arg_names) == len(arg_types) == pocl_kernel.num_args)
        assert num_threads_from_arg in arg_names

        self.pocl_kernel = pocl_kernel
        self.arg_names = arg_names
        self.arg_types = arg_types
        self.num_threads_from_arg = num_threads_from_arg
        self.command_queue = command_queue
        self.wait_on_call = wait_on_call

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
                assert isinstance(vv, cla.Array)
                assert vv.context == self.pocl_kernel.context
                arg_list.append(vv.base_data[vv.offset:])
            else:
                raise ValueError(f'Type {tt} not recognized')

        event = self.pocl_kernel(self.command_queue,
                (kwargs[self.num_threads_from_arg],),
                None, *arg_list)

        if self.wait_on_call:
            event.wait()

        return event

class XfPoclFFT(object):
    def __init__(self, platform, data, axes, wait_on_call=True):

        self.platform = platform
        self.axes = axes
        self.wait_on_call = wait_on_call

        assert len(data.shape) > max(axes)

        # Check internal dimensions are powers of two
        for ii in axes[:-1]:
            nn = data.shape[ii]
            frac_part, _ = np.modf(np.log(nn)/np.log(2))
            assert np.isclose(frac_part, 0) , ('PyOpenCL FFT requires'
                    ' all dimensions apart from the last to be powers of two!')

        import gpyfft
        self._fftobj = gpyfft.fft.FFT(platform.pocl_context,
                platform.command_queue, data, axes=axes)

    def transform(self, data):
        """The transform is done inplace"""

        event, = self._fftobj.enqueue_arrays(data)
        if self.wait_on_call:
            event.wait()
        return event

    def itransform(self, data):
        """The transform is done inplace"""

        event, = self._fftobj.enqueue_arrays(data, forward=False)
        if self.wait_on_call:
            event.wait()
        return event
