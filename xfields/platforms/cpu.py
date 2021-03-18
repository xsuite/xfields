import ctypes

import cppyy

import numpy as np

from .base import XfBasePlatform
from .default_kernels import cpu_default_kernels

class MinimalDotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr)

class XfCpuPlatform(object):

    """

       Creates a CPU Platform object, that allows performing the computations
       on conventionla CPUs.

    Args:
        default_kernels (bool): If ``True``, the Xfields defult kernels are
            automatically imported.
    Returns:
        XfCpuPlatform: platform object.

    """

    def __init__(self, default_kernels=True):

        self._kernels = MinimalDotDict()

        if default_kernels:
            self.add_kernels(src_files=cpu_default_kernels['src_files'],
                    kernel_descriptions=cpu_default_kernels['kernel_descriptions'])

    def add_kernels(self, src_code='', src_files=[], kernel_descriptions={}):

        """
        Adds user-defined kernels to to the platform. The kernel source
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
            void my_mul(const int n, const float* x1,
                        const float* x2, float* y) {
                int tid;
                for (tid=0; tid<n; tid++){
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


        src_content = src_code;
        for ff in src_files:
            with open(ff, 'r') as fid:
                src_content += ('\n\n' + fid.read())

        ker_names = kernel_descriptions.keys()

        skip_compile = False
        for kk in ker_names:
            if hasattr(cppyy.gbl, kk):
                skip_compile = True
                break

        if skip_compile:
            print('Warning! Compilation is skipped because some of'
                  ' the kernels already exist! To recompile all '
                  'please restart python')
        else:
            cppyy.cppdef(src_content)

        for nn in ker_names:
            kk = getattr(cppyy.gbl, nn)
            aa = kernel_descriptions[nn]['args']
            aa_types, aa_names = zip(*aa)
            self.kernels[nn] = XfCpuKernel(cppyy_kernel=kk,
                arg_names=aa_names, arg_types=aa_types)

    def nparray_to_platform_mem(self, arr):
        """
        Moves a numpy array to the device memory. No action is performed by
        this function in the CPU platform. The method is provided
        so that the CPU platform has an identical API to the GPU ones.

        Args:
            arr (numpy.ndarray): Array to be transferred

        Returns:
            numpy.ndarray: The same array (no copy!).

        """
        return arr

    def nparray_from_platform_mem(self, dev_arr):
        """
        Moves an array to the device to a numpy array. No action is performed by
        this function in the CPU platform. The method is provided so that the CPU
        platform has an identical API to the GPU ones.

        Args:
            dev_arr (numpy.ndarray): Array to be transferred/
        Returns:
            numpy.ndarray: The same data copied to a numpy array.

        """
        return dev_arr

    @property
    def nplike_lib(self):
        """
        Module containing all the numpy features. Numpy members should be accessed
        through ``nplike_lib`` to keep compatibility with the other platforms.

        """

        return np

    def synchronize(self):
        """
        Ensures that all computations submitted to the platform are completed.
        No action is performed by this function in the CPU platform. The method
        is provided so that the CPU platform has an identical API to the GPU ones.
        """
        pass

    def zeros(self, *args, **kwargs):
        """
        Allocates an array of zeros on the device. The function has the same
        interface of numpy.zeros"""
        return self.nplike_lib.zeros(*args, **kwargs)

    def plan_FFT(self, data, axes):
        """
        Generate an FFT plan object to be executed on the platform.

        Args:
            data (numpy.ndarray): Array having type and shape for which the FFT
                needs to be planned.
            axes (sequence of ints): Axes along which the FFT needs to be
                performed.
        Returns:
            XfCpuFFT: FFT plan for the required array shape, type and axes.

        Example:

        .. code-block:: python

            plan = platform.plan_FFT(data, axes=(0,1))

            data2 = 2*data

            # Forward tranform (in place)
            plan.transform(data2)

            # Inverse tranform (in place)
            plan.itransform(data2)
        """
        return XfCpuFFT(data, axes)

    @property
    def kernels(self):

        """
        Dictionary containing all the kernels that have been imported to the platform.
        The syntax ``platform.kernels.mykernel`` can also be used.

        Example:

        .. code-block:: python

            src_code = r'''
            void my_mul(const int n, const float* x1,
                        const float* x2, float* y) {
                int tid;
                for (tid=0; tid<n; tid++){
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
            # or as follows:
            platform.kernels['my_mul'](n=len(a1), x1=a1, x2=a2)

        """

        return self._kernels


class XfCpuKernel(object):

    def __init__(self, cppyy_kernel, arg_names, arg_types):

        assert (len(arg_names) == len(arg_types))

        self.cppyy_kernel = cppyy_kernel
        self.arg_names = arg_names
        self.arg_types = arg_types

        c_argtypes = []
        for tt in arg_types:
            if tt[0] == 'scalar':
                if np.issubdtype(tt[1], np.integer):
                    c_argtypes.append(int)
                else:
                    c_argtypes.append(tt[1])
            elif tt[0] == 'array':
                c_argtypes.append(None) # Not needed for cppyy
            else:
                raise ValueError(f'Type {tt} not recognized')
        self.c_arg_types = c_argtypes

    @property
    def num_args(self):
        return len(self.arg_names)

    def __call__(self, **kwargs):
        assert len(kwargs.keys()) == self.num_args
        arg_list = []
        for nn, tt, ctt in zip(self.arg_names, self.arg_types, self.c_arg_types):
            vv = kwargs[nn]
            if tt[0] == 'scalar':
                assert np.isscalar(vv)
                arg_list.append(ctt(vv))
            elif tt[0] == 'array':
                arg_list.append(vv.ctypes.data_as(ctypes.POINTER(
                    np.ctypeslib.as_ctypes_type(tt[1]))))
            else:
                raise ValueError(f'Type {tt} not recognized')

        event = self.cppyy_kernel(*arg_list)


class XfCpuFFT(object):
    def __init__(self, data, axes):

        self.axes = axes

        # I perform one fft to have numpy cache the plan
        _ = np.fft.ifftn(np.fft.fftn(data, axes=axes), axes=axes)

    def transform(self, data):
        """The transform is done inplace"""
        data[:] = np.fft.fftn(data, axes=self.axes)[:]

    def itransform(self, data):
        """The transform is done inplace"""
        data[:] = np.fft.ifftn(data, axes=self.axes)[:]
