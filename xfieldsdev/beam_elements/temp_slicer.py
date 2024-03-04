import numpy as np
from scipy import special
import xobjects as xo
import xpart as xp
from ..general import _pkg_root

_digitize_kernel = xo.Kernel(
            c_name="digitize",
            args=[xo.Arg(xp.Particles._XoStruct, name='particles'),
                  xo.Arg(xo.Float64, const=True, pointer=True, name='particles_zeta'),
                  xo.Arg(xo.Float64, const=True, pointer=True, name='bin_edges'),
                  xo.Arg(xo.Int64, name='n_slices'),
                  xo.Arg(xo.Int64, pointer=True, name='particles_slice')]
)

_compute_slice_moments_kernel = xo.Kernel(
            c_name="compute_slice_moments",
            args=[xo.Arg(xp.Particles._XoStruct, name='particles'),
                  xo.Arg(xo.Int64, pointer=True, name='particles_slice'),
                  xo.Arg(xo.Float64, pointer=True, name='moments'),
                  xo.Arg(xo.Int64, name='n_slices'),
                  xo.Arg(xo.Int64, name='threshold_num_macroparticles')]
)


_compute_slice_moments_cuda_sums_per_slice_kernel = xo.Kernel(
            c_name="compute_slice_moments_cuda_sums_per_slice",
            args=[xo.Arg(xp.Particles._XoStruct, name='particles'),
                  xo.Arg(xo.Int64, pointer=True, name='particles_slice'),
                  xo.Arg(xo.Float64, pointer=True, name='moments'),
                  xo.Arg(xo.Int64, const=True, name='num_macroparticles'),
                  xo.Arg(xo.Int64, const=True, name='n_slices'),
                  xo.Arg(xo.Int64, const=True, name='shared_mem_size_bytes')
                  ],
            n_threads="num_macroparticles",
)

_compute_slice_moments_cuda_moments_from_sums_kernel = xo.Kernel(
            c_name="compute_slice_moments_cuda_moments_from_sums",
            args=[xo.Arg(xo.Float64, pointer=True, name='moments'),
                  xo.Arg(xo.Int64, const=True, name='n_slices'),
                  xo.Arg(xo.Int64, const=True, name='weight'),
                  xo.Arg(xo.Int64, const=True, name='threshold_num_macroparticles')],
            n_threads="n_slices",
)

_temp_slicer_kernels = {'digitize': _digitize_kernel,
                        'compute_slice_moments':_compute_slice_moments_kernel,
                        'compute_slice_moments_cuda_sums_per_slice':_compute_slice_moments_cuda_sums_per_slice_kernel,
                        'compute_slice_moments_cuda_moments_from_sums':_compute_slice_moments_cuda_moments_from_sums_kernel,
                        }


class TempSlicer(xo.HybridClass):

    _xofields = {
        '_dummy': xo.Int64, # Not to have zero-size xobject
    }

    # I add undescores in front of the names so that I can define custom
    # properties
    _rename = {nn: '_'+nn for nn in _xofields}

    _extra_c_sources = [_pkg_root.joinpath('headers/compute_slice_moments.h'),
                        ]

    _depends_on = [xp.Particles]

    _kernels = _temp_slicer_kernels

    def __init__(self, _context=None,
                 _buffer=None,
                 _offset=None,
                 _xobject=None,
                 n_slices = 11,
                 sigma_z = 0.1,
                 mode="unibin"):

        assert isinstance(n_slices, int) and n_slices>0, ("'n_slices' must be a positive integer!")
        assert mode in ["unicharge", "unibin", "shatilov"], ("Accepted values for 'mode': 'unicharge', 'unibin', 'shatilov'")

        # bin params are in units of RMS bunch length
        if mode=="unicharge":
            z_k_arr, l_k_arr, w_k_arr, dz_k_arr = self.unicharge(n_slices)
        elif mode=="unibin":
            z_k_arr, l_k_arr, w_k_arr, dz_k_arr = self.unibin(n_slices)
        elif mode=="shatilov":
            z_k_arr, l_k_arr, w_k_arr, dz_k_arr = self.shatilov(n_slices)

        self.num_slices  = n_slices
        self.sigma_z     = sigma_z
        self.bin_centers = z_k_arr * sigma_z
        self.bin_edges   = l_k_arr * sigma_z
        self.bin_weights = w_k_arr
        self.bin_widths_beamstrahlung = dz_k_arr * sigma_z

        if _xobject is not None:
            self.xoinitialize(_xobject=_xobject, _context=_context,
                             _buffer=_buffer, _offset=_offset)
            return

        self.xoinitialize(
                 _context=_context,
                 _buffer=_buffer,
                 _offset=_offset)

        self.compile_kernels(only_if_needed=False)

    def rho(self, z):
        """
        Gaussian charge density.
        Rho has a unit of [1/m].
        z = z_k/sigma_z [1], normalized by bunch length in the frame where the slicing takes place
        """
        return np.exp(-z**2/(2)) / (np.sqrt(2*np.pi))

    def unicharge(self, n_slices):
        """
        Uniform charge slicing.
        """

        # these are units of sigma_z
        z_k_arr_unicharge = np.zeros(n_slices)  # should be n_slices long, ordered from + to -
        l_k_arr_unicharge = np.zeros(n_slices+1)  # bin edges, n_slices+1 long
        w_k_arr_unicharge = np.zeros(n_slices)  # bin weights, used for bunch intensity normalization
        half = int((n_slices + 1) / 2)
        n_odd = n_slices % 2
        w_k_arr_unicharge[:half] = 1 / n_slices  # fill up initial values, e.g. n_slices=300-> fill up elements [0,149]; 301: [0,150]
        l_k_arr_unicharge[0] = -5  # leftmost bin edge
        w_k_sum = 0 # sum of weights: integral of gaussian up to l_k
        rho_upper = 0 # start from top of distribution (positive end, l_upper=inf)

        # go from bottom end toward 0 (=middle of Gaussian)
        for j in range(half):

            w_k_sum += 2*w_k_arr_unicharge[j] # integrate rho up to and including bin j

            # get bin center
            if n_odd and j == half-1:  # center bin (z_c=0)
                z_k_arr_unicharge[j] = 0
            else:  # all other bins
                rho_lower = rho_upper

                # get upper bin boundary
                arg = w_k_sum - 1
                l_upper = np.sqrt(2)*special.erfinv(arg)
                l_k_arr_unicharge[j+1] = l_upper
                rho_upper = self.rho(l_upper)

                # get z_k: center of momentum
                z_k_arr_unicharge[j] = (rho_upper - rho_lower) / w_k_arr_unicharge[j]

        # mirror for positive half
        z_k_arr_unicharge[half:] = -z_k_arr_unicharge[n_slices-half-1::-1]  # bin centers
        w_k_arr_unicharge[half:] =  w_k_arr_unicharge[n_slices-half-1::-1]  # bin weights, used for bunch intensity normalization
        l_k_arr_unicharge[half:] = -l_k_arr_unicharge[n_slices-half::-1]  # bin edges
        dz_k_arr_unicharge       = np.diff(l_k_arr_unicharge)  # for beamstrahlung
        l_k_arr_unicharge        = l_k_arr_unicharge[::-1]

        return z_k_arr_unicharge, l_k_arr_unicharge, w_k_arr_unicharge, dz_k_arr_unicharge

    def unibin(self, n_slices):
        """
        Uniform bin slicing.
        """

        # these are units of sigma_z
        z_k_list_unibin = []  # should be n_slices long, ordered from + to -

        m = 1 if not n_slices%2 else 0

        # dmitry goes from +n_slices/2 to -n_slices/2-1 (50-(-51) for 101 slices); hirata goes from n_slices to 0
        for k in range(int(n_slices/2), -int(n_slices/2)-(1-m), -1):

            # slices extend from -N*sigma to +N*sigma
            N = 5
            z_k = (2*k - m) / (n_slices - 1) * N * special.erf(np.sqrt(n_slices / 6))
            z_k_list_unibin.append(z_k)

        z_k_arr_unibin = np.array(z_k_list_unibin)  # bin centers
        w_k_arr_unibin = np.exp(-z_k_arr_unibin**2/2) # proportional, but these are not yet not normalized
        w_k_arr_unibin = w_k_arr_unibin / np.sum(w_k_arr_unibin) # bin weights, used for bunch intensity normalization
        dz_i = -np.diff(z_k_arr_unibin)[0]
        l_k_arr_unibin = np.hstack([z_k_arr_unibin+dz_i/2, z_k_arr_unibin[-1]-dz_i/2])  # bin edges
        dz_k_array_unibin = np.ones(n_slices)*dz_i  # for beamstrahlung

        return z_k_arr_unibin, l_k_arr_unibin, w_k_arr_unibin, dz_k_array_unibin

    def shatilov(self, n_slices):
        """
        This method is a mix between uniform bin and charge. It finds the slice centers by iteration.
        """

        # these are units of sigma_z
        z_k_arr_shatilov = np.zeros(n_slices)  # should be n_slices long, ordered from + to -
        l_k_arr_shatilov = np.zeros(n_slices+1)  # bin edges, n_slices+1 long
        w_k_arr_shatilov = np.zeros(n_slices)  # bin weights, used for bunch intensity normalization
        half = int((n_slices + 1) / 2)
        n_odd = n_slices % 2
        w_k_arr_shatilov[:half] = 1 / n_slices  # fill up initial values, e.g. n_slices=300-> fill up elements [0,149]; 301: [0,150]
        l_k_arr_shatilov[0] = -5  # leftmost bin edge

        k_max = min(1000, 20*n_slices)  # max iterations for l_k

        for i in range(k_max+1):
            w_k_sum = 0 # sum of weights: integral of gaussian up to l_k
            rho_upper = 0 # start from top of distribution (positive end, l_upper=inf)

            # go from bottom toward 0 (=middle of Gaussian)
            for j in range(half):

                w_k_sum += 2*w_k_arr_shatilov[j] # integrate rho up to including current bin

                # get z_k
                if n_odd and j == half-1:  # center bin (z_c=0)
                    z_k_arr_shatilov[j] = 0
                else:  # all other bins
                    rho_lower = rho_upper

                    arg = w_k_sum - 1
                    l_upper = np.sqrt(2)*special.erfinv(arg)

                    l_k_arr_shatilov[j+1] = l_upper

                    rho_upper = self.rho(l_upper)  # to cancel 1/sigma_z in rho

                    # get z_k: center of momentum
                    z_k_arr_shatilov[j] = (rho_upper - rho_lower) / w_k_arr_shatilov[j]

                # get w_k
                if i < k_max:
                    w_k_arr_shatilov[j] = np.exp( -z_k_arr_shatilov[j]**2 / 4 )

            # renormalize w_k
            if i < k_max:
                w_int = 2*np.sum(w_k_arr_shatilov[:half]) - n_odd * w_k_arr_shatilov[half-1]
                w_k_arr_shatilov[:half] = w_k_arr_shatilov[:half] / w_int

        # mirror for negative half
        z_k_arr_shatilov[half:] = -z_k_arr_shatilov[n_slices-half-1::-1]  # bin centers
        w_k_arr_shatilov[half:] =  w_k_arr_shatilov[n_slices-half-1::-1]  # bin weights, used for bunch intensity normalization
        l_k_arr_shatilov[half:] = -l_k_arr_shatilov[n_slices-half::-1]  # bin edges
        dz_k_arr_shatilov       = np.diff(l_k_arr_shatilov)  # for beamstrahlung
        l_k_arr_shatilov        = l_k_arr_shatilov[::-1]

        return z_k_arr_shatilov, l_k_arr_shatilov, w_k_arr_shatilov, dz_k_arr_shatilov

    def get_slice_indices(self, particles):
        context = particles._context
        if isinstance(context, xo.ContextPyopencl):
            raise NotImplementedError

        bin_edges = context.nparray_to_context_array(self.bin_edges)

        if isinstance(context, xo.ContextCupy):
            digitize = particles._context.nplike_lib.digitize  # only works with cpu and cupy
            indices = digitize(particles.zeta, bin_edges, right=True)
        else:  # OpenMP implementation of binary search for CPU
            indices = particles._context.nplike_lib.zeros_like(particles.zeta, dtype=particles._context.nplike_lib.int64)
            self._context.kernels.digitize(particles = particles, particles_zeta = particles.zeta,
                                                    bin_edges = bin_edges, n_slices = self.num_slices,
                                                    particles_slice = indices)

        indices -= 1 # In digitize, 0 means before the first edge
        indices[particles.state <=0 ] = -1

        indices_out = context.zeros(shape=indices.shape, dtype=np.int64)
        indices_out[:] = indices
        return indices_out

    def assign_slices(self, particles):
        particles.slice = self.get_slice_indices(particles)

    def compute_moments(self, particles, update_assigned_slices=True, threshold_num_macroparticles=20):
        context = particles._context
        if isinstance(context, xo.ContextPyopencl):
            raise NotImplementedError

        if update_assigned_slices:
            self.assign_slices(particles)

        if isinstance(context, xo.ContextCupy):
            slice_moments = self._context.zeros(self.num_slices*(6+10+1+6+10),dtype=np.float64)  # sums (16) + count (1) + moments (16)
            self._context.kernels.compute_slice_moments_cuda_sums_per_slice(particles=particles, particles_slice=particles.slice,
                                                           moments=slice_moments, num_macroparticles=np.int64(len(particles.slice)),
                                                           n_slices=np.int64(self.num_slices), shared_mem_size_bytes=np.int64(self.num_slices*17*8))

            self._context.kernels.compute_slice_moments_cuda_moments_from_sums(moments=slice_moments, n_slices=np.int64(self.num_slices),
                                                           weight=particles.weight.get()[0], threshold_num_macroparticles=np.int64(threshold_num_macroparticles))
            return slice_moments[int(self.num_slices*16):]

        # context CPU with OpenMP
        else:

            slice_moments = self._context.zeros(self.num_slices*(1+6+10),dtype=np.float64)

            # np.cumsum[-1] =/= np.sum due to different order of summation
            # use np.isclose instead of ==; np.sum does pariwise sum which orders values differently thus causing a numerical error
            # see: https://stackoverflow.com/questions/69610452/why-does-the-last-entry-of-numpy-cumsum-not-necessarily-equal-numpy-sum
            self._context.kernels.compute_slice_moments(particles=particles, particles_slice=particles.slice,
                                                    moments=slice_moments, n_slices=self.num_slices,
                                                    threshold_num_macroparticles=threshold_num_macroparticles)
            return slice_moments
