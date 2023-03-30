import numpy as np
from scipy import special
import xobjects as xo

class TempSlicer:
    def __init__(self, n_slices, sigma_z, mode="unibin"):

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

        digitize = particles._context.nplike_lib.digitize  # only works with cpu and cupy
        indices = digitize(particles.zeta, bin_edges, right=True)
        indices -= 1 # In digitize, 0 means before the first edge
        indices[particles.state <=0 ] = -1

        indices_out = context.zeros(shape=indices.shape, dtype=np.int64)
        indices_out[:] = indices
        return indices_out

    def assign_slices(self, particles):
        particles.slice = self.get_slice_indices(particles)

    def compute_moments(self, particles, update_assigned_slices=True, threshold_num_macroparticles=20):
        if update_assigned_slices:
            self.assign_slices(particles)

        slice_moments = np.zeros(self.num_slices*(1+6+10),dtype=float)
        for i_slice in range(self.num_slices):
            mask = (particles.slice == i_slice) & (particles.state >0)  # skip lost particles (1: alive, 0 lost)
            slice_moments[i_slice]                   = 0 if len(particles.x[mask]) < threshold_num_macroparticles else len(particles.x[mask])                                    # nb part
            slice_moments[self.num_slices+i_slice]   = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float(particles.x[mask].sum())/slice_moments[i_slice]     # <x>
            slice_moments[2*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float(particles.px[mask].sum())/slice_moments[i_slice]    # <px>
            slice_moments[3*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float(particles.y[mask].sum())/slice_moments[i_slice]     # <y>
            slice_moments[4*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float(particles.py[mask].sum())/slice_moments[i_slice]    # <py>
            slice_moments[5*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float(particles.zeta[mask].sum())/slice_moments[i_slice]  # <z>
            slice_moments[6*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float(particles.delta[mask].sum())/slice_moments[i_slice] # <pz> # TODO mhy pzeta doesn't work?

            x_diff  = 0 if len(particles.x[mask]) < threshold_num_macroparticles else particles.x[mask]-slice_moments[self.num_slices+i_slice]
            px_diff = 0 if len(particles.x[mask]) < threshold_num_macroparticles else particles.px[mask]-slice_moments[2*self.num_slices+i_slice]
            y_diff  = 0 if len(particles.x[mask]) < threshold_num_macroparticles else particles.y[mask]-slice_moments[3*self.num_slices+i_slice]
            py_diff = 0 if len(particles.x[mask]) < threshold_num_macroparticles else particles.py[mask]-slice_moments[4*self.num_slices+i_slice]
            slice_moments[7*self.num_slices+i_slice]  = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((x_diff**2).sum())/slice_moments[i_slice]             # Sigma_11
            slice_moments[8*self.num_slices+i_slice]  = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((x_diff*px_diff).sum())/slice_moments[i_slice]      # Sigma_12
            slice_moments[9*self.num_slices+i_slice]  = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((x_diff*y_diff).sum())/slice_moments[i_slice]       # Sigma_13
            slice_moments[10*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((x_diff*py_diff).sum())/slice_moments[i_slice]     # Sigma_14
            slice_moments[11*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((px_diff**2).sum())/slice_moments[i_slice]           # Sigma_22
            slice_moments[12*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((px_diff*y_diff).sum())/slice_moments[i_slice]     # Sigma_23
            slice_moments[13*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((px_diff*py_diff).sum())/slice_moments[i_slice]    # Sigma_24
            slice_moments[14*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((y_diff**2).sum())/slice_moments[i_slice]            # Sigma_33
            slice_moments[15*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((y_diff*py_diff).sum())/slice_moments[i_slice]     # Sigma_34
            slice_moments[16*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((py_diff**2).sum())/slice_moments[i_slice]           # Sigma_44

        for i_slice in range(self.num_slices):
            slice_moments[i_slice] *= particles.weight[0]

        return slice_moments