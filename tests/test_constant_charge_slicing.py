import numpy as np

import xfields as xf


constant_charge_slicing_gaussian = (
xf.config_tools.beambeam_config_tools.config_tools.constant_charge_slicing_gaussian)


def test_constant_charge_slicing_gaussian():

    sigma_z = 0.4
    n_particles = 1e11
    n_slices = 7

    z_centroids, z_cuts, N_part_per_slice = constant_charge_slicing_gaussian(
        N_part_tot=n_particles, sigmaz=sigma_z, N_slices=n_slices)

    z = np.linspace(-5*sigma_z, 5*sigma_z, 1000000)
    f_distrib = n_particles/(sigma_z * np.sqrt(2 * np.pi))*np.exp(-z**2/(2*sigma_z**2))

    assert np.isclose(np.trapz(f_distrib, z), n_particles, atol=0, rtol=1e-6)
    assert np.isclose(np.sum(N_part_per_slice), n_particles, atol=0, rtol=1e-14)

    for ii in range(len(z_cuts)-1):
        assert np.isclose(
            np.trapz(f_distrib[np.logical_and(z>=z_cuts[ii], z<z_cuts[ii+1])],
                    z[np.logical_and(z>=z_cuts[ii], z<z_cuts[ii+1])]),
                    N_part_per_slice[ii], atol=0, rtol=1e-3)
        assert z_cuts[ii] < z_centroids[ii+1] < z_cuts[ii+1]
        assert np.isclose(
            np.trapz(
                (z[np.logical_and(z>=z_cuts[ii], z<z_cuts[ii+1])]
                * f_distrib[np.logical_and(z>=z_cuts[ii], z<z_cuts[ii+1])])
                / N_part_per_slice[ii],
                z[np.logical_and(z>=z_cuts[ii], z<z_cuts[ii+1])]),
            z_centroids[ii+1], atol=10e-6, rtol=0)
