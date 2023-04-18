import pathlib

import numpy as np
from scipy import constants as cst

import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp

from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()


@for_all_test_contexts(excluding='ContextPyopencl')
def test_beambeam3d_beamstrahlung_single_collision(test_context):
    ###########
    # ttbar 2 #
    ###########
    bunch_intensity     = 2.3e11  # [1]
    energy              = 182.5  # [GeV]
    p0c                 = 182.5e9  # [eV]
    mass0               = .511e6  # [eV]
    phi                 = 15e-3  # [rad] half xing
    physemit_x          = 1.46e-09  # [m]
    physemit_y          = 2.9e-12  # [m]
    beta_x              = 1  # [m]
    beta_y              = .0016  # [m]
    sigma_x             = np.sqrt(physemit_x*beta_x)  # [m]
    sigma_px            = np.sqrt(physemit_x/beta_x)  # [m]
    sigma_y             = np.sqrt(physemit_y*beta_y)  # [m]
    sigma_py            = np.sqrt(physemit_y/beta_y)  # [m]
    sigma_z_tot         = .00254  # [m] sr+bs
    sigma_delta_tot     = .00192  # [m]
    n_macroparticles_b1 = int(1e6)

    n_slices = 100

    #############
    # particles #
    #############

    #e-
    particles_b1 = xp.Particles(
                _context = test_context,
                q0        = -1,
                p0c       = p0c,
                mass0     = mass0,
                x         = sigma_x        *np.random.randn(n_macroparticles_b1),
                y         = sigma_y        *np.random.randn(n_macroparticles_b1),
                zeta      = sigma_z_tot    *np.random.randn(n_macroparticles_b1),
                px        = sigma_px       *np.random.randn(n_macroparticles_b1),
                py        = sigma_py       *np.random.randn(n_macroparticles_b1),
                delta     = sigma_delta_tot*np.random.randn(n_macroparticles_b1),
                )

    particles_b1.name = "b1"

    slicer = xf.TempSlicer(n_slices=n_slices, sigma_z=sigma_z_tot, mode="unicharge")

    el_beambeam_b1 = xf.BeamBeamBiGaussian3D(
    _context=test_context,
    config_for_update = None,
    other_beam_q0=1,
    phi=phi,
    alpha=0,
    # decide between round or elliptical kick formula
    min_sigma_diff     = 1e-28,
    # slice intensity [num. real particles] n_slices inferred from length of this
    slices_other_beam_num_particles = slicer.bin_weights * bunch_intensity,
    # unboosted strong beam moments
    slices_other_beam_zeta_center = slicer.bin_centers,
    slices_other_beam_Sigma_11    = n_slices*[sigma_x**2],
    slices_other_beam_Sigma_22    = n_slices*[sigma_px**2],
    slices_other_beam_Sigma_33    = n_slices*[sigma_y**2],
    slices_other_beam_Sigma_44    = n_slices*[sigma_py**2],
    # only if BS on
    slices_other_beam_zeta_bin_width_star_beamstrahlung = slicer.bin_widths_beamstrahlung / np.cos(phi),  # boosted dz
    # has to be set
    slices_other_beam_Sigma_12    = n_slices*[0],
    slices_other_beam_Sigma_34    = n_slices*[0],
    )

    #########################
    # track for 1 collision #
    #########################

    line = xt.Line(elements = [el_beambeam_b1])
    line.build_tracker(_context=test_context)

    assert line._needs_rng == False

    line.configure_radiation(model_beamstrahlung='quantum')
    assert line._needs_rng == True

    record = line.start_internal_logging_for_elements_of_type(
        xf.BeamBeamBiGaussian3D, capacity={"beamstrahlungtable": int(3e5)})
    line.track(particles_b1, num_turns=1)
    line.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

    record.move(_context=xo.context_default)

    ###########################################
    # test 1: compare spectrum with guineapig #
    ###########################################

    fname = (test_data_folder
        / "beamstrahlung/guineapig_ttbar2_beamstrahlung_photon_energies_gev.txt")
    guinea_photons = np.loadtxt(fname)  # contains about 250k photons emitted from 1e6 macroparticles in 1 collision
    n_bins = 10
    bins = np.logspace(
        np.log10(1e-14), np.log10(1e1), n_bins)
    xsuite_hist = np.histogram(record.beamstrahlungtable.photon_energy/1e9,
                              bins=bins)[0]
    guinea_hist = np.histogram(guinea_photons, bins=bins)[0]

    bin_rel_errors = np.abs(xsuite_hist - guinea_hist) / guinea_hist
    print(f"bin relative errors [1]: {bin_rel_errors}")

    # test if relative error in the last 5 bins is smaller than 1e-1
    assert np.allclose(xsuite_hist[-5:], guinea_hist[-5:], rtol=1e-1, atol=0)

    ############################################
    # test 2: compare beamstrahlung parameters #
    ############################################

    # average and maximum BS parameter
    # https://www.researchgate.net/publication/2278298_Beam-Beam_Phenomena_In_Linear_Colliders
    # page 20
    r0 = cst.e**2/(4*np.pi*cst.epsilon_0*cst.m_e*cst.c**2) # - if pp

    upsilon_max = (
        2 * r0**2 * energy/(mass0*1e-9) * bunch_intensity
        / (1/137*sigma_z_tot*(sigma_x + 1.85*sigma_y)))
    upsilon_avg = (5/6 * r0**2 * energy/(mass0*1e-9) * bunch_intensity
                   / (1/137*sigma_z_tot*(sigma_x + sigma_y)))

    # get rid of padded zeros in table
    photon_critical_energy = np.array(
        sorted(set(record.beamstrahlungtable.photon_critical_energy))[1:])
    primary_energy         = np.array(
        sorted(set(        record.beamstrahlungtable.primary_energy))[1:])

    upsilon_avg_sim = np.mean(0.67 * photon_critical_energy / primary_energy)
    upsilon_max_sim = np.max(0.67 * photon_critical_energy / primary_energy)

    print("Y max. [1]:", upsilon_max)
    print("Y avg. [1]:", upsilon_avg)
    print("Y max. [1]:", upsilon_max_sim)
    print("Y avg. [1]:", upsilon_avg_sim)
    print("Y max. ratio [1]:", upsilon_max_sim / upsilon_max)
    print("Y avg. ratio [1]:", upsilon_avg_sim / upsilon_avg)

    assert np.abs(upsilon_max_sim / upsilon_max - 1) < 2e-2
    assert np.abs(upsilon_avg_sim / upsilon_avg - 1) < 5e-2


@for_all_test_contexts(excluding='ContextPyopencl')
def test_beambeam3d_collective_beamstrahlung_single_collision(test_context):
    ###########
    # ttbar 2 #
    ###########
    bunch_intensity     = 2.3e11  # [1]
    energy              = 182.5  # [GeV]
    p0c                 = 182.5e9  # [eV]
    mass0               = .511e6  # [eV]
    phi                 = 15e-3  # [rad] half xing
    u_sr                = 9.2  # [GeV]
    u_bs                = .0114  # [GeV]
    k2_factor           = .4  # [1]
    qx                  = .554  # [1] half arc
    qy                  = .588  # [1]
    qs                  = .0436  # [1]
    physemit_x          = 1.46e-09  # [m]
    physemit_y          = 2.9e-12  # [m]
    beta_x              = 1  # [m]
    beta_y              = .0016  # [m]
    sigma_x             = np.sqrt(physemit_x*beta_x)  # [m]
    sigma_px            = np.sqrt(physemit_x/beta_x)  # [m]
    sigma_y             = np.sqrt(physemit_y*beta_y)  # [m]
    sigma_py            = np.sqrt(physemit_y/beta_y)  # [m]
    sigma_z             = .00194  # [m] sr
    sigma_z_tot         = .00254  # [m] sr+bs
    sigma_delta         = .0015  # [m]
    sigma_delta_tot     = .00192  # [m]
    beta_s              = sigma_z/sigma_delta  # [m]
    physemit_s          = sigma_z*sigma_delta  # [m]
    physemit_s_tot      = sigma_z_tot*sigma_delta_tot  # [m]
    n_macroparticles_b1 = int(1e6)
    n_macroparticles_b2 = int(1e6)

    n_slices = 100

    #############
    # particles #
    #############

    #e-
    particles_b1 = xp.Particles(
                _context = test_context,
                q0        = -1,
                p0c       = p0c,
                mass0     = mass0,
                x         = sigma_x        *np.random.randn(n_macroparticles_b1),
                y         = sigma_y        *np.random.randn(n_macroparticles_b1),
                zeta      = sigma_z_tot    *np.random.randn(n_macroparticles_b1),
                px        = sigma_px       *np.random.randn(n_macroparticles_b1),
                py        = sigma_py       *np.random.randn(n_macroparticles_b1),
                delta     = sigma_delta_tot*np.random.randn(n_macroparticles_b1),
                )

    particles_b1.name = "b1"

    slicer = xf.TempSlicer(n_slices=n_slices, sigma_z=sigma_z_tot, mode="unicharge")

    # this is different w.r.t WS test
    config_for_update=xf.ConfigForUpdateBeamBeamBiGaussian3D(
                    pipeline_manager=None,
                    element_name="beambeam",
                    slicer=slicer,
                    update_every=None, # Never updates (test in weakstrong mode)
                    )

    el_beambeam_b1 = xf.BeamBeamBiGaussian3D(
    _context=test_context,
    config_for_update = config_for_update,
    other_beam_q0=1,
    phi=phi,
    alpha=0,
    # decide between round or elliptical kick formula
    min_sigma_diff     = 1e-28,
    # slice intensity [num. real particles] n_slices inferred from length of this
    slices_other_beam_num_particles = slicer.bin_weights * bunch_intensity,
    # unboosted strong beam moments
    slices_other_beam_zeta_center = slicer.bin_centers,
    slices_other_beam_Sigma_11    = n_slices*[sigma_x**2],
    slices_other_beam_Sigma_22    = n_slices*[sigma_px**2],
    slices_other_beam_Sigma_33    = n_slices*[sigma_y**2],
    slices_other_beam_Sigma_44    = n_slices*[sigma_py**2],
    # only if BS on
    slices_other_beam_zeta_bin_width_star_beamstrahlung = slicer.bin_widths_beamstrahlung / np.cos(phi),  # boosted dz
    # has to be set
    slices_other_beam_Sigma_12    = n_slices*[0],
    slices_other_beam_Sigma_34    = n_slices*[0],
    )

    el_beambeam_b1.name = "beambeam"

    #########################
    # track for 1 collision #
    #########################

    line = xt.Line(elements = [el_beambeam_b1])
    line.build_tracker(_context=test_context)

    assert line._needs_rng == False

    line.configure_radiation(model_beamstrahlung='quantum')
    assert line._needs_rng == True

    record = line.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D, capacity={"beamstrahlungtable": int(3e5)})
    line.track(particles_b1, num_turns=1)
    line.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

    record.move(_context=xo.context_default)

    ###########################################
    # test 1: compare spectrum with guineapig #
    ###########################################

    fname = test_data_folder / "beamstrahlung/guineapig_ttbar2_beamstrahlung_photon_energies_gev.txt"
    guinea_photons = np.loadtxt(fname)  # contains about 250k photons emitted from 1e6 macroparticles in 1 collision
    n_bins = 10
    bins = np.logspace(np.log10(1e-14), np.log10(1e1), n_bins)
    xsuite_hist = np.histogram(record.beamstrahlungtable.photon_energy/1e9, bins=bins)[0]
    guinea_hist = np.histogram(guinea_photons, bins=bins)[0]

    bin_rel_errors = np.abs(xsuite_hist - guinea_hist) / guinea_hist
    print(f"bin relative errors [1]: {bin_rel_errors}")

    # test if relative error in the last 5 bins is smaller than 1e-1
    assert np.allclose(xsuite_hist[-5:], guinea_hist[-5:], rtol=1e-1, atol=0)

    ############################################
    # test 2: compare beamstrahlung parameters #
    ############################################

    # average and maximum BS parameter
    # https://www.researchgate.net/publication/2278298_Beam-Beam_Phenomena_In_Linear_Colliders
    # page 20
    r0 = cst.e**2/(4*np.pi*cst.epsilon_0*cst.m_e*cst.c**2) # - if pp

    upsilon_max =   2 * r0**2 * energy/(mass0*1e-9) * bunch_intensity / (1/137*sigma_z_tot*(sigma_x + 1.85*sigma_y))
    upsilon_avg = 5/6 * r0**2 * energy/(mass0*1e-9) * bunch_intensity / (1/137*sigma_z_tot*(sigma_x + sigma_y))

    # get rid of padded zeros in table
    photon_critical_energy = np.array(sorted(set(record.beamstrahlungtable.photon_critical_energy))[1:])
    primary_energy         = np.array(sorted(set(        record.beamstrahlungtable.primary_energy))[1:])

    upsilon_avg_sim = np.mean(0.67 * photon_critical_energy / primary_energy)
    upsilon_max_sim = np.max(0.67 * photon_critical_energy / primary_energy)

    print("Y max. [1]:", upsilon_max)
    print("Y avg. [1]:", upsilon_avg)
    print("Y max. [1]:", upsilon_max_sim)
    print("Y avg. [1]:", upsilon_avg_sim)
    print("Y max. ratio [1]:", upsilon_max_sim / upsilon_max)
    print("Y avg. ratio [1]:", upsilon_avg_sim / upsilon_avg)

    assert np.abs(upsilon_max_sim / upsilon_max - 1) < 2e-2
    assert np.abs(upsilon_avg_sim / upsilon_avg - 1) < 5e-2
