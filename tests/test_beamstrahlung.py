import pathlib
import pytest

import numpy as np
from scipy import constants as cst

import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp

from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts
def test_beambeam3d_beamstrahlung_ws_no_config(test_context):

    if isinstance(test_context, xo.ContextCupy):
        print(f"[test.py] default_blocksize: {test_context.default_block_size}")

    print(repr(test_context))

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
        xf.BeamBeamBiGaussian3D, capacity={"beamstrahlungtable": int(3e5), "bhabhatable": int(0), "lumitable": int(0)})
    line.track(particles_b1, num_turns=1)
    line.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

    record.move(_context=xo.context_default)

    ###################################################
    # Change to average beamstrahlung and track again #
    ###################################################
    el_beambeam_b1.slices_other_beam_sqrtSigma_11_beamstrahlung = (
        test_context.nparray_to_context_array(np.array(n_slices*[sigma_x])))
    el_beambeam_b1.slices_other_beam_sqrtSigma_33_beamstrahlung = (
        test_context.nparray_to_context_array(np.array(n_slices*[sigma_y])))
    el_beambeam_b1.slices_other_beam_sqrtSigma_55_beamstrahlung = (
        test_context.nparray_to_context_array(slicer.bin_weights * sigma_z_tot))

    line.configure_radiation(model_beamstrahlung='mean')
    record_avg = line.start_internal_logging_for_elements_of_type(
        xf.BeamBeamBiGaussian3D, capacity={"beamstrahlungtable": int(3e5), "bhabhatable": int(0), "lumitable": int(0)})
    line.track(particles_b1, num_turns=1)
    line.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

    record_avg.move(_context=xo.context_default)

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

    assert np.abs(upsilon_max_sim / upsilon_max - 1) < 5e-2
    assert np.abs(upsilon_avg_sim / upsilon_avg - 1) < 1e-1

    ############################################################################
    # test 3: compare beamstrahlung average photon energy with the two methods #
    ############################################################################

    photon_energy_mean = np.mean(sorted(set(record.beamstrahlungtable.photon_energy))[1:])
    assert np.all(np.abs(record_avg.beamstrahlungtable.photon_energy - photon_energy_mean) / photon_energy_mean < 1e-1)

@for_all_test_contexts
def test_beambeam3d_beamstrahlung_ws_config(test_context):

    if isinstance(test_context, xo.ContextPyopencl):
        pytest.skip("Not implemented for OpenCL")
        return

    if isinstance(test_context, xo.ContextCupy):
        pytest.skip("Not implemented for cupy")
        return

    if isinstance(test_context, xo.ContextCupy):
        print(f"[test.py] default_blocksize: {test_context.default_block_size}")

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

    slicer = xf.TempSlicer(_context=test_context, n_slices=n_slices,
                           sigma_z=sigma_z_tot, mode="unicharge")

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

    record = line.start_internal_logging_for_elements_of_type(
        xf.BeamBeamBiGaussian3D, capacity={"beamstrahlungtable": int(3e5), "bhabhatable": int(0), "lumitable": int(0)})
    line.track(particles_b1, num_turns=1)
    line.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

    record.move(_context=xo.context_default)

    ###################################################
    # Change to average beamstrahlung and track again #
    ###################################################

    el_beambeam_b1.slices_other_beam_sqrtSigma_11_beamstrahlung = test_context.nplike_lib.array(n_slices*[sigma_x]) 
    el_beambeam_b1.slices_other_beam_sqrtSigma_33_beamstrahlung = test_context.nplike_lib.array(n_slices*[sigma_y]) 
    el_beambeam_b1.slices_other_beam_sqrtSigma_55_beamstrahlung = test_context.nplike_lib.array(slicer.bin_weights) * sigma_z_tot

    line.configure_radiation(model_beamstrahlung='mean')
    record_avg = line.start_internal_logging_for_elements_of_type(
        xf.BeamBeamBiGaussian3D, capacity={"beamstrahlungtable": int(3e5), "bhabhatable": int(0), "lumitable": int(0)})
    line.track(particles_b1, num_turns=1)
    line.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

    record_avg.move(_context=xo.context_default)

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

    assert np.abs(upsilon_max_sim / upsilon_max - 1) < 5e-2
    assert np.abs(upsilon_avg_sim / upsilon_avg - 1) < 1e-1

    ############################################################################
    # test 3: compare beamstrahlung average photon energy with the two methods #
    ############################################################################

    photon_energy_mean = np.mean(sorted(set(record.beamstrahlungtable.photon_energy))[1:])
    assert np.all(np.abs(record_avg.beamstrahlungtable.photon_energy - photon_energy_mean) / photon_energy_mean < 1e-1)

@for_all_test_contexts
def test_beambeam3d_beamstrahlung_qss(test_context):

    if isinstance(test_context, xo.ContextPyopencl):
        pytest.skip("Not implemented for OpenCL")
        return

    if isinstance(test_context, xo.ContextCupy):
        pytest.skip("Not implemented for cupy")
        return

    if isinstance(test_context, xo.ContextCupy):
        print(f"[test.py] default_blocksize: {test_context.default_block_size}")

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
    n_macroparticles_b2 = int(1e6)

    n_slices = 100
    if isinstance(test_context, xo.ContextCupy):
        test_context.default_shared_mem_size_bytes=n_slices*17*8

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
                weight=bunch_intensity/n_macroparticles_b1,
                )

    # e+
    particles_b2 = xp.Particles(
                _context = test_context,
                q0        = 1,
                p0c       = p0c,
                mass0     = mass0,
                x         = sigma_x        *np.random.randn(n_macroparticles_b2),
                y         = sigma_y        *np.random.randn(n_macroparticles_b2),
                zeta      = sigma_z_tot    *np.random.randn(n_macroparticles_b2),
                px        = sigma_px       *np.random.randn(n_macroparticles_b2),
                py        = sigma_py       *np.random.randn(n_macroparticles_b2),
                delta     = sigma_delta_tot*np.random.randn(n_macroparticles_b2),
                weight=bunch_intensity/n_macroparticles_b2,
                )

    particles_b1.name = "b1"
    particles_b2.name = "b2"

    particles_b1.init_pipeline('b1')
    particles_b2.init_pipeline('b2')

    slicer = xf.TempSlicer(_context=test_context, n_slices=n_slices, sigma_z=sigma_z_tot, mode="unicharge")

    pipeline_manager = xt.PipelineManager()
    pipeline_manager.add_particles('b1',0)
    pipeline_manager.add_particles('b2',0)
    pipeline_manager.add_element('IP1')
    pipeline_manager.add_element('IP2')

    config_for_update_b1_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
        pipeline_manager=pipeline_manager,
        element_name='IP1',
        partner_particles_name = 'b2',
        slicer=slicer,
        update_every=1,
        quasistrongstrong=True
        )

    config_for_update_b2_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
        pipeline_manager=pipeline_manager,
        element_name='IP1',
        partner_particles_name = 'b1',
        slicer=slicer,
        update_every=1,
        quasistrongstrong=True
        )

    bbeamIP1_b1 = xf.BeamBeamBiGaussian3D(
                _context=test_context,
                other_beam_q0 = particles_b2.q0,
                phi = phi, 
                alpha=0,
                config_for_update = config_for_update_b1_IP1,
    )
    bbeamIP1_b2 = xf.BeamBeamBiGaussian3D(
                _context=test_context,
                other_beam_q0 = particles_b1.q0,
                phi = -phi, # boost beam 1 slices by -phi
                alpha=0,
                config_for_update = config_for_update_b2_IP1,
    )

    #########################
    # track for 1 collision #
    #########################

    line_b1 = xt.Line(elements = [bbeamIP1_b1,])
    line_b2 = xt.Line(elements = [bbeamIP1_b2,])

    line_b1.build_tracker(_context=test_context)
    line_b2.build_tracker(_context=test_context)

    assert line_b1._needs_rng == False
    assert line_b2._needs_rng == False

    line_b1.configure_radiation(model_beamstrahlung='quantum')
    line_b2.configure_radiation(model_beamstrahlung='quantum')

    assert line_b1._needs_rng == True
    assert line_b2._needs_rng == True

    particles_b1.name = "b1"
    particles_b2.name = "b2"

    branch_b1 = xt.PipelineBranch(line_b1, particles_b1)
    branch_b2 = xt.PipelineBranch(line_b2, particles_b2)
    multitracker = xt.PipelineMultiTracker(branches=[branch_b1,branch_b2])

    record_ss_b1 = line_b1.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D, 
                                                                capacity={
                                                                    "beamstrahlungtable": int(3e5),
                                                                    "bhabhatable": int(0),
                                                                    "lumitable": int(0),
                                                                })
    record_ss_b2 = line_b2.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D, 
                                                                capacity={
                                                                    "beamstrahlungtable": int(3e5),
                                                                    "bhabhatable": int(0),
                                                                    "lumitable": int(0),
                                                                })

    multitracker.track(num_turns=1)
    line_b1.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)
    line_b2.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

    record_ss_b1.move(_context=xo.context_default)
    record_ss_b2.move(_context=xo.context_default)

    ###################################################
    # Change to average beamstrahlung and track again #
    ###################################################

    bbeamIP1_b1.slices_other_beam_sqrtSigma_11_beamstrahlung = test_context.nplike_lib.array(n_slices*[sigma_x])
    bbeamIP1_b1.slices_other_beam_sqrtSigma_33_beamstrahlung = test_context.nplike_lib.array(n_slices*[sigma_y])
    bbeamIP1_b1.slices_other_beam_sqrtSigma_55_beamstrahlung = test_context.nplike_lib.array(slicer.bin_weights) * sigma_z_tot

    bbeamIP1_b2.slices_other_beam_sqrtSigma_11_beamstrahlung = test_context.nplike_lib.array(n_slices*[sigma_x])
    bbeamIP1_b2.slices_other_beam_sqrtSigma_33_beamstrahlung = test_context.nplike_lib.array(n_slices*[sigma_y]) 
    bbeamIP1_b2.slices_other_beam_sqrtSigma_55_beamstrahlung = test_context.nplike_lib.array(slicer.bin_weights) * sigma_z_tot 

    line_b1.configure_radiation(model_beamstrahlung='mean')
    line_b2.configure_radiation(model_beamstrahlung='mean')

    record_avg_b1 = line_b1.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D, capacity={"beamstrahlungtable": int(3e5), "bhabhatable": int(0), "lumitable": int(0)})
    record_avg_b2 = line_b2.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D, capacity={"beamstrahlungtable": int(3e5), "bhabhatable": int(0), "lumitable": int(0)})

    multitracker.track(num_turns=1)

    line_b1.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)
    line_b2.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

    record_avg_b1.move(_context=xo.context_default)
    record_avg_b2.move(_context=xo.context_default)

    ###########################################
    # test 1: compare spectrum with guineapig #
    ###########################################

    fname = test_data_folder / "beamstrahlung/guineapig_ttbar2_beamstrahlung_photon_energies_gev.txt"
    guinea_photons = np.loadtxt(fname)  # contains about 250k photons emitted from 1e6 macroparticles in 1 collision
    n_bins = 10
    bins = np.logspace(np.log10(1e-14), np.log10(1e1), n_bins)
    guinea_hist = np.histogram(guinea_photons, bins=bins)[0]

    xsuite_ss_b1_hist = np.histogram(record_ss_b1.beamstrahlungtable.photon_energy/1e9, bins=bins)[0]
    xsuite_ss_b2_hist = np.histogram(record_ss_b2.beamstrahlungtable.photon_energy/1e9, bins=bins)[0]

    ss_b1_bin_rel_errors = np.abs(xsuite_ss_b1_hist - guinea_hist) / guinea_hist
    ss_b2_bin_rel_errors = np.abs(xsuite_ss_b2_hist - guinea_hist) / guinea_hist
    print(f"QSS beam 1 bin relative errors [1]: {ss_b1_bin_rel_errors}")
    print(f"QSS beam 2 bin relative errors [1]: {ss_b2_bin_rel_errors}")

    # test if relative error in the last 5 bins is smaller than 1e-1
    assert np.allclose(xsuite_ss_b2_hist[-5:], guinea_hist[-5:], rtol=1e-1, atol=0)
    assert np.allclose(xsuite_ss_b2_hist[-5:], guinea_hist[-5:], rtol=1e-1, atol=0)

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
    ss_b1_photon_critical_energy = np.array(sorted(set(record_ss_b1.beamstrahlungtable.photon_critical_energy))[1:])
    ss_b2_photon_critical_energy = np.array(sorted(set(record_ss_b2.beamstrahlungtable.photon_critical_energy))[1:])

    ss_b1_primary_energy         = np.array(sorted(set(        record_ss_b1.beamstrahlungtable.primary_energy))[1:])
    ss_b2_primary_energy         = np.array(sorted(set(        record_ss_b2.beamstrahlungtable.primary_energy))[1:])

    ss_b1_upsilon_avg_sim = np.mean(0.67 * ss_b1_photon_critical_energy / ss_b1_primary_energy)
    ss_b2_upsilon_avg_sim = np.mean(0.67 * ss_b2_photon_critical_energy / ss_b2_primary_energy)

    ss_b1_upsilon_max_sim = np.max(0.67 * ss_b1_photon_critical_energy / ss_b1_primary_energy)
    ss_b2_upsilon_max_sim = np.max(0.67 * ss_b2_photon_critical_energy / ss_b2_primary_energy)

    print("Y max. [1]:", upsilon_max)
    print("Y avg. [1]:", upsilon_avg)

    print("QSS beam 1 Y max. [1]:", ss_b1_upsilon_max_sim)
    print("QSS beam 1 Y avg. [1]:", ss_b1_upsilon_avg_sim)
    print("QSS beam 1 Y max. ratio [1]:", ss_b1_upsilon_max_sim / upsilon_max)
    print("QSS beam 1 Y avg. ratio [1]:", ss_b1_upsilon_avg_sim / upsilon_avg)

    print("QSS beam 2 Y max. [1]:", ss_b2_upsilon_max_sim)
    print("QSS beam 2 Y avg. [1]:", ss_b2_upsilon_avg_sim)
    print("QSS beam 2 Y max. ratio [1]:", ss_b2_upsilon_max_sim / upsilon_max)
    print("QSS beam 2 Y avg. ratio [1]:", ss_b2_upsilon_avg_sim / upsilon_avg)

    assert np.abs(ss_b1_upsilon_max_sim / upsilon_max - 1) < 5e-2
    assert np.abs(ss_b1_upsilon_avg_sim / upsilon_avg - 1) < 1e-1

    assert np.abs(ss_b2_upsilon_max_sim / upsilon_max - 1) < 5e-2
    assert np.abs(ss_b2_upsilon_avg_sim / upsilon_avg - 1) < 1e-1

    ############################################################################
    # test 3: compare beamstrahlung average photon energy with the two methods #
    ############################################################################

    ss_b1_photon_energy_mean = np.mean(sorted(set(record_ss_b1.beamstrahlungtable.photon_energy))[1:])
    ss_b2_photon_energy_mean = np.mean(sorted(set(record_ss_b2.beamstrahlungtable.photon_energy))[1:])

    assert np.all(np.abs(record_avg_b1.beamstrahlungtable.photon_energy - ss_b1_photon_energy_mean) / ss_b1_photon_energy_mean < 1e-1)
    assert np.all(np.abs(record_avg_b2.beamstrahlungtable.photon_energy - ss_b2_photon_energy_mean) / ss_b2_photon_energy_mean < 1e-1)

@for_all_test_contexts
def test_beambeam3d_beamstrahlung_ss(test_context):

    if isinstance(test_context, xo.ContextPyopencl):
        pytest.skip("Not implemented for OpenCL")
        return

    if isinstance(test_context, xo.ContextCupy):
        pytest.skip("Not implemented for cupy")
        return

    if isinstance(test_context, xo.ContextCupy):
        print(f"[test.py] default_blocksize: {test_context.default_block_size}")

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
    n_macroparticles_b2 = int(1e6)

    n_slices = 100
    if isinstance(test_context, xo.ContextCupy):
        test_context.default_shared_mem_size_bytes=n_slices*17*8


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
                weight=bunch_intensity/n_macroparticles_b1,
                )

    # e+
    particles_b2 = xp.Particles(
                _context = test_context,
                q0        = 1,
                p0c       = p0c,
                mass0     = mass0,
                x         = sigma_x        *np.random.randn(n_macroparticles_b2),
                y         = sigma_y        *np.random.randn(n_macroparticles_b2),
                zeta      = sigma_z_tot    *np.random.randn(n_macroparticles_b2),
                px        = sigma_px       *np.random.randn(n_macroparticles_b2),
                py        = sigma_py       *np.random.randn(n_macroparticles_b2),
                delta     = sigma_delta_tot*np.random.randn(n_macroparticles_b2),
                weight=bunch_intensity/n_macroparticles_b2,
                )

    particles_b1.name = "b1"
    particles_b2.name = "b2"

    particles_b1.init_pipeline('b1')
    particles_b2.init_pipeline('b2')

    slicer = xf.TempSlicer(_context=test_context, n_slices=n_slices, sigma_z=sigma_z_tot, mode="unicharge")

    pipeline_manager = xt.PipelineManager()
    pipeline_manager.add_particles('b1',0)
    pipeline_manager.add_particles('b2',0)
    pipeline_manager.add_element('IP1')
    pipeline_manager.add_element('IP2')

    config_for_update_b1_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
        pipeline_manager=pipeline_manager,
        element_name='IP1',
        partner_particles_name = 'b2',
        slicer=slicer,
        update_every=1,
        quasistrongstrong=False
        )

    config_for_update_b2_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
        pipeline_manager=pipeline_manager,
        element_name='IP1',
        partner_particles_name = 'b1',
        slicer=slicer,
        update_every=1,
        quasistrongstrong=False
        )

    bbeamIP1_b1 = xf.BeamBeamBiGaussian3D(
                _context=test_context,
                other_beam_q0 = particles_b2.q0,
                phi = phi,
                alpha=0,
                config_for_update = config_for_update_b1_IP1,
    )
    bbeamIP1_b2 = xf.BeamBeamBiGaussian3D(
                _context=test_context,
                other_beam_q0 = particles_b1.q0,
                phi = -phi, # boost beam 1 slices by -phi
                alpha=0,
                config_for_update = config_for_update_b2_IP1,
    )

    #########################
    # track for 1 collision #
    #########################

    line_b1 = xt.Line(elements = [bbeamIP1_b1,])
    line_b2 = xt.Line(elements = [bbeamIP1_b2,])

    line_b1.build_tracker(_context=test_context)
    line_b2.build_tracker(_context=test_context)

    assert line_b1._needs_rng == False
    assert line_b2._needs_rng == False

    line_b1.configure_radiation(model_beamstrahlung='quantum')
    line_b2.configure_radiation(model_beamstrahlung='quantum')

    assert line_b1._needs_rng == True
    assert line_b2._needs_rng == True

    particles_b1.name = "b1"
    particles_b2.name = "b2"

    branch_b1 = xt.PipelineBranch(line_b1, particles_b1)
    branch_b2 = xt.PipelineBranch(line_b2, particles_b2)
    multitracker = xt.PipelineMultiTracker(branches=[branch_b1,branch_b2])

    record_ss_b1 = line_b1.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D, 
                                                                capacity={
                                                                    "beamstrahlungtable": int(3e5),
                                                                    "bhabhatable": int(0),
                                                                    "lumitable": int(0),

                                                                })
    record_ss_b2 = line_b2.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D, 
                                                                capacity={
                                                                    "beamstrahlungtable": int(3e5),
                                                                    "bhabhatable": int(0),
                                                                    "lumitable": int(0),
                                                                })

    multitracker.track(num_turns=1)
    line_b1.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)
    line_b2.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

    record_ss_b1.move(_context=xo.context_default)
    record_ss_b2.move(_context=xo.context_default)

    ###################################################
    # Change to average beamstrahlung and track again #
    ###################################################

    bbeamIP1_b1.slices_other_beam_sqrtSigma_11_beamstrahlung = test_context.nplike_lib.array(n_slices*[sigma_x])
    bbeamIP1_b1.slices_other_beam_sqrtSigma_33_beamstrahlung = test_context.nplike_lib.array(n_slices*[sigma_y])
    bbeamIP1_b1.slices_other_beam_sqrtSigma_55_beamstrahlung = test_context.nplike_lib.array(slicer.bin_weights) * sigma_z_tot

    bbeamIP1_b2.slices_other_beam_sqrtSigma_11_beamstrahlung = test_context.nplike_lib.array(n_slices*[sigma_x])
    bbeamIP1_b2.slices_other_beam_sqrtSigma_33_beamstrahlung = test_context.nplike_lib.array(n_slices*[sigma_y])
    bbeamIP1_b2.slices_other_beam_sqrtSigma_55_beamstrahlung = test_context.nplike_lib.array(slicer.bin_weights) * sigma_z_tot

    line_b1.configure_radiation(model_beamstrahlung='mean')
    line_b2.configure_radiation(model_beamstrahlung='mean')

    record_avg_b1 = line_b1.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D, capacity={"beamstrahlungtable": int(3e5), "bhabhatable": int(0), "lumitable": int(0)})
    record_avg_b2 = line_b2.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D, capacity={"beamstrahlungtable": int(3e5), "bhabhatable": int(0), "lumitable": int(0)})

    multitracker.track(num_turns=1)

    line_b1.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)
    line_b2.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

    record_avg_b1.move(_context=xo.context_default)
    record_avg_b2.move(_context=xo.context_default)

    ###########################################
    # test 1: compare spectrum with guineapig #
    ###########################################

    fname = test_data_folder / "beamstrahlung/guineapig_ttbar2_beamstrahlung_photon_energies_gev.txt"
    guinea_photons = np.loadtxt(fname)  # contains about 250k photons emitted from 1e6 macroparticles in 1 collision
    n_bins = 10
    bins = np.logspace(np.log10(1e-14), np.log10(1e1), n_bins)
    guinea_hist = np.histogram(guinea_photons, bins=bins)[0]

    xsuite_ss_b1_hist = np.histogram(record_ss_b1.beamstrahlungtable.photon_energy/1e9, bins=bins)[0]
    xsuite_ss_b2_hist = np.histogram(record_ss_b2.beamstrahlungtable.photon_energy/1e9, bins=bins)[0]

    ss_b1_bin_rel_errors = np.abs(xsuite_ss_b1_hist - guinea_hist) / guinea_hist
    ss_b2_bin_rel_errors = np.abs(xsuite_ss_b2_hist - guinea_hist) / guinea_hist
    print(f"SS beam 1 bin relative errors [1]: {ss_b1_bin_rel_errors}")
    print(f"SS beam 2 bin relative errors [1]: {ss_b2_bin_rel_errors}")

    # test if relative error in the last 5 bins is smaller than 1e-1
    assert np.allclose(xsuite_ss_b2_hist[-5:], guinea_hist[-5:], rtol=1e-1, atol=0)
    assert np.allclose(xsuite_ss_b2_hist[-5:], guinea_hist[-5:], rtol=1e-1, atol=0)

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
    ss_b1_photon_critical_energy = np.array(sorted(set(record_ss_b1.beamstrahlungtable.photon_critical_energy))[1:])
    ss_b2_photon_critical_energy = np.array(sorted(set(record_ss_b2.beamstrahlungtable.photon_critical_energy))[1:])

    ss_b1_primary_energy         = np.array(sorted(set(        record_ss_b1.beamstrahlungtable.primary_energy))[1:])
    ss_b2_primary_energy         = np.array(sorted(set(        record_ss_b2.beamstrahlungtable.primary_energy))[1:])

    ss_b1_upsilon_avg_sim = np.mean(0.67 * ss_b1_photon_critical_energy / ss_b1_primary_energy)
    ss_b2_upsilon_avg_sim = np.mean(0.67 * ss_b2_photon_critical_energy / ss_b2_primary_energy)

    ss_b1_upsilon_max_sim = np.max(0.67 * ss_b1_photon_critical_energy / ss_b1_primary_energy)
    ss_b2_upsilon_max_sim = np.max(0.67 * ss_b2_photon_critical_energy / ss_b2_primary_energy)

    print("Y max. [1]:", upsilon_max)
    print("Y avg. [1]:", upsilon_avg)

    print("SS beam 1 Y max. [1]:", ss_b1_upsilon_max_sim)
    print("SS beam 1 Y avg. [1]:", ss_b1_upsilon_avg_sim)
    print("SS beam 1 Y max. ratio [1]:", ss_b1_upsilon_max_sim / upsilon_max)
    print("SS beam 1 Y avg. ratio [1]:", ss_b1_upsilon_avg_sim / upsilon_avg)

    print("SS beam 2 Y max. [1]:", ss_b2_upsilon_max_sim)
    print("SS beam 2 Y avg. [1]:", ss_b2_upsilon_avg_sim)
    print("SS beam 2 Y max. ratio [1]:", ss_b2_upsilon_max_sim / upsilon_max)
    print("SS beam 2 Y avg. ratio [1]:", ss_b2_upsilon_avg_sim / upsilon_avg)

    assert np.abs(ss_b1_upsilon_max_sim / upsilon_max - 1) < 5e-2
    assert np.abs(ss_b1_upsilon_avg_sim / upsilon_avg - 1) < 1e-1

    assert np.abs(ss_b2_upsilon_max_sim / upsilon_max - 1) < 5e-2
    assert np.abs(ss_b2_upsilon_avg_sim / upsilon_avg - 1) < 1e-1

    ############################################################################
    # test 3: compare beamstrahlung average photon energy with the two methods #
    ############################################################################

    ss_b1_photon_energy_mean = np.mean(sorted(set(record_ss_b1.beamstrahlungtable.photon_energy))[1:])
    ss_b2_photon_energy_mean = np.mean(sorted(set(record_ss_b2.beamstrahlungtable.photon_energy))[1:])

    assert np.all(np.abs(record_avg_b1.beamstrahlungtable.photon_energy - ss_b1_photon_energy_mean) / ss_b1_photon_energy_mean < 1e-1)
    assert np.all(np.abs(record_avg_b2.beamstrahlungtable.photon_energy - ss_b2_photon_energy_mean) / ss_b2_photon_energy_mean < 1e-1)
