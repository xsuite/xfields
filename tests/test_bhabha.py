import pathlib
import pytest

import numpy as np
from scipy import constants as cst

import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp

from xobjects.test_helpers import for_all_test_contexts

bins = np.logspace(np.log10(1e-4), np.log10(2e2), 20)
guinea_hist = np.array([2693, 3137, 2912, 2482, 2156])  # from GP legacy ttbar2 100 slices
cx = 1e-2

@for_all_test_contexts
def test_beambeam3d_bhabha_ws_no_config(test_context):

    if isinstance(test_context, xo.ContextCupy):
        import cupy as cp

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
    n_macroparticles_b1 = int(1e4)

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
                weight=bunch_intensity/n_macroparticles_b1,
                )

    particles_b1.name = "b1"

    slicer = xf.TempSlicer(n_slices=n_slices, sigma_z=sigma_z_tot, mode="unicharge")

    el_beambeam_b1 = xf.BeamBeamBiGaussian3D(
            _context=test_context,
            config_for_update = None,  # need to set sig_12 and sig_34
            other_beam_q0=1,
            phi=phi,
            alpha=0,
            # decide between round or elliptical kick formula
            min_sigma_diff     = 1e-28,
            # slice intensity [num. real particles] n_slices inferred from length of this
            slices_other_beam_num_particles      = slicer.bin_weights * bunch_intensity,
            # unboosted strong beam moments
            slices_other_beam_zeta_center = slicer.bin_centers,
            slices_other_beam_Sigma_11    = n_slices*[sigma_x**2],
            slices_other_beam_Sigma_22    = n_slices*[sigma_px**2],
            slices_other_beam_Sigma_33    = n_slices*[sigma_y**2],
            slices_other_beam_Sigma_44    = n_slices*[sigma_py**2],
            # has to be set
            slices_other_beam_Sigma_12    = n_slices*[0],
            slices_other_beam_Sigma_34    = n_slices*[0],
            compt_x_min=cx,
            flag_beamsize_effect=1,
    )

    #########################
    # track for 1 collision #
    #########################

    line = xt.Line(elements = [el_beambeam_b1])
    line.build_tracker(_context=test_context)

    assert line._needs_rng == False

    line.configure_radiation(model_bhabha='quantum')
    assert line._needs_rng == True

    record_ws_b1 = line.start_internal_logging_for_elements_of_type(
        xf.BeamBeamBiGaussian3D,
        capacity={
            "beamstrahlungtable": int(0),
            "bhabhatable": int(3e4),
            "lumitable": int(0)})
    line.track(particles_b1, num_turns=1)
    line.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

    record_ws_b1.move(_context=xo.context_default)

    ###########################################
    # test 1: compare spectrum with guineapig #
    ###########################################

    xsuite_ws_b1_hist  = np.histogram(
        record_ws_b1.bhabhatable.photon_energy/1e9, bins=bins)[0][-5:]
    ws_b1_bin_rel_errors = np.abs(xsuite_ws_b1_hist - guinea_hist) / (guinea_hist)
    print(f"WS beam 1 bin relative errors [1]: {ws_b1_bin_rel_errors}")

    # test if relative error in the last 5 bins is smaller than 2e-1
    assert np.allclose(xsuite_ws_b1_hist[-5:], guinea_hist, rtol=2e-1, atol=0)

@for_all_test_contexts
def test_beambeam3d_bhabha_ws_config(test_context):

    if isinstance(test_context, xo.ContextPyopencl):
        pytest.skip("Not implemented for OpenCL")
        return

    if isinstance(test_context, xo.ContextCupy):
        pytest.skip("Not implemented for cupy")
        return

    if isinstance(test_context, xo.ContextCupy):
        import cupy as cp

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
    n_macroparticles_b1 = int(1e4)

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
                weight=bunch_intensity/n_macroparticles_b1,
                )

    particles_b1.name = "b1"

    slicer = xf.TempSlicer(_context=test_context, n_slices=n_slices, sigma_z=sigma_z_tot, mode="unicharge")

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
            slices_other_beam_num_particles      = slicer.bin_weights * bunch_intensity,
            # unboosted strong beam moments
            slices_other_beam_zeta_center = slicer.bin_centers,
            slices_other_beam_Sigma_11    = n_slices*[sigma_x**2],
            slices_other_beam_Sigma_22    = n_slices*[sigma_px**2],
            slices_other_beam_Sigma_33    = n_slices*[sigma_y**2],
            slices_other_beam_Sigma_44    = n_slices*[sigma_py**2],
            compt_x_min=cx,
            flag_beamsize_effect=1,
    )

    el_beambeam_b1.name = "beambeam"

    #########################
    # track for 1 collision #
    #########################

    line = xt.Line(elements = [el_beambeam_b1])
    line.build_tracker(_context=test_context)

    assert line._needs_rng == False

    line.configure_radiation(model_bhabha='quantum')
    assert line._needs_rng == True

    record_ws_b1 = line.start_internal_logging_for_elements_of_type(
        xf.BeamBeamBiGaussian3D, capacity={"beamstrahlungtable": int(0), "bhabhatable": int(3e4), "lumitable": int(0)})
    line.track(particles_b1, num_turns=1)
    line.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

    record_ws_b1.move(_context=xo.context_default)

    ###########################################
    # test 1: compare spectrum with guineapig #
    ###########################################

    xsuite_ws_b1_hist  = np.histogram( record_ws_b1.bhabhatable.photon_energy/1e9, bins=bins)[0][-5:]
    ws_b1_bin_rel_errors = np.abs(xsuite_ws_b1_hist - guinea_hist) / (guinea_hist)
    print(f"WS beam 1 bin relative errors [1]: {ws_b1_bin_rel_errors}")

    # test if relative error in the last 5 bins is smaller than 2e-1
    assert np.allclose(xsuite_ws_b1_hist[-5:], guinea_hist, rtol=2e-1, atol=0)

@for_all_test_contexts
def test_beambeam3d_bhabha_qss(test_context):

    if isinstance(test_context, xo.ContextPyopencl):
        pytest.skip("Not implemented for OpenCL")
        return

    if isinstance(test_context, xo.ContextCupy):
        pytest.skip("Not implemented for cupy")
        return

    if isinstance(test_context, xo.ContextCupy):
        import cupy as cp

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
    n_macroparticles_b1 = int(1e4)
    n_macroparticles_b2 = int(1e4)

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
        quasistrongstrong=True,
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
                compt_x_min=cx,
                flag_beamsize_effect=1,
    )
    bbeamIP1_b2 = xf.BeamBeamBiGaussian3D(
                _context=test_context,
                other_beam_q0 = particles_b1.q0,
                phi = -phi, # boost beam 1 slices by -phi
                alpha=0,
                config_for_update = config_for_update_b2_IP1,
                compt_x_min=cx,
                flag_beamsize_effect=1,
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

    line_b1.configure_radiation(model_bhabha='quantum')
    line_b2.configure_radiation(model_bhabha='quantum')

    assert line_b1._needs_rng == True
    assert line_b2._needs_rng == True

    particles_b1.name = "b1"
    particles_b2.name = "b2"

    branch_b1 = xt.PipelineBranch(line_b1, particles_b1)
    branch_b2 = xt.PipelineBranch(line_b2, particles_b2)
    multitracker = xt.PipelineMultiTracker(branches=[branch_b1,branch_b2])

    record_qss_b1 = line_b1.start_internal_logging_for_elements_of_type(
        xf.BeamBeamBiGaussian3D, capacity={"beamstrahlungtable": int(0), "bhabhatable": int(3e4), "lumitable": int(0)})
    record_qss_b2 = line_b2.start_internal_logging_for_elements_of_type(
        xf.BeamBeamBiGaussian3D, capacity={"beamstrahlungtable": int(0), "bhabhatable": int(3e4), "lumitable": int(0)})

    multitracker.track(num_turns=1)
    line_b1.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)
    line_b2.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

    record_qss_b1.move(_context=xo.context_default)
    record_qss_b2.move(_context=xo.context_default)

    ###########################################
    # test 1: compare spectrum with guineapig #
    ###########################################

    xsuite_qss_b1_hist = np.histogram(record_qss_b1.bhabhatable.photon_energy/1e9, bins=bins)[0][-5:]
    xsuite_qss_b2_hist = np.histogram(record_qss_b2.bhabhatable.photon_energy/1e9, bins=bins)[0][-5:]
    qss_b1_bin_rel_errors = np.abs(xsuite_qss_b1_hist - guinea_hist) / (guinea_hist)
    qss_b2_bin_rel_errors = np.abs(xsuite_qss_b2_hist - guinea_hist) / (guinea_hist)
    print(f"QSS beam 1 bin relative errors [1]: {qss_b1_bin_rel_errors}")
    print(f"QSS beam 2 bin relative errors [1]: {qss_b2_bin_rel_errors}")

    # test if relative error in the last 5 bins is smaller than 2e-1
    assert np.allclose(xsuite_qss_b1_hist[-5:], guinea_hist, rtol=2e-1, atol=0)
    assert np.allclose(xsuite_qss_b2_hist[-5:], guinea_hist, rtol=2e-1, atol=0)

@for_all_test_contexts
def test_beambeam3d_bhabha_ss(test_context):

    if isinstance(test_context, xo.ContextPyopencl):
        pytest.skip("Not implemented for OpenCL")
        return

    if isinstance(test_context, xo.ContextCupy):
        pytest.skip("Not implemented for cupy")
        return

    if isinstance(test_context, xo.ContextCupy):
        import cupy as cp

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
    n_macroparticles_b1 = int(1e4)
    n_macroparticles_b2 = int(1e4)

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
        quasistrongstrong=False,
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
                compt_x_min=cx,
                flag_beamsize_effect=1,
    )
    bbeamIP1_b2 = xf.BeamBeamBiGaussian3D(
                _context=test_context,
                other_beam_q0 = particles_b1.q0,
                phi = -phi, # boost beam 1 slices by -phi
                alpha=0,
                config_for_update = config_for_update_b2_IP1,
                compt_x_min=cx,
                flag_beamsize_effect=1,
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

    line_b1.configure_radiation(model_bhabha='quantum')
    line_b2.configure_radiation(model_bhabha='quantum')

    assert line_b1._needs_rng == True
    assert line_b2._needs_rng == True

    particles_b1.name = "b1"
    particles_b2.name = "b2"

    branch_b1 = xt.PipelineBranch(line_b1, particles_b1)
    branch_b2 = xt.PipelineBranch(line_b2, particles_b2)
    multitracker = xt.PipelineMultiTracker(branches=[branch_b1,branch_b2])

    record_ss_b1 = line_b1.start_internal_logging_for_elements_of_type(
        xf.BeamBeamBiGaussian3D, capacity={"beamstrahlungtable": int(0), "bhabhatable": int(3e4), "lumitable": int(0)})
    record_ss_b2 = line_b2.start_internal_logging_for_elements_of_type(
        xf.BeamBeamBiGaussian3D, capacity={"beamstrahlungtable": int(0), "bhabhatable": int(3e4), "lumitable": int(0)})

    multitracker.track(num_turns=1)
    line_b1.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)
    line_b2.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

    record_ss_b1.move(_context=xo.context_default)
    record_ss_b2.move(_context=xo.context_default)

    ###########################################
    # test 1: compare spectrum with guineapig #
    ###########################################

    xsuite_ss_b1_hist = np.histogram(record_ss_b1.bhabhatable.photon_energy/1e9, bins=bins)[0][-5:]
    xsuite_ss_b2_hist = np.histogram(record_ss_b2.bhabhatable.photon_energy/1e9, bins=bins)[0][-5:]
    ss_b1_bin_rel_errors = np.abs(xsuite_ss_b1_hist - guinea_hist) / (guinea_hist)
    ss_b2_bin_rel_errors = np.abs(xsuite_ss_b2_hist - guinea_hist) / (guinea_hist)
    print(f"SS beam 1 bin relative errors [1]: {ss_b1_bin_rel_errors}")
    print(f"SS beam 2 bin relative errors [1]: {ss_b2_bin_rel_errors}")

    # test if relative error in the last 5 bins is smaller than 2e-1
    assert np.allclose(xsuite_ss_b1_hist[-5:], guinea_hist, rtol=2e-1, atol=0)
    assert np.allclose(xsuite_ss_b2_hist[-5:], guinea_hist, rtol=2e-1, atol=0)

