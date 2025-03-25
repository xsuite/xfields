import pathlib
import pytest

import numpy as np

import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp

from xobjects.test_helpers import for_all_test_contexts, fix_random_seed


@for_all_test_contexts(excluding="ContextPyopencl")
@fix_random_seed(54329842)
def test_iptocp_strongstrong_px(test_context):

    if isinstance(test_context, xo.ContextCupy):
        import cupy as cp

    print(repr(test_context))
    
    ##############################
    # generic FCC-ee like params #
    ##############################
    
    mass0           = 0.511e6  # [eV]
    p0c             = 100e9  # [eV]
    bunch_intensity = 1e11  # [e]
    sigma_x         = 1e-6  # [m]
    sigma_y         = 1e-9  # [m]
    sigma_z         = 1e-3  # [m]
    alpha           = 0  # crossing plane: x-z [0 rad]
    n_macroparticles = int(1e4)
    
    # test different crossing angles
    phi_arr = np.array([1e-2, 1e-4, 0])  # [rad] half xing, px = dx/ds = tg(phi) ~ phi
    for phi in phi_arr:
        print(f"Checking for half xing={phi} [rad]")
        # so that IP=/=CP
        z1 = 0
        z2 = sigma_z
        px = phi
        
        #############
        # particles #
        #############
        
        #e-
        beam_1 = xp.Particles(
                    _context=test_context, 
                    q0        = -1,
                    p0c       = p0c,
                    mass0     = mass0,
                    x         = sigma_x        *np.random.randn(n_macroparticles),
                    y         = sigma_y        *np.random.randn(n_macroparticles), # if 0 infinite bbforce causes nan
                    zeta      = z1,
                    px        = px,
                    py        = 0,
                    delta     = 0,
                    weight=bunch_intensity/n_macroparticles
                    )
        
        # e+
        beam_2 = xp.Particles(
                    _context=test_context, 
                    q0        = 1,
                    p0c       = p0c,
                    mass0     = mass0,
                    x         = sigma_x        *np.random.randn(n_macroparticles),
                    y         = sigma_y        *np.random.randn(n_macroparticles),
                    zeta      = z2,
                    px        = -px, # beam 2 x axis points to opposite direction
                    py        = 0,
                    delta     = 0,
                    weight=bunch_intensity/n_macroparticles
                    )
        
        # setup 1: with crossing angle with px=0
        
        #e-
        beam_1_10 = beam_1.copy()
        beam_1_10.px = 0
        beam_1_1 = beam_1_10.copy()
        
        # e+
        beam_2_10 = beam_2.copy()
        beam_2_10.px = 0
        beam_2_1  = beam_2_10.copy()
        
        # setup 2: without crossing angle with px=phi
        
        #e-
        beam_1_20 = beam_1.copy()
        beam_1_2  = beam_1_20.copy()
        
        # e+
        beam_2_20 = beam_2.copy()
        beam_2_2  = beam_2_20.copy()
        
        beam_1_1.name = "b11"
        beam_2_1.name = "b21"   
        beam_1_2.name = "b12"
        beam_2_2.name = "b22"   
        
        beam_1_1.init_pipeline('b11')
        beam_2_1.init_pipeline('b21')
        beam_1_2.init_pipeline('b12')
        beam_2_2.init_pipeline('b22')
        
        # slicer with 1 slices, just to assign the slice id to all particles
        slicer = xf.TempSlicer(_context=test_context, n_slices=1, sigma_z=sigma_z, mode="unicharge")
        
        #######################################
        # track with crossing angle with px=0 #
        #######################################
        
        pipeline_manager = xt.PipelineManager()
        pipeline_manager.add_particles(beam_1_1.name,0)
        pipeline_manager.add_particles(beam_2_1.name,0)
        pipeline_manager.add_element('IP1')
        pipeline_manager.add_element('IP2')
        
        
        config_for_update_b1_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
           pipeline_manager=pipeline_manager,
           element_name='IP1',
           partner_particles_name=beam_2_1.name,
           slicer=slicer,
           update_every=1,
           quasistrongstrong=False,
        
           )
        
        config_for_update_b2_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
           pipeline_manager=pipeline_manager,
           element_name='IP1',
           partner_particles_name=beam_1_1.name,
           slicer=slicer,
           update_every=1,
           quasistrongstrong=False,
        
           )
        
        # beambeam elements
        bbeamIP1_b1 = xf.BeamBeamBiGaussian3D(
                    _context=test_context,
                    other_beam_q0 = beam_2_1.q0,
                    phi = phi, 
                    alpha=alpha,
                    config_for_update = config_for_update_b1_IP1,
        )
        bbeamIP1_b2 = xf.BeamBeamBiGaussian3D(
                    _context=test_context,
                    other_beam_q0 = beam_1_1.q0,
                    phi = -phi, # boost beam 2 slices by -phi
                    alpha=alpha, 
                    config_for_update = config_for_update_b2_IP1,
        )
        
        # init tracker
        line_b1 = xt.Line(elements = [bbeamIP1_b1,])
        line_b2 = xt.Line(elements = [bbeamIP1_b2,])
        
        line_b1.build_tracker(_context=test_context)
        line_b2.build_tracker(_context=test_context)
        
        branch_b1 = xt.PipelineBranch(line_b1, beam_1_1)
        branch_b2 = xt.PipelineBranch(line_b2, beam_2_1)
        multitracker = xt.PipelineMultiTracker(branches=[branch_b1,branch_b2])
        
        # track
        multitracker.track(num_turns=1)
        
        ############################################
        # track without crossing angle with px=phi #
        ############################################
        
        pipeline_manager = xt.PipelineManager()
        pipeline_manager.add_particles(beam_1_2.name,0)
        pipeline_manager.add_particles(beam_2_2.name,0)
        pipeline_manager.add_element('IP1')
        pipeline_manager.add_element('IP2')
        
        
        config_for_update_b1_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
           pipeline_manager=pipeline_manager,
           element_name='IP1',
           partner_particles_name=beam_2_2.name,
           slicer=slicer,
           update_every=1,
           quasistrongstrong=False,
        
           )
        
        config_for_update_b2_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
           pipeline_manager=pipeline_manager,
           element_name='IP1',
           partner_particles_name=beam_1_2.name,
           slicer=slicer,
           update_every=1,
           quasistrongstrong=False,
        
           )
        
        # beambeam elements
        bbeamIP1_b1 = xf.BeamBeamBiGaussian3D(
                    _context=test_context,
                    other_beam_q0 = beam_2_2.q0,
                    phi = 0, 
                    alpha=alpha,
                    config_for_update = config_for_update_b1_IP1,
        )
        bbeamIP1_b2 = xf.BeamBeamBiGaussian3D(
                    _context=test_context,
                    other_beam_q0 = beam_1_2.q0,
                    phi = 0, # boost beam 2 slices by -phi
                    alpha=alpha, 
                    config_for_update = config_for_update_b2_IP1,
        )
        
        # init tracker
        line_b1 = xt.Line(elements = [bbeamIP1_b1,])
        line_b2 = xt.Line(elements = [bbeamIP1_b2,])
        
        line_b1.build_tracker(_context=test_context)
        line_b2.build_tracker(_context=test_context)
        
        branch_b1 = xt.PipelineBranch(line_b1, beam_1_2)
        branch_b2 = xt.PipelineBranch(line_b2, beam_2_2)
        multitracker = xt.PipelineMultiTracker(branches=[branch_b1,branch_b2])
        
        # track
        multitracker.track(num_turns=1)
        
        ##########
        # checks #
        ##########
        
        # px from 2 setups after 1 kick should be similar (not exactly same due to h=/=0 at inv. boost)
        abserr_b1 = np.abs((beam_1_1.px + phi - beam_1_2.px))
        abserr_b2 = np.abs((beam_2_1.px - phi - beam_2_2.px))
        
        print(f"beam 1 max. error: { np.max(abserr_b1)}, beam 2 max. error: { np.max(abserr_b2)}")
        print(f"beam 1 avg. error: {np.mean(abserr_b1)}, beam 2 avg. error: {np.mean(abserr_b2)}")
        
        tol = max(phi*1e-5, 1e-10)  # when phi=0 1e-16 is better
        assert np.all(abserr_b1<=tol), "beam 1: error too large!"
        assert np.all(abserr_b2<=tol), "beam 2: error too large!"
        
        
@for_all_test_contexts(excluding="ContextPyopencl")
@fix_random_seed(672367)
def test_iptocp_strongstrong_py(test_context):

    if isinstance(test_context, xo.ContextCupy):
        import cupy as cp

    print(repr(test_context))
    
    ##############################
    # generic FCC-ee like params #
    ##############################
    
    mass0           = 0.511e6  # [eV]
    p0c             = 100e9  # [eV]
    bunch_intensity = 1e11  # [e]
    sigma_x         = 1e-6  # [m]
    sigma_y         = 1e-9  # [m]
    sigma_z         = 1e-3  # [m]
    alpha = np.pi/2  # crossing plane: y-z [pi/2 rad]
    n_macroparticles = int(1e4)
    
    # test different crossing angles
    phi_arr = np.array([1e-2, 1e-4, 0])  # [rad] half xing, py = dy/ds = tg(phi) ~ phi
    for phi in phi_arr:
        print(f"Checking for half xing={phi} [rad]")
        # so that IP=/=CP
        z1 = 0
        z2 = sigma_z
        py = phi
        
        #############
        # particles #
        #############
        
        #e-
        beam_1 = xp.Particles(
                    _context=test_context, 
                    q0        = -1,
                    p0c       = p0c,
                    mass0     = mass0,
                    x         = sigma_x        *np.random.randn(n_macroparticles),
                    y         = sigma_y        *np.random.randn(n_macroparticles), # if 0 infinite bbforce causes nan
                    zeta      = z1,
                    px        = 0,
                    py        = py,
                    delta     = 0,
                    weight=bunch_intensity/n_macroparticles
                    )
        
        # e+
        beam_2 = xp.Particles(
                    _context=test_context, 
                    q0        = 1,
                    p0c       = p0c,
                    mass0     = mass0,
                    x         = sigma_x        *np.random.randn(n_macroparticles),
                    y         = sigma_y        *np.random.randn(n_macroparticles),
                    zeta      = z2,
                    px        = 0,
                    py        = py, # beam 2 y axis points to same direction
                    delta     = 0,
                    weight=bunch_intensity/n_macroparticles
                    )
        
        # setup 1: with crossing angle with py=0
        
        #e-
        beam_1_10 = beam_1.copy()
        beam_1_10.py = 0
        beam_1_1 = beam_1_10.copy()
        
        # e+
        beam_2_10 = beam_2.copy()
        beam_2_10.py = 0
        beam_2_1  = beam_2_10.copy()
        
        # setup 2: without crossing angle with py=phi
        
        #e-
        beam_1_20 = beam_1.copy()
        beam_1_2  = beam_1_20.copy()
        
        # e+
        beam_2_20 = beam_2.copy()
        beam_2_2  = beam_2_20.copy()
        
        beam_1_1.name = "b11"
        beam_2_1.name = "b21"   
        beam_1_2.name = "b12"
        beam_2_2.name = "b22"   
        
        beam_1_1.init_pipeline('b11')
        beam_2_1.init_pipeline('b21')
        beam_1_2.init_pipeline('b12')
        beam_2_2.init_pipeline('b22')
        
        # slicer with 1 slices, just to assign the slice id to all particles
        slicer = xf.TempSlicer(_context=test_context, n_slices=1, sigma_z=sigma_z, mode="unicharge")
        
        #######################################
        # track with crossing angle with px=0 #
        #######################################
        
        pipeline_manager = xt.PipelineManager()
        pipeline_manager.add_particles(beam_1_1.name,0)
        pipeline_manager.add_particles(beam_2_1.name,0)
        pipeline_manager.add_element('IP1')
        pipeline_manager.add_element('IP2')
        
        
        config_for_update_b1_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
           pipeline_manager=pipeline_manager,
           element_name='IP1',
           partner_particles_name=beam_2_1.name,
           slicer=slicer,
           update_every=1,
           quasistrongstrong=False,
        
           )
        
        config_for_update_b2_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
           pipeline_manager=pipeline_manager,
           element_name='IP1',
           partner_particles_name=beam_1_1.name,
           slicer=slicer,
           update_every=1,
           quasistrongstrong=False,
        
           )
        
        # beambeam elements
        bbeamIP1_b1 = xf.BeamBeamBiGaussian3D(
                    _context=test_context,
                    other_beam_q0 = beam_2_1.q0,
                    phi = phi, 
                    alpha=alpha,
                    config_for_update = config_for_update_b1_IP1,
        )
        bbeamIP1_b2 = xf.BeamBeamBiGaussian3D(
                    _context=test_context,
                    other_beam_q0 = beam_1_1.q0,
                    phi = phi, # boost beam 2 slices by phi since we are in y-z plane
                    alpha=alpha, 
                    config_for_update = config_for_update_b2_IP1,
        )
        
        # init tracker
        line_b1 = xt.Line(elements = [bbeamIP1_b1,])
        line_b2 = xt.Line(elements = [bbeamIP1_b2,])
        
        line_b1.build_tracker(_context=test_context)
        line_b2.build_tracker(_context=test_context)
        
        branch_b1 = xt.PipelineBranch(line_b1, beam_1_1)
        branch_b2 = xt.PipelineBranch(line_b2, beam_2_1)
        multitracker = xt.PipelineMultiTracker(branches=[branch_b1,branch_b2])
        
        # track
        multitracker.track(num_turns=1)
        
        ############################################
        # track without crossing angle with px=phi #
        ############################################
        
        pipeline_manager = xt.PipelineManager()
        pipeline_manager.add_particles(beam_1_2.name,0)
        pipeline_manager.add_particles(beam_2_2.name,0)
        pipeline_manager.add_element('IP1')
        pipeline_manager.add_element('IP2')
        
        
        config_for_update_b1_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
           pipeline_manager=pipeline_manager,
           element_name='IP1',
           partner_particles_name=beam_2_2.name,
           slicer=slicer,
           update_every=1,
           quasistrongstrong=False,
        
           )
        
        config_for_update_b2_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
           pipeline_manager=pipeline_manager,
           element_name='IP1',
           partner_particles_name=beam_1_2.name,
           slicer=slicer,
           update_every=1,
           quasistrongstrong=False,
        
           )
        
        # beambeam elements
        bbeamIP1_b1 = xf.BeamBeamBiGaussian3D(
                    _context=test_context,
                    other_beam_q0 = beam_2_2.q0,
                    phi = 0, 
                    alpha=alpha,
                    config_for_update = config_for_update_b1_IP1,
        )
        bbeamIP1_b2 = xf.BeamBeamBiGaussian3D(
                    _context=test_context,
                    other_beam_q0 = beam_1_2.q0,
                    phi = 0,
                    alpha=alpha, 
                    config_for_update = config_for_update_b2_IP1,
        )
        
        # init tracker
        line_b1 = xt.Line(elements = [bbeamIP1_b1,])
        line_b2 = xt.Line(elements = [bbeamIP1_b2,])
        
        line_b1.build_tracker(_context=test_context)
        line_b2.build_tracker(_context=test_context)
        
        branch_b1 = xt.PipelineBranch(line_b1, beam_1_2)
        branch_b2 = xt.PipelineBranch(line_b2, beam_2_2)
        multitracker = xt.PipelineMultiTracker(branches=[branch_b1,branch_b2])
        
        # track
        multitracker.track(num_turns=1)
        
        ##########
        # checks #
        ##########
        
        # py from 2 setups after 1 kick should be similar (not exactly same due to h=/=0 at inv. boost)
        abserr_b1 = np.abs((beam_1_1.py + phi - beam_1_2.py))
        abserr_b2 = np.abs((beam_2_1.py + phi - beam_2_2.py))
        
        print(f"beam 1 max. error: { np.max(abserr_b1)}, beam 2 max. error: { np.max(abserr_b2)}")
        print(f"beam 1 avg. error: {np.mean(abserr_b1)}, beam 2 avg. error: {np.mean(abserr_b2)}")
        
        tol = max(phi*1e-5, 1e-10)  # when phi=0 1e-16 is better
        assert np.all(abserr_b1<=tol), "beam 1: error too large!"
        assert np.all(abserr_b2<=tol), "beam 2: error too large!"
