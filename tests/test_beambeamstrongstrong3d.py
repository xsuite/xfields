# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp

def test_beambeamstrongstrong3d():

   context = xo.ContextCpu(omp_num_threads=0)

   ####################
   # Pipeline manager #
   ####################
   pipeline_manager = xt.PipelineManager()
   pipeline_manager.add_particles('B1b1',0)
   pipeline_manager.add_particles('B2b1',0)
   pipeline_manager.add_element('IP1')

   #################################
   # Generate particles            #
   #################################

   n_macroparticles = int(1e4)
   bunch_intensity_B1 = 2.3E4
   bunch_intensity_B2 = 1.3E4
   physemit_x = 2E-6*0.938/7E3
   physemit_y = 2E-6*0.938/7E3
   beta_x_IP1 = 1.0
   beta_y_IP1 = 2.0
   sigma_z = 0.08
   sigma_delta = 1E-4
   beta_s = sigma_z/sigma_delta
   Qx = 0.31
   Qy = 0.32
   Qs = 2.1E-3

   #Offsets in sigma
   mean_x_init = 0.1
   mean_y_init = 0.1

   p0c = 7000e9

   particles_b1 = xp.Particles(_context=context,
      p0c=p0c,
      x=np.sqrt(physemit_x*beta_x_IP1)*(np.random.randn(n_macroparticles)+mean_x_init),
      px=np.sqrt(physemit_x/beta_x_IP1)*np.random.randn(n_macroparticles),
      y=np.sqrt(physemit_y*beta_y_IP1)*(np.random.randn(n_macroparticles)+mean_y_init),
      py=np.sqrt(physemit_y/beta_y_IP1)*np.random.randn(n_macroparticles),
      zeta=sigma_z*np.random.randn(n_macroparticles),
      delta=sigma_delta*np.random.randn(n_macroparticles),
      weight=bunch_intensity_B1/n_macroparticles
   )
   particles_b1.init_pipeline('B1b1')
   particles_b2 = xp.Particles(_context=context,
      p0c=p0c,
      x=np.sqrt(physemit_x*beta_x_IP1)*(np.random.randn(n_macroparticles)-mean_x_init),
      px=np.sqrt(physemit_x/beta_x_IP1)*np.random.randn(n_macroparticles),
      y=np.sqrt(physemit_y*beta_y_IP1)*(np.random.randn(n_macroparticles)+mean_y_init),
      py=np.sqrt(physemit_y/beta_y_IP1)*np.random.randn(n_macroparticles),
      zeta=sigma_z*np.random.randn(n_macroparticles),
      delta=sigma_delta*np.random.randn(n_macroparticles),
      weight=bunch_intensity_B2/n_macroparticles
   )
   particles_b2.init_pipeline('B2b1')

   n_slices = 5
   slicer = xf.TempSlicer(sigma_z=sigma_z, n_slices=n_slices)
   slicer.assign_slices(particles_b1)
   moments_b1 = slicer.compute_moments(particles_b1,update_assigned_slices=False)
   slicer.assign_slices(particles_b2)
   moments_b2 = slicer.compute_moments(particles_b2,update_assigned_slices=False)

   #############
   # Beam-beam #
   #############

   config_for_update_b1_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
      pipeline_manager=pipeline_manager,
      element_name='IP1',
      partner_particles_name = 'B2b1',
      slicer=slicer,
      update_every=1
      )
   config_for_update_b2_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
      pipeline_manager=pipeline_manager,
      element_name='IP1',
      partner_particles_name = 'B1b1',
      slicer=slicer,
      update_every=1
      )
   config_for_update_b1_IP2=xf.ConfigForUpdateBeamBeamBiGaussian3D(
      pipeline_manager=pipeline_manager,
      element_name='IP2',
      partner_particles_name = 'B2b1',
      slicer=slicer,
      update_every=1
      )
   config_for_update_b2_IP2=xf.ConfigForUpdateBeamBeamBiGaussian3D(
      pipeline_manager=pipeline_manager,
      element_name='IP2',
      partner_particles_name = 'B1b1',
      slicer=slicer,
      update_every=1
      )

   bbeamIP1_b1 = xf.BeamBeamBiGaussian3D(
               _context=context,
               other_beam_q0 = particles_b2.q0,
               phi = 0.0,alpha=0.0,
               config_for_update = config_for_update_b1_IP1)

   bbeamIP1_b2 = xf.BeamBeamBiGaussian3D(
               _context=context,
               other_beam_q0 = particles_b1.q0,
               phi = 0.0,alpha=0.0,
               config_for_update = config_for_update_b2_IP1)

   #################################################################
   # Tracker                                                       #
   #################################################################

   elements_b1 = [bbeamIP1_b1]
   elements_b2 = [bbeamIP1_b2]
   line_b1 = xt.Line(elements=elements_b1)
   line_b2 = xt.Line(elements=elements_b2)
   line_b1.build_tracker()
   line_b2.build_tracker()
   branch_b1 = xt.PipelineBranch(line_b1,particles_b1)
   branch_b2 = xt.PipelineBranch(line_b2,particles_b2)
   multitracker = xt.PipelineMultiTracker(branches=[branch_b1,branch_b2])

   #################################################################
   # Tracking                                                      #
   #################################################################

   nTurn = 1
   multitracker.track(num_turns=nTurn,turn_by_turn_monitor=True)

   #################################################################
   # Test: Check update of beam-beam element with the other        #
   #       beam's property                                         #
   #################################################################

   assert np.isclose(np.max(np.abs(bbeamIP1_b1.partner_moments-moments_b2)),0.0)
   assert np.isclose(np.max(np.abs(bbeamIP1_b2.partner_moments-moments_b1)),0.0)

