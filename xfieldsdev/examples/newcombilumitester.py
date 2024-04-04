
import time

import numpy as np
from matplotlib import pyplot as plt
import xobjects as xo
import xtrack as xt
import xfieldsdev as xf
print(xf.__file__)
import xpart as xp



#testing
###################
#To run this code:
#mpirun -np 2 python xsuite_dev/xtrack/examples/L0beambeam.py
lumi_qss_b1 = []
lumi_averages_b1 = []

combilumi_qss_b1 = []
combilumi_averages_b1 = []

lumi_qss_b1_nobeambeam = []
lumi_averages_b1_nobeambeam = []

combilumi_qss_b1_nobeambeam = []
combilumi_averages_b1_nobeambeam = []


context = xo.ContextCpu(omp_num_threads=0)


#################################
# Generate particles            #
#################################

p0c = 7e12
n_macroparticles = int(1.2e4)
bunch_intensity = 2.5e11 #7.8e10
physemit_x = (2E-6*xp.PROTON_MASS_EV)/p0c #(2.95E-6*xp.PROTON_MASS_EV)/p0c
physemit_y = (2E-6*xp.PROTON_MASS_EV)/p0c #(2.95E-6*xp.PROTON_MASS_EV)/p0c
beta_x_IP1 = 19.2
beta_y_IP1 = 19.2
sigma_z = 0.08
sigma_delta = 1E-4
beta_s = sigma_z/sigma_delta
Qx = 63.31
Qy = 60.32
Qs = 2.1E-3
frev = 11245.5 


colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y']
xshift = [0, 1, 2, 3, 4, 5, 6]
yshift = [0, 1, 2, 3, 4, 5, 6]

for i in range(len(xshift)):
    context = xo.ContextCpu(omp_num_threads=0)

####################
# Pipeline manager #
####################

# Retrieving MPI info


# Building the pipeline manager based on a MPI communicator
    pipeline_manager = xt.PipelineManager()

    # Add information about the Particles instance that require
    # communication through the pipeline manager.
    # Each Particles instance must be identifiable by a unique name
    # The second argument is the MPI rank in which the Particles object
    # lives.
    pipeline_manager.add_particles('B1b1',0)
    pipeline_manager.add_particles('B2b1',1)
    # Add information about the elements that require communication
    # through the pipeline manager.
    # Each Element instance must be identifiable by a unique name
    # All Elements are instanciated in all ranks
    pipeline_manager.add_element('IP1')

    print('Initialising particles')
    particles1 = xp.Particles(_context=context,
        p0c=p0c,
        x=np.sqrt(physemit_x*beta_x_IP1)
            *(np.random.randn(n_macroparticles)),
        px=np.sqrt(physemit_x/beta_x_IP1)
            *np.random.randn(n_macroparticles),
        y=np.sqrt(physemit_y*beta_y_IP1)
            *(np.random.randn(n_macroparticles)),
        py=np.sqrt(physemit_y/beta_y_IP1)
            *np.random.randn(n_macroparticles),
        zeta=sigma_z*np.random.randn(n_macroparticles),
        delta=sigma_delta*np.random.randn(n_macroparticles),
        weight=bunch_intensity/n_macroparticles
    )
    
    particles2 = xp.Particles(_context=context,
        p0c=p0c,
        x=np.sqrt(physemit_x*beta_x_IP1)
            *(np.random.randn(n_macroparticles)),
        px=np.sqrt(physemit_x/beta_x_IP1)
            *np.random.randn(n_macroparticles),
        y=np.sqrt(physemit_y*beta_y_IP1)
            *(np.random.randn(n_macroparticles)),
        py=np.sqrt(physemit_y/beta_y_IP1)
            *np.random.randn(n_macroparticles),
        zeta=sigma_z*np.random.randn(n_macroparticles),
        delta=sigma_delta*np.random.randn(n_macroparticles),
        weight=bunch_intensity/n_macroparticles
    )
    # Initialise the Particles object with its unique name
    # (must match the info provided to the pipeline manager)
    particles1.init_pipeline('B1b1')
    particles2.init_pipeline('B1b2')
    # Keep in memory the name of the Particles object with which this
    # rank will communicate
    # (must match the info provided to the pipeline manager)

#############
# Beam-beam #
#############

    nb_slice = 1 #normally this is 11
    slicer = xf.TempSlicer(sigma_z=sigma_z, n_slices=nb_slice, mode = "shatilov") #or shatilov
    config_for_update_IP1_b1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
        pipeline_manager=pipeline_manager,
        element_name='IP1', # The element name must be unique and match the
                       # one given to the pipeline manager
        partner_particles_name = 'B1b2', # the name of the
                       # Particles object with which to collide
        slicer=slicer,
        update_every=10,
        n_lumigrid_cells= 256*256
        #update_every=1 # Setup for strong-strong simulation
        )
    config_for_update_IP1_b2=xf.ConfigForUpdateBeamBeamBiGaussian3D(
        pipeline_manager=pipeline_manager,
        element_name='IP1', # The element name must be unique and match the
                       # one given to the pipeline manager
        partner_particles_name = 'B1b1', # the name of the
                       # Particles object with which to collide
        slicer=slicer,
        update_every=10,
        n_lumigrid_cells = 256*256
        #update_every=1 # Setup for strong-strong simulation
        )


    print('build bb elements...')
    bbeamIP1_b1 = xf.BeamBeamBiGaussian3D(
                _context=context,
                other_beam_q0 = particles2.q0,
                phi = 0,alpha=0.0,                    
                config_for_update = config_for_update_IP1_b1,
                ref_shift_x = xshift[i]*np.sqrt(physemit_x*beta_x_IP1)/2,
                ref_shift_y = yshift[i]*np.sqrt(physemit_y*beta_y_IP1)/2,
                flag_luminosity = 1,
                flag_combilumi = 1,
                number_of_particles = n_macroparticles,
                beam_intensity = bunch_intensity,
                other_beam_intensity = bunch_intensity,
                x_rms = np.sqrt(physemit_x*beta_x_IP1),
                y_rms = np.sqrt(physemit_y*beta_y_IP1))

    bbeamIP1_b2 = xf.BeamBeamBiGaussian3D(
                _context=context,
                other_beam_q0 = particles1.q0,
                phi = 0,alpha=0.0,                    
                beam_intensity = bunch_intensity,
                other_beam_intensity = bunch_intensity,
                number_of_particles = n_macroparticles,
                x_rms = np.sqrt(physemit_x*beta_x_IP1),
                y_rms = np.sqrt(physemit_y*beta_y_IP1),
                
                config_for_update = config_for_update_IP1_b2)
#################################################################
# arcs (here they are all the same with half the phase advance) #
#################################################################

    arc1_b1 = xt.LineSegmentMap(
            betx = beta_x_IP1,bety = beta_y_IP1,
            qx = Qx/2, qy = Qy/2,bets = beta_s, qs=Qs/2)

    arc2_b1= xt.LineSegmentMap(
            betx = beta_x_IP1,bety = beta_y_IP1,
            qx = Qx/2, qy = Qy/2,bets = beta_s, qs=Qs/2)

    arc1_b2 = xt.LineSegmentMap(
            betx = beta_x_IP1,bety = beta_y_IP1,
            qx = Qx/2, qy = Qy/2,bets = beta_s, qs=Qs/2)

    arc2_b2 = xt.LineSegmentMap(
            betx = beta_x_IP1,bety = beta_y_IP1,
            qx = Qx/2, qy = Qy/2,bets = beta_s, qs=Qs/2)
#################################################################
# Tracker                                                       #
#################################################################

# In this example there is one Particles object per rank
# We build its line and tracker as usual
    elements1 = [arc1_b1,bbeamIP1_b1,arc2_b2]
    elements2 = [arc1_b2,bbeamIP1_b2,arc2_b2]
    line1 = xt.Line(elements=elements1)
    line2 = xt.Line(elements=elements2)
    line1.build_tracker()
    line2.build_tracker()
# A pipeline branch is a line and Particles object that will
# be tracked through it
    branch1 = xt.PipelineBranch(line1, particles1)
    branch2 = xt.PipelineBranch(line2, particles2)
# The multitracker can deal with a set of branches (here only one)
    multitracker = xt.PipelineMultiTracker(branches=[branch1, branch2])

    num_turns= 1000
    record_qss_b1 = line1.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D, 
                                                                capacity={
                                                                    "beamstrahlungtable": int(0),
                                                                    "bhabhatable": int(0),
                                                                    "lumitable": num_turns,
                                                                    "combilumitable": num_turns,
                                                                    })
                                                                
                                                                


    multitracker.track(num_turns=num_turns)
    line1.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

    record_qss_b1.move(_context=xo.context_default)

 
    lumi_b1 = record_qss_b1.lumitable.luminosity
    combilumi_b1 = record_qss_b1.combilumitable.combilumi
    
    lumi_qss_b1.append(lumi_b1)
    combilumi_qss_b1.append(lumi_b1)

    lumi_averages_b1.append(np.mean(lumi_b1[200:]))
    combilumi_averages_b1.append(np.mean(lumi_b1[200:]))
    