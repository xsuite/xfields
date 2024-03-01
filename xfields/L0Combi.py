#L0 with combi formula

%load_ext wurlitzer

import time

import numpy as np
from matplotlib import pyplot as plt
import xobjects as xo
import xtrack as xt
import xfields as xf
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
    particles = xp.Particles(_context=context,
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
    particles.init_pipeline('B1b1')
    # Keep in memory the name of the Particles object with which this
    # rank will communicate
    # (must match the info provided to the pipeline manager)
    partner_particles_name = 'B2b1'

#############
# Beam-beam #
#############

    nb_slice = 1 #normally this is 11
    slicer = xf.TempSlicer(sigma_z=sigma_z, n_slices=nb_slice, mode = "shatilov") #or shatilov
    config_for_update_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
        pipeline_manager=pipeline_manager,
        element_name='IP1', # The element name must be unique and match the
                       # one given to the pipeline manager
        partner_particles_name = partner_particles_name, # the name of the
                       # Particles object with which to collide
        slicer=slicer,
        update_every=10,
        #update_every=1 # Setup for strong-strong simulation
        )



    print('build bb elements...')
    bbeamIP1 = xf.BeamBeamBiGaussian3D(
                _context=context,
                other_beam_q0 = particles.q0,
                phi = 0,alpha=0.0,
                config_for_update = config_for_update_IP1,
                ref_shift_x = xshift[i]*np.sqrt(physemit_x*beta_x_IP1),
                ref_shift_y = yshift[i]*np.sqrt(physemit_y*beta_y_IP1),
                flag_luminosity = 1,
                flag_combilumi = 1)

#################################################################
# arcs (here they are all the same with half the phase advance) #
#################################################################

    arc12 = xt.LineSegmentMap(
            betx = beta_x_IP1,bety = beta_y_IP1,
            qx = Qx/2, qy = Qy/2,bets = beta_s, qs=Qs/2)

    arc21 = xt.LineSegmentMap(
            betx = beta_x_IP1,bety = beta_y_IP1,
            qx = Qx/2, qy = Qy/2,bets = beta_s, qs=Qs/2)

#################################################################
# Tracker                                                       #
#################################################################

# In this example there is one Particles object per rank
# We build its line and tracker as usual
    elements = [arc12,bbeamIP1,arc21]
    line = xt.Line(elements=elements)
    line.build_tracker()
# A pipeline branch is a line and Particles object that will
# be tracked through it
    branch = xt.PipelineBranch(line, particles)
# The multitracker can deal with a set of branches (here only one)
    multitracker = xt.PipelineMultiTracker(branches=[branch])
    x = context.nparray_from_context_array(particles.x)
    y = context.nparray_from_context_array(particles.y)
    print(np.mean(x))
    print(np.mean(y))
    num_turns= 1000
    record_qss_b1 = line.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D, 
                                                                capacity={
                                                                    "beamstrahlungtable": int(0),
                                                                    "bhabhatable": int(0),
                                                                    "lumitable": num_turns,
                                                                    "combilumitable": num_turns,
                                                                    })
                                                                    #
                                                                


    multitracker.track(num_turns=num_turns)
    line.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

    record_qss_b1.move(_context=xo.context_default)

 
    lumi_b1 = record_qss_b1.lumitable.luminosity
    combilumi_b1 = record_qss_b1.combilumitable.combilumi
    
    lumi_qss_b1.append(lumi_b1)
    combilumi_qss_b1.append(lumi_b1)

    lumi_averages_b1.append(np.mean(lumi_b1[200:]))
    combilumi_averages_b1.append(np.mean(lumi_b1[200:]))
    
    
    ########################
    #NO beam beam
    ########################
    
# for i in range(len(xshift)):
#     context = xo.ContextCpu(omp_num_threads=0)

# ####################
# # Pipeline manager #
# ####################

# # Retrieving MPI info


# # Building the pipeline manager based on a MPI communicator
#     pipeline_manager = xt.PipelineManager()

#     # Add information about the Particles instance that require
#     # communication through the pipeline manager.
#     # Each Particles instance must be identifiable by a unique name
#     # The second argument is the MPI rank in which the Particles object
#     # lives.
#     pipeline_manager.add_particles('B1b1',0)
#     pipeline_manager.add_particles('B2b1',1)
#     # Add information about the elements that require communication
#     # through the pipeline manager.
#     # Each Element instance must be identifiable by a unique name
#     # All Elements are instanciated in all ranks
#     pipeline_manager.add_element('IP1')

#     print('Initialising particles')
#     particles = xp.Particles(_context=context,
#         p0c=p0c,
#         x=np.sqrt(physemit_x*beta_x_IP1)
#             *(np.random.randn(n_macroparticles)),
#         px=np.sqrt(physemit_x/beta_x_IP1)
#             *np.random.randn(n_macroparticles),
#         y=np.sqrt(physemit_y*beta_y_IP1)
#             *(np.random.randn(n_macroparticles)),
#         py=np.sqrt(physemit_y/beta_y_IP1)
#             *np.random.randn(n_macroparticles),
#         zeta=sigma_z*np.random.randn(n_macroparticles),
#         delta=sigma_delta*np.random.randn(n_macroparticles),
#         weight=bunch_intensity/n_macroparticles
#     )
#     # Initialise the Particles object with its unique name
#     # (must match the info provided to the pipeline manager)
#     particles.init_pipeline('B1b1')
#     # Keep in memory the name of the Particles object with which this
#     # rank will communicate
#     # (must match the info provided to the pipeline manager)
#     partner_particles_name = 'B2b1'

# #############
# # Beam-beam #
# #############

#     nb_slice = 1 #normally this is 11
#     slicer = xf.TempSlicer(sigma_z=sigma_z, n_slices=nb_slice, mode = "shatilov") #or shatilov
#     config_for_update_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
#         pipeline_manager=pipeline_manager,
#         element_name='IP1', # The element name must be unique and match the
#                    # one given to the pipeline manager
#         partner_particles_name = partner_particles_name, # the name of the
#                    # Particles object with which to collide
#         slicer=slicer,
#         update_every=10,
#     #update_every=1 # Setup for strong-strong simulation
#         )



#     print('build bb elements...')
#     bbeamIP1 = xf.BeamBeamBiGaussian3D(
#                 _context=context,
#                 other_beam_q0 = 0,
#                 phi = 0,alpha=0.0,
#                 config_for_update = config_for_update_IP1,
#                 other_beam_shift_x = xshift[i]*np.sqrt(physemit_x*beta_x_IP1)/2,
#                 #other_beam_shift_y = yshift[i]*np.sqrt(physemit_y*beta_y_IP1)/2,
#                 flag_luminosity = 1,
#                 flag_combilumi = 1)


# #################################################################
# # arcs (here they are all the same with half the phase advance) #
# #################################################################

#     arc12 = xt.LineSegmentMap(
#             betx = beta_x_IP1,bety = beta_y_IP1,
#             qx = Qx/2, qy = Qy/2,bets = beta_s, qs=Qs/2)

#     arc21 = xt.LineSegmentMap(
#             betx = beta_x_IP1,bety = beta_y_IP1,
#             qx = Qx/2, qy = Qy/2,bets = beta_s, qs=Qs/2)

# #################################################################
# # Tracker                                                       #
# #################################################################

# # In this example there is one Particles object per rank
# # We build its line and tracker as usual
#     elements = [arc12, bbeamIP1, arc21]
#     line = xt.Line(elements=elements)
#     line.build_tracker()
# # A pipeline branch is a line and Particles object that will
# # be tracked through it
#     branch = xt.PipelineBranch(line, particles)
# # The multitracker can deal with a set of branches (here only one)
#     multitracker = xt.PipelineMultiTracker(branches=[branch])
    
    
#     x = context.nparray_from_context_array(particles.x)
#     y = context.nparray_from_context_array(particles.y)
    
#     print(np.mean(x))
#     print(np.mean(y))
    
#     num_turns= 1000
#     record_qss_b1 = line.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D, 
#                                                             capacity={
#                                                                 "beamstrahlungtable": int(0),
#                                                                 "bhabhatable": int(0),
#                                                                 "lumitable": num_turns,
#                                                                 "combilumitable": num_turns
#                                                             })


#     multitracker.track(num_turns=num_turns)
#     line.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

#     record_qss_b1.move(_context=xo.context_default)
 
#     lumi_b1_nobeambeam = record_qss_b1.lumitable.luminosity
#     combilumi_b1_nobeambeam = record_qss_b1.combilumitable.luminosity
    
#     lumi_qss_b1_nobeambeam.append(lumi_b1_nobeambeam)
#     combilumi_qss_b1_nobeambeam.append(combilumi_b1_nobeambeam)

#     lumi_averages_b1_nobeambeam.append(np.mean(lumi_b1_nobeambeam))
#     combilumi_averages_b1_nobeambeam.append(np.mean(combilumi_b1_nobeambeam))
#################################################################
# Tracking  (Same as usual)                                     #
#################################################################


#################################################################
# Post-processing: raw data and spectrum                        #
#################################################################
# Complete source: xfields/examples/002_beambeam/006_beambeam6d_strongstrong_pipeline_MPI2Procs.py

lumis = []
separation = [0, 1, 2, 3, 4, 5, 6]

def Lumi_analytical(Nb, N1, N2, frev, Delta_i, sig_i, sig_x, sig_y):
    W = np.exp(-Delta_i**2/(4*sig_i**2))
    return ((Nb * N1 * N2 * frev * W)/(4 * np.pi * 100 * sig_x * 100 * sig_y))

for i in range(len(separation)):
    lumis.append(Lumi_analytical(2808), bunch_intensity, bunch_intensity, frev, separation[i]*np.sqrt(physemit_x*beta_x_IP1),np.sqrt(physemit_x*beta_x_IP1), np.sqrt(physemit_x*beta_x_IP1), np.sqrt(physemit_y*beta_x_IP1))

fig0, ax1 = plt.subplots()
fig1, ax2 = plt.subplots()

ax1.set_title("Luminosity as a function of beam separation")
ax1.set_xlabel("Separation (sigma)")
ax1.set_ylabel("Luminosity")
ax1.plot(separation, lumis, label = "Analytical")
ax1.plot(xshift, (2808*frev*np.array(lumi_averages_b1))/10000, label = "With beam beam") 
ax1.plot(xshift, (2808*frev*np.array(lumi_averages_b1_nobeambeam))/10000, label = "Without beam beam")
ax1.plot(xshift, (np.array(combilumi_averages_b1)), label = "Combi with beam beam") 
ax1.plot(xshift, (np.array(combilumi_averages_b1_nobeambeam)), label = "Combi without beam beam")
ax1.legend()


ax2.set_title("L/L0")
ax2.plot(xshift, np.array(lumi_averages_b1)/np.array(lumi_averages_b1_nobeambeam))
plt.show()
