# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import time

import numpy as np
from matplotlib import pyplot as plt
from mpi4py import MPI
import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp

context = xo.ContextCpu(omp_num_threads=0)

####################
# Pipeline manager #
####################

# Retrieving MPI info
comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
my_rank   = comm.Get_rank()

# Building the pipeline manager based on a MPI communicator
pipeline_manager = xt.PipelineManager(comm)
if nprocs >= 2:
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
    pipeline_manager.add_element('IP2')
else:
    print('Need at least 2 MPI processes for this test')
    exit()

#################################
# Generate particles            #
#################################

n_macroparticles = int(1e4)
bunch_intensity = 2.3e11
physemit_x = 2E-6*0.938/7E3
physemit_y = 2E-6*0.938/7E3
beta_x_IP1 = 1.0
beta_y_IP1 = 1.0
beta_x_IP2 = 1.0
beta_y_IP2 = 1.0
sigma_z = 0.08
sigma_delta = 1E-4
beta_s = sigma_z/sigma_delta
Qx = 0.285
Qy = 0.32
Qs = 2.1E-3

#Offsets [sigma] the two beams with opposite direction
#to enhance oscillation of the pi-mode 
mean_x_init = 0.1
mean_y_init = 0.1

p0c = 7000e9

print('Initialising particles')
if my_rank == 0:
    particles = xp.Particles(_context=context,
        p0c=p0c,
        x=np.sqrt(physemit_x*beta_x_IP1)
            *(np.random.randn(n_macroparticles)+mean_x_init),
        px=np.sqrt(physemit_x/beta_x_IP1)
            *np.random.randn(n_macroparticles),
        y=np.sqrt(physemit_y*beta_y_IP1)
            *(np.random.randn(n_macroparticles)+mean_y_init),
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
elif my_rank == 1:
    particles = xp.Particles(_context=context,
        p0c=p0c,
        x=np.sqrt(physemit_x*beta_x_IP1)
            *(np.random.randn(n_macroparticles)+mean_x_init),
        px=np.sqrt(physemit_x/beta_x_IP1)
            *np.random.randn(n_macroparticles),
        y=np.sqrt(physemit_y*beta_y_IP1)
            *(np.random.randn(n_macroparticles)-mean_y_init),
        py=np.sqrt(physemit_y/beta_y_IP1)
            *np.random.randn(n_macroparticles),
        zeta=sigma_z*np.random.randn(n_macroparticles),
        delta=sigma_delta*np.random.randn(n_macroparticles),
        weight=bunch_intensity/n_macroparticles
    )
    # Initialise the Particles object with its unique name
    # (must match the info provided to the pipeline manager)
    particles.init_pipeline('B2b1')
    # Keep in memory the name of the Particles object with which this
    # rank will communicate
    # (must match the info provided to the pipeline manager)
    partner_particles_name = 'B1b1'
else:
    # Stop the execution of ranks that are not needed
    print('No need for rank',my_rank)
    exit()

#############
# Beam-beam #
#############

nb_slice = 11
slicer = xf.TempSlicer(sigma_z=sigma_z, n_slices=nb_slice)
config_for_update_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
   pipeline_manager=pipeline_manager,
   element_name='IP1', # The element name must be unique and match the
                       # one given to the pipeline manager
   partner_particles_name = partner_particles_name, # the name of the
                       # Particles object with which to collide
   slicer=slicer,
   update_every=1 # Setup for strong-strong simulation
   )
config_for_update_IP2=xf.ConfigForUpdateBeamBeamBiGaussian3D(
   pipeline_manager=pipeline_manager,
   element_name='IP2', # The element name must be unique and match the
                       #  one given to the pipeline manager
   partner_particles_name = partner_particles_name, # the name of the
                       #Particles object with which to collide
   slicer=slicer,
   update_every=1 # Setup for strong-strong simulation
   )



print('build bb elements...')
bbeamIP1 = xf.BeamBeamBiGaussian3D(
            _context=context,
            other_beam_q0 = particles.q0,
            phi = 500E-6,alpha=0.0,
            config_for_update = config_for_update_IP1)

bbeamIP2 = xf.BeamBeamBiGaussian3D(
            _context=context,
            other_beam_q0 = particles.q0,
            phi = 500E-6,alpha=np.pi/2,
            config_for_update = config_for_update_IP2)

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
elements = [bbeamIP1,arc12,bbeamIP2,arc21]
line = xt.Line(elements=elements)
line.build_tracker()
# A pipeline branch is a line and Particles object that will
# be tracked through it
branch = xt.PipelineBranch(line, particles)
# The multitracker can deal with a set of branches (here only one)
multitracker = xt.PipelineMultiTracker(branches=[branch])

#################################################################
# Tracking  (Same as usual)                                     #
#################################################################
print('Tracking...')
time0 = time.time()
nTurn = 1024
multitracker.track(num_turns=nTurn,turn_by_turn_monitor=True)
print('Done with tracking.',(time.time()-time0)/1024,'[s/turn]')


#################################################################
# Post-processing: raw data and spectrum                        #
#################################################################

# Show beam oscillation spectrum (only one rank)
if my_rank == 0:
    positions_x_b1 = np.average(line.record_last_track.x,axis=0)
    positions_y_b1 = np.average(line.record_last_track.y,axis=0)
    plt.figure(1)
    freqs = np.fft.fftshift(np.fft.fftfreq(nTurn))
    mask = freqs > 0
    myFFT = np.fft.fftshift(np.fft.fft(positions_x_b1))
    plt.semilogy(freqs[mask], (np.abs(myFFT[mask])),label='Horizontal')
    myFFT = np.fft.fftshift(np.fft.fft(positions_y_b1))
    plt.semilogy(freqs[mask], (np.abs(myFFT[mask])),label='Vertical')
    plt.xlabel('Frequency [$f_{rev}$]')
    plt.ylabel('Amplitude')
    plt.legend(loc=0)
    plt.show()



