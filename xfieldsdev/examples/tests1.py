import time

import numpy as np
from matplotlib import pyplot as plt
import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp

context = xo.ContextCpu(omp_num_threads=0)

####################
# Pipeline manager #
####################

#################################
# Generate particles            #
################################

p0c = 7000e9
n_macroparticles = int(500)
bunch_intensity = 1.15e11
physemit_x = (3.75E-6*xp.PROTON_MASS_EV)/p0c 
physemit_y = (3.75E-6*xp.PROTON_MASS_EV)/p0c 
beta_x_IP1 = 0.55
beta_y_IP1 = 0.55
beta_x_IP2 = 0.55
beta_y_IP2 = 0.55
sigma_z = 0.08
sigma_delta = 1E-4
beta_s = sigma_z/sigma_delta
Qx = 62.31
Qy = 60.32
Qs = 2.1E-3

#Offsets [sigma] the two beams with opposite direction
#to enhance oscillation of the pi-mode 
mean_x_init = 0.1
mean_y_init = 0.1

pipeline_manager = xt.PipelineManager()
pipeline_manager.add_particles('B1b1',0)
pipeline_manager.add_particles('B2b1',0)
pipeline_manager.add_element('IP1')
pipeline_manager.add_element('IP2')

print('Initialising particles')

particles_b1 = xp.Particles(_context=context,
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
particles_b1.init_pipeline('B1b1')
    # Initialise the Particles object with its unique name
    # (must match the info provided to the pipeline manager)
    # Keep in memory the name of the Particles object with which this
    # rank will communicate
    # (must match the info provided to the pipeline manager)

particles_b2 = xp.Particles(_context=context,
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
particles_b2.init_pipeline('B2b1')

slicer = xf.TempSlicer(sigma_z=sigma_z, n_slices=2)
config_for_update_IP1_b1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
   pipeline_manager=pipeline_manager,
   element_name='IP1', # The element name must be unique and match the
                       # one given to the pipeline manager
   partner_particles_name = 'B2b1', # the name of the
                       # Particles object with which to collide
   slicer=slicer,
   update_every=1 # Setup for strong-strong simulation
   )
config_for_update_IP2_b1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
   pipeline_manager=pipeline_manager,
   element_name='IP2', # The element name must be unique and match the
                       #  one given to the pipeline manager
   partner_particles_name = 'B2b1', # the name of the
                       #Particles object with which to collide
   slicer=slicer,
   update_every=1 # Setup for strong-strong simulation
   )

config_for_update_IP1_b2=xf.ConfigForUpdateBeamBeamBiGaussian3D(
   pipeline_manager=pipeline_manager,
   element_name='IP1', # The element name must be unique and match the
                       # one given to the pipeline manager
   partner_particles_name = 'B1b1', # the name of the
                       # Particles object with which to collide
   slicer=slicer,
   update_every=1 # Setup for strong-strong simulation
   )
config_for_update_IP2_b2=xf.ConfigForUpdateBeamBeamBiGaussian3D(
   pipeline_manager=pipeline_manager,
   element_name='IP2', # The element name must be unique and match the
                       #  one given to the pipeline manager
   partner_particles_name = 'B1b1', # the name of the
                       #Particles object with which to collide
   slicer=slicer,
   update_every=1 # Setup for strong-strong simulation
   )

print('build bb elements...')
bbeamIP1_b1 = xf.BeamBeamBiGaussian3D(
            _context=context,
            other_beam_q0 = particles_b2.q0,
            phi = 0,alpha=0.0,
            config_for_update = config_for_update_IP1_b1)

bbeamIP2_b1 = xf.BeamBeamBiGaussian3D(
            _context=context,
            other_beam_q0 = particles_b2.q0,
            phi = 0,alpha=np.pi/2,
            config_for_update = config_for_update_IP2_b1)

bbeamIP1_b2 = xf.BeamBeamBiGaussian3D(
            _context=context,
            other_beam_q0 = particles_b1.q0,
            phi = 0,alpha=0.0,
            config_for_update = config_for_update_IP1_b2)

bbeamIP2_b2 = xf.BeamBeamBiGaussian3D(
            _context=context,
            other_beam_q0 = particles_b1.q0,
            phi = 0,alpha=np.pi/2,
            config_for_update = config_for_update_IP2_b2)

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
elements_b1 = [bbeamIP1_b1,arc12,bbeamIP2_b1,arc21]
elements_b2 = [bbeamIP1_b2, arc12, bbeamIP2_b2, arc21]
line_b1 = xt.Line(elements=elements_b1)
line_b2 = xt.Line(elements=elements_b2)
line_b1.build_tracker(context)
line_b2.build_tracker(context)
# A pipeline branch is a line and Particles object that will
# be tracked through it
branch_b1 = xt.PipelineBranch(line_b1, particles_b1)
branch_b2 = xt.PipelineBranch(line_b2, particles_b2)
# The multitracker can deal with a set of branches (here only one)
multitracker = xt.PipelineMultiTracker(branches=[branch_b1, branch_b2])

#################################################################
# Tracking  (Same as usual)                                     #
#################################################################
print('Tracking...')
time0 = time.time()
nTurn = 1024
multitracker.track(num_turns=nTurn,turn_by_turn_monitor=True)
print('Done with tracking.',(time.time()-time0)/1024,'[s/turn]')
print('total time - ', time.time()-time0)


#################################################################
# Post-processing: raw data and spectrum                        #
#################################################################

# Show beam oscillation spectrum (only one rank)

positions_x_b1 = np.average(line_b1.record_last_track.x,axis=0)
positions_y_b1 = np.average(line_b1.record_last_track.y,axis=0)
plt.figure(1)
freqs = np.fft.fftshift(np.fft.fftfreq(nTurn))
mask = freqs > 0
myFFT = np.fft.fftshift(np.fft.fft(positions_x_b1))
plt.plot(freqs[mask], (np.abs(myFFT[mask])),label='Horizontal b1')
myFFT = np.fft.fftshift(np.fft.fft(positions_y_b1))
plt.plot(freqs[mask], (np.abs(myFFT[mask])),label='Vertical b1')
plt.xlabel('Frequency [$f_{rev}$]')
plt.ylabel('Amplitude')
plt.legend(loc=0)
plt.title("Beam 1")

positions_x_b2 = np.average(line_b2.record_last_track.x,axis=0)
positions_y_b2 = np.average(line_b2.record_last_track.y,axis=0)
plt.figure(2)
freqs = np.fft.fftshift(np.fft.fftfreq(nTurn))
mask = freqs > 0
myFFT = np.fft.fftshift(np.fft.fft(positions_x_b2))
plt.plot(freqs[mask], (np.abs(myFFT[mask])),label='Horizontal b2')
myFFT = np.fft.fftshift(np.fft.fft(positions_y_b2))
plt.plot(freqs[mask], (np.abs(myFFT[mask])),label='Vertical b2')
plt.xlabel('Frequency [$f_{rev}$]')
plt.ylabel('Amplitude')
plt.legend(loc=0)
plt.title("Beam 2")

plt.show()

