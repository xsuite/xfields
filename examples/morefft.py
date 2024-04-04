
#Strong-strong centroid investigations 

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import math
import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp
import logging
logging.basicConfig(level=logging.WARNING)

context = xo.ContextCpu(omp_num_threads=0)

# Machine and beam parameters


p0c = 7e12 
n_macroparticles = int(1.2e4)
bunch_intensity = 1.1e11 #7.8e10
physemit_x = (3.75E-6*xp.PROTON_MASS_EV)/p0c #(2.95E-6*xp.PROTON_MASS_EV)/p0c
physemit_y = (3.75E-6*xp.PROTON_MASS_EV)/p0c #(2.95E-6*xp.PROTON_MASS_EV)/p0c
beta_x = 0.55
beta_y  = 0.55
sigma_z = 0.08
sigma_delta = 1E-4
beta_s = sigma_z/sigma_delta
Qx = 62.31
Qy = 60.32
Qs = 2.1E-3
nTurn = 8192

xshift = [0]
yshift = [0]

turns = np.arange(0,8192,1)

constshift = [0]

##############################
# Plot the beam-beam force   #
##############################

for j in range(len(constshift)):

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    for i in range(len(yshift)):
        r_0 = 0.84e-12
        gamma = 7460
        sigma = np.sqrt(physemit_x*beta_x)
        #bbp = (bunch_intensity*r_0*beta_x)/(4*np.pi*gamma*sigma**2)
        bbp = 0.0037
        print(bbp)
        pipeline_manager2 = xt.PipelineManager()
        pipeline_manager2.add_particles('B1b2', 0)
        pipeline_manager2.add_particles('B2b2',1)
        pipeline_manager2.add_element('IP1')
        
        particles = xp.Particles(_context=context,
            p0c=p0c, mass0=xp.PROTON_MASS_EV, q0=1)

        
        nb_slice = 1 #11 normally
        slicer = xf.TempSlicer(sigma_z=sigma_z, n_slices=nb_slice,  mode = "shatilov")
        config_for_update_b1_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
            pipeline_manager=pipeline_manager2,
            element_name='IP1', # The element name must be unique and match the
                        # one given to the pipeline manager
            partner_particles_name = "B2b2", # the name of the
                        # Particles object with which to collide
            slicer=slicer,
            update_every=1 # Setup for strong-strong simulation
            )
        config_for_update_b2_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
            pipeline_manager=pipeline_manager2,
            element_name='IP1', # The element name must be unique and match the
                        #  one given to the pipeline manager
            partner_particles_name = "B1b2", # the name of the
                        #Particles object with which to collide
            slicer=slicer,
            update_every=1 # Setup for strong-strong simulation
            )

        
        bbeam2 = xf.BeamBeamBiGaussian3D(
                    _context=context,
                    other_beam_q0 = particles.q0,
                    phi = 0, alpha = 0.0,
                    config_for_update = config_for_update_b1_IP1,
                    ref_shift_x = yshift[i]*np.sqrt(physemit_x*beta_x),
                    ref_shift_y = yshift[i]*np.sqrt(physemit_x*beta_x))
        
        bbeam22 = xf.BeamBeamBiGaussian3D(
                    _context=context,
                    other_beam_q0 = particles.q0,
                    phi= 0, alpha = 0.0,
                    config_for_update = config_for_update_b2_IP1)

        

        ##############################
        # Tune footprint             #
        ##############################
    
        # Build linear lattice
                
        arc21b1 = xt.LineSegmentMap(
            betx = beta_x,bety = beta_y,
            qx = Qx/2, qy = Qy/2,bets = beta_s, qs=Qs/2)
                    
        arc22b1 = xt.LineSegmentMap(
            betx = beta_x,bety = beta_y,
            qx = Qx/2, qy = Qy/2,bets = beta_s, qs=Qs/2)
                            
        arc21b2 = xt.LineSegmentMap(
            betx = beta_x,bety = beta_y,
            qx = Qx/2, qy = Qy/2,bets = beta_s, qs=Qs/2)
                                
        arc22b2 = xt.LineSegmentMap(
            betx = beta_x,bety = beta_y,
            qx = Qx/2, qy = Qy/2,bets = beta_s, qs=Qs/2)
                                    
        
        # Build line and tracker with a beam-beam element and linear lattice
        elements2b1 = [arc21b1, bbeam2, arc22b1]
        elements2b2 = [arc21b2, bbeam22, arc22b2]
        
        element_names2b1 = ['arc21b1','bbeam2', 'arc22b1']
        element_names2b2 = ['arc21b2', 'bbeam22', 'arc22b2']
        line2b1 = xt.Line(elements=elements2b1, element_names = element_names2b1)
        line2b2 = xt.Line(elements=elements2b2, element_names = element_names2b2)
        

        
        monitor_bbeam2 = xt.BeamSizeMonitor(start_at_turn=0, stop_at_turn=8192)       
        line2b1.insert_element(index='bbeam2', element=monitor_bbeam2, name='mon2')
        monitor_bbeam22 = xt.BeamSizeMonitor(start_at_turn=0, stop_at_turn=8192)       
        line2b2.insert_element(index='bbeam22', element=monitor_bbeam22, name='mon22')
        line2b1.build_tracker()
        line2b2.build_tracker()
        particles2b1 = xp.generate_matched_gaussian_bunch( 
        num_particles=n_macroparticles, total_intensity_particles=bunch_intensity,
        nemitt_x= 3.75E-6, nemitt_y=3.75E-6, sigma_z=sigma_z, line = line2b1, particle_ref = particles)
        particles2b2 = xp.generate_matched_gaussian_bunch( 
        num_particles=n_macroparticles, total_intensity_particles=bunch_intensity,
        nemitt_x= 3.75E-6, nemitt_y=3.75E-6, sigma_z=sigma_z, line = line2b2, particle_ref = particles)
        
        particles2b1.init_pipeline('B1b2')
        particles2b2.init_pipeline('B2b2')
        branch_2b1 = xt.PipelineBranch(line2b1, particles2b1)
        branch_2b2 = xt.PipelineBranch(line2b2, particles2b2)
        multitracker2 = xt.PipelineMultiTracker(branches=[branch_2b1, branch_2b2])
        multitracker2.track(num_turns = nTurn, turn_by_turn_monitor = True)
    
        line2b1.discard_tracker()
        line2b2.discard_tracker()

        print('Tracking...')


    #################################################################
    # Post-processing: raw data and spectrum                        #
    #################################################################
        ax1.plot((np.fft.fftfreq(len(turns))-0.31)/bbp, (np.abs(np.fft.fft(monitor_bbeam2.x_mean))), label = "Xmean_b1 with %.2f"%yshift[i])
        ax1.plot((np.fft.fftfreq(len(turns))-0.31)/bbp, (np.abs(np.fft.fft(monitor_bbeam22.x_mean))), label = "Xmean_b2 with %.2f"%yshift[i])
        ax1.set_xlim(-7.5,7.5)
        ax2.plot((np.fft.fftfreq(len(turns))-0.32)/bbp, (np.abs(np.fft.fft(monitor_bbeam2.y_mean))), label = "Ymean_b1 with %.2f"%yshift[i])
        ax2.plot((np.fft.fftfreq(len(turns))-0.32)/bbp, (np.abs(np.fft.fft(monitor_bbeam22.y_mean))), label = "Ymean_b2 with %.2f"%yshift[i])
        ax2.set_xlim(-7.5, 7.5)
ax1.set_title("X mean FFT")
ax1.set_xlabel("Frequency")
ax1.set_ylabel('Amplitude')
ax1.legend()

ax2.set_title("Y mean FFT")
ax2.set_xlabel("Frequency")
ax2.set_ylabel('Amplitude')
ax2.legend()

        
plt.show()
