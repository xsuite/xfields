

#L0 with combi formula

import time

import numpy as np
from matplotlib import pyplot as plt
import xobjects as xo
import xtrack as xt
import xfieldsdev as xf
import xpart as xp

###################
#To run this code:
#mpirun -np 2 python xsuite_dev/xtrack/examples/L0beambeam.py
lumi_qss_b1 = []
lumi_averages_b1 = []

lumi_qss_b1_nobeambeam = []
lumi_averages_b1_nobeambeam = []


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
nTurn = 100


colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y']
xshift = [0, 1, 2, 3, 4, 5, 6]
yshift = [0, 1, 2, 3, 4, 5, 6]

for shift in xshift:
    context = xo.ContextCpu(omp_num_threads=0)

    pipeline_manager = xt.PipelineManager()
    pipeline_manager.add_particles('b1',0)
    pipeline_manager.add_particles('b2',0)
    pipeline_manager.add_element('IP1')

    particles_b1 = xp.Particles(_context=context,
        p0c=p0c,
        x=np.sqrt(physemit_x*beta_x_IP1)*(np.random.randn(n_macroparticles)),
        px=np.sqrt(physemit_x/beta_x_IP1)*np.random.randn(n_macroparticles),
        y=np.sqrt(physemit_y*beta_y_IP1)*(np.random.randn(n_macroparticles)),
        py=np.sqrt(physemit_y/beta_y_IP1)*np.random.randn(n_macroparticles),
        zeta=sigma_z*np.random.randn(n_macroparticles),
        delta=sigma_delta*np.random.randn(n_macroparticles),
        weight=bunch_intensity/n_macroparticles
    )
    particles_b1.init_pipeline('b1')
    particles_b2 = xp.Particles(_context=context,
        p0c=p0c,
        x=np.sqrt(physemit_x*beta_x_IP1)*(np.random.randn(n_macroparticles)),
        px=np.sqrt(physemit_x/beta_x_IP1)*np.random.randn(n_macroparticles),
        y=np.sqrt(physemit_y*beta_y_IP1)*(np.random.randn(n_macroparticles)),
        py=np.sqrt(physemit_y/beta_y_IP1)*np.random.randn(n_macroparticles),
        zeta=sigma_z*np.random.randn(n_macroparticles),
        delta=sigma_delta*np.random.randn(n_macroparticles),
        weight=bunch_intensity/n_macroparticles
    )
    particles_b2.init_pipeline('b2')

    #############
    # Beam-beam #
    #############
    slicer = xf.TempSlicer(sigma_z=sigma_z, n_slices=1, mode = 'shatilov')
    config_for_update_b1_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
    pipeline_manager=pipeline_manager,
    element_name='IP1',
    partner_particles_name = 'b2',
    slicer=slicer,
    update_every=1,
    n_lumigrid_cells = 24*24
    )
    config_for_update_b2_IP1=xf.ConfigForUpdateBeamBeamBiGaussian3D(
    pipeline_manager=pipeline_manager,
    element_name='IP1',
    partner_particles_name = 'b1',
    slicer=slicer,
    update_every=1,
    n_lumigrid_cells=24*24
    )

    print('build bb elements...')
    bbeamIP1_b1 = xf.BeamBeamBiGaussian3D(
                _context=context,
                other_beam_q0 = particles_b2.q0,
                phi = 0,alpha=0,
                config_for_update = config_for_update_b1_IP1,
                ref_shift_x = shift*np.sqrt(physemit_x*beta_x_IP1)/2,
                flag_luminosity=1,
                flag_combilumi=1,
                beam_intensity=bunch_intensity,
                other_beam_intensity=bunch_intensity)
    bbeamIP1_b2 = xf.BeamBeamBiGaussian3D(
                _context=context,
                other_beam_q0 = particles_b1.q0,
                phi = 0,alpha=0,
                config_for_update = config_for_update_b2_IP1,
                ref_shift_x = shift*np.sqrt(physemit_x*beta_x_IP1)/2)



    #################################################################
    # arcs (here they are all the same with half the phase advance) #
    #################################################################

    arc = xt.LineSegmentMap(
            betx = beta_x_IP1,bety = beta_y_IP1,
            qx = Qx, qy = Qy,bets = beta_s, qs=Qs)

    #################################################################
    # Tracker                                                       #
    #################################################################

    elements_b1 = [bbeamIP1_b1,arc]
    elements_b2 = [bbeamIP1_b2,arc]
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

    record_qss_b1 = line_b1.start_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D, 
                                                            capacity={
                                                                "beamstrahlungtable": int(0),
                                                                "bhabhatable": int(0),
                                                                "lumitable": nTurn,
                                                                "combilumitable": nTurn
                                                            })


    print('Tracking...')
    time0 = time.time()

    multitracker.track(num_turns=nTurn,turn_by_turn_monitor=True)
    print('Done with tracking.',(time.time()-time0)/16384,'[s/turn]')
    line_b1.stop_internal_logging_for_elements_of_type(xf.BeamBeamBiGaussian3D)

    record_qss_b1.move(_context=xo.context_default)
 
    lumi_b1_nobeambeam = record_qss_b1.lumitable.luminosity
    
    lumi_qss_b1_nobeambeam.append(lumi_b1_nobeambeam)

    lumi_averages_b1_nobeambeam.append(np.mean(lumi_b1_nobeambeam))


lumis = []
separation = [0, 1, 2, 3, 4, 5, 6]

def Lumi_analytical(Nb, N1, N2, frev, Delta_i, sig_i, sig_x, sig_y):
    W = np.exp(-Delta_i**2/(4*sig_i**2))
    return ((Nb * N1 * N2 * frev * W)/(4 * np.pi * 100 * sig_x * 100 * sig_y))

for i in range(len(separation)):
    lumis.append(Lumi_analytical(n_macroparticles, bunch_intensity, bunch_intensity, frev, separation[i]*np.sqrt(physemit_x*beta_x_IP1),np.sqrt(physemit_x*beta_x_IP1), np.sqrt(physemit_x*beta_x_IP1), np.sqrt(physemit_y*beta_x_IP1)))

fig0, ax1 = plt.subplots()
fig1, ax2 = plt.subplots()

ax1.set_title("Luminosity as a function of beam separation")
ax1.set_xlabel("Separation (sigma)")
ax1.set_ylabel("Luminosity")
ax1.plot(separation, lumis, label = "Analytical")
ax1.plot(xshift, (frev*np.array(lumi_averages_b1)), label = "With beam beam") #Here i am confused- this is in m^-2 or cm^-2
ax1.plot(xshift, (frev*np.array(lumi_averages_b1_nobeambeam)), label = "Without beam beam")
ax1.legend()


ax2.set_title("L/L0")
ax2.plot(xshift, np.array(lumi_averages_b1)/np.array(lumi_averages_b1_nobeambeam))
plt.show()