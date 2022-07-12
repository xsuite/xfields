import mpi4py
mpi4py.rc.threads = False
from mpi4py import MPI
import numpy as np
from scipy import constants as cst
from matplotlib import pyplot as plt

print('Thread level:',MPI.Query_thread(),f'({MPI.THREAD_SINGLE},{MPI.THREAD_FUNNELED},{MPI.THREAD_SERIALIZED},{MPI.THREAD_MULTIPLE})')

import xobjects as xo
import xtrack as xt
import xpart as xp
from xfields.beam_elements.beambeam import BeamBeamBiGaussian2D
from xtrack.pipeline import PipelineManager

context = xo.ContextCpu(omp_num_threads=0)

proton_mass = cst.value('proton mass energy equivalent in MeV')*1E6
n_turn = int(1E4)
n_macroparticles = int(1E4)
bunch_intensity = 2E11
gamma = 7E12/proton_mass
betar = np.sqrt(1-1/gamma**2)
epsn_x = 2E-6
epsn_y = 2E-6
betastar_x = 1.0
betastar_y = 1.0
sigma_z = 0.08
sigma_delta = 1E-4
Q_x = 0.31
Q_y = 0.32
beta_s = sigma_z/sigma_delta
Q_s = 1E-3

particles = xp.Particles(_context=context,
                     #particlenumber_per_mp=bunch_intensity/n_macroparticles,
                     q0 = 1,
                     mass0 = proton_mass,
                     gamma0 = gamma,
                     x=np.sqrt(epsn_x*betastar_x/gamma)*(np.random.randn(n_macroparticles)),
                     px=np.sqrt(epsn_x/betastar_x/gamma)*np.random.randn(n_macroparticles),
                     y=np.sqrt(epsn_y*betastar_y/gamma)*(np.random.randn(n_macroparticles)),
                     py=np.sqrt(epsn_y/betastar_y/gamma)*np.random.randn(n_macroparticles),
                     zeta=sigma_z*np.random.randn(n_macroparticles),
                     delta=sigma_delta*np.random.randn(n_macroparticles),
                     )

arc12 = xt.LinearTransferMatrix(alpha_x_0 = 0.0, beta_x_0 = betastar_x, disp_x_0 = 0.0,
                       alpha_x_1 = 0.0, beta_x_1 = betastar_x, disp_x_1 = 0.0,
                       alpha_y_0 = 0.0, beta_y_0 = betastar_y, disp_y_0 = 0.0,
                       alpha_y_1 = 0.0, beta_y_1 = betastar_y, disp_y_1 = 0.0,
                       Q_x = Q_x/2, Q_y = Q_y/2,
                       beta_s = beta_s, Q_s = -Q_s/2,
                       energy_ref_increment=0.0,energy_increment=0)

arc21 = xt.LinearTransferMatrix(alpha_x_0 = 0.0, beta_x_0 = betastar_x, disp_x_0 = 0.0,
                       alpha_x_1 = 0.0, beta_x_1 = betastar_x, disp_x_1 = 0.0,
                       alpha_y_0 = 0.0, beta_y_0 = betastar_y, disp_y_0 = 0.0,
                       alpha_y_1 = 0.0, beta_y_1 = betastar_y, disp_y_1 = 0.0,
                       Q_x = Q_x/2, Q_y = Q_y/2,
                       beta_s = beta_s, Q_s = -Q_s/2,
                       energy_ref_increment=0.0,energy_increment=0)

beamBeam_IP1 = BeamBeamBiGaussian2D(_context=context,min_sigma_diff=1e-10,q0=1,beta0=betar,
                                    update_on_track = True)
beamBeam_IP2 = BeamBeamBiGaussian2D(_context=context,min_sigma_diff=1e-10,q0=1,beta0=betar,
                                    update_on_track = True)

####### Setting up pipeline #######
my_rank = MPI.COMM_WORLD.Get_rank()
pipeline_manager = PipelineManager(communicator = MPI.COMM_WORLD)
if my_rank == 0:
    name = 'B1b1'
    partner_name = 'B2b1'
else:
    name = 'B2b1'
    partner_name = 'B1b1'
particles.init_pipeline(name=name)
beamBeam_IP1.init_pipeline(pipeline_manager=pipeline_manager,name = 'BBIP1',partners_names=[partner_name])
beamBeam_IP2.init_pipeline(pipeline_manager=pipeline_manager,name = 'BBIP2',partners_names=[partner_name])
pipeline_manager.add_particles('B1b1',0)
pipeline_manager.add_particles('B2b1',1)
pipeline_manager.add_element(beamBeam_IP1.name)
pipeline_manager.add_element(beamBeam_IP2.name)
####################################

tracker = xt.Tracker(
    line=xt.Line(elements=[beamBeam_IP1,
                           arc12,
                           beamBeam_IP2,
                           arc21]),
                           enable_pipeline_hold=True)

positions = np.zeros(n_turn,dtype=float)
for turn in range(n_turn):
    print(f'Rank {my_rank}, start tracking at turn {turn}')
    session = tracker.track(particles, num_turns=1)
    while session is not None and session.on_hold:
       print(f'Rank {my_rank}, is on hold at turn {turn}')
       session = tracker.resume(session)
    positions[turn] = np.average(particles.x)

myFFT = np.fft.fftshift(np.log10(np.abs(np.fft.fft(positions))))
freqs = np.fft.fftshift(np.fft.fftfreq(len(positions)))
plt.plot(freqs,myFFT)
plt.xlim([0.27,0.34])
plt.show()



