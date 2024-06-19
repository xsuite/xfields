import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

import xtrack as xt
import xfields as xf


class MinistryOfSillyWakes:

    def __init__(self, factor=1.0):
        self.factor = factor

    def __call__(self, z):
        ret_val = np.copy(z)*self.factor
        ret_val[z > 0] = 0.0
        return ret_val


# Filling scheme
filling_scheme = np.zeros(10)
filling_scheme[0] = 1
filling_scheme[4] = 1
filling_scheme[-1] = 1
filled_slots = np.nonzero(filling_scheme)[0]
bunch_numbers_0 = np.array([0, 1], dtype=int)
bunch_numbers_1 = np.array([2], dtype=int)

print('initialising pipeline')
use_mpi_communicator = True
if use_mpi_communicator:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    if nprocs == 1:
        rank_for_b1 = 0
    else:
        rank_for_b1 = 1
    my_rank = comm.Get_rank()
    if my_rank > 1:
        print('aborting rank', my_rank)
        exit()
else:
    comm = xt.pipeline.core.PipelineCommunicator()
    rank_for_b1 = 0
    my_rank = 0

pipeline_manager = xt.PipelineManager(comm)
pipeline_manager.add_particles(f'b0', 0)
pipeline_manager.add_particles(f'b1', rank_for_b1)
pipeline_manager.add_element('wake')

bunch_spacing = 25E-9*c
sigma_zeta = bunch_spacing/20

zeta_0 = []
for bunch_number in bunch_numbers_0:
    zeta_0.append(np.linspace(-sigma_zeta, sigma_zeta, 1000) -
                  filled_slots[bunch_number]*bunch_spacing)
zeta_0 = np.hstack(zeta_0)

ioffset = np.argmin(np.abs(zeta_0))

print('Initialising particles')
particles_0 = xt.Particles(p0c=7E12, zeta=zeta_0)
particles_0.init_pipeline('b0')
particles_0.x[ioffset] += 1.0
particles_0.y[ioffset] += 1.0

zeta_1 = []
for bunch_number in bunch_numbers_1:
    zeta_1.append(np.linspace(-sigma_zeta, sigma_zeta, 1000) -
                  filled_slots[bunch_number]*bunch_spacing)
zeta_1 = np.hstack(zeta_1)
particles_1 = xt.Particles(p0c=7E12, zeta=zeta_1)
particles_1.init_pipeline('b1')

print('Initialising wake')
n_slices = 100
n_turns_wake = 1
circumference = 27E3
wfx_0 = xf.WakeComponent(
    source_moments=['num_particles', 'x'],
    kick='px',
    scale_kick=None,
    function=MinistryOfSillyWakes(1.0),
    zeta_range=(-1.1*sigma_zeta, 1.1*sigma_zeta),
    num_slices=n_slices,
    bunch_spacing_zeta=bunch_spacing,
    filling_scheme=filling_scheme,
    bunch_numbers=bunch_numbers_0,
    num_turns=n_turns_wake,
    circumference=circumference,
)
wfx_0.init_pipeline(pipeline_manager=pipeline_manager,
                    element_name='wake', partners_names=['b1'])
wfx_1 = xf.WakeComponent(
    source_moments=['num_particles', 'x'],
    kick='px',
    scale_kick=None,
    function=MinistryOfSillyWakes(1.0),
    zeta_range=(-1.1*sigma_zeta, 1.1*sigma_zeta),
    num_slices=n_slices,  # per bunch
    bunch_spacing_zeta=bunch_spacing,
    filling_scheme=filling_scheme,
    bunch_numbers=bunch_numbers_1,
    num_turns=n_turns_wake,
    circumference=circumference,
)
wfx_1.init_pipeline(pipeline_manager=pipeline_manager, element_name='wake',
                    partners_names=['b0'])


print('Initialising lines')
line_0 = xt.Line(elements=[wfx_0])
line_1 = xt.Line(elements=[wfx_1])
print('Initialising multitracker')
line_0.build_tracker()
line_1.build_tracker()
multitracker = xt.PipelineMultiTracker(
    branches=[xt.PipelineBranch(line=line_0, particles=particles_0),
              xt.PipelineBranch(line=line_1, particles=particles_1),
              ])

print('Tracking')

multitracker.track(num_turns=1)
print('plotting')
plt.figure(0)
plt.plot(particles_0.zeta, particles_0.x, '.b')
plt.plot(particles_0.zeta, particles_0.y, '.g')
plt.plot(particles_1.zeta, particles_1.x, 'xb')
plt.plot(particles_1.zeta, particles_1.y, 'xg')
plt.figure(1)
plt.plot(particles_0.zeta, particles_0.px, '.b')
plt.plot(particles_0.zeta, particles_0.py, '.g')
plt.plot(particles_1.zeta, particles_1.px, 'xb')
plt.plot(particles_1.zeta, particles_1.py, 'xg')


for slot, filled in enumerate(filling_scheme):
    if filled:
        plt.figure(0)
        plt.axvline(-slot*bunch_spacing, color='k', ls='--')
        plt.figure(1)
        plt.axvline(-slot*bunch_spacing, color='k', ls='--')
print('done')
plt.show()
