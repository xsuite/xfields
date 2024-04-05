import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e, m_p

import xtrack as xt
import xfields as xf

# Filling scheme
n_slots = 1000
filling_scheme = np.array(np.floor(np.random.rand(n_slots)+0.1),dtype=int)
filling_scheme[0] = 1
filled_slots = np.nonzero(filling_scheme)[0]
n_bunches = len(filled_slots)
n_bunches_0 = int(np.floor(n_bunches/2))
bunch_numbers_0 = np.arange(n_bunches_0,dtype=int)
bunch_numbers_1 = np.arange(n_bunches_0,n_bunches,dtype=int)

print('initialising pipeline')
comm = xt.pipeline.core.PipelineCommunicator()
pipeline_manager = xt.PipelineManager(comm)
pipeline_manager.add_particles(f'b0',0)
pipeline_manager.add_particles(f'b1',0)
pipeline_manager.add_element('wake')

bunch_spacing = 25E-9*c
sigma_zeta = bunch_spacing/20

zeta_0 = []
for bunch_number in bunch_numbers_0:
    zeta_0.append(np.linspace(-sigma_zeta,sigma_zeta,1000)-filled_slots[bunch_number]*bunch_spacing)
zeta_0 = np.hstack(zeta_0)

ioffset = np.argmin(np.abs(zeta_0))

print('Initialising particles')
particles_0 = xt.Particles(p0c=7E12,zeta=zeta_0)
particles_0.init_pipeline('b0')
particles_0.x[ioffset] += 1.0
particles_0.y[ioffset] += 0.0

zeta_1 = []
for bunch_number in bunch_numbers_1:
    zeta_1.append(np.linspace(-sigma_zeta,sigma_zeta,1000)-filled_slots[bunch_number]*bunch_spacing)
zeta_1 = np.hstack(zeta_1)
particles_1 = xt.Particles(p0c=7E12,zeta=zeta_1)
particles_1.init_pipeline('b1')

print('Initialising wake')
n_slices = 100
n_turns_wake = 1
circumference = n_slots * bunch_spacing
wake_table_name = xf.general._pkg_root.joinpath('../test_data/HLLHC_wake.dat')
wake_file_columns = ['time', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y','dipole_xy','dipole_yx']
components = ['dipole_x']
wf_0 = xf.MultiWakefield.from_table(wake_table_name,wake_file_columns, use_components = components,
    zeta_range=(-1.1*sigma_zeta,1.1*sigma_zeta),
    num_slices=n_slices,  # per bunch
    bunch_spacing_zeta=bunch_spacing,
    filling_scheme=filling_scheme,
    bunch_numbers = bunch_numbers_0,
    num_turns=n_turns_wake,
    circumference=circumference,
)
wf_0.init_pipeline(pipeline_manager=pipeline_manager,element_name = 'wake', partners_names = ['b1'])
wf_1 = xf.MultiWakefield.from_table(wake_table_name,wake_file_columns, use_components = components,
    zeta_range=(-1.1*sigma_zeta,1.1*sigma_zeta),
    num_slices=n_slices,  # per bunch
    bunch_spacing_zeta=bunch_spacing,
    filling_scheme=filling_scheme,
    bunch_numbers = bunch_numbers_1,
    num_turns=n_turns_wake,
    circumference=circumference,
)
wf_1.init_pipeline(pipeline_manager=pipeline_manager,element_name = 'wake', partners_names = ['b0'])

print('Initialising lines')
line_0 = xt.Line(elements=[wf_0])
line_1 = xt.Line(elements=[wf_1])
print('Initialising multitracker')
line_0.build_tracker()
line_1.build_tracker()
multitracker = xt.PipelineMultiTracker(
    branches=[xt.PipelineBranch(line=line_0, particles=particles_0),
            xt.PipelineBranch(line=line_1, particles=particles_1),
            ])
print('Tracking')
pipeline_manager.verbose = True
multitracker.track(num_turns=1)
print('plotting')
plt.figure(0)
plt.plot(particles_0.zeta, particles_0.x, '.b')
plt.plot(particles_0.zeta, particles_0.y, '.g')
plt.plot(particles_1.zeta, particles_1.x, 'xb')
plt.plot(particles_1.zeta, particles_1.y, 'xg')
plt.figure(1)
plt.semilogy(particles_0.zeta, np.abs(particles_0.px), '.b')
plt.semilogy(particles_0.zeta, np.abs(particles_0.py), '.g')
plt.semilogy(particles_1.zeta, np.abs(particles_1.px), 'xb')
plt.semilogy(particles_1.zeta, np.abs(particles_1.py), 'xg')

for slot,filled in enumerate(filling_scheme):
    if filled:
        plt.figure(0)
        plt.axvline(-slot*bunch_spacing,color='k',ls='--')
        #plt.figure(1)
        #plt.axvline(-slot*bunch_spacing,color='k',ls='--')
print('done')
scaling_constant = -particles_0.q0**2 * e**2 / (particles_0.p0c[0] * e)
wake_data = np.loadtxt(wake_table_name)
for iwake_component,wake_component in enumerate(wake_file_columns):
    if wake_component in components:
        plt.figure(1)
        plt.semilogy(-1E-9*wake_data[:,0]*c,1E15*np.abs(scaling_constant*wake_data[:,iwake_component]),'-r',label=wake_component)
        plt.figure(1)
        interpolated_wake_0 = np.interp(particles_0.zeta,-1E-9*np.flip(wake_data[:,0])*c,-1E15*scaling_constant*np.flip(wake_data[:,iwake_component]))
        interpolated_wake_1 = np.interp(particles_1.zeta,-1E-9*np.flip(wake_data[:,0])*c,-1E15*scaling_constant*np.flip(wake_data[:,iwake_component]))
        plt.semilogy(particles_0.zeta,np.abs(interpolated_wake_0),'-xy',label=wake_component)
        plt.semilogy(particles_1.zeta,np.abs(interpolated_wake_1),'-oy',label=wake_component)
        plt.figure(10)
        norm = np.average(np.abs(interpolated_wake_0))
        plt.plot(particles_0.zeta,np.abs(particles_0.px-interpolated_wake_0)/norm,'xb')
        plt.plot(particles_1.zeta,np.abs(particles_1.px-interpolated_wake_1)/norm,'ob')
plt.show()


