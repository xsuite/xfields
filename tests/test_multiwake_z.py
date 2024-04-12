import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e, m_p

import xtrack as xt
import xfields as xf

# Filling scheme
n_slots = 1
filling_scheme = np.array(np.floor(np.random.rand(n_slots)+0.1),dtype=int)
filling_scheme[0] = 1
filled_slots = np.nonzero(filling_scheme)[0]
n_bunches = len(filled_slots)
bunch_numbers_0 = np.arange(n_bunches,dtype=int)

print('initialising pipeline')
comm = xt.pipeline.core.PipelineCommunicator()
pipeline_manager = xt.PipelineManager(comm)
pipeline_manager.add_particles(f'b0',0)
pipeline_manager.add_particles(f'b1',0)
pipeline_manager.add_element('wake')

bunch_spacing = 25E-9*c
sigma_zeta = bunch_spacing/20
zeta_range = (-1.1*sigma_zeta,1.1*sigma_zeta)
n_slices = 1001

zeta_0 = []
for bunch_number in bunch_numbers_0:
    zeta_0.append(np.linspace(zeta_range[0],zeta_range[1],n_slices)-filled_slots[bunch_number]*bunch_spacing)
zeta_0 = np.hstack(zeta_0)

ioffset = np.argmin(np.abs(zeta_0))
print('Initialising particles')
particles_0 = xt.Particles(p0c=7E12,zeta=zeta_0)
particles_0.init_pipeline('b0')
particles_0.x[ioffset] += 1.0
particles_0.y[ioffset] += 0.0

print('Initialising wake')
n_turns_wake = 1
circumference = n_slots * bunch_spacing
wake_table_name = xf.general._pkg_root.joinpath('../test_data/HLLHC_wake.dat')
wake_file_columns = ['time', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y','dipole_xy','dipole_yx']
components = ['dipole_x']
wf_0 = xf.MultiWakefield.from_table(wake_table_name,wake_file_columns, use_components = components,
    zeta_range=zeta_range,
    num_slices=n_slices,  # per bunch
    bunch_spacing_zeta=bunch_spacing,
    filling_scheme=filling_scheme,
    bunch_numbers = bunch_numbers_0,
    num_turns=n_turns_wake,
    circumference=circumference,
)
print('Initialising lines')
line_0 = xt.Line(elements=[wf_0])
print('Initialising multitracker')
line_0.build_tracker()
print('Tracking')
line_0.track(particles_0,num_turns=1)
print('plotting')
plt.figure(3)
plt.plot(particles_0.zeta, particles_0.px, '.b')
wake = wf_0.wakefields[0]
for i in range(len(wake.slicer.zeta_centers)):
    plt.axvline(wake.slicer.zeta_centers[i],color='k',ls=':')
    if i > 1:
        plt.axvline(0.5*(wake.slicer.zeta_centers[i-1]+wake.slicer.zeta_centers[i]),color='k',ls='-')
scaling_constant = -particles_0.q0**2 * e**2 / (particles_0.p0c[0] * e)
wake_data = np.loadtxt(wake_table_name)
for iwake_component,wake_component in enumerate(wake_file_columns):
    if wake_component in components:
        plt.figure(3)
        plt.plot(wake.z_wake[0],scaling_constant*wake.G_aux[0],'-+y')
        plt.plot(-1E-9*wake_data[:,0]*c,-1E15*scaling_constant*wake_data[:,iwake_component],'-r',label=wake_component)
        interpolated_wake_0 = np.interp(particles_0.zeta,-1E-9*np.flip(wake_data[:,0])*c,-1E15*scaling_constant*np.flip(wake_data[:,iwake_component]))
        plt.plot(particles_0.zeta,interpolated_wake_0,'-xy',label=wake_component)
plt.xlim([-0.01,0.001])
plt.show()


