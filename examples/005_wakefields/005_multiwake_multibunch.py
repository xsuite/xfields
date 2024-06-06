import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e, m_p

import xtrack as xt
import xfields as xf

class MinistryOfSillyWakes:

    def __init__(self,factor=1.0):
        self.factor = factor

    def __call__(self, z):
        retVal = np.copy(z)*self.factor
        retVal[z>0] = 0.0
        return retVal

# Filling scheme
filling_scheme = np.zeros(10)
filling_scheme[0] = 1
filling_scheme[4] = 1
filling_scheme[-1] = 1
filled_slots = np.nonzero(filling_scheme)[0]

bunch_spacing = 25E-9*c
sigma_zeta = bunch_spacing/20

zeta = []
for filled_slot in filled_slots:
    zeta.append(np.linspace(-sigma_zeta,sigma_zeta,1000)-filled_slot*bunch_spacing)
zeta = np.hstack(zeta)

ioffset = np.argmin(np.abs(zeta))

print('Initialising particles')
particles = xt.Particles(p0c=7E12,zeta=zeta)
particles.x[ioffset] += 1.0
particles.y[ioffset] += 1.0

print('Initialising wakes')
wfx = xf.WakeComponent(
    source_moments=['num_particles', 'x'],
    kick='px',
    scale_kick=None,
    function=MinistryOfSillyWakes(1.0)
)

wfy = xf.WakeComponent(
    source_moments=['num_particles', 'y'],
    kick='py',
    scale_kick=None,
    function=MinistryOfSillyWakes(-1.0)
)

n_slices = 1000
n_turns_wake = 1
circumference = 27E3
wf = xf.Wakefield(
    components=[wfx, wfy],
    zeta_range=(-1.1*sigma_zeta,1.1*sigma_zeta),
    num_slices=n_slices,  # per bunch
    bunch_spacing_zeta=bunch_spacing,
    filling_scheme=filling_scheme,
    bunch_numbers=np.arange(len(filled_slots)),
    num_turns=n_turns_wake,
    circumference=circumference,
)

print('Initialising line')
line = xt.Line(elements=[wf],
               element_names=['wf'])
print('Initialising tracker')
line.build_tracker()
print('Tracking')
line.track(particles, num_turns=1)
print('plotting')
plt.figure(0)
plt.plot(particles.zeta, particles.x, '.b')
plt.plot(particles.zeta, particles.y, '.g')
plt.figure(1)
plt.plot(particles.zeta, particles.px, '.b')
plt.plot(particles.zeta, particles.py, '.g')


for slot,filled in enumerate(filling_scheme):
    if filled:
        plt.figure(0)
        plt.axvline(-slot*bunch_spacing,color='k',ls='--')
        plt.figure(1)
        plt.axvline(-slot*bunch_spacing,color='k',ls='--')
print('done')
plt.show()


