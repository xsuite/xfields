"""
Example to generate five SPS Pb particles, spaced out in initial longitudinal zeta, while changing synchrotron tune 
"""
import numpy as np
import time
import xtrack as xt
import xpart as xp
import xobjects as xo
import xfields as xf
import matplotlib.pyplot as plt


def change_synchrotron_tune_by_factor(A, line, sigma_z, Nb):
    """
    Scale synchrotron tune Qs while keeping bucket half-height delta constant, also adjusting
    bunch length and bunch intensity accordingly for identical bunch filling factor and space charge effects

    Parameters
    ----------
    line : xtrack.Line
        line used in tracking
    A : float
        factor by which to scale the synchrotron tune
    sigma_z : float
        original bunch length
    Nb : float
        original bunch intensity

    Returns
    -------
    line_new : xtrack.Line
        line with updated RF voltage and harmonic
    sigma_z_new : float
        updated new bunch length
    Nb_new : float
        updated new bunch intensity
    """

    # Find RF cavity number 
    nn = 'actcse.31632' # 'actcse.31637' for protons
    
    line[nn].voltage *= A # scale voltage by desired factor
    line[nn].frequency *= A # in reality scale harmonic number, but translates directly to frequency
    sigma_z_new = sigma_z / A  # adjust bunch length such that space charge effects remain the same
    Nb_new = Nb / A # adjust bunch intensity such that space charge effects remain the same
    
    return line, sigma_z_new, Nb_new


# Initialize chosen context, number of turns, particles and space charge interactions
context = xo.ContextCpu(omp_num_threads='auto')
num_turns = 10_000
number_of_particles = 5
num_spacecharge_interactions = 1080
scale_factor_Qs = 2.0  # by how many times to scale the nominal synchrotron tune

# Beam parameters, will be used for space charge
Nb = 2.46e8 # bunch_intensity measured 2.46e8 Pb ions per bunch on 2023-10-16
sigma_z =  0.225
nemitt_x = 1.3e-6
nemitt_y = 0.9e-6

# Import line
line = xt.Line.from_json('sps_ion_2021.json')
line, sigma_z, Nb = change_synchrotron_tune_by_factor(scale_factor_Qs, line, sigma_z, Nb) # update synchrotron tune, scale bucket length and SC parameters
twiss = line.twiss()
print('\nUpdated parameters to sigma_z = {:.4f} and Nb = {:.3e}'.format(sigma_z, Nb))
print('New Qs = {:.6f} when Qs changed by factor {}\n'.format(twiss['qs'], scale_factor_Qs))

# Generate particles spread out in lognitudinal space make linear spacing between close to center of RF bucket and to separatrix
zetas = np.linspace(0.05, 0.7 / scale_factor_Qs, num=number_of_particles)
particles = xp.build_particles(line = line, particle_ref = line.particle_ref,
                            x_norm=0.1, y_norm=0.1, delta=0.0, zeta=zetas,
                            nemitt_x = nemitt_x, nemitt_y = nemitt_y, _context=context) # default transverse amplitude is 0.1 sigmas

# Install frozen space charge, emulating a Gaussian bunch
lprofile = xf.LongitudinalProfileQGaussian(
        number_of_particles = Nb,
        sigma_z = sigma_z,
        z0=0.,
        q_parameter=1.0)

# Install frozen space charge as base 
xf.install_spacecharge_frozen(line = line,
                   particle_ref = line.particle_ref,
                   longitudinal_profile = lprofile,
                   nemitt_x = nemitt_x, nemitt_y = nemitt_y,
                   sigma_z = sigma_z,
                   num_spacecharge_interactions = num_spacecharge_interactions)

line.build_tracker(_context = context)

tt = line.get_table()
tt_sc = tt.rows[tt.element_type=='SpaceChargeBiGaussian']
for nn in tt_sc.name:
    line[nn].z_kick_num_integ_per_sigma = 5

# Start dictionary
zeta_vals = np.zeros([len(particles.zeta), num_turns])
delta_vals = np.zeros([len(particles.delta), num_turns])
zeta_vals[:, 0] = particles.zeta
delta_vals[:, 0] = particles.delta

# Track particles
time00 = time.time()
for turn in range(1, num_turns):
    if turn % 10 == 0:
        print(f'Tracking turn {turn}')
    line.track(particles)
    zeta_vals[:, turn] = particles.zeta
    delta_vals[:, turn] = particles.delta
time01 = time.time()
dt0 = time01-time00
print('\nTracking time: {:.1f} s = {:.1f} min'.format(dt0, dt0/60))

# Plot relative zeta values
fig, ax = plt.subplots(1, 1, figsize = (8, 4.5))
turns = np.arange(len(zeta_vals[0]))
i = -2 # index of particle to plot - select penultimate
ax.plot(turns, zeta_vals[i, :] / zeta_vals[i, 0], alpha=0.8, label='Qs changed by factor {}'.format(scale_factor_Qs))
ax.set_xlabel('Turns')
ax.set_ylabel(r'$\zeta$ / $\zeta_{0}$')
ax.set_ylim(0.95, 1.05)
ax.legend(loc='upper left', fontsize=10)
plt.tight_layout()

# Plot particle evolution in longitudinal phase space
fig2, ax2 = plt.subplots(1, 1, figsize = (8, 4.5))
for i in range(number_of_particles):
    ax2.scatter(zeta_vals[i, :], delta_vals[i, :] * 1e3, c=range(num_turns), marker='.')
ax2.set_xlabel(r'$\zeta$ [m]')
ax2.set_ylabel(r'$\delta$ [1e-3]')

# Adding color bar for the number of turns
cbar = plt.colorbar(ax2.collections[0], ax=ax2)
cbar.set_label('Number of Turns')
plt.tight_layout()

plt.show()