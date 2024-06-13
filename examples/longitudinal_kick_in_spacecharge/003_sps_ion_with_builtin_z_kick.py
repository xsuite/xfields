"""
Example with SPS Pb ions to install frozen space charge elements that also apply longitudinal kicks
"""
import numpy as np
import time
import xtrack as xt
import xpart as xp
import xobjects as xo
import xfields as xf
import matplotlib.pyplot as plt

test_on_gpu = False

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
if test_on_gpu:
    context = xo.ContextCupy()
else:
    context = xo.ContextCpu(omp_num_threads='auto')

num_turns = 5000
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
p0 = xp.build_particles(line = line, particle_ref = line.particle_ref,
                            x_norm=0.1, y_norm=0.1, delta=0.0, zeta=zetas,
                            nemitt_x = nemitt_x, nemitt_y = nemitt_y, _context=context) # default transverse amplitude is 0.1 sigmas


# Install frozen space charge, emulating a Gaussian bunch
lprofile = xf.LongitudinalProfileQGaussian(
        number_of_particles = Nb,
        sigma_z = sigma_z,
        z0=0.,
        q_parameter=1.0)

##### Install frozen space charge, but directly configuring SC elements with z kick
xf.install_spacecharge_frozen(line = line,
                   particle_ref = line.particle_ref,
                   longitudinal_profile = lprofile,
                   nemitt_x = nemitt_x, nemitt_y = nemitt_y,
                   sigma_z = sigma_z,
                   num_spacecharge_interactions = num_spacecharge_interactions,
                   z_kick_num_integ_per_sigma=5)

line.build_tracker(_context = context)
line.enable_time_dependent_vars = True


line.track(p0.copy(), num_turns=num_turns, with_progress=True,
              log=xt.Log(zeta=lambda l, p: p.zeta.copy()))
log_with_builtin_kick = line.log_last_track

zeta_with_builtin_kick = np.stack(log_with_builtin_kick['zeta'])

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
i_part_plot = -1
plt.plot(zeta_with_builtin_kick[:, i_part_plot], label='With built-in z kick')
z0 = p0.zeta[i_part_plot]
plt.ylim([z0*0.99, z0*1.05])
plt.xlabel('Turns')
plt.ylabel(r'zeta [m]')
plt.grid()
plt.legend()
plt.show()