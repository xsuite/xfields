import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, physical_constants
from scipy.signal import hilbert
from scipy.stats import linregress

import xtrack as xt
import xobjects as xo
import xpart as xp
import xfields as xf

longitudinal_mode = 'nonlinear'

# Simulation settings
n_turns = 10_000
n_turns_wake = 1
n_macroparticles = int(1e5)
num_slices = 100

circumference = 26658.8832
bucket_length_m = circumference / 35640

wake_table_filename = xf.general._pkg_root.joinpath(
    '../test_data/HLLHC_wake.dat')
wake_file_columns = ['time', 'longitudinal', 'dipole_x', 'dipole_y',
                     'quadrupole_x', 'quadrupole_y', 'dipole_xy',
                     'quadrupole_xy', 'dipole_yx', 'quadrupole_yx',
                     'constant_x', 'constant_y']
wf = xf.Wakefield.from_table(
    wake_table_filename, wake_file_columns,
    use_components=['dipole_x', 'dipole_y'],
    zeta_range=(-0.5*bucket_length_m, 0.5*bucket_length_m),
    num_slices=100,
    num_turns=1.,
    circumference=circumference
)

one_turn_map = xt.LineSegmentMap(
    length=circumference, betx=70., bety=80.,
    qx=62.31, qy=60.32,
    longitudinal_mode='linear_fixed_qs',
    dqx=-10., dqy=-10.,  # <-- to see fast mode-0 instability
    qs=2e-3, bets=731.27
)

# Generate line
line = xt.Line(elements=[one_turn_map, wf],
               element_names=['one_turn_map', 'wf'])


line.particle_ref = xt.Particles(p0c=7e12)
line.build_tracker()

# Generate particles
particles = xp.generate_matched_gaussian_bunch(line=line,
                    num_particles=100_000, total_intensity_particles=2.3e11,
                    nemitt_x=2e-6, nemitt_y=2e-6, sigma_z=0.075)

# Apply a distortion to the bunch to trigger an instability
amplitude = 1e-3
particles.x += amplitude
particles.y += amplitude

flag_plot = True

mean_x_xt = np.zeros(n_turns)
mean_y_xt = np.zeros(n_turns)

plt.ion()

fig1 = plt.figure(figsize=(6.4*1.7, 4.8))
ax_x = fig1.add_subplot(121)
line1_x, = ax_x.plot(mean_x_xt, 'r-', label='average x-position')
line2_x, = ax_x.plot(mean_x_xt, 'm-', label='exponential fit')
ax_x.set_ylim(-3.5, -1)
ax_x.set_xlim(0, n_turns)
ax_y = fig1.add_subplot(122, sharex=ax_x)
line1_y, = ax_y.plot(mean_y_xt, 'b-', label='average y-position')
line2_y, = ax_y.plot(mean_y_xt, 'c-', label='exponential fit')
ax_y.set_ylim(-3.5, -1)
ax_y.set_xlim(0, n_turns)

plt.xlabel('turn')
plt.ylabel('log10(average x-position)')
plt.legend()

turns = np.linspace(0, n_turns - 1, n_turns)

for i_turn in range(n_turns):
    line.track(particles, num_turns=1)

    mean_x_xt[i_turn] = np.mean(particles.x)
    mean_y_xt[i_turn] = np.mean(particles.y)

    if i_turn % 50 == 0:
        print(f'Turn: {i_turn}')

    if i_turn % 50 == 0 and i_turn > 1:
        i_fit_end = np.argmax(mean_x_xt)  # i_turn
        i_fit_start = int(i_fit_end * 0.9)

        # compute x instability growth rate
        ampls_x_xt = np.abs(hilbert(mean_x_xt))
        fit_x_xt = linregress(turns[i_fit_start: i_fit_end],
                              np.log(ampls_x_xt[i_fit_start: i_fit_end]))

        # compute y instability growth rate

        ampls_y_xt = np.abs(hilbert(mean_y_xt))
        fit_y_xt = linregress(turns[i_fit_start: i_fit_end],
                              np.log(ampls_y_xt[i_fit_start: i_fit_end]))

        line1_x.set_xdata(turns[:i_turn])
        line1_x.set_ydata(np.log10(np.abs(mean_x_xt[:i_turn])))
        line2_x.set_xdata(turns[:i_turn])
        line2_x.set_ydata(np.log10(np.exp(fit_x_xt.intercept +
                                          fit_x_xt.slope*turns[:i_turn])))

        line1_y.set_xdata(turns[:i_turn])
        line1_y.set_ydata(np.log10(np.abs(mean_y_xt[:i_turn])))
        line2_y.set_xdata(turns[:i_turn])
        line2_y.set_ydata(np.log10(np.exp(fit_y_xt.intercept +
                                          fit_y_xt.slope*turns[:i_turn])))
        print(f'xtrack h growth rate: {fit_x_xt.slope}')
        print(f'xtrack v growth rate: {fit_y_xt.slope}')

        fig1.canvas.draw()
        fig1.canvas.flush_events()

        out_folder = '.'
        np.save(f'{out_folder}/mean_x.npy', mean_x_xt)
        np.save(f'{out_folder}/mean_y.npy', mean_y_xt)
