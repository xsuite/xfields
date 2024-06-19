import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, physical_constants
from scipy.signal import hilbert
from scipy.stats import linregress

import xtrack as xt
import xobjects as xo
import xpart as xp
import xfields as xf

context = xo.ContextCpu(omp_num_threads=0)


def _anharmonicities_from_octupole_current_settings(i_octupole_focusing,
                                                    i_octupole_defocusing):
    """Calculate the constants of proportionality app_x, app_y and
    app_xy (== app_yx) for the amplitude detuning introduced by the
    LHC octupole magnets (aka. LHC Landau octupoles) from the
    electric currents i_octupole_focusing [A] and i_octupole_defocusing [A]
    flowing through the magnets. The maximum current is given by
    i_max = +/- 550 [A]. The values app_x, app_y, app_xy obtained
    from the formulae are proportional to the strength of detuning
    for one complete turn around the accelerator, i.e. one-turn
    values.
    The calculation is based on formulae (3.6) taken from 'The LHC
    transverse coupled-bunch instability' by N. Mounet, EPFL PhD
    Thesis, 2012. Values (hard-coded numbers below) are valid for
    LHC Landau octupoles before LS1. Beta functions in x and y are
    correctly taken into account. Note that here, the values of
    app_x, app_y and app_xy are not normalized to the reference
    momentum p0. This is done only during the calculation of the
    detuning in the corresponding detune method of the
    AmplitudeDetuningSegment.
    More detailed explanations and references on how the formulae
    were obtained are given in the PhD thesis (pg. 85ff) cited
    above.

    Acknowledgments: copied from PyHEADTAIL
    """
    i_max = 550.  # [A]
    e_max = 7000.  # [GeV]

    app_x = e_max * (267065. * i_octupole_focusing / i_max -
                     7856. * i_octupole_defocusing / i_max)
    app_y = e_max * (9789. * i_octupole_focusing / i_max -
                     277203. * i_octupole_defocusing / i_max)
    app_xy = e_max * (-102261. * i_octupole_focusing / i_max +
                      93331. * i_octupole_defocusing / i_max)

    # Convert to SI units.
    convert_to_si = e / (1.e-9 * c)
    app_x *= convert_to_si
    app_y *= convert_to_si
    app_xy *= convert_to_si

    return app_x, app_y, app_xy

longitudinal_mode = 'nonlinear'

# Simulation settings
n_turns = 10**4
n_turns_wake = 1
flatten = False
flag_plot = True
n_macroparticles = 10**5
num_slices = 100

# Beam settings
intensity = 2.3e11
E0 = physical_constants['proton mass energy equivalent in MeV'][0]*1e6
p0c = 7000e9
p0 = p0c * e / c
gamma = p0c/E0
beta = np.sqrt(1-1/gamma**2)

# Machine settings
h_RF = 35640
bunch_spacing_buckets = 10
accQ_x = 60.31
accQ_y = 60.32
chroma = -10  # we choose negative chromaticity to see a fast instability
epsn_x = 2e-6
epsn_y = 2e-6
taub = 1e-9
sigma_z = taub*beta*c/4
circumference = 26658.883
averageRadius = circumference / (2 * np.pi)
alpha = 53.86**-2
momentumCompaction = alpha
eta = momentumCompaction - 1.0 / gamma ** 2
V = 16e6
Q_s = np.sqrt((e*V*h_RF*eta)/(2*np.pi*beta*c*p0))
sigma_delta = Q_s * sigma_z / (averageRadius * eta)
beta_s = sigma_z / sigma_delta
f_rev = beta*c/circumference
f_rf = f_rev*h_RF
beta_x = averageRadius/accQ_x
beta_y = averageRadius/accQ_y
bucket_length = circumference / h_RF

zeta_range = (-0.5*bucket_length, 0.5*bucket_length)


i_oct = 0

app_x, app_y, app_xy = _anharmonicities_from_octupole_current_settings(
    i_octupole_focusing=i_oct, i_octupole_defocusing=i_oct,
)

# Initialise wakes
n_bunches_wake = 120  # Can be longer than filling scheme

wake_table_name = xf.general._pkg_root.joinpath(
    '../test_data/HLLHC_wake.dat')
wake_file_columns = ['time', 'longitudinal', 'dipole_x', 'dipole_y',
                     'quadrupole_x', 'quadrupole_y', 'dipole_xy',
                     'quadrupole_xy', 'dipole_yx', 'quadrupole_yx',
                     'constant_x', 'constant_y']
components = ['dipole_x', 'dipole_y',
              'quadrupole_x', 'quadrupole_y', 'dipole_xy',
              'quadrupole_xy', 'dipole_yx', 'quadrupole_yx',
              'constant_x', 'constant_y']
wf = xf.Wakefield.from_table(
    wake_table_name, wake_file_columns,
    use_components=components,
    zeta_range=zeta_range,
    num_slices=num_slices,  # per bunch
    bunch_spacing_zeta=bucket_length*bunch_spacing_buckets,
    num_turns=n_turns_wake,
    circumference=circumference
)

if longitudinal_mode == 'linear':
    betatron_map = xt.LineSegmentMap(
        length=circumference, betx=beta_x, bety=beta_y,
        qx=accQ_x, qy=accQ_y,
        longitudinal_mode='linear_fixed_qs',
        dqx=chroma, dqy=chroma,
        det_xx=app_x/p0, det_yy=app_y/p0, det_xy=app_xy/p0, det_yx=app_xy/p0,
        qs=Q_s, bets=beta_s
    )
elif longitudinal_mode == 'nonlinear':
    betatron_map = xt.LineSegmentMap(
        length=circumference, betx=beta_x, bety=beta_y,
        qx=accQ_x, qy=accQ_y,
        longitudinal_mode='nonlinear',
        dqx=chroma, dqy=chroma,
        voltage_rf=V, frequency_rf=f_rf,
        det_xx=app_x/p0, det_yy=app_y/p0, det_xy=app_xy/p0, det_yx=app_xy/p0,
        lag_rf=180, momentum_compaction_factor=momentumCompaction
    )
else:
    raise ValueError("longitudinal_mode must be 'linear' or 'nonlinear'")


# Generate line
line = xt.Line(elements=[betatron_map, wf],
               element_names=['betatron_map', 'wf'])


line.particle_ref = xt.Particles(p0c=p0c)
line.build_tracker(_context=context)

# Generate particles
particles = xp.generate_matched_gaussian_bunch(
         num_particles=n_macroparticles, total_intensity_particles=intensity,
         nemitt_x=epsn_x, nemitt_y=epsn_y, sigma_z=sigma_z,
         line=line, _context=context
)

# Apply a distortion to the bunch to trigger an instability
amplitude = 1e-3
particles.x += amplitude
particles.y += amplitude

flag_plot = True

mean_x_xt = np.zeros(n_turns)
mean_y_xt = np.zeros(n_turns)

plt.ion()

fig1 = plt.figure()
ax = fig1.add_subplot(111)
line1, = ax.plot(mean_x_xt, 'r-', label='average x-position')
line2, = ax.plot(mean_x_xt, 'm-', label='exponential fit')
ax.set_ylim(-3.5, -1)
ax.set_xlim(0, n_turns)

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

        line1.set_xdata(turns[:i_turn])
        line1.set_ydata(np.log10(np.abs(mean_x_xt[:i_turn])))
        line2.set_xdata(turns[:i_turn])
        line2.set_ydata(np.log10(np.exp(fit_x_xt.intercept +
                                 fit_x_xt.slope*turns[:i_turn])))
        print(f'xtrack h growth rate: {fit_x_xt.slope}')

        fig1.canvas.draw()
        fig1.canvas.flush_events()

        out_folder = '.'
        np.save(f'{out_folder}/mean_x.npy', mean_x_xt)
        np.save(f'{out_folder}/mean_y.npy', mean_y_xt)
