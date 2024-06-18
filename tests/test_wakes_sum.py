import numpy as np
from scipy.signal import hilbert
from scipy.stats import linregress

import xtrack as xt
import xpart as xp
import xfields as xf

from xobjects.test_helpers import for_all_test_contexts

@for_all_test_contexts
def test_wakes_sum(test_context):
    # Simulation settings
    n_turns = 3_000
    circumference = 26658.8832
    bucket_length_m = circumference / 35640

    wake_table_filename = xf.general._pkg_root.joinpath(
        '../test_data/HLLHC_wake.dat')
    wake_file_columns = ['time', 'longitudinal', 'dipole_x', 'dipole_y',
                        'quadrupole_x', 'quadrupole_y', 'dipole_xy',
                        'quadrupole_xy', 'dipole_yx', 'quadrupole_yx',
                        'constant_x', 'constant_y']
    wf_df = xf.Wakefield.table_from_headtail_file(
        wake_file=wake_table_filename,
        wake_file_columns=wake_file_columns
    )
    wf0 = xf.Wakefield.from_table(
        wf_df,
        use_components=['dipole_x', 'dipole_y'],
        zeta_range=(-0.5*bucket_length_m, 0.5*bucket_length_m),
        num_slices=20,
        num_turns=1.,
        circumference=circumference
    )

    wf = wf0 + wf0

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
    particles = xp.generate_matched_gaussian_bunch(
        line=line, num_particles=100_000, total_intensity_particles=2.3e11,
        nemitt_x=2e-6, nemitt_y=2e-6, sigma_z=0.075, _context=test_context)

    # Apply a distortion to the bunch to trigger an instability
    amplitude = 1e-3
    particles.x += amplitude
    particles.y += amplitude

    mean_x_xt = np.zeros(n_turns)
    mean_y_xt = np.zeros(n_turns)

    for i_turn in range(n_turns):
        line.track(particles, num_turns=1)

        mean_x_xt[i_turn] = np.mean(particles.x)
        mean_y_xt[i_turn] = np.mean(particles.y)


    turns = np.linspace(0, n_turns - 1, n_turns)

    # compute x instability growth rate
    ampls_x_xt = np.abs(hilbert(mean_x_xt))
    fit_x_xt = linregress(turns[-2000:],
                          np.log(ampls_x_xt[-2000:]))

    # compute y instability growth rate
    ampls_y_xt = np.abs(hilbert(mean_y_xt))
    fit_y_xt = linregress(turns[-2000:],
                          np.log(ampls_y_xt[-2000:]))

    assert np.isclose(fit_x_xt.slope, 0.0006687412001520158, atol=1e-5)
    assert np.isclose(fit_y_xt.slope, 0.0009959528632964905, atol=1e-5)
