import numpy as np
from scipy.constants import c
from scipy.constants import physical_constants
from scipy.signal import hilbert
from scipy.stats import linregress

import xfields as xf
import xtrack as xt
import xpart as xp
from xobjects.test_helpers import for_all_test_contexts


exclude_contexts = ['ContextPyopencl', 'ContextCupy']

@for_all_test_contexts(excluding=exclude_contexts)
def test_transverse_damper_mpi(test_context):
    longitudinal_mode = 'nonlinear'
    # Machine settings
    n_turns = 1000

    n_macroparticles = 10000
    intensity = 8e9

    alpha = 53.86**-2

    e0 = physical_constants['proton mass energy equivalent in MeV'][0]*1e6
    en = 450e9
    gamma = en/e0
    beta = np.sqrt(1-1/gamma**2)

    h_rf = 35640
    bunch_spacing_buckets = 10
    num_bunches = 4
    n_slices = 500

    epsn_x = 2e-6
    epsn_y = 2e-6
    taub = 0.9e-9
    sigma_z = taub*beta*c/4

    circumference = 26658.883

    momentum_compaction = alpha
    v_rf = 4e6
    f_rev = beta*c/circumference
    f_rf = f_rev*h_rf

    bucket_length = circumference / h_rf
    zeta_range = (-0.5*bucket_length, 0.5*bucket_length)

    filling_scheme = np.zeros(int(h_rf/bunch_spacing_buckets))
    filling_scheme[0: num_bunches] = 1

    one_turn_map = xt.LineSegmentMap(
        length=circumference,
        betx=70, bety=80,
        qx=62.31, qy=60.32,
        longitudinal_mode=longitudinal_mode,
        voltage_rf=v_rf, frequency_rf=f_rf, lag_rf=180,
        momentum_compaction_factor=momentum_compaction,
    )

    gain_x = 0.01
    gain_y = 0.01

    transverse_damper = xf.TransverseDamper(
        gain_x=gain_x, gain_y=gain_y,
        filling_scheme=filling_scheme,
        zeta_range=zeta_range,
        num_slices=n_slices,
        bunch_spacing_zeta=bunch_spacing_buckets*bucket_length,
        circumference=circumference
    )

    line = xt.Line(elements=[one_turn_map, transverse_damper],
                   element_names=['one_turn_map', 'transverse_damper'])

    line.particle_ref = xt.Particles(p0c=450e9)
    line.build_tracker(_context=test_context)

    particles = xp.generate_matched_gaussian_multibunch_beam(
        _context=test_context,
        filling_scheme=filling_scheme,
        bunch_num_particles=n_macroparticles,
        bunch_intensity_particles=intensity,
        nemitt_x=epsn_x, nemitt_y=epsn_y, sigma_z=sigma_z,
        line=line, bunch_spacing_buckets=bunch_spacing_buckets,
        rf_harmonic=[h_rf], rf_voltage=[v_rf],
        particle_ref=line.particle_ref,
        prepare_line_and_particles_for_mpi_wake_sim=True
    )

    # apply a distortion to the bunches
    amplitude = 1e-3
    particles.px += amplitude
    particles.py += amplitude

    mean_x = np.zeros((n_turns))
    mean_y = np.zeros((n_turns))

    for i_turn in range(n_turns):
        line.track(particles, num_turns=1)
        mean_x[i_turn] = np.mean(particles.x)
        mean_y[i_turn] = np.mean(particles.y)

    turns = np.linspace(0, n_turns-1, n_turns)

    i_fit_start = 200
    i_fit_end = 600

    ampls_x = np.abs(hilbert(mean_x))
    fit_x = linregress(turns[i_fit_start: i_fit_end],
                        np.log(ampls_x[i_fit_start: i_fit_end]))

    assert np.isclose(-fit_x.slope*2, gain_x, atol=1e-4, rtol=0)

    ampls_y = np.abs(hilbert(mean_y))
    fit_y = linregress(turns[i_fit_start: i_fit_end],
                        np.log(ampls_y[i_fit_start: i_fit_end]))

    assert np.isclose(-fit_y.slope*2, gain_y, atol=1e-4, rtol=0)
