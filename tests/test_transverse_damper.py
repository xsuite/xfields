import numpy as np
from scipy.constants import c, e
from scipy.constants import physical_constants
from scipy.signal import hilbert
from scipy.stats import linregress

import xfields as xf
import xtrack as xt
import xpart as xp
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_beambeam3d_beamstrahlung_ws_no_config(test_context):
    longitudinal_mode = 'nonlinear'
    # Machine settings
    n_turns = 1000

    n_macroparticles = 10000 #int(10**6/4)  # per bunch
    intensity = 8e9

    alpha = 53.86**-2

    E0 = physical_constants['proton mass energy equivalent in MeV'][0]*1e6
    E = 450e9
    p0 = E * e / c
    gamma = E/E0
    beta = np.sqrt(1-1/gamma**2)

    h_RF = 35640
    bunch_spacing_buckets = 10
    n_bunches = 2
    n_slices = 500

    accQ_x = 60.275
    accQ_y = 60.295
    chroma = 0

    epsn_x = 2e-6
    epsn_y = 2e-6
    taub = 0.9e-9
    sigma_z = taub*beta*c/4

    circumference = 26658.883
    average_radius = circumference / (2 * np.pi)

    momentum_compaction = alpha
    eta = momentum_compaction - 1.0 / gamma ** 2
    V = 4e6
    Q_s = np.sqrt((e*V*h_RF*eta)/(2*np.pi*beta*c*p0))
    sigma_delta = Q_s * sigma_z / (average_radius * eta)
    beta_s = sigma_z / sigma_delta
    f_rev = beta*c/circumference
    f_rf = f_rev*h_RF

    beta_x = average_radius/accQ_x
    beta_y = average_radius/accQ_y

    bucket_length = circumference / h_RF
    zeta_range = (-0.5*bucket_length, 0.5*bucket_length)

    filling_scheme = np.zeros(int(h_RF/bunch_spacing_buckets))
    filling_scheme[0:n_bunches] = 1
    filled_slots = np.nonzero(filling_scheme)[0]

    betatron_map = xt.LineSegmentMap(
        length=circumference, betx=beta_x, bety=beta_y,
        qx=accQ_x, qy=accQ_y,
        longitudinal_mode=longitudinal_mode,
        voltage_rf=V, frequency_rf=f_rf, lag_rf=180,
        momentum_compaction_factor=momentum_compaction,
        dqx=chroma, dqy=chroma
    )

    gain_x = 0.01
    gain_y = 0.01

    transverse_damper = xf.TransverseDamper(
        gain_x=gain_x, gain_y=gain_y,
        num_bunches=n_bunches,
        filling_scheme=filling_scheme,
        filled_slots=filled_slots,
        zeta_range=zeta_range,
        num_slices=n_slices,
        bunch_spacing_zeta=bunch_spacing_buckets*bucket_length,
        circumference=circumference
    )

    line = xt.Line(elements=[[betatron_map, transverse_damper][0]],
                   element_names=[['betatron_map', 'transverse_damper'][0]])

    line.particle_ref = xt.Particles(p0c=450e9)
    line.build_tracker(_context=test_context)

    particles = xp.generate_matched_gaussian_multibunch_beam(
        _context=test_context,
        filling_scheme=filling_scheme,
        num_particles=n_macroparticles,
        total_intensity_particles=intensity,
        nemitt_x=epsn_x, nemitt_y=epsn_y, sigma_z=sigma_z,
        line=line, bunch_spacing_buckets=bunch_spacing_buckets,
        bunch_numbers=filled_slots,
        rf_harmonic=[h_RF], rf_voltage=[V],
        particle_ref=line.particle_ref,

    )

    # apply a distortion to the bunches
    amplitude = 1e-3
    particles.px += amplitude
    particles.py += amplitude

    mean_x = np.zeros((n_turns, n_bunches))
    mean_y = np.zeros((n_turns, n_bunches))

    from tqdm import tqdm

    for i_turn in tqdm(range(n_turns)):
        line.track(particles, num_turns=1)
        transverse_damper.track(particles, i_turn)
        for ib in range(n_bunches):
            mean_x[i_turn, ib] = np.mean(particles.x[n_macroparticles*ib:
                                                     n_macroparticles*(ib+1)])
            mean_y[i_turn, ib] = np.mean(particles.y[n_macroparticles*ib:
                                                     n_macroparticles*(ib+1)])

    turns = np.linspace(0, n_turns-1, n_turns)

    i_fit_start = 200
    i_fit_end = 600

    for i_bunch in range(n_bunches):
        ampls_x = np.abs(hilbert(mean_x[:, i_bunch]))
        fit_x = linregress(turns[i_fit_start: i_fit_end],
                           np.log(ampls_x[i_fit_start: i_fit_end]))

        assert np.isclose(-fit_x.slope*2, gain_x, atol=1e-4, rtol=0)

        ampls_y = np.abs(hilbert(mean_y[:, i_bunch]))
        fit_y = linregress(turns[i_fit_start: i_fit_end],
                           np.log(ampls_y[i_fit_start: i_fit_end]))

        assert np.isclose(-fit_y.slope*2, gain_y, atol=1e-4, rtol=0)
