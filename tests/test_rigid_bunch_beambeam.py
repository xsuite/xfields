# copyright ############################### #
# This file is part of the Xfields Package. #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
import pytest

import xfields as xf
import xobjects as xo
import xtrack as xt
from xfields import BeamBeamRigidBunchStudy
from xfields.config_tools.beambeam_config_tools.rigid_bunch_mode import (
    _BEAMBEAM_EXTRA_KEY,
    _discover_installation,
)
from xfields.config_tools.beambeam_config_tools.config_tools import (
    compute_beambeam_geometry,
    compute_twiss_and_madpoints_at_bb,
)


N_CELLS = 8
CELL_LENGTH = 10.0
N_SLOTS = 8
SLOT_LENGTH = CELL_LENGTH * N_CELLS / N_SLOTS
NEMITT_X = 2.0e-6
NEMITT_Y = 2.5e-6


def _make_toy_ring(suffix, shared_ips):
    """Stable FODO ring with boundaries at every possible BB encounter."""
    elements = []
    names = []
    for ii in range(N_CELLS):
        if ii == 0:
            marker_name = 'ip1'
        elif ii == 3:
            marker_name = 'ip2'
        else:
            marker_name = f'cell{ii}_{suffix}'
        marker = shared_ips.get(marker_name, xt.Marker())

        # The installer places encounters 1 um downstream of their nominal
        # position. These small drifts and markers make those positions element
        # boundaries, including the +/- half-slot long-range positions.
        elements.extend([
            marker,
            xt.Multipole(knl=[0, 0.05]),
            xt.Drift(length=1e-6),
            xt.Marker(),
            xt.Drift(length=0.5 * CELL_LENGTH - 1e-6),
            xt.Multipole(knl=[0, -0.05]),
            xt.Drift(length=1e-6),
            xt.Marker(),
            xt.Drift(length=0.5 * CELL_LENGTH - 1e-6),
        ])
        names.extend([
            marker_name,
            f'qf{ii}_{suffix}',
            f'deps_a{ii}_{suffix}',
            f'edge_a{ii}_{suffix}',
            f'drift_a{ii}_{suffix}',
            f'qd{ii}_{suffix}',
            f'deps_b{ii}_{suffix}',
            f'edge_b{ii}_{suffix}',
            f'drift_b{ii}_{suffix}',
        ])

    line = xt.Line(
        elements=elements,
        element_names=names,
        particle_ref=xt.Particles(p0c=7e12),
    )
    # Keep the line boundary away from both beam--beam regions.
    line.cycle(name_first_element=f'cell6_{suffix}', inplace=True)
    line.twiss_default['method'] = '4d'
    return line


def _make_toy_environment():
    shared_ips = {'ip1': xt.Marker(), 'ip2': xt.Marker()}
    return xt.Environment(lines={
        'cw': _make_toy_ring('cw', shared_ips),
        'acw': _make_toy_ring('acw', shared_ips),
    })


def _toy_filling():
    filling_pattern_cw = np.zeros(N_SLOTS, dtype=int)
    filling_pattern_acw = np.zeros(N_SLOTS, dtype=int)
    filling_pattern_cw[[0, 2, 5]] = 1
    filling_pattern_acw[[0, 3, 6]] = 1
    intensity_cw = np.full(N_SLOTS, 1.0e11)
    intensity_acw = np.full(N_SLOTS, 1.0e11)
    intensity_cw[[0, 2, 5]] = [1.0e11, 2.0e11, 3.0e11]
    intensity_acw[[0, 3, 6]] = [1.5e11, 2.5e11, 3.5e11]
    return (filling_pattern_cw, filling_pattern_acw,
            intensity_cw, intensity_acw)


def _install_toy_rigid_bunch_beambeam():
    env = _make_toy_environment()
    (filling_pattern_cw, filling_pattern_acw,
     intensity_cw, intensity_acw) = _toy_filling()

    env.xfields.install_beambeam_interactions(
        clockwise_line='cw', anticlockwise_line='acw',
        ip_names=['ip1', 'ip2'],
        num_long_range_encounters_per_side=1,
        harmonic_number=N_SLOTS,
        bunch_spacing_buckets=1,
        mode='rigid_bunch')
    assert env.extra_config['xfields_beambeam']['mode'] == 'rigid_bunch'
    assert not hasattr(env, '_beam_beam_rigid_bunch_study')
    bb_names_cw = [name for name in env.cw.element_names
                   if name.startswith('bb_')]
    bb_names_acw = [name for name in env.acw.element_names
                    if name.startswith('bb_')]
    assert len(bb_names_cw) == 6
    assert len(bb_names_acw) == 6
    for name in bb_names_cw:
        assert len(env.cw[name].own_beam_zeta) == N_SLOTS
        assert len(env.cw[name].other_beam_zeta) == N_SLOTS
    for name in bb_names_acw:
        assert len(env.acw[name].own_beam_zeta) == N_SLOTS
        assert len(env.acw[name].other_beam_zeta) == N_SLOTS
    for line, names in ((env.cw, bb_names_cw), (env.acw, bb_names_acw)):
        for name in names:
            metadata = line[name].extra[_BEAMBEAM_EXTRA_KEY]
            assert metadata['other_element_name'] != name
            assert metadata['ip_name'] in ('ip1', 'ip2')

    # Installation and its full-slot elements are serializable before filling-
    # dependent physics parameters are loaded by configuration.
    env = xt.Environment.from_dict(env.to_dict())
    installation = _discover_installation(env)
    assert list(installation.elements['cw']) == sorted(
        installation.elements['cw'])
    assert list(installation.elements['acw']) == sorted(
        installation.elements['acw'])
    study = env.xfields.configure_beambeam_interactions(
        num_particles={'cw': intensity_cw, 'acw': intensity_acw},
        nemitt_x=NEMITT_X,
        nemitt_y=NEMITT_Y,
        filled_slots_cw=np.flatnonzero(filling_pattern_cw),
        filled_slots_acw=np.flatnonzero(filling_pattern_acw))
    assert not hasattr(env, '_beam_beam_rigid_bunch_study')
    return (env, study, filling_pattern_cw, filling_pattern_acw,
            intensity_cw, intensity_acw)


def test_rigid_bunch_beambeam_toy_installation_and_configuration():
    # Characterize the consolidated install/configure path through a normalized
    # set of encounter, element and solution properties.
    (env, study, filling_pattern_cw, filling_pattern_acw,
     intensity_cw, intensity_acw) = _install_toy_rigid_bunch_beambeam()
    assert isinstance(study, BeamBeamRigidBunchStudy)

    expected_encounters = [
        'bb_ip1_ho', 'bb_ip1_r01', 'bb_ip1_l01',
        'bb_ip2_ho', 'bb_ip2_r01', 'bb_ip2_l01',
    ]
    assert study.enc_names == expected_encounters
    assert study.bb_names_cw == [
        'bb_ho.c1b1_00', 'bb_lr.r1b1_01', 'bb_lr.l1b1_01',
        'bb_ho.c2b1_00', 'bb_lr.r2b1_01', 'bb_lr.l2b1_01',
    ]
    assert study.bb_names_acw == [
        'bb_ho.c1b2_00', 'bb_lr.r1b2_01', 'bb_lr.l1b2_01',
        'bb_ho.c2b2_00', 'bb_lr.r2b2_01', 'bb_lr.l2b2_01',
    ]
    assert study.ip_offsets == {'ip1': 0, 'ip2': 6}
    assert study.n_slots == N_SLOTS
    assert study.bunch_spacing_zeta == SLOT_LENGTH
    xo.assert_allclose(study.filling_pattern_cw, filling_pattern_cw,
                       rtol=0, atol=0)
    xo.assert_allclose(study.filling_pattern_acw, filling_pattern_acw,
                       rtol=0, atol=0)
    xo.assert_allclose(study.filled_slots_cw, [0, 2, 5], rtol=0, atol=0)
    xo.assert_allclose(study.filled_slots_acw, [0, 3, 6], rtol=0, atol=0)
    xo.assert_allclose(
        study.num_particles_cw,
        intensity_cw[[0, 2, 5]], rtol=0, atol=0)
    xo.assert_allclose(
        study.num_particles_acw,
        intensity_acw[[0, 3, 6]], rtol=0, atol=0)

    expected_offsets = [0, 1, 7, 6, 7, 5]
    assert [study.geom[name]['offset'] for name in expected_encounters] \
        == expected_offsets
    assert [study.geom[name]['signed_n'] for name in expected_encounters] \
        == [0, 1, -1, 0, 1, -1]
    for geom in study.geom.values():
        assert geom['sep_x'] == 0
        assert geom['sep_y'] == 0

    xo.assert_allclose(
        env.cw.get_table()['s', study.bb_names_cw],
        [20.000001, 25.000001, 15.000001,
         50.000001, 55.000001, 45.000001],
        rtol=0, atol=2e-14)
    xo.assert_allclose(
        env.acw.get_table()['s', study.bb_names_acw],
        [20.000001, 15.000001, 25.000001,
         50.000001, 45.000001, 55.000001],
        rtol=0, atol=2e-14)

    beta0_cw = float(env.cw.particle_ref.beta0[0])
    gamma0_cw = float(env.cw.particle_ref.gamma0[0])
    beta0_acw = float(env.acw.particle_ref.beta0[0])
    gamma0_acw = float(env.acw.particle_ref.gamma0[0])
    for base, offset in zip(expected_encounters, expected_offsets):
        bb_cw = study.bb_cw[base]
        bb_acw = study.bb_acw[base]
        assert isinstance(bb_cw, xf.BeamBeamBiGaussianRigidBunch2D)
        assert isinstance(bb_acw, xf.BeamBeamBiGaussianRigidBunch2D)
        assert bb_cw.coherent == 1
        assert bb_acw.coherent == 1
        assert bb_cw.zeta_period == N_SLOTS * SLOT_LENGTH
        assert bb_acw.zeta_period == N_SLOTS * SLOT_LENGTH
        assert bb_cw.zeta_match_tol == 0.1 * SLOT_LENGTH
        assert bb_acw.zeta_match_tol == 0.1 * SLOT_LENGTH
        assert bb_cw.zeta_offset == -offset * SLOT_LENGTH
        assert bb_acw.zeta_offset == offset * SLOT_LENGTH

        # Element grids cover every RF slot and increase in zeta for binary
        # search, hence they reverse the public physical-slot order.
        all_zeta = -np.arange(N_SLOTS) * SLOT_LENGTH
        xo.assert_allclose(bb_cw.own_beam_zeta, all_zeta[::-1], rtol=0, atol=0)
        xo.assert_allclose(bb_acw.own_beam_zeta, all_zeta[::-1], rtol=0, atol=0)
        assert len(bb_cw.other_beam_zeta) == N_SLOTS
        assert len(bb_acw.other_beam_zeta) == N_SLOTS

        geom = study.geom[base]
        # The shared covariance API includes relativistic beta in the
        # normalized-to-geometric emittance conversion.
        xo.assert_allclose(
            geom['Sigma_11_cw'],
            geom['betx_cw'] * NEMITT_X / (beta0_cw * gamma0_cw),
            rtol=1e-14)
        xo.assert_allclose(
            geom['Sigma_33_cw'],
            geom['bety_cw'] * NEMITT_Y / (beta0_cw * gamma0_cw),
            rtol=1e-14)
        xo.assert_allclose(
            geom['Sigma_11_acw'],
            geom['betx_acw'] * NEMITT_X / (beta0_acw * gamma0_acw),
            rtol=1e-14)
        xo.assert_allclose(
            geom['Sigma_33_acw'],
            geom['bety_acw'] * NEMITT_Y / (beta0_acw * gamma0_acw),
            rtol=1e-14)
        xo.assert_allclose(
            bb_cw.own_beam_Sigma_11, geom['Sigma_11_cw'], rtol=0, atol=0)
        xo.assert_allclose(
            bb_cw.own_beam_Sigma_13, geom['Sigma_13_cw'], rtol=0, atol=0)
        xo.assert_allclose(
            bb_cw.own_beam_Sigma_33, geom['Sigma_33_cw'], rtol=0, atol=0)
        xo.assert_allclose(
            bb_cw.other_beam_Sigma_11, geom['Sigma_11_acw'], rtol=0, atol=0)
        xo.assert_allclose(
            bb_cw.other_beam_Sigma_13, geom['Sigma_13_acw'], rtol=0, atol=0)
        xo.assert_allclose(
            bb_cw.other_beam_Sigma_33, geom['Sigma_33_acw'], rtol=0, atol=0)

    # The study geometry is the normalized view of the shared per-encounter
    # Reduced Twiss tables and MadPoints.
    names_by_ip = {'cw': {}, 'acw': {}}
    for base, ip, _ in study.enc_specs:
        names_by_ip['cw'].setdefault(ip, []).append(
            study.bb_name(base, beam='cw'))
        names_by_ip['acw'].setdefault(ip, []).append(
            study.bb_name(base, beam='acw'))
    twiss_and_madpoints = compute_twiss_and_madpoints_at_bb(
        line_cw=env.cw, line_acw=env.acw,
        element_names_by_ip=names_by_ip,
        nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y,
        survey_separation=False)
    for base, ip, _ in study.enc_specs:
        geom = study.geom[base]
        encounter = compute_beambeam_geometry(
            twiss_and_madpoints=twiss_and_madpoints,
            element_name_cw=study.bb_name(base, beam='cw'),
            element_name_acw=study.bb_name(base, beam='acw'))
        for beam in ('cw', 'acw'):
            xo.assert_allclose(
                geom[f'betx_{beam}'],
                encounter[beam]['betx'], rtol=0, atol=0)
            xo.assert_allclose(
                geom[f'bety_{beam}'],
                encounter[beam]['bety'], rtol=0, atol=0)
        for beam in ('cw', 'acw'):
            for component in (11, 13, 33):
                xo.assert_allclose(
                    geom[f'Sigma_{component}_{beam}'],
                    encounter[beam]['sigma'][component],
                    rtol=0, atol=0)
        xo.assert_allclose(
            geom['sep_x'], encounter['separation_x'], rtol=0, atol=0)
        xo.assert_allclose(
            geom['sep_y'], encounter['separation_y'], rtol=0, atol=0)

    env.cw['beambeam_scale'] = 0.37
    for name in study.bb_names_cw:
        assert env.cw[name].scale_strength == 0.37
    for name in study.bb_names_acw:
        assert env.acw[name].scale_strength == 0.37
    env.cw['beambeam_scale'] = 1.0

    reduced = study.second_order_maps()
    assert reduced.enc_names == study.enc_names
    assert reduced.geom == study.geom
    for name in reduced.bb_names_cw:
        assert isinstance(
            reduced.cw_line[name], xf.BeamBeamBiGaussianRigidBunch2D)
    for name in reduced.bb_names_acw:
        assert isinstance(
            reduced.acw_line[name], xf.BeamBeamBiGaussianRigidBunch2D)

    with pytest.raises(RuntimeError, match=(
            'did not converge.*require_convergence=False')):
        reduced.solve(
            max_iterations=1, tol_sigma=0,
            twiss_mode='fast_orbit', show_progress=False)

    solution = reduced.solve(
        max_iterations=2,
        tol_sigma=0,
        twiss_mode='fast_orbit',
        show_progress=False,
        require_convergence=False,
    )
    assert isinstance(solution, xf.RigidBunchTwiss)
    assert solution.converged is False
    assert solution.num_iterations == 2
    assert np.isfinite(solution.max_orbit_change)

    converged_solution = reduced.solve(
        max_iterations=3,
        twiss_mode='fast_orbit',
        show_progress=False,
    )
    assert converged_solution.converged is True
    assert converged_solution.num_iterations <= 3
    assert converged_solution.max_orbit_change < 1e-4

    mbtw_cw, mbtw_acw = solution.cw, solution.acw
    assert len(mbtw_cw) == len(study.filled_slots_cw)
    assert len(mbtw_acw) == len(study.filled_slots_acw)

    for bb in reduced.bb_cw.values():
        assert bb.num_other_bunches == N_SLOTS
        xo.assert_allclose(
            bb.other_beam_num_particles,
            (filling_pattern_acw * intensity_acw)[::-1], rtol=0, atol=0)
    for bb in reduced.bb_acw.values():
        assert bb.num_other_bunches == N_SLOTS
        xo.assert_allclose(
            bb.other_beam_num_particles,
            (filling_pattern_cw * intensity_cw)[::-1], rtol=0, atol=0)

    # At the +1-slot encounter, physical slot 0 faces an empty slot and gets no
    # kick, while the remaining two bunches have partners. This checks the
    # offset-sign conversion together with the public negative-zeta convention.
    for beam, bb in (
            ('cw', reduced.bb_cw['bb_ip1_r01']),
            ('acw', reduced.bb_acw['bb_ip1_r01'])):
        probe = xt.Particles(
            p0c=7e12,
            x=np.full(3, 1.0e-3),
            zeta=reduced.bunch_zeta(beam=beam))
        bb.track(probe)
        assert probe.px[0] == 0
        assert np.all(np.abs(probe.px[1:]) > 0)

    study.load_solution(solution)
    for base in expected_encounters:
        for full_bb, reduced_bb in (
                (study.bb_cw[base], reduced.bb_cw[base]),
                (study.bb_acw[base], reduced.bb_acw[base])):
            assert full_bb.num_other_bunches == N_SLOTS
            for field in (
                    'other_beam_zeta', 'other_beam_shift_x',
                    'other_beam_shift_y',
                    'other_beam_num_particles'):
                xo.assert_allclose(
                    getattr(full_bb, field),
                    getattr(reduced_bb, field),
                    rtol=0, atol=0)

    # Dynamic-beta covariance arrives in public filled-slot order from Twiss
    # and is reordered with the negative-zeta grids inside each element.
    dynamic_solution = reduced.solve(
        max_iterations=1,
        tol_sigma=0,
        dynamic_beta=True,
        twiss_mode='fast',
        show_progress=False,
        require_convergence=False,
    )
    mbtw_cw_dyn = dynamic_solution.cw
    mbtw_acw_dyn = dynamic_solution.acw
    gamma0_cw = float(reduced.cw_line.particle_ref.gamma0[0])
    gamma0_acw = float(reduced.acw_line.particle_ref.gamma0[0])
    indices_cw = N_SLOTS - 1 - study.filled_slots_cw
    indices_acw = N_SLOTS - 1 - study.filled_slots_acw
    for base in expected_encounters:
        name_cw = reduced.bb_name(base, beam='cw')
        name_acw = reduced.bb_name(base, beam='acw')
        Sigma_11_cw = mbtw_cw_dyn['betx', name_cw] * NEMITT_X / gamma0_cw
        Sigma_11_acw = mbtw_acw_dyn['betx', name_acw] * NEMITT_X / gamma0_acw
        xo.assert_allclose(
            np.asarray(reduced.bb_cw[base].own_beam_Sigma_11)[indices_cw],
            Sigma_11_cw, rtol=1e-14)
        xo.assert_allclose(
            np.asarray(reduced.bb_acw[base].own_beam_Sigma_11)[indices_acw],
            Sigma_11_acw, rtol=1e-14)
        xo.assert_allclose(
            np.asarray(reduced.bb_cw[base].other_beam_Sigma_11)[indices_acw],
            Sigma_11_acw, rtol=1e-14)
        xo.assert_allclose(
            np.asarray(reduced.bb_acw[base].other_beam_Sigma_11)[indices_cw],
            Sigma_11_cw, rtol=1e-14)

    # A changed filling updates the full-slot arrays in place. No element is
    # rebuilt, and empty slots retain zero opposing population.
    filling_pattern_cw_new = np.zeros(N_SLOTS, dtype=int)
    filling_pattern_acw_new = np.zeros(N_SLOTS, dtype=int)
    filling_pattern_cw_new[[1, 4]] = 1
    filling_pattern_acw_new[[0, 2, 5, 7]] = 1
    env.cw['beambeam_scale'] = 0.29
    original_cw = dict(study.bb_cw)
    original_acw = dict(study.bb_acw)
    study.apply_filling_pattern(
        filling_pattern_cw=filling_pattern_cw_new,
        filling_pattern_acw=filling_pattern_acw_new)

    xo.assert_allclose(study.filled_slots_cw, [1, 4], rtol=0, atol=0)
    xo.assert_allclose(study.filled_slots_acw, [0, 2, 5, 7], rtol=0, atol=0)
    for base, bb in study.bb_cw.items():
        assert bb is original_cw[base]
        assert len(bb.own_beam_zeta) == N_SLOTS
        assert len(bb.other_beam_zeta) == N_SLOTS
        assert bb.num_own_bunches == N_SLOTS
        assert bb.num_other_bunches == N_SLOTS
        assert bb.scale_strength == 0.29
        xo.assert_allclose(
            bb.other_beam_num_particles, np.zeros(N_SLOTS), rtol=0, atol=0)
    for base, bb in study.bb_acw.items():
        assert bb is original_acw[base]
        assert len(bb.own_beam_zeta) == N_SLOTS
        assert len(bb.other_beam_zeta) == N_SLOTS
        assert bb.num_own_bunches == N_SLOTS
        assert bb.num_other_bunches == N_SLOTS
        assert bb.scale_strength == 0.29
        xo.assert_allclose(
            bb.other_beam_num_particles, np.zeros(N_SLOTS), rtol=0, atol=0)


def test_rigid_bunch_configuration_is_rediscovered_and_repeatable():
    (env, first, filling_pattern_cw, filling_pattern_acw,
     intensity_cw, intensity_acw) = _install_toy_rigid_bunch_beambeam()

    env.cw['beambeam_scale'] = 0.41
    second = env.xfields.configure_beambeam_interactions(
        num_particles={'cw': intensity_cw, 'acw': intensity_acw},
        nemitt_x=NEMITT_X,
        nemitt_y=NEMITT_Y)
    second.apply_filling_pattern(
        filling_pattern_cw=filling_pattern_cw,
        filling_pattern_acw=filling_pattern_acw)

    assert second is not first
    assert second.geom.keys() == first.geom.keys()
    for encounter_name in first.geom:
        assert second.geom[encounter_name] == first.geom[encounter_name]
    assert env.cw['beambeam_scale'] == 0.41
    assert env.acw['beambeam_scale'] == 0.41
    assert not hasattr(env, '_beam_beam_rigid_bunch_study')


def test_rigid_bunch_configuration_restores_scale_expression_on_error(
        monkeypatch):
    (env, _, _, _, intensity_cw,
     intensity_acw) = _install_toy_rigid_bunch_beambeam()

    env['scale_source'] = 0.23
    env['beambeam_scale'] = 2 * env.ref['scale_source']
    previous_expression = str(env.ref['beambeam_scale'].xdeps.expr)

    def fail_geometry(self):
        raise RuntimeError('geometry failed')

    monkeypatch.setattr(BeamBeamRigidBunchStudy, '_compute_geometry',
                        fail_geometry)
    with pytest.raises(RuntimeError, match='geometry failed'):
        env.xfields.configure_beambeam_interactions(
            num_particles={'cw': intensity_cw, 'acw': intensity_acw},
            nemitt_x=NEMITT_X,
            nemitt_y=NEMITT_Y)

    assert str(env.ref['beambeam_scale'].xdeps.expr) == previous_expression
    assert env.cw['beambeam_scale'] == 0.46
    env['scale_source'] = 0.31
    assert env.cw['beambeam_scale'] == 0.62
    assert env.acw['beambeam_scale'] == 0.62


def test_rigid_bunch_pattern_contract_matches_beam_stats_monitor():
    # Beam-beam and BeamStatsMonitor interpret filling patterns as occupancy
    # patterns over physical slots. Slot i is centred at
    # zeta = -i * bunch_spacing_zeta; intensities are a separate input.
    filling_pattern = np.array([1, 0, 1, 1])
    bunch_spacing_zeta = 5.0
    slot_intensities = np.array([1.0e11, 0.0, 2.0e11, 3.0e11])

    line = xt.Line(
        elements=[xt.Drift(length=4 * bunch_spacing_zeta)],
        particle_ref=xt.Particles(p0c=7e12))
    study = BeamBeamRigidBunchStudy(
        line, line, ips=[], num_long_range_encounters_per_side=0,
        harmonic_number=4, bunch_spacing_buckets=1,
        nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y,
        num_particles={'cw': 4.0e11, 'acw': slot_intensities})
    study.apply_filling_pattern(
        filling_pattern_cw=filling_pattern,
        filling_pattern_acw=filling_pattern)

    monitor = xt.BeamStatsMonitor(
        filling_pattern=filling_pattern,
        selected_slots=[0, 3],
        bunch_spacing_zeta=bunch_spacing_zeta)

    xo.assert_allclose(study.filled_slots_cw, [0, 2, 3], rtol=0, atol=0)
    xo.assert_allclose(study.filled_slots_cw, monitor.filled_slots,
                       rtol=0, atol=0)
    xo.assert_allclose(study.bunch_zeta(beam='cw'), [0, -10, -15],
                       rtol=0, atol=0)
    with pytest.raises(ValueError, match="'cw'.*'acw'"):
        study.bunch_zeta(beam=False)
    xo.assert_allclose(
        monitor.zeta_centers_unwrapped(line_length=20)[0], [0, -15],
        rtol=0, atol=0)
    xo.assert_allclose(study.num_particles_cw,
                       np.full(3, 4.0e11), rtol=0, atol=0)
    xo.assert_allclose(study.num_particles_acw,
                       [1.0e11, 2.0e11, 3.0e11], rtol=0, atol=0)

    with pytest.raises(ValueError, match='slot-indexed array'):
        invalid = BeamBeamRigidBunchStudy(
            line, line, ips=[], num_long_range_encounters_per_side=0,
            harmonic_number=4, bunch_spacing_buckets=1,
            nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y,
            num_particles={
                'cw': [1.0e11, 2.0e11, 3.0e11],
                'acw': 1.0e11})
        invalid.apply_filling_pattern(
            filling_pattern_cw=filling_pattern,
            filling_pattern_acw=filling_pattern)

    with pytest.raises(ValueError, match='only zero and one'):
        study.apply_filling_pattern(
            filling_pattern_cw=[1, 0, 2, 1],
            filling_pattern_acw=filling_pattern)

    source_slots = np.array([3, 0, 2], dtype=np.int64)
    study.apply_filling_pattern(
        filled_slots_cw=source_slots,
        filled_slots_acw=[0, 2, 3])
    source_slots[:] = 1
    exposed_slots = study.filled_slots_cw
    exposed_pattern = study.filling_pattern_cw
    exposed_slots[:] = 1
    exposed_pattern[:] = 0
    xo.assert_allclose(study.filled_slots_cw, [0, 2, 3], rtol=0, atol=0)
    xo.assert_allclose(
        study.filling_pattern_cw, filling_pattern, rtol=0, atol=0)

    with pytest.raises(ValueError, match='Only one'):
        study.apply_filling_pattern(
            filling_pattern_cw=filling_pattern,
            filled_slots_cw=[0, 2, 3],
            filled_slots_acw=[0, 2, 3])
