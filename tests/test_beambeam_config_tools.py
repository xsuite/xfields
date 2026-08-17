# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import numpy as np
import pandas as pd

import xfields as xf
import xobjects as xo
import xtrack as xt

from xfields.config_tools.beambeam_config_tools.config_tools import (
    find_bb_separations,
)
from xfields.config_tools.beambeam_config_tools._madpoint import MadPoint
from xfields.config_tools.beambeam_config_tools.weak_strong import (
    _BEAMBEAM_CONFIG_KEY,
    _BEAMBEAM_EXTRA_KEY,
    _delay_in_slots,
    _discover_installation,
)


def test_generate_beambeam_encounter_table():
    encounters = xf.generate_beambeam_encounter_table(
        ip_names=['ip1', 'ip2'],
        num_long_range_encounters_per_side={'ip1': 2, 'ip2': 1},
        bunch_spacing_zeta=10.0,
        delay_at_ips_slots={'ip1': 0, 'ip2': 6},
        n_slots=8,
    )

    assert list(encounters[[
        'ip_name', 'encounter_type', 'identifier',
    ]].itertuples(index=False, name=None)) == [
        ('ip1', 'head_on', 0),
        ('ip1', 'long_range', 1),
        ('ip1', 'long_range', -1),
        ('ip1', 'long_range', 2),
        ('ip1', 'long_range', -2),
        ('ip2', 'head_on', 0),
        ('ip2', 'long_range', 1),
        ('ip2', 'long_range', -1),
    ]
    xo.assert_allclose(encounters['s_from_ip_cw'].to_numpy(),
                       [0, 5, -5, 10, -10, 0, 5, -5], rtol=0, atol=0)
    xo.assert_allclose(encounters['s_from_ip_acw'].to_numpy(),
                       [0, -5, 5, -10, 10, 0, -5, 5], rtol=0, atol=0)
    xo.assert_allclose(encounters['delay_in_slots_cw'].to_numpy(),
                       [0, 1, -1, 2, -2, 6, 7, 5], rtol=0, atol=0)
    xo.assert_allclose(encounters['delay_in_slots_acw'].to_numpy(),
                       [0, -1, 1, -2, 2, 2, 1, 3], rtol=0, atol=0)


def test_conventional_encounters_keep_positions_and_delays():
    shared_ips = {'ip1': xt.Marker(), 'ip2': xt.Marker()}
    env = xt.Environment(lines={
        'cw': _make_conventional_toy_ring('cw', shared_ips),
        'acw': _make_conventional_toy_ring('acw', shared_ips),
    })
    env.xfields.install_beambeam_interactions(
        clockwise_line='cw', anticlockwise_line='acw',
        ip_names=['ip1', 'ip2'],
        num_long_range_encounters_per_side=[1, 1],
        num_slices_head_on=3,
        harmonic_number=8, bunch_spacing_buckets=1,
        sigmaz=0.1, delay_at_ips_slots=[0, 6])

    installation = _discover_installation(env)
    for orientation, expected_delays in (
            ('clockwise', [-1, 1, 5, 7]),
            ('anticlockwise', [1, -1, 3, 1])):
        records = installation.elements[orientation]
        long_range = sorted(
            (record for record in records
             if record.metadata['label'] == 'bb_lr'),
            key=lambda record: (
                record.metadata['ip_name'], record.metadata['identifier']))
        xo.assert_allclose(
            [_delay_in_slots(installation, orientation, record)
             for record in long_range],
            expected_delays, rtol=0, atol=0)

        head_on = [record for record in records
                   if record.metadata['label'] == 'bb_ho']
        assert sum(record.metadata['ip_name'] == 'ip1'
                   for record in head_on) == 3
        assert sum(record.metadata['ip_name'] == 'ip2'
                   for record in head_on) == 3
        expected_head_on_delays = (
            [0, 6] if orientation == 'clockwise' else [0, 2])
        for ip_name, expected_delay in zip(
                ['ip1', 'ip2'], expected_head_on_delays):
            assert {_delay_in_slots(installation, orientation, record)
                    for record in head_on
                    if record.metadata['ip_name'] == ip_name} == {
                        expected_delay}


def _make_conventional_toy_ring(suffix, shared_ips):
    elements = []
    names = []
    kick_x = 1e-6 if suffix == 'cw' else -2e-6
    kick_y = -0.5e-6 if suffix == 'cw' else 1.5e-6
    bend_angle = 2e-3 if suffix == 'cw' else 1e-3
    for ii in range(8):
        if ii == 2:
            marker_name = 'ip1'
        elif ii == 5:
            marker_name = 'ip2'
        else:
            marker_name = f'cell{ii}_{suffix}'
        marker = shared_ips.get(marker_name, xt.Marker())
        elements.extend([
            marker,
            xt.Multipole(
                knl=[kick_x if ii == 0 else 0, 0.05],
                ksl=[kick_y if ii == 0 else 0]),
            xt.Drift(length=2.0),
            xt.Bend(length=1.0, angle=bend_angle),
            xt.Drift(length=2.0),
            xt.Multipole(knl=[0, -0.05]),
            xt.Drift(length=2.0),
            xt.Bend(length=1.0, angle=-bend_angle),
            xt.Drift(length=2.0),
        ])
        names.extend([
            marker_name,
            f'qf{ii}_{suffix}',
            f'drift_a0{ii}_{suffix}',
            f'bend_a{ii}_{suffix}',
            f'drift_a1{ii}_{suffix}',
            f'qd{ii}_{suffix}',
            f'drift_b0{ii}_{suffix}',
            f'bend_b{ii}_{suffix}',
            f'drift_b1{ii}_{suffix}',
        ])

    line = xt.Line(
        elements=elements,
        element_names=names,
        particle_ref=xt.Particles(p0c=7e12),
    )
    line.twiss_default['method'] = '4d'
    return line


def test_conventional_install_and_configure_characterization():
    # Exercise the established sliced-head-on/weak-strong workflow end to end
    # on a small two-ring model. This locks down its installed metadata and the
    # fields ultimately loaded into representative 2D and 3D elements before
    # the conventional path is migrated to shared geometry helpers.
    shared_ips = {'ip1': xt.Marker(), 'ip2': xt.Marker()}
    env = xt.Environment(lines={
        'cw': _make_conventional_toy_ring('cw', shared_ips),
        'acw': _make_conventional_toy_ring('acw', shared_ips),
        'unused': xt.Line(
            elements=[xt.Drift(length=1)],
            particle_ref=xt.Particles(p0c=7e12)),
    })
    env.xfields.install_beambeam_interactions(
        clockwise_line='cw', anticlockwise_line='acw',
        ip_names=['ip1', 'ip2'],
        num_long_range_encounters_per_side=[1, 1],
        num_slices_head_on=3,
        harmonic_number=8, bunch_spacing_buckets=1,
        sigmaz=0.1, delay_at_ips_slots=[0, 6])

    assert not hasattr(env, '_bb_config')
    assert env.extra_config[_BEAMBEAM_CONFIG_KEY] == {
        'version': 1,
        'mode': 'weak_strong',
        'clockwise_line': 'cw',
        'anticlockwise_line': 'acw',
        'ip_names': ['ip1', 'ip2'],
        'n_slots': 8,
        'delay_at_ips_slots': {'ip1': 0, 'ip2': 6},
    }
    for line in (env.cw, env.acw):
        for element_name in line.element_names:
            if element_name.startswith('bb_'):
                assert _BEAMBEAM_EXTRA_KEY in line[element_name].extra
                metadata = line[element_name].extra[_BEAMBEAM_EXTRA_KEY]
                assert not ({'orientation', 'line_name', 'other_line_name',
                             'ip_index', 'n_slots', 'delay_in_slots'}
                            & metadata.keys())

    # Environment-wide and encounter-local state survive serialization.
    env = xt.Environment.from_dict(env.to_dict())
    installation = _discover_installation(env)
    records_cw = installation.elements['clockwise']
    records_acw = installation.elements['anticlockwise']
    assert len(records_cw) == 10
    assert len(records_acw) == 10
    assert [record.name for record in records_cw] == sorted(
        record.name for record in records_cw)
    assert [record.name for record in records_acw] == sorted(
        record.name for record in records_acw)
    assert {record.metadata['label'] for record in records_cw} == {
        'bb_ho', 'bb_lr'}
    assert {record.metadata['label'] for record in records_acw} == {
        'bb_ho', 'bb_lr'}
    assert {record.name: record.metadata['other_element_name']
            for record in records_cw} == {
        record.name: record.name.replace('b1_', 'b2_')
        for record in records_cw}
    assert {record.name: record.metadata['other_element_name']
            for record in records_acw} == {
        record.name: record.name.replace('b2_', 'b1_')
        for record in records_acw}

    xo.assert_allclose(
        env.cw.get_table()['s', [
            'bb_lr.l1b1_01', 'bb_lr.r1b1_01',
            'bb_lr.l2b1_01', 'bb_lr.r2b1_01']],
        [15, 25, 45, 55], rtol=0, atol=1e-14)
    xo.assert_allclose(
        env.acw.get_table()['s', [
            'bb_lr.l1b2_01', 'bb_lr.r1b2_01',
            'bb_lr.l2b2_01', 'bb_lr.r2b2_01']],
        [25, 15, 55, 45], rtol=0, atol=1e-14)

    # Save the bare optics used by configuration for independent covariance
    # and closed-orbit checks. The ACW table is reversed by the conventional
    # configuration code before partner data are transferred.
    tw_cw = env.cw.twiss()
    tw_acw = env.acw.twiss().reverse()
    cov_cw = tw_cw.get_beam_covariance(nemitt_x=2e-6, nemitt_y=2.5e-6)
    cov_acw = tw_acw.get_beam_covariance(nemitt_x=2e-6, nemitt_y=2.5e-6)

    encounter_instances = pd.DataFrame([
        {
            'ip_name': record.metadata['ip_name'],
            'encounter_type': record.metadata['label'],
            'identifier': record.metadata['identifier'],
        }
        for record in records_cw])
    shared_geometry, _ = xf.compute_beambeam_geometry(
        encounter_table=encounter_instances,
        line_cw=env.cw, line_acw=env.acw,
        element_names_cw=[record.name for record in records_cw],
        element_names_acw=[record.metadata['other_element_name']
                           for record in records_cw],
        nemitt_x=2e-6, nemitt_y=2.5e-6,
        survey_separation=True,
        twiss_cw=tw_cw, twiss_acw=tw_acw,
        acw_is_reversed=True)
    for orientation, twiss in (('cw', tw_cw), ('acw', tw_acw)):
        names = shared_geometry[f'element_name_{orientation}']
        for coordinate in ('x', 'px', 'y', 'py'):
            xo.assert_allclose(
                shared_geometry[f'{coordinate}_{orientation}'].to_numpy(),
                [twiss[coordinate, name] for name in names],
                rtol=0, atol=0)

    surveys_cw = {
        ip: env.cw.survey(element0=ip, reverse=False)
        for ip in ('ip1', 'ip2')}
    surveys_acw = {
        ip: env.acw.survey(element0=ip, reverse=False).reverse()
        for ip in ('ip1', 'ip2')}
    legacy_separation_cw = {}
    legacy_separation_acw = {}
    for row in shared_geometry.itertuples(index=False):
        point_cw = MadPoint(
            row.element_name_cw, use_twiss=True, use_survey=True,
            xsuite_survey=surveys_cw[row.ip_name], xsuite_twiss=tw_cw)
        point_acw = MadPoint(
            row.element_name_acw, use_twiss=True, use_survey=True,
            xsuite_survey=surveys_acw[row.ip_name], xsuite_twiss=tw_acw)
        sep_x_cw, sep_y_cw = find_bb_separations(
            points_weak=[point_cw], points_strong=[point_acw])
        sep_x_acw, sep_y_acw = find_bb_separations(
            points_weak=[point_acw], points_strong=[point_cw])
        legacy_separation_cw[row.element_name_cw] = (
            sep_x_cw[0], sep_y_cw[0])
        legacy_separation_acw[row.element_name_acw] = (
            sep_x_acw[0], sep_y_acw[0])
        xo.assert_allclose(row.separation_x_cw, sep_x_cw[0],
                           rtol=0, atol=1e-14)
        xo.assert_allclose(row.separation_y_cw, sep_y_cw[0],
                           rtol=0, atol=1e-14)
        xo.assert_allclose(row.separation_x_acw, sep_x_acw[0],
                           rtol=0, atol=1e-14)
        xo.assert_allclose(row.separation_y_acw, sep_y_acw[0],
                           rtol=0, atol=1e-14)

    env.xfields.configure_beambeam_interactions(
        num_particles=1e11,
        nemitt_x=2e-6, nemitt_y=2.5e-6,
        crab_strong_beam=False)

    lr_cw = env.cw['bb_lr.r1b1_01']
    assert isinstance(lr_cw, xf.BeamBeamBiGaussian2D)
    xo.assert_allclose(lr_cw.other_beam_num_particles, 1e11, rtol=0, atol=0)
    xo.assert_allclose(
        lr_cw.other_beam_Sigma_11,
        cov_acw['Sigma11', 'bb_lr.r1b2_01'], rtol=1e-14)
    xo.assert_allclose(
        lr_cw.other_beam_Sigma_33,
        cov_acw['Sigma33', 'bb_lr.r1b2_01'], rtol=1e-14)
    xo.assert_allclose(
        lr_cw.other_beam_shift_x,
        legacy_separation_cw['bb_lr.r1b1_01'][0],
        rtol=0, atol=1e-14)
    xo.assert_allclose(
        lr_cw.other_beam_shift_y,
        legacy_separation_cw['bb_lr.r1b1_01'][1],
        rtol=0, atol=1e-14)
    xo.assert_allclose(
        lr_cw.ref_shift_x, tw_cw['x', 'bb_lr.r1b1_01'], rtol=0, atol=1e-14)
    xo.assert_allclose(
        lr_cw.ref_shift_y, tw_cw['y', 'bb_lr.r1b1_01'], rtol=0, atol=1e-14)
    assert lr_cw.other_beam_q0 == env.acw.particle_ref.q0
    assert lr_cw.other_beam_beta0 == env.acw.particle_ref.beta0[0]

    ho_cw = env.cw['bb_ho.c1b1_00']
    assert isinstance(ho_cw, xf.BeamBeamBiGaussian3D)
    assert ho_cw.num_slices_other_beam == 1
    xo.assert_allclose(
        ho_cw.slices_other_beam_num_particles[0], 1e11 / 3,
        rtol=1e-14)
    for sigma_name in ('11', '12', '22', '33', '34', '44'):
        xo.assert_allclose(
            getattr(ho_cw, f'slices_other_beam_Sigma_{sigma_name}')[0],
            cov_acw[f'Sigma{sigma_name}', 'bb_ho.c1b2_00'], rtol=1e-14)
    for sigma_name in ('13', '14', '23', '24'):
        assert getattr(ho_cw, f'slices_other_beam_Sigma_{sigma_name}')[0] == 0
    xo.assert_allclose(
        ho_cw.other_beam_shift_x,
        legacy_separation_cw['bb_ho.c1b1_00'][0],
        rtol=0, atol=1e-14)
    xo.assert_allclose(
        ho_cw.other_beam_shift_y,
        legacy_separation_cw['bb_ho.c1b1_00'][1],
        rtol=0, atol=1e-14)
    delta_px = (tw_cw['px', 'bb_ho.c1b1_00']
                - tw_acw['px', 'bb_ho.c1b2_00'])
    delta_py = (tw_cw['py', 'bb_ho.c1b1_00']
                - tw_acw['py', 'bb_ho.c1b2_00'])
    xo.assert_allclose(
        2 * ho_cw.phi * np.cos(ho_cw.alpha), delta_px,
        rtol=0, atol=1e-14)
    xo.assert_allclose(
        2 * ho_cw.phi * np.sin(ho_cw.alpha), delta_py,
        rtol=0, atol=1e-14)
    assert ho_cw.other_beam_q0 == env.acw.particle_ref.q0
    assert abs(lr_cw.post_subtract_px) > 0
    assert abs(lr_cw.post_subtract_py) > 0
    assert abs(ho_cw.post_subtract_px) > 0
    assert abs(ho_cw.post_subtract_py) > 0

    # Check the counter-rotating transformation on representative ACW lenses.
    lr_acw = env.acw['bb_lr.r1b2_01']
    ho_acw = env.acw['bb_ho.c1b2_00']
    assert isinstance(lr_acw, xf.BeamBeamBiGaussian2D)
    assert isinstance(ho_acw, xf.BeamBeamBiGaussian3D)
    xo.assert_allclose(
        lr_acw.other_beam_Sigma_11,
        cov_cw['Sigma11', 'bb_lr.r1b1_01'], rtol=1e-14)
    xo.assert_allclose(
        lr_acw.other_beam_shift_x,
        -legacy_separation_acw['bb_lr.r1b2_01'][0],
        rtol=0, atol=1e-14)
    xo.assert_allclose(
        lr_acw.other_beam_shift_y,
        legacy_separation_acw['bb_lr.r1b2_01'][1],
        rtol=0, atol=1e-14)
    xo.assert_allclose(
        ho_acw.slices_other_beam_Sigma_33[0],
        cov_cw['Sigma33', 'bb_ho.c1b1_00'], rtol=1e-14)
    xo.assert_allclose(
        ho_acw.other_beam_shift_x,
        -legacy_separation_acw['bb_ho.c1b2_00'][0],
        rtol=0, atol=1e-14)
    xo.assert_allclose(
        ho_acw.other_beam_shift_y,
        legacy_separation_acw['bb_ho.c1b2_00'][1],
        rtol=0, atol=1e-14)
    delta_px_acw = (tw_acw['px', 'bb_ho.c1b2_00']
                    - tw_cw['px', 'bb_ho.c1b1_00'])
    delta_py_acw = -(tw_acw['py', 'bb_ho.c1b2_00']
                     - tw_cw['py', 'bb_ho.c1b1_00'])
    xo.assert_allclose(
        2 * ho_acw.phi * np.cos(ho_acw.alpha), delta_px_acw,
        rtol=0, atol=1e-14)
    xo.assert_allclose(
        2 * ho_acw.phi * np.sin(ho_acw.alpha), delta_py_acw,
        rtol=0, atol=1e-14)

    assert env['beambeam_scale'] == 1
    env['beambeam_scale'] = 0.37
    for line, records in ((env.cw, records_cw), (env.acw, records_acw)):
        for record in records:
            xo.assert_allclose(line[record.name].scale_strength, 0.37,
                               rtol=0, atol=0)

    # Configuration is repeatable: it first makes the tagged lenses inactive,
    # reanalyses the bare lines, and then restores fully configured elements.
    env.xfields.configure_beambeam_interactions(
        num_particles=1e11,
        nemitt_x=2e-6, nemitt_y=2.5e-6,
        crab_strong_beam=False)
    xo.assert_allclose(
        env.cw['bb_lr.r1b1_01'].other_beam_shift_x,
        legacy_separation_cw['bb_lr.r1b1_01'][0],
        rtol=0, atol=1e-14)
    assert env['beambeam_scale'] == 1


def test_conventional_element_state_drives_filling_pattern():
    shared_ips = {'ip1': xt.Marker(), 'ip2': xt.Marker()}
    env = xt.Environment(lines={
        'cw': _make_conventional_toy_ring('cw', shared_ips),
        'acw': _make_conventional_toy_ring('acw', shared_ips),
    })
    env.xfields.install_beambeam_interactions(
        clockwise_line='cw', anticlockwise_line='acw',
        ip_names=['ip1', 'ip2'],
        num_long_range_encounters_per_side=[1, 1],
        num_slices_head_on=3,
        harmonic_number=8, bunch_spacing_buckets=1,
        sigmaz=0.1, delay_at_ips_slots=[0, 6])
    env = xt.Environment.from_dict(env.to_dict())
    env.xfields.configure_beambeam_interactions(
        num_particles=1e11, nemitt_x=2e-6, nemitt_y=2.5e-6,
        crab_strong_beam=False)

    filling_cw = np.zeros(8, dtype=int)
    filling_acw = np.zeros(8, dtype=int)
    filling_cw[[0, 2]] = 1
    filling_acw[[0, 1, 6]] = 1
    env.xfields.apply_filling_pattern(
        filling_pattern_cw=filling_cw,
        filling_pattern_acw=filling_acw,
        i_bunch_cw=0, i_bunch_acw=0)

    assert env.cw['bb_ho.c1b1_00'].scale_strength == 1
    assert env.cw['bb_lr.r1b1_01'].scale_strength == 1
    assert env.cw['bb_lr.l1b1_01'].scale_strength == 0
    assert env.cw['bb_ho.c2b1_00'].scale_strength == 1
    assert env.acw['bb_ho.c2b2_00'].scale_strength == 1
    assert env.acw['bb_lr.r2b2_01'].scale_strength == 0

    env['beambeam_scale'] = 0.4
    assert env.cw['bb_lr.r1b1_01'].scale_strength == 0.4
    assert env.cw['bb_lr.l1b1_01'].scale_strength == 0


def test_conventional_single_beam_antisymmetry_configuration():
    # Exercise the one-line LHC configuration used when the missing opposing
    # beam is reconstructed from the optics antisymmetry around each IP. This
    # path deliberately retains its legacy Twiss/survey covariance handling.
    env = xt.Environment(lines={
        'cw': _make_conventional_toy_ring('cw', shared_ips={}),
    })
    env.xfields.install_beambeam_interactions(
        clockwise_line='cw', anticlockwise_line=None,
        ip_names=['ip1', 'ip2'],
        num_long_range_encounters_per_side=[1, 1],
        num_slices_head_on=3,
        harmonic_number=8, bunch_spacing_buckets=1,
        sigmaz=0.1)

    twiss = env.cw.twiss()
    covariance = twiss.get_beam_covariance(
        nemitt_x=2e-6, nemitt_y=2.5e-6)
    env.xfields.configure_beambeam_interactions(
        num_particles=1e11,
        nemitt_x=2e-6, nemitt_y=2.5e-6,
        crab_strong_beam=False,
        use_antisymmetry=True,
        separation_bumps={'ip1': 'x', 'ip2': 'y'})

    lr_right = env.cw['bb_lr.r1b1_01']
    xo.assert_allclose(
        lr_right.other_beam_Sigma_11,
        covariance['Sigma11', 'bb_lr.l1b1_01'], rtol=1e-14)
    xo.assert_allclose(
        lr_right.other_beam_Sigma_33,
        covariance['Sigma33', 'bb_lr.l1b1_01'], rtol=1e-14)
    xo.assert_allclose(lr_right.other_beam_num_particles, 1e11,
                       rtol=0, atol=0)
    xo.assert_allclose(lr_right.scale_strength, 1, rtol=0, atol=0)
