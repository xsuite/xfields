# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import numpy as np

import xfields as xf
import xobjects as xo
import xtrack as xt

from xfields.config_tools.beambeam_config_tools.config_tools import (
    _compute_delays,
    generate_set_of_bb_encounters_1beam,
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
    # The conventional weak-strong installer expands each logical head-on
    # encounter into slices. Long-range positions and bunch-pairing delays must
    # nevertheless come from the same logical description used by rigid-bunch
    # installation.
    kwargs = dict(
        circumference=80.0,
        harmonic_number=8,
        bunch_spacing_buckets=1,
        numberOfHOSlices=3,
        bunch_particle_charge=1,
        sigt=0.1,
        relativistic_beta=1,
        ip_names=['ip1', 'ip2'],
        numberOfLRPerIRSide=[1, 1],
    )
    encounters_cw = generate_set_of_bb_encounters_1beam(
        beam_name='b1', other_beam_name='b2', **kwargs)
    encounters_acw = generate_set_of_bb_encounters_1beam(
        beam_name='b2', other_beam_name='b1', **kwargs)
    _compute_delays(
        encounters_cw, encounters_acw,
        delay_at_ips_slots=[0, 6], ip_names=['ip1', 'ip2'],
        harmonic_number=8, bunch_spacing_buckets=1)

    for table, orientation in (
            (encounters_cw, 'cw'), (encounters_acw, 'acw')):
        lr = table[table['label'] == 'bb_lr'].sort_values(
            ['ip_name', 'identifier'])
        xo.assert_allclose(lr['atPosition'].to_numpy(), [-5, 5, -5, 5],
                           rtol=0, atol=0)
        expected_delays = ([-1, 1, 5, 7] if orientation == 'cw'
                           else [1, -1, 3, 1])
        xo.assert_allclose(lr['delay_in_slots'].to_numpy(), expected_delays,
                           rtol=0, atol=0)

        head_on = table[table['label'] == 'bb_ho']
        assert len(head_on[head_on['ip_name'] == 'ip1']) == 3
        assert len(head_on[head_on['ip_name'] == 'ip2']) == 3
        expected_head_on_delays = ([0, 6] if orientation == 'cw' else [0, 2])
        for ip_name, expected_delay in zip(
                ['ip1', 'ip2'], expected_head_on_delays):
            xo.assert_allclose(
                head_on[head_on['ip_name'] == ip_name][
                    'delay_in_slots'].to_numpy(),
                expected_delay, rtol=0, atol=0)


def _make_conventional_toy_ring(suffix, shared_ips):
    elements = []
    names = []
    kick_x = 1e-6 if suffix == 'cw' else -2e-6
    kick_y = -0.5e-6 if suffix == 'cw' else 1.5e-6
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
            xt.Drift(length=5.0),
            xt.Multipole(knl=[0, -0.05]),
            xt.Drift(length=5.0),
        ])
        names.extend([
            marker_name,
            f'qf{ii}_{suffix}',
            f'drift_a{ii}_{suffix}',
            f'qd{ii}_{suffix}',
            f'drift_b{ii}_{suffix}',
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
    })
    env.xfields.install_beambeam_interactions(
        clockwise_line='cw', anticlockwise_line='acw',
        ip_names=['ip1', 'ip2'],
        num_long_range_encounters_per_side=[1, 1],
        num_slices_head_on=3,
        harmonic_number=8, bunch_spacing_buckets=1,
        sigmaz=0.1, delay_at_ips_slots=[0, 6])

    df_cw = env._bb_config['dataframes']['clockwise']
    df_acw = env._bb_config['dataframes']['anticlockwise']
    assert len(df_cw) == 10
    assert len(df_acw) == 10
    assert list(df_cw.index) == sorted(df_cw.index)
    assert list(df_acw.index) == sorted(df_acw.index)
    assert set(df_cw['label']) == {'bb_ho', 'bb_lr'}
    assert set(df_acw['label']) == {'bb_ho', 'bb_lr'}
    assert dict(df_cw['other_elementName']) == {
        name: name.replace('b1_', 'b2_') for name in df_cw.index}
    assert dict(df_acw['other_elementName']) == {
        name: name.replace('b2_', 'b1_') for name in df_acw.index}

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

    encounter_instances = df_cw[[
        'ip_name', 'label', 'identifier']].rename(
            columns={'label': 'encounter_type'}).reset_index(drop=True)
    shared_geometry, _ = xf.compute_beambeam_geometry(
        encounter_table=encounter_instances,
        line_cw=env.cw, line_acw=env.acw,
        element_names_cw=df_cw.index,
        element_names_acw=df_cw['other_elementName'],
        nemitt_x=2e-6, nemitt_y=2.5e-6,
        survey_separation=False,
        twiss_cw=tw_cw, twiss_acw=tw_acw)
    for orientation, twiss in (('cw', tw_cw), ('acw', tw_acw)):
        names = shared_geometry[f'element_name_{orientation}']
        for coordinate in ('x', 'px', 'y', 'py'):
            xo.assert_allclose(
                shared_geometry[f'{coordinate}_{orientation}'].to_numpy(),
                [twiss[coordinate, name] for name in names],
                rtol=0, atol=0)

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
        tw_acw['x', 'bb_lr.r1b2_01'] - tw_cw['x', 'bb_lr.r1b1_01'],
        rtol=0, atol=1e-14)
    xo.assert_allclose(
        lr_cw.other_beam_shift_y,
        tw_acw['y', 'bb_lr.r1b2_01'] - tw_cw['y', 'bb_lr.r1b1_01'],
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
        tw_acw['x', 'bb_ho.c1b2_00'] - tw_cw['x', 'bb_ho.c1b1_00'],
        rtol=0, atol=1e-14)
    xo.assert_allclose(
        ho_cw.other_beam_shift_y,
        tw_acw['y', 'bb_ho.c1b2_00'] - tw_cw['y', 'bb_ho.c1b1_00'],
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
        ho_acw.slices_other_beam_Sigma_33[0],
        cov_cw['Sigma33', 'bb_ho.c1b1_00'], rtol=1e-14)

    assert env['beambeam_scale'] == 1
    env['beambeam_scale'] = 0.37
    for line, dataframe in ((env.cw, df_cw), (env.acw, df_acw)):
        for name in dataframe.index:
            xo.assert_allclose(line[name].scale_strength, 0.37,
                               rtol=0, atol=0)
