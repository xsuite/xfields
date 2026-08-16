# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import xfields as xf
import xobjects as xo

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
