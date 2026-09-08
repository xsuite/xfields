# copyright ############################### #
# This file is part of the Xfields Package. #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""Convert the LHC operational filling format to Xsuite-friendly files."""

import json
from pathlib import Path


HERE = Path(__file__).parent
SOURCE = HERE / '25ns_2460b_2448_2092_2239_144bpi_20inj.json'
FULL_OUTPUT = HERE / 'filling_25ns_2460b.json'
EXAMPLE_OUTPUT = HERE / 'filling_25ns_104b.json'

# The small dataset contains a 48-bunch window of the longest CW train and the
# regions paired with it by head-on collisions in the default LHC example.
EXAMPLE_WINDOW = 48
EXAMPLE_IP_OFFSETS = (0, 891, 2670)


def _filled_slots(filling_pattern, *, name, num_slots):
    if len(filling_pattern) != num_slots:
        raise ValueError(
            f'`{name}` has {len(filling_pattern)} entries; expected '
            f'{num_slots}.')
    if any(value not in (0, 1) for value in filling_pattern):
        raise ValueError(f'`{name}` must be a binary filling pattern.')
    return [slot for slot, value in enumerate(filling_pattern) if value]


def _longest_contiguous_run(filled_slots):
    runs = []
    current_run = []
    for slot in filled_slots:
        if current_run and slot != current_run[-1] + 1:
            runs.append(current_run)
            current_run = []
        current_run.append(slot)
    if current_run:
        runs.append(current_run)
    return max(runs, key=len)


def _example_subset(filled_slots_cw, filled_slots_acw, num_slots):
    reference_slots = _longest_contiguous_run(
        filled_slots_cw)[:EXAMPLE_WINDOW]
    candidate_slots = {
        (slot + sign * offset) % num_slots
        for slot in reference_slots
        for offset in EXAMPLE_IP_OFFSETS
        for sign in (-1, 1)
    }
    return (
        sorted(candidate_slots.intersection(filled_slots_cw)),
        sorted(candidate_slots.intersection(filled_slots_acw)),
    )


def _write_filling(path, *, name, source, num_slots,
                   filled_slots_cw, filled_slots_acw):
    filling = {
        'name': name,
        'source': source,
        'num_slots': num_slots,
        'filled_slots_cw': filled_slots_cw,
        'filled_slots_acw': filled_slots_acw,
    }
    with open(path, 'w') as fid:
        json.dump(filling, fid, indent=2)
        fid.write('\n')
    print(f'Wrote {path}')


def main():
    with open(SOURCE) as fid:
        lhc_filling = json.load(fid)

    # In this LHC model beam 1 is clockwise and beam 2 is anticlockwise.
    filling_pattern_cw = lhc_filling['schemebeam1']
    filling_pattern_acw = lhc_filling['schemebeam2']
    num_slots = len(filling_pattern_cw)

    filled_slots_cw = _filled_slots(
        filling_pattern_cw, name='schemebeam1', num_slots=num_slots)
    filled_slots_acw = _filled_slots(
        filling_pattern_acw, name='schemebeam2', num_slots=num_slots)

    _write_filling(
        FULL_OUTPUT,
        name=lhc_filling['schemeName'],
        source=SOURCE.name,
        num_slots=num_slots,
        filled_slots_cw=filled_slots_cw,
        filled_slots_acw=filled_slots_acw)

    example_slots_cw, example_slots_acw = _example_subset(
        filled_slots_cw, filled_slots_acw, num_slots)
    _write_filling(
        EXAMPLE_OUTPUT,
        name='25ns_104b_example_subset',
        source=FULL_OUTPUT.name,
        num_slots=num_slots,
        filled_slots_cw=example_slots_cw,
        filled_slots_acw=example_slots_acw)


if __name__ == '__main__':
    main()
