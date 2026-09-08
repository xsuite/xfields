# copyright ############################### #
# This file is part of the Xfields Package. #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""Convert the LHC operational filling format to an Xsuite-friendly file."""

import json
from pathlib import Path


HERE = Path(__file__).parent
SOURCE = HERE / '25ns_2460b_2448_2092_2239_144bpi_20inj.json'
OUTPUT = HERE / 'filling_25ns_2460b.json'


def _filled_slots(filling_pattern, *, name, num_slots):
    if len(filling_pattern) != num_slots:
        raise ValueError(
            f'`{name}` has {len(filling_pattern)} entries; expected '
            f'{num_slots}.')
    if any(value not in (0, 1) for value in filling_pattern):
        raise ValueError(f'`{name}` must be a binary filling pattern.')
    return [slot for slot, value in enumerate(filling_pattern) if value]


def main():
    with open(SOURCE) as fid:
        lhc_filling = json.load(fid)

    # In this LHC model beam 1 is clockwise and beam 2 is anticlockwise.
    filling_pattern_cw = lhc_filling['schemebeam1']
    filling_pattern_acw = lhc_filling['schemebeam2']
    num_slots = len(filling_pattern_cw)

    xsuite_filling = {
        'name': lhc_filling['schemeName'],
        'source': SOURCE.name,
        'num_slots': num_slots,
        'filled_slots_cw': _filled_slots(
            filling_pattern_cw, name='schemebeam1', num_slots=num_slots),
        'filled_slots_acw': _filled_slots(
            filling_pattern_acw, name='schemebeam2', num_slots=num_slots),
    }

    with open(OUTPUT, 'w') as fid:
        json.dump(xsuite_filling, fid, indent=2)
        fid.write('\n')
    print(f'Wrote {OUTPUT}')


if __name__ == '__main__':
    main()
