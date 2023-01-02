import json

import numpy as np
import xtrack as xt

with open('../../test_data/hllhc14_for_bb_tests/line_b1.json', 'r') as fid:
    dct_b1 = json.load(fid)
line_b1 = xt.Line.from_dict(dct_b1)

tracker = line_b1.build_tracker()

target_qx = 62.315
target_qy = 60.315

tw = tracker.twiss()
print('Match tune')
tracker.match(vary=['kqtf.b1', 'kqtd.b1'],
    targets = [('qx', target_qx, 1e-4), ('qy', target_qy, 1e-4)])

print('Match chromaticity')
tracker.match(vary=['ksf.b1', 'ksd.b1',],
    targets = [('dqx', 15, .1), ('dqy', 15, .1)])

print('Match tune and chromaticity')
tracker.match(vary=['kqtf.b1', 'kqtd.b1', 'ksf.b1', 'ksd.b1'],
    targets = [('qx', 62.315, 1e-4), ('qy', 60.325, 1e-4), ('dqx', 15, .1), ('dqy', 15, .1)])
