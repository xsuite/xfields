import json

import numpy as np
import xtrack as xt

with open('../../test_data/hllhc14_for_bb_tests/line_b1.json', 'r') as fid:
    dct_b1 = json.load(fid)
line_b1 = xt.Line.from_dict(dct_b1)

tracker = line_b1.build_tracker()

tw = tracker.twiss()

tracker.match(vary=['kqtf.b1', 'kqtd.b1'],
    targets = [('qx', 62.315), ('qy', 60.325)])

tracker.match(vary=['ksf.b1', 'ksd.b1',],
    targets = [('dqx', 15), ('dqy', 15)])

tracker.match(vary=['kqtf.b1', 'kqtd.b1', 'ksf.b1', 'ksd.b1'],
    targets = [('qx', 62.315), ('qy', 60.325), ('dqx', 15), ('dqy', 15)])
