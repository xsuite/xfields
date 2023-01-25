import json

import numpy as np
import xtrack as xt

with open('../../test_data/hllhc14_for_bb_tests/line_b1.json', 'r') as fid:
    dct_b1 = json.load(fid)
line_b1 = xt.Line.from_dict(dct_b1)

tracker = line_b1.build_tracker()

target_qx = 62.315
target_qy = 60.325

tw = tracker.twiss()

tracker.match(
    vary=[
        xt.Vary('kqtf.b1', step=1e-8),
        xt.Vary('kqtd.b1', step=1e-8),
        xt.Vary('ksf.b1', step=1e-8),
        xt.Vary('ksd.b1', step=1e-8),
    ],
    targets = [
        xt.Target('qx', target_qx, tol=1e-4),
        xt.Target('qy', target_qy, tol=1e-4),
        xt.Target('dqx', 15., tol=0.05),
        xt.Target('dqy', 15., tol=0.05)])


