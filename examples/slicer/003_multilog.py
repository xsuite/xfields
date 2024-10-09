import xtrack as xt
import xpart as xp
import xfields as xf

import numpy as np

line = xt.Line.from_json(
    '../../../xtrack/test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.build_tracker()

line.vars['vrf400'] = 16

class MyLog:
    def __init__(self):
        self._store = ['a', 'b']

    def __call__(self, line, particle):
        return {'a': 1, 'b': 2}

p = line.build_particles(x=0)

line.track(p, num_turns=10, log=xt.Log(moments=MyLog()))