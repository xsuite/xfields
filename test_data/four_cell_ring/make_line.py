# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

from pathlib import Path

import numpy as np

import xtrack as xt


def build_line():
    """
    Build a four-cell electron ring.

    The returned line is intentionally independent of any specific collective
    effect. Examples can install additional elements and apertures as needed.
    """
    lbend = 3.0
    angle = np.pi / 2

    lquad = 0.3
    k1qf = 0.1
    k1qd = 0.7

    env = xt.Environment()

    line = env.new_line(components=[
        env.new('mqf.1', xt.Quadrupole, length=lquad, k1=k1qf),
        env.new('d1.1',  xt.Drift, length=1.0),
        env.new('mb1.1', xt.Bend, length=lbend, angle=angle),
        env.new('d2.1',  xt.Drift, length=1.0),

        env.new('mqd.1', xt.Quadrupole, length=lquad, k1=-k1qd),
        env.new('d3.1',  xt.Drift, length=1.0),
        env.new('mb2.1', xt.Bend, length=lbend, angle=angle),
        env.new('d4.1',  xt.Drift, length=1.0),

        env.new('mqf.2', xt.Quadrupole, length=lquad, k1=k1qf),
        env.new('d1.2',  xt.Drift, length=1.0),
        env.new('mb1.2', xt.Bend, length=lbend, angle=angle),
        env.new('d2.2',  xt.Drift, length=1.0),

        env.new('mqd.2', xt.Quadrupole, length=lquad, k1=-k1qd),
        env.new('d3.2',  xt.Drift, length=1.0),
        env.new('mb2.2', xt.Bend, length=lbend, angle=angle),
        env.new('d4.2',  xt.Drift, length=1.0),
    ])

    line.set_particle_ref('electron', p0c=1e9)
    line.configure_bend_model(core='full', edge=None)

    return line


if __name__ == '__main__':
    out_path = Path(__file__).with_name('line.json')
    build_line().to_json(out_path)
