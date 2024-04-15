from pathlib import Path

import numpy as np
from conftest import get_madx_ibs_growth_rates, get_ref_particle_from_madx_beam
from cpymad.madx import Madx

XT_TEST_DATA = Path(__file__).parent.parent.parent / "xtrack/test_data/"


def test_it_works():
    elena_seq = XT_TEST_DATA / "elena" / "elena.seq"
    elena_beam = XT_TEST_DATA / "elena" / "highenergy.beam"
    elena_strengths = XT_TEST_DATA / "elena" / "highenergy.str"
    madx = Madx()
    madx.call(elena_seq.as_posix())
    madx.call(str(elena_strengths))
    madx.call(str(elena_beam))
    madx.use(sequence="elena")
    get_madx_ibs_growth_rates(madx)
    get_ref_particle_from_madx_beam(madx)
