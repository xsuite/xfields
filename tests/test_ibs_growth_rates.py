from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import xtrack as xt
from conftest import (
    get_madx_ibs_growth_rates,
    get_parameters_from_madx_beam,
    get_ref_particle_from_madx_beam,
)
from cpymad.madx import Madx

from xfields.ibs import get_intrabeam_scattering_growth_rates

# /!\ This assumes xtrack repo is sitting next to xfields repo
XT_TEST_DATA = Path(__file__).parent.parent.parent / "xtrack" / "test_data/"


def test_clic_dr_growth_rates():
    """Compare to MAD-X for the CLIC DR."""
    # -----------------------------------------------------
    # Load ELENA sequence, beam and strengths in MAD-X
    clic_dr_dir = XT_TEST_DATA / "clic_dr"
    madx = Madx(stdout=False)
    madx.call(str(clic_dr_dir / "sequence.madx"))
    madx.use(sequence="ring")
    mad_Tx, mad_Ty, mad_Tz = get_madx_ibs_growth_rates(madx)
    # -----------------------------------------------------
    # Get equivalent line and parameters
    line = xt.Line.from_madx_sequence(madx.sequence.ring)
    line.particle_ref = get_ref_particle_from_madx_beam(madx)
    tw = line.twiss(method="4d")
    npart, gemitt_x, gemitt_y, sigd, bl = get_parameters_from_madx_beam(madx)
    # -----------------------------------------------------
    # Get growth rates with Nagaitsev formalism
    nag_rates = get_intrabeam_scattering_growth_rates(
        twiss=tw,
        formalism="nagaitsev",
        total_beam_intensity=npart,
        gemitt_x=gemitt_x,
        gemitt_y=gemitt_y,
        sigma_delta=sigd,
        bunch_length=bl,
    )
    # -----------------------------------------------------
    # Get growth rates with Bjorken-Mtingwa formalism
    bm_rates = get_intrabeam_scattering_growth_rates(
        twiss=tw,
        formalism="Bjorken-Mtingwa",
        total_beam_intensity=npart,
        gemitt_x=gemitt_x,
        gemitt_y=gemitt_y,
        sigma_delta=sigd,
        bunch_length=bl,
    )
    # -----------------------------------------------------
    # Compare the results - Nagaitsev
    assert_allclose(nag_rates.Tx, mad_Tx, atol=1e-14, rtol=5e-2)
    assert_allclose(nag_rates.Ty, mad_Ty, atol=1e-14, rtol=5e-2)
    assert_allclose(nag_rates.Tz, mad_Tz, atol=1e-14, rtol=5e-2)
    # Compare the results - Bjorken-Mtingwa
    assert_allclose(bm_rates.Tx, mad_Tx, atol=1e-14, rtol=5e-2)
    assert_allclose(bm_rates.Ty, mad_Ty, atol=1e-14, rtol=5e-2)
    assert_allclose(bm_rates.Tz, mad_Tz, atol=1e-14, rtol=5e-2)
