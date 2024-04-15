from pathlib import Path

import numpy as np
import xtrack as xt
from conftest import (
    get_madx_ibs_growth_rates,
    get_parameters_from_madx_beam,
    get_ref_particle_from_madx_beam,
)
from cpymad.madx import Madx

from xfields.ibs import get_intrabeam_scattering_growth_rates

XT_TEST_DATA = Path(__file__).parent.parent.parent / "xtrack" / "test_data/"


def test_clic_growth_rates():
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
        num_particles=npart,
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
        num_particles=npart,
        gemitt_x=gemitt_x,
        gemitt_y=gemitt_y,
        sigma_delta=sigd,
        bunch_length=bl,
    )
    # -----------------------------------------------------
    # Compare the results - Nagaitsev
    assert np.isclose(nag_rates.Tx, mad_Tx, rtol=5e-2)
    assert np.isclose(nag_rates.Ty, mad_Ty, rtol=5e-2)
    assert np.isclose(nag_rates.Tz, mad_Tz, rtol=5e-2)
    # Compare the results - Bjorken-Mtingwa
    assert np.isclose(bm_rates.Tx, mad_Tx, rtol=5e-2)
    assert np.isclose(bm_rates.Ty, mad_Ty, rtol=5e-2)
    assert np.isclose(bm_rates.Tz, mad_Tz, rtol=5e-2)
