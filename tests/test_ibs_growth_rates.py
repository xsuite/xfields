from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import xtrack as xt
from conftest import (
    get_madx_ibs_growth_rates,
    get_parameters_from_madx_beam,
    get_ref_particle_from_madx_beam,
    set_madx_beam_parameters,
)
from cpymad.madx import Madx

from xfields.ibs import get_intrabeam_scattering_growth_rates

<<<<<<< HEAD
# /!\ This assumes xtrack repo is sitting next to xfields repo
XT_TEST_DATA = Path(__file__).parent.parent.parent / "xtrack" / "test_data/"
=======
# ------------------------------------------------------------------------
# We compare our values to the ones of MAD-X, hence in the numpy function
# ours should be the first argument.
#
# We also use an absolute tolerance of 1e-8 by definition, as growth rates
# so small or smaller should just not be considered since the corresponding
# damping / growth time is crazy big:
#     if T = 1e-8 [1/s],
#     then tau = 1/T > 3 years,
#     and we are pretty safe from IBS.
>>>>>>> d5d364f (add comment explanation of parameters order, and absolute tolerance value used)


def test_clic_dr_growth_rates():
    """Compare to MAD-X for the CLIC DR."""
    # -----------------------------------------------------
<<<<<<< HEAD
    # Load ELENA sequence, beam and strengths in MAD-X
    clic_dr_dir = XT_TEST_DATA / "clic_dr"
=======
    # Have MAD-X load CLIC DR sequence, beam etc.
    clic_dr_dir = XTRACK_TEST_DATA / "clic_dr"
>>>>>>> d5d364f (add comment explanation of parameters order, and absolute tolerance value used)
    madx = Madx(stdout=False)
    madx.call(str(clic_dr_dir / "sequence.madx"))
    madx.use(sequence="ring")
    # -----------------------------------------------------
    # Set beam parameters and get growth rates
    set_madx_beam_parameters(
        madx,
        total_beam_intensity=5e9,
        nemitt_x=5.66e-7,
        nemitt_y=3.70e-9,
        sigma_delta=1.75e-3,
        bunch_length=1.58e-3,
    )
    mad_Tx, mad_Ty, mad_Tz = get_madx_ibs_growth_rates(madx)
    # -----------------------------------------------------
    # Get equivalent xtrack.Line and parameters
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
<<<<<<< HEAD
<<<<<<< HEAD
    # Compare the results - Nagaitsev
    assert_allclose(nag_rates.Tx, mad_Tx, atol=1e-14, rtol=5e-2)
    assert_allclose(nag_rates.Ty, mad_Ty, atol=1e-14, rtol=5e-2)
    assert_allclose(nag_rates.Tz, mad_Tz, atol=1e-14, rtol=5e-2)
=======
    # Compare the results - Nagaitsev (atol since very small values)
    assert_allclose(nag_rates.Tx, mad_Tx, atol=1e-10, rtol=5e-2)
    assert_allclose(nag_rates.Ty, mad_Ty, atol=1e-10, rtol=5e-2)
    assert_allclose(nag_rates.Tz, mad_Tz, atol=1e-10, rtol=5e-2)
>>>>>>> d5d364f (add comment explanation of parameters order, and absolute tolerance value used)
=======
    # Compare the results - Nagaitsev
    assert_allclose(nag_rates.Tx, mad_Tx, atol=1e-8, rtol=11.5e-2)
    assert_allclose(nag_rates.Ty, mad_Ty, atol=1e-8, rtol=5e-2)
    assert_allclose(nag_rates.Tz, mad_Tz, atol=1e-8, rtol=5e-2)
>>>>>>> ad7675c (needed to set beam parameters as xtrack files leave default emittances etc)
    # Compare the results - Bjorken-Mtingwa
    assert_allclose(bm_rates.Tx, mad_Tx, atol=1e-8, rtol=11.5e-2)
    assert_allclose(bm_rates.Ty, mad_Ty, atol=1e-8, rtol=5e-2)
    assert_allclose(bm_rates.Tz, mad_Tz, atol=1e-8, rtol=5e-2)

