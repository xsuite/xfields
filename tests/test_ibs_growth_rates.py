<<<<<<< HEAD
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
=======
import pytest
>>>>>>> 93664ef (parametrize bunched, do not test nagaitsev in coasting as it makes big approximation)
import xtrack as xt
<<<<<<< HEAD
from conftest import (
=======
from ibs_conftest import (
    XTRACK_TEST_DATA,
>>>>>>> 897a3be (PR comments)
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


@pytest.mark.parametrize("bunched", [True, False])
def test_clic_dr_growth_rates(bunched):
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
        bunched=bunched,
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
        bunched=bunched,
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
        bunched=bunched,
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
<<<<<<< HEAD
    assert_allclose(nag_rates.Tx, mad_Tx, atol=1e-8, rtol=11.5e-2)
    assert_allclose(nag_rates.Ty, mad_Ty, atol=1e-8, rtol=5e-2)
    assert_allclose(nag_rates.Tz, mad_Tz, atol=1e-8, rtol=5e-2)
>>>>>>> ad7675c (needed to set beam parameters as xtrack files leave default emittances etc)
=======
    if bunched is True:  # in Nagaitsev coasting makes big assumptions
        assert_allclose(nag_rates.Tx, mad_Tx, atol=1e-8, rtol=11.5e-2)
        assert_allclose(nag_rates.Ty, mad_Ty, atol=1e-8, rtol=5e-2)
        assert_allclose(nag_rates.Tz, mad_Tz, atol=1e-8, rtol=5e-2)
>>>>>>> 93664ef (parametrize bunched, do not test nagaitsev in coasting as it makes big approximation)
    # Compare the results - Bjorken-Mtingwa
    assert_allclose(bm_rates.Tx, mad_Tx, atol=1e-8, rtol=11.5e-2)
    assert_allclose(bm_rates.Ty, mad_Ty, atol=1e-8, rtol=5e-2)
    assert_allclose(bm_rates.Tz, mad_Tz, atol=1e-8, rtol=5e-2)


@pytest.mark.parametrize("bunched", [True, False])
def test_sps_injection_protons_growth_rates(bunched):
    """Compare to MAD-X for the SPS injection protons."""
    # -----------------------------------------------------
    # Have MAD-X load CLIC DR sequence, beam etc.
    sps_dir = XTRACK_TEST_DATA / "sps_w_spacecharge"
    madx = Madx(stdout=False)
    madx.call(str(sps_dir / "sps_thin.seq"))
    madx.use(sequence="sps")
    # -----------------------------------------------------
    # Beam is fully setup in file, get growth rates
    madx.sequence.sps.beam.bunched = bunched
    mad_Tx, mad_Ty, mad_Tz = get_madx_ibs_growth_rates(madx)
    # -----------------------------------------------------
    # Get equivalent xtrack.Line and parameters
    line = xt.Line.from_madx_sequence(madx.sequence.sps)
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
        bunched=bunched,
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
        bunched=bunched,
    )
    # -----------------------------------------------------
    # Compare the results - Nagaitsev
    if bunched is True:  # in Nagaitsev coasting makes big assumptions
        # Computed with different formalism than MAD-X so 10% isn't crazy
        assert_allclose(nag_rates.Tx, mad_Tx, atol=1e-8, rtol=2.5e-2)
        assert_allclose(nag_rates.Ty, mad_Ty, atol=1e-8, rtol=10e-2)
        assert_allclose(nag_rates.Tz, mad_Tz, atol=1e-8, rtol=2.5e-2)
    # Compare the results - Bjorken-Mtingwa
    assert_allclose(bm_rates.Tx, mad_Tx, atol=1e-8, rtol=2.5e-2)
    assert_allclose(bm_rates.Ty, mad_Ty, atol=1e-8, rtol=2.5e-2)
    assert_allclose(bm_rates.Tz, mad_Tz, atol=1e-8, rtol=2.5e-2)


@pytest.mark.parametrize("bunched", [True, False])
def test_hllhc14_growth_rates(bunched):
    """
    Compare to MAD-X for the HLLHC14 protons.
    The lattice has vertical dispersion so Nagaitsev
    vertical growth rate will be blatantly wrong.
    """
    # -----------------------------------------------------
    # Have MAD-X load SPS sequence, beam etc. as done in
    # the script next to it that creates the xtrack.Line
    hllhc14_dir = XTRACK_TEST_DATA / "hllhc14_input_mad"
    madx = Madx(stdout=False)
    madx.call(str(hllhc14_dir / "final_seq.madx"))
    madx.use(sequence="lhcb1")
    # -----------------------------------------------------
    # Beam is fully setup in file, get growth rates
    madx.sequence.lhcb1.beam.bunched = bunched
    mad_Tx, mad_Ty, mad_Tz = get_madx_ibs_growth_rates(madx)
    # -----------------------------------------------------
    # Get equivalent xtrack.Line and parameters
    line = xt.Line.from_madx_sequence(madx.sequence.lhcb1)
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
        bunched=bunched,
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
        bunched=bunched,
    )
    # -----------------------------------------------------
    # Compare the results - Nagaitsev (don't compare vertical
    # as lattice has Dy and formalism is wrong in this case)
    if bunched is True:  # in Nagaitsev coasting makes big assumptions
        assert_allclose(nag_rates.Tx, mad_Tx, atol=1e-8, rtol=4e-2)
        assert_allclose(nag_rates.Tz, mad_Tz, atol=1e-8, rtol=2.5e-2)
    # Compare the results - Bjorken-Mtingwa
    assert_allclose(bm_rates.Tx, mad_Tx, atol=1e-8, rtol=4e-2)
    assert_allclose(bm_rates.Ty, mad_Ty, atol=1e-8, rtol=2.5e-2)
    assert_allclose(bm_rates.Tz, mad_Tz, atol=1e-8, rtol=2.5e-2)
