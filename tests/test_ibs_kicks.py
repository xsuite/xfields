import pytest
import xtrack as xt
from ibs_conftest import XTRACK_TEST_DATA
from numpy.testing import assert_allclose

from xfields.ibs import IBSAnalyticalKick, IBSKineticKick

# ----- Test API behaviour ----- #


@pytest.mark.parametrize("update_every", [10, 50])
@pytest.mark.parametrize("name", ["ibskicker", "elementname"])
def test_line_configuration_providing_element(update_every, name):
    """
    Pass the kick element to line configuration method
    and check the right flags are set. The line we use
    does not matter.
    """
    # -----------------------------------------------------
    # Load the line with a .particle_ref and create element
    sps_dir = XTRACK_TEST_DATA / "sps_w_spacecharge"
    linefile = sps_dir / "line_no_spacecharge_and_particle.json"
    line = xt.Line.from_json(linefile)
    ibskick = IBSKineticKick(num_slices=50)  # same for IBSAnalyticalKick
    # -----------------------------------------------------
    # Check the flags are as they should be before configuration
    assert ibskick.update_every is None
    assert ibskick._name is None
    assert ibskick._twiss is None
    assert ibskick._scale_strength == 0
    # -----------------------------------------------------
    # Configure and check flags are set as they should be
    line.configure_intrabeam_scattering(
        element=ibskick,
        name=name,  # kwarg given to .insert_element()
        at_s=100,  # kwarg given to .insert_element()
        update_every=update_every,
    )
    assert ibskick.update_every == update_every
    assert ibskick._name == name
    assert isinstance(ibskick._twiss, xt.TwissTable)
    assert ibskick._scale_strength == 1


@pytest.mark.parametrize("update_every", [10, 50])
@pytest.mark.parametrize("name", ["ibskicker", "elementname"])
def test_line_configuration_manual_insertion(update_every, name):
    """
    Manually insert the kick element in the line then
    calling the configuration method and checking the
    right flags are set. The line we use does not matter.
    """
    # -----------------------------------------------------
    # Load the line with a .particle_ref and create element
    sps_dir = XTRACK_TEST_DATA / "sps_w_spacecharge"
    linefile = sps_dir / "line_no_spacecharge_and_particle.json"
    line = xt.Line.from_json(linefile)
    # -----------------------------------------------------
    # Create element, check flags and manually insert it
    ibskick = IBSAnalyticalKick(formalism="b&m", num_slices=50)  # same for IBSKineticKick
    line.insert_element(element=ibskick, index=-1, name=name)
    assert ibskick.update_every is None
    assert ibskick._name is None
    assert ibskick._twiss is None
    assert ibskick._scale_strength == 0
    # -----------------------------------------------------
    # Configure and check flags are set as they should be
    line.configure_intrabeam_scattering(update_every=update_every)
    assert ibskick.update_every == update_every
    assert ibskick._name == name
    assert isinstance(ibskick._twiss, xt.TwissTable)
    assert ibskick._scale_strength == 1


# def test_configuration_raises_on_more_than_one_element():
#     pass


# def test_configuration_raises_on_below_transition():
#     pass


# ----- Test coefficients computation ----- #


# def test_kick_coefficients():
#     pass


# def test_kinetic_coefficients():
#     pass
