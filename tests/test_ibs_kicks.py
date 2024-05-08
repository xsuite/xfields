import pytest
import xpart as xp
import xtrack as xt
from cpymad.madx import Madx
from ibs_conftest import XTRACK_TEST_DATA, get_ref_particle_from_madx_beam
from numpy.testing import assert_allclose
from xobjects.test_helpers import for_all_test_contexts

from xfields.ibs import IBSAnalyticalKick, IBSKineticKick
from xfields.ibs._kicks import DiffusionCoefficients, FrictionCoefficients, IBSKickCoefficients

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


@pytest.mark.parametrize("num_insertions", (0, 3))
def test_configuration_raises_if_not_one_element(num_insertions):
    """
    Inserting an invalid number of elements in the line
    and checking the configuration method raises. The line
    we use does not matter.
    """
    # -----------------------------------------------------
    # Load the line with a .particle_ref and create element
    sps_dir = XTRACK_TEST_DATA / "sps_w_spacecharge"
    linefile = sps_dir / "line_no_spacecharge_and_particle.json"
    line = xt.Line.from_json(linefile)
    ibskick = IBSKineticKick(num_slices=50)  # same for IBSAnalyticalKick
    # -----------------------------------------------------
    # Inserting element(s) in the line 'num_insertions' times
    for i in range(num_insertions):
        line.insert_element(element=ibskick, index=-1, name=f"ibskick{i}")
    # -----------------------------------------------------
    # Attempting configuration and checking it raises
    with pytest.raises(AssertionError):
        line.configure_intrabeam_scattering(update_every=10)


def test_configuration_raises_on_below_transition_analytical_kick():
    """
    Try to configure an IBSAnalyticalKick in a line
    operating below transition energy and check it raises.
    """
    # -----------------------------------------------------
    # Have MAD-X load ELENA sequence (it is below transition)
    elena_dir = XTRACK_TEST_DATA / "elena"
    madx = Madx(stdout=False)
    madx.call(str(elena_dir / "elena.seq"))
    madx.call(str(elena_dir / "highenergy.beam"))
    madx.call(str(elena_dir / "highenergy.str"))
    madx.use(sequence="elena")
    # -----------------------------------------------------
    # Get the equivalent xtrack.Line and create element
    line = xt.Line.from_madx_sequence(madx.sequence.elena)
    line.particle_ref = get_ref_particle_from_madx_beam(madx)
    ibskick = IBSAnalyticalKick(formalism="B&M", num_slices=50)
    # -----------------------------------------------------
    # Insert element in the line and attempt configuration
    line.insert_element(element=ibskick, index=10, name="ibskick")
    with pytest.raises(AssertionError):
        line.configure_intrabeam_scattering(update_every=10)


# ----- Test coefficients computation ----- #


@for_all_test_contexts(excluding="ContextPyopencl")
@pytest.mark.parametrize("formalism", ["b&m", "nagaitsev"])
def test_kick_coefficients(test_context, formalism):
    """
    We get a line (without vertical disp) and generate a large
    particle distribution to check that the kick coefficients
    computed after configuration are correct values (reference
    values were computed independently with xibs, and checked
    against values from old Michalis scripts.)
    """
    # -----------------------------------------------------
    # Load the line with a .particle_ref and create element
    sps_dir = XTRACK_TEST_DATA / "sps_w_spacecharge"
    linefile = sps_dir / "line_no_spacecharge_and_particle.json"
    line = xt.Line.from_json(linefile)
    ibskick = IBSAnalyticalKick(formalism=formalism, num_slices=50)
    # -----------------------------------------------------
    # Configure in line and generate particles distribution
    line.configure_intrabeam_scattering(element=ibskick, name="ibskick", index=0, update_every=1)
    particles = xp.generate_matched_gaussian_bunch(
        num_particles=250_000,
        nemitt_x=1.2e-5,
        nemitt_y=1.2e-5,
        sigma_z=23e-2,
        total_intensity_particles=1.3e11,
        line=line,
        _context=test_context,
    )
    # -----------------------------------------------------
    # Compute the kick coefficients and check they are correct
    # We compare to expected values for the parameters above
    # with a little tolerance due to distribution generation
    coeffs = ibskick.compute_kick_coefficients(particles)
    refs = IBSKickCoefficients(Kx=9.616535584664553e-10, Ky=0, Kz=1.67606352e-08)
    assert_allclose(coeffs.Kx, refs.Kx, rtol=1e-2)
    assert_allclose(coeffs.Ky, refs.Ky, rtol=1e-2)
    assert_allclose(coeffs.Kz, refs.Kz, rtol=1e-2)


@for_all_test_contexts(excluding="ContextPyopencl")
def test_kinetic_coefficients(test_context):
    """
    We get a line and generate a large particle distribution
    to check that the kinetic coefficients computed after
    configuration are correct values (reference values were
    computed independently with xibs, and checked against
    values from old Michalis scripts.)
    """
    # -----------------------------------------------------
    # Load the line with a .particle_ref and create element
    sps_dir = XTRACK_TEST_DATA / "sps_w_spacecharge"
    linefile = sps_dir / "line_no_spacecharge_and_particle.json"
    line = xt.Line.from_json(linefile)
    ibskick = IBSKineticKick(num_slices=50)
    # -----------------------------------------------------
    # Configure in line and generate particles distribution
    line.configure_intrabeam_scattering(element=ibskick, name="ibskick", index=0, update_every=1)
    particles = xp.generate_matched_gaussian_bunch(
        num_particles=250_000,
        nemitt_x=1.2e-5,
        nemitt_y=1.2e-5,
        sigma_z=23e-2,
        total_intensity_particles=1.3e11,
        line=line,
        _context=test_context,
    )
    # -----------------------------------------------------
    # Compute the kinetic coefficients and check they are correct
    # We compare to expected values for the parameters above
    # with a little tolerance due to distribution generation
    diffs, fricts = ibskick.compute_kinetic_coefficients(particles)
    diff_refs = DiffusionCoefficients(Dx=1.45736905e-05, Dy=2.33723969e-06, Dz=1.21668536e-05)
    frict_refs = FrictionCoefficients(Fx=7.34149897e-06, Fy=2.95337318e-06, Fz=5.15410959e-06)
    assert_allclose(diffs.Dx, diff_refs.Dx, rtol=1e-2)
    assert_allclose(diffs.Dy, diff_refs.Dy, rtol=1e-2)
    assert_allclose(diffs.Dz, diff_refs.Dz, rtol=1e-2)
    assert_allclose(fricts.Fx, frict_refs.Fx, rtol=1e-2)
    assert_allclose(fricts.Fy, frict_refs.Fy, rtol=1e-2)
    assert_allclose(fricts.Fz, frict_refs.Fz, rtol=1e-2)
