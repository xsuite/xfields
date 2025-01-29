from turtle import rt

import pytest
import xobjects as xo
import xtrack as xt
from ibs_conftest import XTRACK_TEST_DATA
from matplotlib.image import resample

import xfields as xf

BUNCH_INTENSITY: float = 6.2e9  # 1C bunch intensity

# ----- Fixture for the (configured) BESSY III line ----- #


@pytest.fixture(scope="module")
def bessy3_line_with_radiation() -> xt.Line:
    """
    Loads the BESSY III lattice as a Line and
    configures radiation before returning it.
    """
    # -------------------------------------------
    # Load the line with a particle_ref
    bess3_dir = XTRACK_TEST_DATA / "bessy3"
    linefile = bess3_dir / "line.json"
    line = xt.Line.from_json(linefile)
    # -------------------------------------------
    # Build tracker and configure radiation
    line.build_tracker()
    line.matrix_stability_tol = 1e-2
    line.configure_radiation(model="mean")
    line.compensate_radiation_energy_loss()
    # Run twiss in fixture to compile kernels once
    line.twiss(eneloss_and_damping=True)
    return line


# ----- Test Functions vs Analytical Formulae ----- #


@pytest.mark.parametrize("emittance_coupling_factor", [0.02, 0.1, 1])
def test_equilibrium_vs_analytical_constraint_coupling(
    emittance_coupling_factor, bessy3_line_with_radiation: xt.Line
):
    """TODO: Succinct description of what is tested."""
    # -------------------------------------------
    # Get the twiss with SR effects from the configured line
    tw = bessy3_line_with_radiation.twiss(eneloss_and_damping=True)
    # -------------------------------------------
    # Compute the equilibrium emittances - coupling constraint
    result = tw.compute_equilibrium_emittances_from_sr_and_ibs(
        formalism="Nagaitsev",  # No Dy in the line, faster
        total_beam_intensity=BUNCH_INTENSITY,
        emittance_coupling_factor=emittance_coupling_factor,
        emittance_constraint="coupling",
    )
    # -------------------------------------------
    # Check results vs analytical estimations
    factor = 1 + emittance_coupling_factor * (tw.partition_numbers[1] / tw.partition_numbers[0])
    # Check the horizontal equilibrium emittance
    xo.assert_allclose(
        result.eq_sr_ibs_gemitt_x,
        result.gemitt_x[0] / (1 - result.Tx[-1] / 2 / (tw.damping_constants_s[0] * factor)),
        rtol=1e-2,
    )
    # Check the emittance coupling constraint (also checks vert. eq. emittance)
    xo.assert_allclose(result.gemitt_y / result.gemitt_x, emittance_coupling_factor, rtol=1e-2)
    # Check the longitudinal equilibrium emittance
    xo.assert_allclose(
        result.eq_sr_ibs_gemitt_zeta,
        result.gemitt_zeta[0] / (1 - result.Tz[-1] / 2 / (tw.damping_constants_s[2])),
        rtol=1e-2,
    )

    # else:
    # Check equilibrium emittance
    # xo.assert_allclose(
    #     emittances_x_list[-1],
    #     tw.eq_gemitt_x / (1 - T_x[-1] / 2 / tw.damping_constants_s[0]),
    #     rtol=5e-2,
    # )
    # # Check equilibrium emittance
    # xo.assert_allclose(
    #     emittances_z_list[-1],
    #     tw.eq_gemitt_zeta / (1 - T_z[-1] / 2 / tw.damping_constants_s[2]),
    #     rtol=2e-2,
    # )
    # # Check emittance coupling constraint
    # xo.assert_allclose(
    #     emittances_y_list[-1] / emittances_x_list[-1],
    #     emittance_coupling_factor,
    # )


@pytest.mark.parametrize("emittance_coupling_factor", [0.02, 0.1, 0.2])
def test_ibs_emittance_coupling_factor(
    emittance_coupling_factor, bessy3_line_with_radiation: xt.Line
):
    """
    As the emittance coupling factor increases, the equilibrium emittance
    cannot be compared anymore to the solution of the differential equation
    describing the emittance evolution in presence of IBS and SR if a
    constraint on the emittance is enforced.
    """
    tw = bessy3_line_with_radiation.twiss(eneloss_and_damping=True)
    #######################################
    # Equilibrium emittances calculations #
    #######################################

    time, emittances_x_list, emittances_y_list, emittances_z_list, T_x, T_y, T_z = (
        xf.ibs.compute_equilibrium_emittances_from_sr_and_ibs(
            tw,
            BUNCH_INTENSITY,
            emittance_coupling_factor=emittance_coupling_factor,
        )
    )

    # Check equilibrium emittance
    xo.assert_allclose(
        emittances_x_list[-1],
        tw.eq_gemitt_x
        / (1 + emittance_coupling_factor)
        / (1 - T_x[-1] / 2 / tw.damping_constants_s[0]),
        rtol=1e-1,
    )
    # Check equilibrium emittance
    xo.assert_allclose(
        emittances_z_list[-1],
        tw.eq_gemitt_zeta / (1 - T_z[-1] / 2 / tw.damping_constants_s[2]),
        rtol=2e-2,
    )
    # Check emittance coupling constraint
    xo.assert_allclose(
        emittances_y_list[-1] / emittances_x_list[-1],
        emittance_coupling_factor,
    )


@pytest.mark.parametrize("emittance_coupling_factor", [0.02, 0.1, 1.0])
def test_ibs_emittance_no_constraint(
    emittance_coupling_factor, bessy3_line_with_radiation: xt.Line
):
    """
    Without any emittance constraint, the equilibrium emittance becomes
    almost identical to the solution of the differential equation describing
    the emittance evolution in presence of IBS and SR.
    """
    tw = bessy3_line_with_radiation.twiss(eneloss_and_damping=True)
    initial_emittances = (
        tw.eq_gemitt_x,
        emittance_coupling_factor * tw.eq_gemitt_x,
        tw.eq_gemitt_zeta,
    )
    emittance_constraint = ""
    natural_emittances = (
        tw.eq_gemitt_x,
        emittance_coupling_factor * tw.eq_gemitt_x,
        tw.eq_gemitt_zeta,
    )

    #######################################
    # Equilibrium emittances calculations #
    #######################################

    time, emittances_x_list, emittances_y_list, emittances_z_list, T_x, T_y, T_z = (
        xf.ibs.compute_equilibrium_emittances_from_sr_and_ibs(
            tw,
            BUNCH_INTENSITY,
            initial_emittances=initial_emittances,
            emittance_coupling_factor=emittance_coupling_factor,
            emittance_constraint=emittance_constraint,
            natural_emittances=natural_emittances,
        )
    )

    # Check equilibrium emittance
    xo.assert_allclose(
        emittances_x_list[-1],
        tw.eq_gemitt_x / (1 - T_x[-1] / 2 / tw.damping_constants_s[0]),
        rtol=2e-2,
    )
    # Check equilibrium emittance
    xo.assert_allclose(
        emittances_y_list[-1],
        emittance_coupling_factor * tw.eq_gemitt_x / (1 - T_y[-1] / 2 / tw.damping_constants_s[1]),
        rtol=2e-2,
    )
    # Check equilibrium emittance
    xo.assert_allclose(
        emittances_z_list[-1],
        tw.eq_gemitt_zeta / (1 - T_z[-1] / 2 / tw.damping_constants_s[2]),
        rtol=2e-2,
    )
