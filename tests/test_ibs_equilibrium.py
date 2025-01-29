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
    """
    Load the BESSY III line and compute ierations until we reach
    an equilibrium with SR and IBS, in the case where we enforce
    a betatron coupling constraint on the transverse planes. The
    resulting values are tested against an analytical estimate.
    TODO: ref for the analytical formula?
    """
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


@pytest.mark.parametrize("emittance_coupling_factor", [0.02, 0.1, 1])
def test_equilibrium_vs_analytical_constraint_excitation(
    emittance_coupling_factor, bessy3_line_with_radiation: xt.Line
):
    """
    Load the BESSY III line and compute ierations until we reach
    an equilibrium with SR and IBS, in the case where we enforce
    an excitation type constraint on the transverse planes. The
    resulting values are tested against an analytical estimate.
    TODO: ref for the analytical formula?
    """
    # -------------------------------------------
    # Get the twiss with SR effects from the configured line
    tw = bessy3_line_with_radiation.twiss(eneloss_and_damping=True)
    # -------------------------------------------
    # Compute the equilibrium emittances - excitation constraint
    result = tw.compute_equilibrium_emittances_from_sr_and_ibs(
        formalism="Nagaitsev",  # No Dy in the line, faster
        total_beam_intensity=BUNCH_INTENSITY,
        emittance_coupling_factor=emittance_coupling_factor,
        emittance_constraint="excitation",
    )
    # -------------------------------------------
    # Check results vs analytical estimations
    # Check the horizontal equilibrium emittance
    xo.assert_allclose(
        result.eq_sr_ibs_gemitt_x,
        result.gemitt_x[0] / (1 - result.Tx[-1] / 2 / tw.damping_constants_s[0]),
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


@pytest.mark.parametrize("initial_factor", [0.02, 0.1, 1])
def test_equilibrium_vs_analytical_no_constraint(
    initial_factor, bessy3_line_with_radiation: xt.Line
):
    """
    Load the BESSY III line and compute ierations until we reach
    an equilibrium with SR and IBS, whithout any constraint on
    the transverse planes. In that case, the equilibrium emittance
    becomes almost identical to the solution of the differential
    equation describing the emittance evolution in presence of IBS
    and SR.
    TODO: ref for the analytical formula?
    """
    # -------------------------------------------
    # Get the twiss with SR effects from the configured line
    tw = bessy3_line_with_radiation.twiss(eneloss_and_damping=True)
    # Determine initial emittances based on a ratio
    init_gemitt_x = tw.eq_gemitt_x
    init_gemitt_y = init_gemitt_x * initial_factor
    init_gemitt_zeta = tw.eq_gemitt_zeta
    # -------------------------------------------
    # Compute the equilibrium emittances - no constraint
    result = tw.compute_equilibrium_emittances_from_sr_and_ibs(
        formalism="Nagaitsev",  # No Dy in the line, faster
        total_beam_intensity=BUNCH_INTENSITY,
        gemitt_x=init_gemitt_x,
        gemitt_y=init_gemitt_y,
        gemitt_zeta=init_gemitt_zeta,
        emittance_constraint=None,
    )
    # -------------------------------------------
    # Check results vs analytical estimations
    # Check the horizontal equilibrium emittance
    xo.assert_allclose(
        result.eq_sr_ibs_gemitt_x,
        result.gemitt_x[0] / (1 - result.Tx[-1] / 2 / tw.damping_constants_s[0]),
        rtol=1e-2,
    )
    # Check the vertical equilibrium emittance
    xo.assert_allclose(
        result.eq_sr_ibs_gemitt_y,
        result.gemitt_y[0] / (1 - result.Ty[-1] / 2 / tw.damping_constants_s[1]),
        rtol=1e-2
    )
    # Check the longitudinal equilibrium emittance
    xo.assert_allclose(
        result.eq_sr_ibs_gemitt_zeta,
        result.gemitt_zeta[0] / (1 - result.Tz[-1] / 2 / (tw.damping_constants_s[2])),
        rtol=1e-2,
    )
