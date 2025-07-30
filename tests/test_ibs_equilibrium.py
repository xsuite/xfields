import pytest
import xobjects as xo
import xtrack as xt
from ibs_conftest import XTRACK_TEST_DATA

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
    linefile = bess3_dir / "bessy3.json"
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


@pytest.mark.parametrize("emittance_coupling_factor", [0.02, 0.1, 1.0])
def test_equilibrium_vs_analytical_constraint_coupling(
    emittance_coupling_factor: float, bessy3_line_with_radiation: xt.Line
):
    """
    Load the BESSY III line and compute ierations until we reach
    an equilibrium with SR and IBS, in the case where we enforce
    a betatron coupling constraint on the transverse planes. The
    resulting values are tested against an analytical estimate.
    """
    # -------------------------------------------
    # Get the twiss with SR effects from the configured line
    tw = bessy3_line_with_radiation.twiss(eneloss_and_damping=True)
    # -------------------------------------------
    # Compute the equilibrium emittances - coupling constraint
    result = tw.get_ibs_and_synrad_emittance_evolution(
        formalism="nagaitsev",  # No Dy in the line, faster
        total_beam_intensity=BUNCH_INTENSITY,
        emittance_coupling_factor=emittance_coupling_factor,
        emittance_constraint="coupling",
    )
    # -------------------------------------------
    # Check results vs analytical estimations (uses last step's growth rate!)
    factor = 1 + emittance_coupling_factor * (tw.partition_numbers[1] / tw.partition_numbers[0])
    # Check the horizontal equilibrium emittance
    xo.assert_allclose(
        result.eq_sr_ibs_gemitt_x,
        result.gemitt_x[0] / (1 - result.Kx[-1] / (tw.damping_constants_s[0] * factor)),
        rtol=1e-2,
    )
    # Check the emittance coupling constraint (also checks vert. eq. emittance)
    xo.assert_allclose(result.gemitt_y / result.gemitt_x, emittance_coupling_factor, rtol=1e-2)
    # Check the longitudinal equilibrium emittance
    xo.assert_allclose(
        result.eq_sr_ibs_gemitt_zeta,
        result.gemitt_zeta[0] / (1 - result.Kz[-1] / (tw.damping_constants_s[2])),
        rtol=1e-2,
    )


@pytest.mark.parametrize("emittance_coupling_factor", [0.02, 0.1, 1.0])
def test_equilibrium_vs_analytical_constraint_excitation(
    emittance_coupling_factor: float, bessy3_line_with_radiation: xt.Line
):
    """
    Load the BESSY III line and compute ierations until we reach
    an equilibrium with SR and IBS, in the case where we enforce
    an excitation type constraint on the transverse planes. The
    resulting values are tested against an analytical estimate.
    """
    # -------------------------------------------
    # Get the twiss with SR effects from the configured line
    tw = bessy3_line_with_radiation.twiss(eneloss_and_damping=True)
    # -------------------------------------------
    # Compute the equilibrium emittances - excitation constraint
    result = tw.get_ibs_and_synrad_emittance_evolution(
        formalism="nagaitsev",  # No Dy in the line, faster
        total_beam_intensity=BUNCH_INTENSITY,
        emittance_coupling_factor=emittance_coupling_factor,
        emittance_constraint="excitation",
    )
    # -------------------------------------------
    # Check results vs analytical estimations (uses last step's growth rate!)
    # Check the horizontal equilibrium emittance
    xo.assert_allclose(
        result.eq_sr_ibs_gemitt_x,
        result.gemitt_x[0] / (1 - result.Kx[-1] / tw.damping_constants_s[0]),
        rtol=1e-2,
    )
    # Check the emittance coupling constraint (also checks vert. eq. emittance)
    xo.assert_allclose(result.gemitt_y / result.gemitt_x, emittance_coupling_factor, rtol=1e-2)
    # Check the longitudinal equilibrium emittance
    xo.assert_allclose(
        result.eq_sr_ibs_gemitt_zeta,
        result.gemitt_zeta[0] / (1 - result.Kz[-1] / (tw.damping_constants_s[2])),
        rtol=1e-2,
    )


# This factor should not be too high as otherwise the starting
# vertical emittance will be way out of realistic values
@pytest.mark.parametrize("initial_factor", [0.01, 0.02, 0.05])
def test_equilibrium_vs_analytical_no_constraint(
    initial_factor: float, bessy3_line_with_radiation: xt.Line
):
    """
    Load the BESSY III line and compute ierations until we reach
    an equilibrium with SR and IBS, whithout any constraint on
    the transverse planes. In that case, the equilibrium emittance
    becomes almost identical to the solution of the differential
    equation describing the emittance evolution in presence of IBS
    and SR.
    """
    # -------------------------------------------
    # Get the twiss with SR effects from the configured line
    tw = bessy3_line_with_radiation.twiss(eneloss_and_damping=True)
    # -------------------------------------------
    # Compute the equilibrium emittances - no constraint
    # No constraint so no renormalization of transverse emittances
    # and SR eq in vertical is 0 so we change it to avoid exact 0
    result = tw.get_ibs_and_synrad_emittance_evolution(
        formalism="nagaitsev",  # No Dy in the line, faster
        total_beam_intensity=BUNCH_INTENSITY,
        gemitt_x=tw.eq_gemitt_x,
        gemitt_y=tw.eq_gemitt_x * initial_factor,
        gemitt_zeta=tw.eq_gemitt_zeta,
        emittance_constraint=None,
    )
    # -------------------------------------------
    # Check results vs analytical estimations (uses last step's growth rate!)
    # Check the horizontal equilibrium emittance
    xo.assert_allclose(
        result.eq_sr_ibs_gemitt_x,
        result.gemitt_x[0] / (1 - result.Kx[-1] / tw.damping_constants_s[0]),
        rtol=1e-2,
    )
    # Check the vertical equilibrium emittance
    xo.assert_allclose(
        result.eq_sr_ibs_gemitt_y,
        result.gemitt_y[0] / (1 - result.Ky[-1] / tw.damping_constants_s[1]),
        atol=5e-7, rtol=1e-2,
    )
    # Check the longitudinal equilibrium emittance
    xo.assert_allclose(
        result.eq_sr_ibs_gemitt_zeta,
        result.gemitt_zeta[0] / (1 - result.Kz[-1] / (tw.damping_constants_s[2])),
        rtol=1e-2,
    )


# ----- Test Functions vs ELEGANT Results ----- #

# Ugly hardcoded results from ELEGANT. The keys here
# are emittance coupling factor and associated are the
# corresponding final values we want to compare to.
# ELEGANT gives IBS rates in emittance convention!
ELEGANT_RESULTS: dict[float, dict[str, float]] = {
    0.02: {
        "eps_x": 1.88319e-10,
        "sigma_z": 0.00491982,
        "sigma_delta": 0.00140017,
        "T_x": 106.648,
        "T_z": 32.8772,
    },
    0.1: {
        "eps_x": 1.46119e-10,
        "sigma_z": 0.00441742,
        "sigma_delta": 0.00125719,
        "T_x": 81.5391,
        "T_z": 25.2408,
    },
    0.5: {
        "eps_x": 1.05835e-10,
        "sigma_z": 0.00404997,
        "sigma_delta": 0.00115261,
        "T_x": 63.7394,
        "T_z": 17.7664,
    },
    1: {
        "eps_x": 8.5549e-11,
        "sigma_z": 0.00395442,
        "sigma_delta": 0.00112542,
        "T_x": 63.1134,
        "T_z": 15.4736,
    },
}


@pytest.mark.parametrize("emittance_coupling_factor", [0.02, 0.1, 0.5, 1.0])
def test_equilibrium_vs_elegant(emittance_coupling_factor: float, bessy3_line_with_radiation: xt.Line):
    """
    Load the BESSY III line and compute ierations until we reach
    an equilibrium with SR and IBS, in the case where we enforce
    a betatron coupling constraint on the transverse planes. The
    resulting values are tested against elegant results under the
    same conditions.

    Note
    ----
        The elegant results are expected (and accepted) to be
        slightly different due to small differences in the TWISS
        functions, IBS growth rates etc themselves.
    """
    # -------------------------------------------
    # Get the twiss with SR effects from the configured line
    tw = bessy3_line_with_radiation.twiss(eneloss_and_damping=True)
    # -------------------------------------------
    # Compute the equilibrium emittances - coupling constraint
    result = tw.get_ibs_and_synrad_emittance_evolution(
        formalism="nagaitsev",  # No Dy in the line, faster
        total_beam_intensity=BUNCH_INTENSITY,
        emittance_coupling_factor=emittance_coupling_factor,
        emittance_constraint="coupling",
    )
    # -------------------------------------------
    # Check Xfields results vs ELEGANT results
    # Check horizontal equilibrium emittance and IBS growth rate
    xo.assert_allclose(
        result.eq_sr_ibs_gemitt_x,
        ELEGANT_RESULTS[emittance_coupling_factor]["eps_x"],
        atol=1e-12,
        rtol=5e-2,
    )
    # Factor of 2 because different conventions between Xsuite and elegant!
    xo.assert_allclose(
        result.Kx[-1],
        ELEGANT_RESULTS[emittance_coupling_factor]["T_x"] / 2,
        rtol=6e-2,
    )
    # Check longitudinal equilibrium emittance and IBS growth rate
    # Different eps_zeta convention between both codes
    xo.assert_allclose(
        result.eq_sr_ibs_gemitt_zeta,
        ELEGANT_RESULTS[emittance_coupling_factor]["sigma_delta"]
        * ELEGANT_RESULTS[emittance_coupling_factor]["sigma_z"],
        rtol=5e-2,
    )
    # Factor of 2 because different conventions between Xsuite and elegant!
    xo.assert_allclose(
        result.Kz[-1],
        ELEGANT_RESULTS[emittance_coupling_factor]["T_z"] / 2,
        rtol=6e-2,
    )


# ----- Some Test Functions for Behaviour ----- #


def test_missing_required_twiss_attributes_raises(bessy3_line_with_radiation: xt.Line):
    """Check we raise if Twiss has no SR equilibrium values."""
    tw = bessy3_line_with_radiation.twiss(eneloss_and_damping=False)  # no tw.eq_*
    # This should tell us we're missing something in the config
    with pytest.raises(
        AttributeError,
        match="TwissTable must contain SR equilibrium emittances and damping constants.",
    ):
        tw.get_ibs_and_synrad_emittance_evolution(
            formalism="Nagaitsev",
            total_beam_intensity=BUNCH_INTENSITY,
            emittance_coupling_factor=1,
            emittance_constraint="coupling",
        )


def test_missing_params_raises(bessy3_line_with_radiation: xt.Line):
    """Check that not provided required params raises."""
    tw = bessy3_line_with_radiation.twiss(eneloss_and_damping=True)
    # These should tell us we're necessary missing arguments
    with pytest.raises(AssertionError, match="Must provide 'formalism'"):
        tw.get_ibs_and_synrad_emittance_evolution(
            formalism=None,
            total_beam_intensity=BUNCH_INTENSITY,
        )
    # Not providing formalism just passes to IBS rates computation which raises
    with pytest.raises(TypeError):
        tw.get_ibs_and_synrad_emittance_evolution(
            # formalism="Nagaitsev",
            total_beam_intensity=BUNCH_INTENSITY,
        )
    with pytest.raises(AssertionError, match="Must provide 'total_beam_intensity'"):
        tw.get_ibs_and_synrad_emittance_evolution(
            formalism="Nagaitsev",
            total_beam_intensity=None,
        )
    # Not providing formalism just passes to IBS rates computation which raises
    with pytest.raises(TypeError):
        tw.get_ibs_and_synrad_emittance_evolution(
            formalism="Nagaitsev",
            # total_beam_intensity=BUNCH_INTENSITY,
        )
    with pytest.raises(AssertionError, match="Must provide 'rtol'"):
        tw.get_ibs_and_synrad_emittance_evolution(
            formalism="Nagaitsev",
            total_beam_intensity=BUNCH_INTENSITY,
            rtol=None,
        )


@pytest.mark.parametrize("emittance_constraint", ["WRONG", "invalid"])
def test_invalid_constraint_raises(emittance_constraint, bessy3_line_with_radiation: xt.Line):
    """Check we raise if the emittance coupling constraint is invalid."""
    tw = bessy3_line_with_radiation.twiss(eneloss_and_damping=True)
    # This should tell us we're missing something in the config
    with pytest.raises(AssertionError, match="Invalid 'emittance_constraint'"):
        tw.get_ibs_and_synrad_emittance_evolution(
            formalism="Nagaitsev",
            total_beam_intensity=BUNCH_INTENSITY,
            emittance_coupling_factor=1,
            emittance_constraint=emittance_constraint,
        )
