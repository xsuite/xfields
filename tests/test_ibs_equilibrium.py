import pytest
import xfields as xf
import xobjects as xo
import xtrack as xt
from ibs_conftest import XTRACK_TEST_DATA
from numpy.testing import assert_allclose

bunch_intensity = 6.2e9 # 1C bunch intensity

bessy3_dir = XTRACK_TEST_DATA / "bessy3"
line = xt.Line.from_json(str(bessy3_dir / "line.json"))
line.build_tracker(_context=xo.ContextCpu())
line.matrix_stability_tol = 1e-2
line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

twiss = line.twiss(eneloss_and_damping=True)

@pytest.mark.parametrize('emittance_constraint', ['coupling', 'excitation'])
def test_ibs_emittance_constraints(emittance_constraint):

    #######################################
    # Equilibrium emittances calculations #
    #######################################

    emittance_coupling_factor = 0.02

    time, emittances_x_list, emittances_y_list, emittances_z_list, T_x, T_y, T_z = xf.ibs.compute_equilibrium_emittances_from_sr_and_ibs(
        twiss, bunch_intensity,
        emittance_coupling_factor=emittance_coupling_factor,
        emittance_constraint=emittance_constraint,
    )

    if emittance_constraint == 'coupling':
        # Check equilibrium emittance
        assert_allclose(
                emittances_x_list[-1],
                twiss.eq_gemitt_x / (1 + emittance_coupling_factor) / (1 - T_x[-1] / 2 / twiss.damping_constants_s[0]),
                rtol=5e-2
                )
        # Check equilibrium emittance
        assert_allclose(
                emittances_z_list[-1],
                twiss.eq_gemitt_zeta / (1 - T_z[-1] / 2 / twiss.damping_constants_s[2]),
                rtol=2e-2
                )
        # Check emittance coupling constraint
        assert_allclose(
                emittances_y_list[-1] / emittances_x_list[-1],
                emittance_coupling_factor,
                rtol=2e-2
                )

    else:
        # Check equilibrium emittance
        assert_allclose(
                emittances_x_list[-1],
                twiss.eq_gemitt_x / (1 - T_x[-1] / 2 / twiss.damping_constants_s[0]),
                rtol=5e-2
                )
        # Check equilibrium emittance
        assert_allclose(
                emittances_z_list[-1],
                twiss.eq_gemitt_zeta / (1 - T_z[-1] / 2 / twiss.damping_constants_s[2]),
                rtol=2e-2
                )
        # Check emittance coupling constraint
        assert_allclose(
                emittances_y_list[-1] / emittances_x_list[-1],
                emittance_coupling_factor,
                )
        
@pytest.mark.parametrize('emittance_coupling_factor', [0.02, 0.1, 0.2])
def test_ibs_emittance_coupling_factor(emittance_coupling_factor):
    """
    As the emittance coupling factor increases, the equilibrium emittance
    cannot be compared anymore to the solution of the differential equation 
    describing the emittance evolution in presence of IBS and SR if a 
    constraint on the emittance is enforced.
    """
    #######################################
    # Equilibrium emittances calculations #
    #######################################

    time, emittances_x_list, emittances_y_list, emittances_z_list, T_x, T_y, T_z = xf.ibs.compute_equilibrium_emittances_from_sr_and_ibs(
        twiss, bunch_intensity,
        emittance_coupling_factor=emittance_coupling_factor,
    )

    # Check equilibrium emittance
    assert_allclose(
            emittances_x_list[-1],
            twiss.eq_gemitt_x / (1 + emittance_coupling_factor) / (1 - T_x[-1] / 2 / twiss.damping_constants_s[0]),
            rtol=1e-1
            )
    # Check equilibrium emittance
    assert_allclose(
            emittances_z_list[-1],
            twiss.eq_gemitt_zeta / (1 - T_z[-1] / 2 / twiss.damping_constants_s[2]),
            rtol=2e-2
            )
    # Check emittance coupling constraint
    assert_allclose(
            emittances_y_list[-1] / emittances_x_list[-1],
            emittance_coupling_factor,
            )
    
@pytest.mark.parametrize('emittance_coupling_factor', [0.02, 0.1, 1.])
def test_ibs_emittance_no_constraint(emittance_coupling_factor):
    """
    Without any emittance constraint, the equilibrium emittance becomes
    almost identical to the solution of the differential equation describing
    the emittance evolution in presence of IBS and SR.
    """
    initial_emittances=(
        twiss.eq_gemitt_x, emittance_coupling_factor*twiss.eq_gemitt_x, 
        twiss.eq_gemitt_zeta
        )
    emittance_constraint=""
    natural_emittances=(
        twiss.eq_gemitt_x, emittance_coupling_factor*twiss.eq_gemitt_x, 
        twiss.eq_gemitt_zeta
        )

    #######################################
    # Equilibrium emittances calculations #
    #######################################

    time, emittances_x_list, emittances_y_list, emittances_z_list, T_x, T_y, T_z = xf.ibs.compute_equilibrium_emittances_from_sr_and_ibs(
        twiss, bunch_intensity,
        initial_emittances=initial_emittances,
        emittance_coupling_factor=emittance_coupling_factor,
        emittance_constraint=emittance_constraint,
        natural_emittances=natural_emittances,
    )

    # Check equilibrium emittance
    assert_allclose(
            emittances_x_list[-1],
            twiss.eq_gemitt_x / (1 - T_x[-1] / 2 / twiss.damping_constants_s[0]),
            rtol=2e-2
            )
    # Check equilibrium emittance
    assert_allclose(
            emittances_y_list[-1],
            emittance_coupling_factor*twiss.eq_gemitt_x / (1 - T_y[-1] / 2 / twiss.damping_constants_s[1]),
            rtol=2e-2
            )
    # Check equilibrium emittance
    assert_allclose(
            emittances_z_list[-1],
            twiss.eq_gemitt_zeta / (1 - T_z[-1] / 2 / twiss.damping_constants_s[2]),
            rtol=2e-2
            )