from pathlib import Path
from typing import Tuple

import xtrack as xt
from cpymad.madx import Madx

# /!\ This assumes xtrack repo is sitting next to xfields repo
XTRACK_TEST_DATA = Path(__file__).parent.parent.parent / "xtrack" / "test_data/"

def set_madx_beam_parameters(
    madx: Madx,
    total_beam_intensity: int,
    gemitt_x: float = None,
    nemitt_x: float = None,
    gemitt_y: float = None,
    nemitt_y: float = None,
    sigma_delta: float = None,
    bunch_length: float = None,
    bunched: bool = True,
) -> None:
    """
    Set some beam parameters to proided values, taking care of conversions
    where needed. This function assumes the desired sequence is in use, and
    has a corresponding beam.

    Parameters
    ----------
    madx : cpymad.madx.Madx
        A `~cpymad.madx.Madx` instance with the desired lattice & beam.
    total_beam_intensity : int
        Number of particles in the beam.
    gemitt_x : float, optional
        Horizontal geometric emittance in [m]. Either this parameter or
        `nemitt_x` is required.
    nemitt_x : float, optional
        Horizontal normalized emittance in [m]. Either this parameter or
        `gemitt_x` is required.
    gemitt_y : float, optional
        Vertical geometric emittance in [m]. Either this parameter or
        `nemitt_y` is required.
    nemitt_y : float, optional
        Vertical normalized emittance in [m]. Either this parameter or
        `gemitt_y` is required.
    sigma_delta : float
        Momentum spread.
    bunch_length : float
        Bunch length in [m].
    bunched : bool
        Whether to tell MAD-X if the beam is bunched.
    """
    # ------------------------------------------------------------------------
    # Get the MAD-X sequence in use and relativistic parameters
    madx.command.twiss()
    seq_name = madx.table.twiss.summary.sequence  # active sequence
    beta0 = madx.sequence[seq_name].beam.beta
    gamma0 = madx.sequence[seq_name].beam.gamma
    # ------------------------------------------------------------------------
    # Ensure we have the parameters we need, convert to geometric if normalized
    assert sigma_delta is not None, "Must provide 'sigma_delta'"
    assert bunch_length is not None, "Must provide 'bunch_length'"
    assert any([gemitt_x, nemitt_x]), "Must provide either 'gemitt_x' or 'nemitt_x'"
    assert any([gemitt_y, nemitt_y]), "Must provide either 'gemitt_y' or 'nemitt_y'"
    if gemitt_x is not None:
        assert nemitt_x is None, "Cannot provide both 'gemitt_x' and 'nemitt_x'"
    if gemitt_y is not None:
        assert nemitt_y is None, "Cannot provide both 'gemitt_y' and 'nemitt_y'"
    if nemitt_x is not None:
        assert gemitt_x is None, "Cannot provide both 'gemitt_x' and 'nemitt_x'"
        gemitt_x = nemitt_x / (gamma0 * beta0)
    if nemitt_y is not None:
        assert gemitt_y is None, "Cannot provide both 'gemitt_y' and 'nemitt_y'"
        gemitt_y = nemitt_y / (gamma0 * beta0)
    # ------------------------------------------------------------------------
    # Set the beam parameters
    madx.sequence[seq_name].beam.npart = total_beam_intensity  # set the number of particles
    madx.sequence[seq_name].beam.ex = gemitt_x  # set the geom emit x (in [m])
    madx.sequence[seq_name].beam.ey = gemitt_y  # set the geom emit y (in [m])
    madx.sequence[seq_name].beam.sige = sigma_delta * (beta0**2)  # set the relative energy spread
    madx.sequence[seq_name].beam.sigt = bunch_length  # set the bunch length (in [m])
    madx.sequence[seq_name].beam.bunched = bunched  # set if the beam is bunched


def get_madx_ibs_growth_rates(madx: Madx) -> Tuple[float, float, float]:
    """
    Calls the IBS module then return horizontal, vertical and longitudinal
    growth rates. A Twiss call is done to make sure it is centered.
    This function assumes the desired sequence is in use, and has a
    corresponding beam with the desired parameters.

    Parameters
    ----------
    madx : cpymad.madx.Madx
        A `~cpymad.madx.Madx` instance with the desired lattice & beam.

    Returns
    -------
    Tuple[float, float, float]
        The horizontal, vertical and longitudinal growth rates.
    """
    madx.command.twiss(centre=True)
    madx.command.ibs()
    madx.input("Tx=1/ibs.tx; Ty=1/ibs.ty; Tl=1/ibs.tl;")
    return madx.globals.Tx, madx.globals.Ty, madx.globals.Tl


def get_ref_particle_from_madx_beam(madx: Madx) -> xt.Particles:
    """
    Create a reference particle from the MAD-X's beam parameters. A
    Twiss call is done to get the active sequence. This function assumes
    the desired sequence is in use, and has a corresponding beam with the
    desired parameters.

    Parameters
    ----------
    madx : cpymad.madx.Madx
        A `~cpymad.madx.Madx` instance with the desired lattice & beam.

    Returns
    -------
    xt.Particles
        A `~xtrack.particles.Particles` instance with the reference particle.
    """
    madx.command.twiss()
    seq_name = madx.table.twiss.summary.sequence  # active sequence
    gamma0 = madx.sequence[seq_name].beam.gamma
    q0 = madx.sequence[seq_name].beam.charge
    mass0 = madx.sequence[seq_name].beam.mass * 1e9  # rest mass | in [GeV] in MAD-X but we want [eV]
    return xt.Particles(q0=q0, mass0=mass0, gamma0=gamma0)


def get_parameters_from_madx_beam(madx: Madx) -> Tuple[float, float, float, float, float]:
    """
    Get beam intensity, transverse emittances, momentum spread and
    bunch length from the MAD-X's beam parameters. A Twiss call is
    done to get the active sequence so we query from the correct beam.
    This function assumes the desired sequence is in use, and has a
    corresponding beam with the desired parameters.

    Parameters
    ----------
    madx : cpymad.madx.Madx
        A `~cpymad.madx.Madx` instance with the desired lattice & beam.

    Returns
    -------
    Tuple[float, float, float, float]
        The beam intensity, horizontal & vertical geometric emittances,
        sigma_delta and bunch_length.
    """
    madx.command.twiss()
    seq_name = madx.table.twiss.summary.sequence  # active sequence
    beta0 = madx.sequence[seq_name].beam.beta
    gemitt_x = madx.sequence[seq_name].beam.ex  # in [m]
    gemitt_y = madx.sequence[seq_name].beam.ey  # in [m]
    sigma_delta = madx.sequence[seq_name].beam.sige / (beta0**2)  # get from energy spread
    bunch_length = madx.sequence[seq_name].beam.sigt  # in [m]
    num_particles = madx.sequence[seq_name].beam.npart
    return num_particles, gemitt_x, gemitt_y, sigma_delta, bunch_length
