import pathlib
from typing import Tuple

import pytest
import xtrack as xt
from cpymad.madx import Madx

XTRACK_TEST_DATA = xt.general._pkg_root  # Need to find a way to access that??


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
