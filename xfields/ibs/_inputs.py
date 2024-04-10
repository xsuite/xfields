# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #
from logging import getLogger
from typing import Self

import numpy as np
import xobjects as xo
import xtrack as xt
from scipy.constants import c

LOGGER = getLogger(__name__)

# Re-implemented BeamParameters and OpticsParameters as xo.HybridClass objects,
# and got rid of the parts initializing from MAD-X.


class BeamParameters(xo.HybridClass):
    """Holds relevant beam parameters for IBS calculations.

    Attributes
    ----------
    n_part : int
        Number of simulated particles.
    particle_charge : int
        Elementary particle charge, in # of Coulomb charges.
    particle_mass_eV : float
        Particle mass in [eV].
    total_energy_eV : float
        Total energy of the simulated particles in [eV].
    gamma_rel : float
        Relativistic gamma of the simulated particles.
    beta_rel : float
        Relativistic gamma of the simulated particles.
    particle_classical_radius_m : float
        The particles' classical radius in [m].
    """

    _xofields = {
        "n_part": xo.Int64,
        "particle_charge": xo.Int64,
        "particle_mass_eV": xo.Float64,
        "total_energy_eV": xo.Float64,
        "gamma_rel": xo.Float64,
        "beta_rel": xo.Float64,
        "particle_classical_radius_m": xo.Float64,
    }

    def __init__(self, particles: xt.Particles) -> None:
        """Init by providing the xt.Particles object."""
        n_part = particles.weight[0] * particles.gamma0.shape[0]
        particle_charge = particles.q0
        particle_mass_eV = particles.mass0
        total_energy_eV = np.sqrt(particles.p0c[0] ** 2 + particles.mass0**2)
        gamma_rel = particles.gamma0[0]
        beta_rel = particles.beta0[0]
        particle_classical_radius_m = particles.get_classical_particle_radius0()

        self.xoinitialize(
            n_part=n_part,
            particle_charge=particle_charge,
            particle_mass_eV=particle_mass_eV,
            total_energy_eV=total_energy_eV,
            gamma_rel=gamma_rel,
            beta_rel=beta_rel,
            particle_classical_radius_m=particle_classical_radius_m,
        )

    @classmethod
    def from_line(cls, line: xt.Line, n_part: int) -> Self:  # noqa: F821
        """
        Convenience constructor to return a `BeamParameters` object from an
        `xtrack.Line`.

        Parameters
        ----------
        line : xtrack.Line
            Line to get parameters from.
        n_part: int
            Number of particles to in the bunch. Mandatory argument
            as it is not possible to infer it from `line.particle_ref`.

        Returns:
            An instantiated `BeamParameters` object.
        """
        result = cls(line.particle_ref)
        result.n_part = n_part  # do not forget to adjust
        return result


class OpticsParameters(xo.HybridClass):
    """Holds relevant optics for IBS calculations.

    Attributes
    ----------
    s : ArrayLike
        Longitudinal positions of the machine elements in [m].
    circumference : float
        Machine circumference in [m].
    slip_factor : float
        Slip factor of the machine (positive above transition).
    revolution_frequency : float
        Revolution frequency of the machine in [Hz].
    betx : ArrayLike
        Horizontal beta functions in [m].
    bety : ArrayLike
        Vertical beta functions in [m].
    alfx : ArrayLike
        Horizontal alpha functions.
    alfy : ArrayLike
        Vertical alpha functions.
    dx : ArrayLike
        Horizontal dispersion functions in [m].
    dy : ArrayLike
        Vertical dispersion functions in [m].
    dpx : ArrayLike
        Horizontal dispersion of px (d px / d delta).
    dpy : ArrayLike
        Vertical dispersion of py (d px / d delta).
    """

    _xofields = {
        "s": xo.Float64[:],
        "circumference": xo.Float64,
        "slip_factor": xo.Float64,
        "revolution_frequency": xo.Float64,
        "betx": xo.Float64[:],
        "bety": xo.Float64[:],
        "alfx": xo.Float64[:],
        "alfy": xo.Float64[:],
        "dx": xo.Float64[:],
        "dy": xo.Float64[:],
        "dpx": xo.Float64[:],
        "dpy": xo.Float64[:],
    }

    def __init__(self, line: xt.Line, **kwargs) -> None:
        """
        Init by providing the xt.Line object.
        Will perform a 4D twiss with the provided kwargs.

        Parameters
        ----------
        line : xtrack.Line
            Line to get parameters from.
        **kwargs :
            Any keyword arguments to be passed to `line.twiss`. The default
            `method` argument is ``4d`` but can be overriden.
        """
        method = kwargs.pop("method", "4d")  # 4D twiss by default, can be overriden
        twiss: xt.TwissTable = line.twiss(method=method, **kwargs)

        if not np.isclose(twiss.c_minus, 0, atol=5e-4):  # there is "some" betatron coupling
            LOGGER.warning(
                f"There is betatron coupling in the machine (|Cminus| = {twiss.c_minus:.3e}), "
                "which is not taken into account in analytical calculations."
            )

        s = twiss.s
        circumference = twiss.circumference
        slip_factor = twiss.slip_factor
        revolution_frequency = twiss.beta0 * c / circumference
        betx = twiss.betx
        bety = twiss.bety
        alfx = twiss.alfx
        alfy = twiss.alfy
        dx = twiss.dx
        dy = twiss.dy
        dpx = twiss.dpx
        dpy = twiss.dpy

        self.xoinitialize(
            s=s,
            circumference=circumference,
            slip_factor=slip_factor,
            revolution_frequency=revolution_frequency,
            betx=betx,
            bety=bety,
            alfx=alfx,
            alfy=alfy,
            dx=dx,
            dy=dy,
            dpx=dpx,
            dpy=dpy,
        )

    @classmethod
    def from_line(cls, line: xt.Line, **kwargs) -> Self:  # noqa: F821
        """Only here so that OpticsParameters.from_line() is also possible."""
        return cls(line, **kwargs)


# ----- Helper functions ----- #

# TODO: determine with Gianni if we want this
# def _is_twiss_centered(twiss: xt.TwissTable) -> bool:
#     r"""
#     Determines if the Twiss was performed at the center of elements.

#     .. hint::
#         This check is taken from the Fortran source code of the ``IBS`` module in ``MAD-X``.
#         We skip all rows in the table until we get to the first element with non-zero length,
#         take note of its `s` position, then the `s` and `l` of the next element. We compare the
#         :math:`\Delta s` to the length and conclude. If the two match, then the :math:`\Delta s`
#         is exactly the length of the second element which means the `s` values are given at the
#         end of elements, and therefore we are not centered. Otherwise, we are.

#     Parameters
#     ----------
#     twiss : xt.TwissTable
#         The result of a Twiss call on the `xt.Line`.

#     Returns
#     -------
#     bool
#         Returns `True` if the Twiss is centered, `False` otherwise.
#     """
#     df = twiss.to_pandas()
#     # Get to the first row with an actual element of non-zero length
#     tw = df[df.l != 0]
#     # Get the "first" value of s and l
#     s0 = tw.s.to_numpy()[0]
#     # Get the s variable at the next element
#     l1 = tw.l.to_numpy()[1]
#     s1 = tw.s.to_numpy()[1]
#     # Compare s1 - s0 with the length of the second element, if it matches then the delta_s corresponds
#     # to the length of the element which means we are getting values at the exit of the elements,
#     # aka the twiss is not centered
#     return not np.isclose(s1 - s0, l1)
