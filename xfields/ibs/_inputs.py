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
    num_particles : int
        Number of simulated particles.
    q0 : int
        Elementary particle charge, in # of Coulomb charges.
    mass0 : float
        Particle reference rest mass in [eV].
    total_energy_eV : float
        Total energy of the simulated particles in [eV].
    gamma0 : float
        Relativistic gamma of the simulated particles.
    beta0 : float
        Relativistic gamma of the simulated particles.
    classical_particle_radius0 : float
        The particles' classical radius in [m].
    """

    _xofields = {
        "num_particles": xo.Int64,
        "q0": xo.Int64,
        "mass0": xo.Float64,
        "total_energy_eV": xo.Float64,  # total energy in [eV] (energy in xt.Particles?)
        "gamma0": xo.Float64,
        "beta0": xo.Float64,
        "classical_particle_radius0": xo.Float64,  # classical_particle_radius0
    }

    def __init__(self, particles: xt.Particles) -> None:
        """Init by providing the xt.Particles object."""
        num_particles = particles.weight[0] * particles.gamma0.shape[0]
        q0 = particles.q0
        mass0 = particles.mass0
        total_energy_eV = np.sqrt(particles.p0c[0] ** 2 + particles.mass0**2)
        gamma0 = particles.gamma0[0]
        beta0 = particles.beta0[0]
        classical_particle_radius0 = particles.get_classical_particle_radius0()

        self.xoinitialize(
            num_particles=num_particles,
            q0=q0,
            mass0=mass0,
            total_energy_eV=total_energy_eV,
            gamma0=gamma0,
            beta0=beta0,
            classical_particle_radius0=classical_particle_radius0,
        )

    @classmethod
    def from_line(cls, line: xt.Line, num_particles: int) -> Self:  # noqa: F821
        """
        Convenience constructor to return a `BeamParameters` object from an
        `xtrack.Line`.

        Parameters
        ----------
        line : xtrack.Line
            Line to get parameters from.
        num_particles: int
            Number of particles to in the bunch. Mandatory argument
            as it is not possible to infer it from `line.particle_ref`.

        Returns:
            An instantiated `BeamParameters` object.
        """
        result = cls(line.particle_ref)
        result.num_particles = num_particles  # do not forget to adjust
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


# TODO: Just one class for inputs
# Takes a TwissTable (but has a .from_line method)
# Is basically the Opticsparameters and uses the tw.particle_on_co for the attrivutes that BeamParams uses
# Still will need user to give npart if not giving a Particles objects

# class IBSInputs(xo.HybridClass):
#     """Holds relevant beam parameters and optics for IBS calculations.

#     Attributes
#     ----------
#     num_particles : int
#         Number of simulated particles.
#     q0 : int
#         Elementary particle charge, in # of Coulomb charges.
#     mass0 : float
#         Particle mass in [eV].
#     total_energy_eV : float
#         Total energy of the simulated particles in [eV].
#     gamma0 : float
#         Relativistic gamma of the simulated particles.
#     beta0 : float
#         Relativistic gamma of the simulated particles.
#     classical_particle_radius0 : float
#         The particles' classical radius in [m].
#     s : ArrayLike
#         Longitudinal positions of the machine elements in [m].
#     circumference : float
#         Machine circumference in [m].
#     slip_factor : float
#         Slip factor of the machine (positive above transition).
#     revolution_frequency : float
#         Revolution frequency of the machine in [Hz].
#     betx : ArrayLike
#         Horizontal beta functions in [m].
#     bety : ArrayLike
#         Vertical beta functions in [m].
#     alfx : ArrayLike
#         Horizontal alpha functions.
#     alfy : ArrayLike
#         Vertical alpha functions.
#     dx : ArrayLike
#         Horizontal dispersion functions in [m].
#     dy : ArrayLike
#         Vertical dispersion functions in [m].
#     dpx : ArrayLike
#         Horizontal dispersion of px (d px / d delta).
#     dpy : ArrayLike
#         Vertical dispersion of py (d px / d delta).
#     """
#     _xofields = {
#         "s": xo.Float64[:],
#         "circumference": xo.Float64,
#         "slip_factor": xo.Float64,
#         "revolution_frequency": xo.Float64,
#         "betx": xo.Float64[:],
#         "bety": xo.Float64[:],
#         "alfx": xo.Float64[:],
#         "alfy": xo.Float64[:],
#         "dx": xo.Float64[:],
#         "dy": xo.Float64[:],
#         "dpx": xo.Float64[:],
#         "dpy": xo.Float64[:],
#     }


#     def __init__(self, twiss: xt.TwissTable) -> None:
#         """
#         Init by providing the xt.TwissTable object corresponding to a specific
#         line configuration.

#         Parameters
#         ----------
#         line : xtrack.TwissTable
#             Twiss results of the `xtrack.Line` configuration.
#         """
#         # First, a check for the coupling situation in the machine
#         if not np.isclose(twiss.c_minus, 0, atol=5e-4):  # there is "some" betatron coupling
#             LOGGER.warning(
#                 f"There is betatron coupling in the machine (|Cminus| = {twiss.c_minus:.3e}), "
#                 "which is not taken into account in analytical calculations."
#             )

#         s = twiss.s
#         circumference = twiss.circumference
#         slip_factor = twiss.slip_factor
#         revolution_frequency = twiss.beta0 * c / circumference
#         betx = twiss.betx
#         bety = twiss.bety
#         alfx = twiss.alfx
#         alfy = twiss.alfy
#         dx = twiss.dx
#         dy = twiss.dy
#         dpx = twiss.dpx
#         dpy = twiss.dpy

#         self.xoinitialize(
#             s=s,
#             circumference=circumference,
#             slip_factor=slip_factor,
#             revolution_frequency=revolution_frequency,
#             betx=betx,
#             bety=bety,
#             alfx=alfx,
#             alfy=alfy,
#             dx=dx,
#             dy=dy,
#             dpx=dpx,
#             dpy=dpy,
#         )

#     @classmethod
#     def from_line(cls, line: xt.Line, **kwargs) -> Self:  # noqa: F821
#         """Only here so that OpticsParameters.from_line() is also possible."""
#         return cls(line, **kwargs)
