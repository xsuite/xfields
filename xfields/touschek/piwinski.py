# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

from math import atan, sqrt, exp, log, pi

import numpy as np
from scipy.integrate import quad
from scipy.special import i0
from scipy.constants import physical_constants

import xtrack as xt

ELECTRON_MASS_EV = xt.ELECTRON_MASS_EV
C_LIGHT_VACUUM = physical_constants['speed of light in vacuum'][0]
CLASSICAL_ELECTRON_RADIUS = physical_constants['classical electron radius'][0]


def piwinski_integral(tm, B1, B2):
    """
    Compute Piwinski integral for Touschek scattering rate calculation.

    The integration variable is k, with t = tan(k)^2. The direct Piwinski
    formula is written as an integral over t from tm to infinity; this
    substitution gives dt = 2*tan(k)*(1 + tan(k)^2) dk and leaves the
    integrand below with an overall factor 2 applied in the rate formula.

    Parameters
    ----------
    tm : float
        Lower limit of the integral in t, i.e. ``beta**2 * delta_m**2`` for
        the momentum acceptance ``delta_m``.
    B1, B2 : float
        Piwinski B1 and B2 coefficients.

    Returns
    -------
    value : float
        Value of the integral.
    """
    km = atan(sqrt(tm))

    def int_piwinski(k):
        t = np.tan(k) ** 2
        fact = (
            (2*t + 1)**2 * (t/tm / (1+t) - 1) / t
            + t
            - sqrt(t*tm * (1 + t))
            - (2 + 1 / (2*t)) * log(t/tm / (1+t))
        )
        if B2 * t < 500:
            intp = fact * exp(-B1*t) * i0(B2*t) * sqrt(1+t)
        else:
            intp = (
                fact
                * exp(B2*t - B1*t)
                / sqrt(2*pi * B2*t)
                * sqrt(1+t)
            )
        return intp

    val, _ = quad(
        int_piwinski,
        km,
        pi / 2,
        epsabs=1e-16,
        epsrel=1e-12,
    )

    return val


def piwinski_b2(B1, B2_squared):
    """
    Piwinski B2 = sqrt(B2_squared).

    Analytically 0 <= B2 <= B1, but B2_squared = B1^2 - (...) can come
    out slightly negative from rounding when B2 ~ 0 (e.g. round beams):
    within a relative 1e-7 of B1^2 (the tolerance of ELEGANT) it is
    taken as 0 instead of giving NaN.

    Parameters
    ----------
    B1 : float
        Piwinski B1 coefficient.
    B2_squared : float
        Square of the Piwinski B2 coefficient.

    Returns
    -------
    B2 : float
        Piwinski B2 coefficient.
    """
    if B2_squared < 0:
        if B2_squared >= -1e-7 * B1**2:
            return 0.0
        raise ValueError(
            f"Piwinski B2^2 = {B2_squared:.6e} < 0 (B1 = {B1:.6e}): "
            "check the optics and beam parameters.")
    return np.sqrt(B2_squared)


def local_piwinski_rate(*, p0c, delta, alfx, betx, alfy, bety,
                        dx, dpx, dy, dpy, delta_neg, delta_pos,
                        gemitt_x, gemitt_y, sigma_z, sigma_delta,
                        bunch_intensity):
    """
    Compute the local Piwinski Touschek scattering rate.

    The rate is the average of the rates of the particles scattered beyond
    the negative and the positive momentum acceptance.

    Parameters
    ----------
    p0c : float
        Reference momentum [eV].
    delta : float
        Relative momentum deviation of the closed orbit.
    alfx, betx, alfy, bety : float
        Twiss parameters.
    dx, dpx, dy, dpy : float
        Dispersion functions.
    delta_neg, delta_pos : float
        Negative and positive momentum acceptance.
    gemitt_x, gemitt_y : float
        Geometric emittances.
    sigma_z : float
        RMS bunch length [m].
    sigma_delta : float
        RMS relative momentum spread.
    bunch_intensity : float
        Number of particles in the bunch.

    Returns
    -------
    rate : float
        Local Touschek scattering rate per bunch [1/s].
    """
    dxt = alfx * dx + betx * dpx # dxt: dx tilde
    dyt = alfy * dy + bety * dpy # dyt: dy tilde

    sigmab_x = np.sqrt(gemitt_x * betx) # Horizontal betatron beam size
    sigma_x = np.sqrt(gemitt_x * betx + dx**2 * sigma_delta**2)

    sigmab_y = np.sqrt(gemitt_y * bety) # Vertical betatron beam size
    sigma_y = np.sqrt(gemitt_y * bety + dy**2 * sigma_delta**2)

    sigma_h = (
        sigma_delta**-2
        + (dx**2 + dxt**2)/sigmab_x**2
        + (dy**2 + dyt**2)/sigmab_y**2
    )**(-0.5)

    p = p0c * (1 + delta)
    gamma = np.sqrt(1 + p**2 / ELECTRON_MASS_EV**2)
    beta = np.sqrt(1 - gamma**-2)

    B1 = (
        betx**2 / (2 * beta**2 * gamma**2 * sigmab_x**2)
        * (1 - sigma_h**2 * dxt**2 / sigmab_x**2)
        + bety**2 / (2 * beta**2 * gamma**2 * sigmab_y**2)
        * (1 - sigma_h**2 * dyt**2 / sigmab_y**2)
    )

    B2 = piwinski_b2(
        B1,
        B1**2
        - betx**2 * bety**2 * sigma_h**2
        / (beta**4 * gamma**4 * sigmab_x**4 * sigmab_y**4
           * sigma_delta**2)
        * (sigma_x**2 * sigma_y**2 - sigma_delta**4 * dx**2 * dy**2)
    )

    tau_neg = beta**2 * (delta_neg**2)
    tau_pos = beta**2 * (delta_pos**2)

    piwinski_integral_neg = piwinski_integral(tau_neg, B1, B2)
    piwinski_integral_pos = piwinski_integral(tau_pos, B1, B2)

    # Factor 2 comes from the t = tan(k)^2 substitution used in
    # piwinski_integral; the manual formula is written directly in t.
    rateN = (
        CLASSICAL_ELECTRON_RADIUS**2 * C_LIGHT_VACUUM
        * bunch_intensity**2
        / (8*np.pi * gamma**2 * sigma_z
           * np.sqrt(sigma_x**2 * sigma_y**2
                     - sigma_delta**4 * dx**2 * dy**2))
        * 2 * np.sqrt(np.pi * (B1**2 - B2**2))
        * piwinski_integral_neg
    )

    rateP = (
        CLASSICAL_ELECTRON_RADIUS**2 * C_LIGHT_VACUUM
        * bunch_intensity**2
        / (8*np.pi * gamma**2 * sigma_z
           * np.sqrt(sigma_x**2 * sigma_y**2
                     - sigma_delta**4 * dx**2 * dy**2))
        * 2 * np.sqrt(np.pi * (B1**2 - B2**2))
        * piwinski_integral_pos
    )

    return (rateN + rateP) / 2


def integrated_piwinski_rates(*, names, s, markers, local_rate, line_length,
                              at=None):
    """
    Integrate the local Piwinski rate over the sections of the markers.

    The rate is integrated with the trapezoidal rule over the rows of the
    line table. The section of a marker goes from the previous marker to
    it, and the section of the first marker from s = 0 (an empty section,
    of zero rate, if it sits at s = 0, as in ELEGANT).

    Parameters
    ----------
    names : sequence of str
        Names of the rows of the line table, including the final
        ``_end_point`` row.
    s : sequence of float
        Longitudinal positions of the rows [m].
    markers : sequence of str
        Names of all the scattering markers of the line, in line order. They
        define the sections.
    local_rate : callable
        ``local_rate(name)`` returns the local Piwinski rate [1/s] at the
        row ``name``.
    line_length : float
        Length of the line [m].
    at : str or None, optional
        If given, only the section of this marker is integrated.

    Returns
    -------
    rates : dict
        Mapping ``{marker: rate}`` with the rate integrated over the section
        of the marker and divided by the line length, i.e. the contribution
        of the section to the ring-averaged rate per bunch [1/s].
    """
    names = list(names)
    s_of = dict(zip(names, s))
    index_of = {nn: ii for ii, nn in enumerate(names)}
    markers = list(markers)
    ii_markers = sorted(index_of[nn] for nn in markers)

    def _step(name, s_before, rate_before, integrated):
        s_name = s_of[name]
        ds = s_name - s_before
        if ds > 0.0:
            rate = local_rate(name)
            integrated += 0.5 * (rate_before + rate) * ds
            return s_name, rate, integrated
        else:
            return s_before, rate_before, integrated

    rates = {}
    if at is None:
        # All the sections, in one pass along the line
        i_marker = 0
        s_before = 0.0
        rate_before = local_rate(names[0])
        integrated = 0.0
        for ii, nn in enumerate(names):
            s_before, rate_before, integrated = _step(
                nn, s_before, rate_before, integrated)
            if ii == ii_markers[i_marker]:
                rates[nn] = integrated / line_length
                integrated = 0.0
                i_marker += 1
                if i_marker == len(ii_markers):
                    break
    else:
        # Only the section of `at`, from the previous marker (or from the
        # start of the line for the first one)
        i_marker = markers.index(at)
        before = markers[i_marker - 1] if i_marker != 0 else names[0]
        s_before = s_of[before]
        rate_before = local_rate(before)
        integrated = 0.0
        for nn in names[index_of[before]:index_of[at] + 1]:
            s_before, rate_before, integrated = _step(
                nn, s_before, rate_before, integrated)
            if nn == at:
                rates[at] = integrated / line_length
                break

    return rates
