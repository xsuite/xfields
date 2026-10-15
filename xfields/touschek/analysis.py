# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

from dataclasses import dataclass

import numpy as np

import xtrack as xt


def _allocated(particles):
    """
    Mask of the slots of the particle buffer that hold a particle.

    The generated particles leave room for twice their number; the unused
    slots have state ``LAST_INVALID_STATE``.
    """
    return particles.state > xt.particles.LAST_INVALID_STATE


def _generated(particles):
    """
    Mask of the Touschek-scattered particles, excluding any secondaries
    added while tracking (e.g. by collimator scattering codes).
    """
    return _allocated(particles) & (
        particles.parent_particle_id == particles.particle_id)


def _lost(particles):
    """
    Mask of the lost particles: any state <= 0, i.e. on an aperture (0) or
    in other ways (negative states, e.g. absorbed in an xcoll collimator).
    """
    return _allocated(particles) & (particles.state <= 0)


def _tracked(particles):
    """
    Whether the particles have been tracked: some are lost, or have
    completed turns.
    """
    allocated = _allocated(particles)
    return bool(np.any(particles.state[allocated] <= 0)
                or np.any(particles.at_turn[allocated] > 0))


def _sample_sums(particles):
    """
    Counts and weights of the Touschek-scattered particles of a sample.

    Only the scattered particles (primaries) are counted: secondaries carry
    the weight of their parent.

    Parameters
    ----------
    particles : xtrack.Particles
        Particle sample.

    Returns
    -------
    sums : dict
        ``num_particles``, ``sum_weight``, ``num_lost_particles`` and
        ``sum_lost_weight``.
    """
    generated = _generated(particles)
    lost = _lost(particles) & generated
    return {
        'num_particles': int(np.sum(generated)),
        'sum_weight': float(np.sum(particles.weight[generated])),
        'num_lost_particles': int(np.sum(lost)),
        'sum_lost_weight': float(np.sum(particles.weight[lost])),
    }


def _replica_variance(replica_sums):
    """
    Variance of the sum of the replicas of a marker.

    Each replica is an independent estimate of the rate of the section, with
    its weights divided by the number of replicas ``R``: the estimate of
    replica ``r`` is ``R * replica_sums[r]`` and their mean is
    ``sum(replica_sums)``. Its variance is estimated as the squared standard
    error of the mean of the ``R`` estimates. The particles of one replica are
    not independent (``pickPart`` keeps the highest-weight ones), so only the
    spread between replicas is used.

    Parameters
    ----------
    replica_sums : sequence of float
        Sum of the weights of interest in each replica.

    Returns
    -------
    variance : float
        Variance of ``sum(replica_sums)``, or ``nan`` for a single replica.
    """
    replica_sums = np.asarray(replica_sums, dtype=float)
    n_replicas = len(replica_sums)
    if n_replicas < 2:
        return np.nan
    return float(n_replicas * np.var(replica_sums, ddof=1))


@dataclass
class TouschekResult:
    """
    Result of a Touschek simulation.

    It is returned by :meth:`xfields.TouschekGenerator.analyse` and
    :meth:`xfields.TouschekGenerator.run`. The rates are per bunch. The result
    separates quantities inferred from the scattering model (the integrated
    Piwinski rates configured on the markers) from quantities obtained from
    the particles (the generated and the lost weight).

    Parameters
    ----------
    element_names : list
        Names of the Touschek markers included in the result.
    local_momentum_acceptance : xtrack.Table
        Local momentum acceptance at the markers, as configured (i.e. scaled),
        with ``name``, ``s``, ``delta_neg`` and ``delta_pos`` columns.
    local_rates : xtrack.Table
        Per-marker diagnostics table. It always contains ``name``, ``s``,
        ``delta_neg``, ``delta_pos``, ``piwinski_rate`` and
        ``integrated_piwinski_rate``. When the particles of each marker are
        known, it also contains ``num_particles`` and ``sum_weight`` and,
        once tracked, ``num_lost_particles`` and ``sum_lost_weight``.
    rate_scattering : float
        Total Touschek scattering rate of the sections of the markers, i.e.
        the sum of their integrated Piwinski rates [1/s].
    lifetime_scattering : float
        Touschek lifetime inferred from ``rate_scattering`` [s].
    rate_tracking : float or None
        Weighted rate of the scattered particles lost in tracking [1/s].
        ``None`` if the particles were not tracked.
    lifetime_tracking : float or None
        Touschek lifetime inferred from ``rate_tracking`` [s]. ``None`` if
        the particles were not tracked.
    tracked : bool
        Whether the particles were tracked.
    particles_by_element : dict or None
        Mapping ``{element_name: xtrack.Particles}`` with the particles of
        each marker, when kept.
    particles : xtrack.Particles or None
        Merged particles, when kept.
    lost_particles : xtrack.Particles or None
        Subset of ``particles`` lost in tracking, when kept. It is a copy taken
        right after tracking.
    rate_generated : float or None
        Total weight of the scattered particles [1/s]: the scattering rate
        beyond the momentum acceptance represented by the particles, i.e.
        about ``weight_retention_fraction`` times ``rate_scattering`` when the
        particles of all the markers are included. ``None`` without particles.
    rate_tracking_error : float or None
        Monte Carlo standard error of ``rate_tracking`` [1/s], from the spread
        between the replicas of each marker. ``None`` unless several replicas
        of some marker are given.
    lifetime_tracking_error : float or None
        Monte Carlo standard error of ``lifetime_tracking`` [s], propagated
        from ``rate_tracking_error`` to first order.

    Attributes
    ----------
    element_names, local_momentum_acceptance, local_rates, rate_scattering,
    lifetime_scattering, rate_tracking, lifetime_tracking, tracked,
    particles_by_element, particles, lost_particles, rate_generated,
    rate_tracking_error, lifetime_tracking_error
        See above.
    """
    element_names: list
    local_momentum_acceptance: xt.Table
    local_rates: xt.Table
    rate_scattering: float
    lifetime_scattering: float
    rate_tracking: float | None
    lifetime_tracking: float | None
    tracked: bool
    particles_by_element: dict | None = None
    particles: xt.Particles | None = None
    lost_particles: xt.Particles | None = None
    rate_generated: float | None = None
    rate_tracking_error: float | None = None
    lifetime_tracking_error: float | None = None
