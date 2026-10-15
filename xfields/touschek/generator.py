# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import time
import warnings

import numpy as np

import xtrack as xt

from ..beam_elements.touschek import (
    TouschekMarker, TouschekRNGState, _make_random_seed,
    _make_xtrack_rng_seed, _resolve_bunch_intensity,
    _resolve_n_scattering_events, _resolve_rng_source,
    _resolve_weight_retention_fraction, _rng_source_name,
    _TOUSCHEK_RNG_ELEGANT)
from .analysis import (
    TouschekResult, _lost, _replica_variance, _sample_sums, _tracked)
from .piwinski import integrated_piwinski_rates, local_piwinski_rate

# Tolerance on the position of the last marker, at the end of the line [m]
_S_TOL = 1e-6

# Relative tolerance of the check of the weight of the particles
_RATE_GENERATED_RTOL = 0.05

# Limits of the centre-of-mass scattering angle [rad], from ELEGANT
_THETA_MIN = 0.00005 * np.pi
_THETA_MAX = 0.99995 * np.pi


def _check_local_momentum_acceptance(local_momentum_acceptance):
    """
    Validate a local momentum acceptance table.

    Parameters
    ----------
    local_momentum_acceptance : xtrack.Table
        Table with ``name``, ``s``, ``delta_neg`` and ``delta_pos`` columns.

    Returns
    -------
    None
    """
    required_cols = {"name", "s", "delta_neg", "delta_pos"}
    if not isinstance(local_momentum_acceptance, xt.Table):
        raise TypeError(
            "`local_momentum_acceptance` must be an `xt.Table` object.")
    missing = required_cols - set(local_momentum_acceptance._col_names)
    if missing:
        raise ValueError(
            f"`local_momentum_acceptance` missing columns: {sorted(missing)}")

    for col in ("s", "delta_neg", "delta_pos"):
        try:
            vals = np.asarray(local_momentum_acceptance[col], dtype=float)
        except Exception:
            raise TypeError(
                f"`{col}` column must be numeric (cannot coerce to float).")
        nan_mask = np.isnan(vals)
        if nan_mask.any():
            bad = list(np.where(nan_mask)[0][:5])
            raise ValueError(f"`{col}` contains NaN at indices {bad}.")
        inf_mask = np.isinf(vals)
        if inf_mask.any():
            bad = list(np.where(inf_mask)[0][:5])
            raise ValueError(f"`{col}` contains inf at indices {bad}.")


def _geometric_emittances(particle_ref, *, nemitt_x, nemitt_y,
                          gemitt_x, gemitt_y):
    """
    Geometric emittances, from either the normalized or the geometric ones.

    Parameters
    ----------
    particle_ref : xtrack.Particles
        Reference particle.
    nemitt_x, nemitt_y : float or None
        Normalized emittances.
    gemitt_x, gemitt_y : float or None
        Geometric emittances.

    Returns
    -------
    gemitt_x, gemitt_y : float
        Geometric emittances.
    """
    nemitt_given = nemitt_x is not None and nemitt_y is not None
    gemitt_given = gemitt_x is not None and gemitt_y is not None

    if nemitt_given and gemitt_given:
        raise ValueError(
            "Provide either normalized emittances (nemitt_x, nemitt_y) "
            "OR geometric emittances (gemitt_x, gemitt_y), not both.")
    if not (nemitt_given or gemitt_given):
        raise ValueError(
            "You must provide either both normalized emittances "
            "(nemitt_x, nemitt_y) OR both geometric emittances "
            "(gemitt_x, gemitt_y).")

    if nemitt_given:
        beta0 = particle_ref.beta0[0]
        gamma0 = particle_ref.gamma0[0]
        return nemitt_x / (beta0 * gamma0), nemitt_y / (beta0 * gamma0)
    return gemitt_x, gemitt_y


def _resolve_seed(seed):
    """
    Seed of the Touschek random streams: drawn if ``None``.
    """
    if seed is None:
        return _make_random_seed()
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)):
        raise TypeError(
            f"`seed` must be an integer (got {type(seed).__name__}).")
    if seed < 0:
        raise ValueError("`seed` must be non-negative.")
    return int(seed)


def _stream_seed(seed, name, replica):
    """
    Seed of the random-number stream of a marker and replica.

    The seed is derived from the configured ``seed`` with
    :class:`numpy.random.SeedSequence`, keyed by the name of the marker and
    by the replica index. The stream of a marker therefore does not depend on
    which other markers or replicas are generated, nor on the order of the
    generation: independent jobs reproduce the particles of a single one.

    Parameters
    ----------
    seed : int
        Seed configured on the marker.
    name : str
        Name of the marker.
    replica : int
        Index of the replica.

    Returns
    -------
    stream_seed : int
        Seed of the stream, in ``[0, 2**31)``.
    """
    sequence = np.random.SeedSequence(
        entropy=int(seed), spawn_key=(int(replica), *name.encode()))
    return int(sequence.generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF)


def _new_rng(marker, seed):
    """
    Random-number generator of a stream seeded with `seed`.

    Parameters
    ----------
    marker : TouschekMarker
        Marker whose generator (``'elegant'`` or ``'xtrack'``) and context
        are used.
    seed : int
        Seed of the stream.

    Returns
    -------
    rng_state : TouschekRNGState or None
        Elegant-compatible state, with ``rng='elegant'``.
    rng_particle : xtrack.Particles or None
        Carrier of the xtrack generator state, with ``rng='xtrack'``.
    """
    context = marker._context
    if marker.rng_source == _TOUSCHEK_RNG_ELEGANT:
        return TouschekRNGState(seed=seed, _context=context), None
    rng_particle = xt.Particles(_context=context)
    rng_particle._init_random_number_generator(
        seeds=_make_xtrack_rng_seed(seed))
    return None, rng_particle


def _on_context(rng_state, rng_particle, context):
    """
    The random-number generator on `context`, copied there if needed.
    """
    if rng_state is not None and rng_state._context is not context:
        rng_state = rng_state.copy(_context=context)
    if rng_particle is not None and rng_particle._context is not context:
        rng_particle = rng_particle.copy(_context=context)
    return rng_state, rng_particle


def _check_configured(name, marker, particle_ref):
    """
    Check that a marker is configured for the reference particle of the line.
    """
    if not marker.p0c > 0:
        raise ValueError(
            f"TouschekMarker '{name}' is not configured: call "
            "`line.xfields.touschek.configure(...)` first.")
    p0c = particle_ref.p0c[0]
    if abs(marker.p0c - p0c) > 1e-10 * p0c:
        raise ValueError(
            f"TouschekMarker '{name}' is configured for p0c = "
            f"{marker.p0c:.10g} eV, but the reference particle of the line "
            f"has p0c = {p0c:.10g} eV: configure the markers again.")


def _effective_nz(nz, delta_neg, delta_pos, sigma_delta):
    """
    Longitudinal sampling cutoff, reduced to stay inside the local momentum
    acceptance.

    The initial coordinates of the Monte Carlo are drawn from a Gaussian
    truncated at ``nz`` sigmas, i.e. up to ``|delta| ~ nz * sigma_delta``.
    Where this exceeds the local momentum acceptance, some particles are
    already outside it before scattering: for scattering angles close to
    zero they are selected as lost with an essentially unchanged state and,
    since the Moller cross section diverges there, with very large weights.
    ``pickPart`` then keeps mostly these few pathological particles, which
    distorts both the local Monte Carlo rate and the Touschek lifetime.

    The cutoff is therefore reduced, element by element, to

        nz_eff = min(nz, 0.85 * min(|delta_neg|, delta_pos) / sigma_delta)

    so that the sampled range always lies inside the (scaled) momentum
    acceptance, with a safety factor 0.85. Tighter cutoffs are applied only
    where the momentum acceptance is restrictive.

    Parameters
    ----------
    nz : float
        Requested longitudinal cutoff, in sigmas.
    delta_neg, delta_pos : float
        Negative and positive local momentum acceptance.
    sigma_delta : float
        RMS relative momentum spread.

    Returns
    -------
    nz_eff : float
        Longitudinal cutoff to be used, in sigmas.
    """
    min_delta_acceptance = min(abs(delta_neg), delta_pos)
    return min(nz, 0.85 * min_delta_acceptance / sigma_delta)


def _configure_markers(line, *, twiss, lma_s, lma_delta_neg, lma_delta_pos,
                       bunch_intensity, gemitt_x, gemitt_y, sigma_z,
                       sigma_delta, n_scattering_events, nx, ny, nz,
                       weight_retention_fraction, rng, seed, at=None,
                       verbose=True):
    """
    Compute the Touschek rates and store the configuration on the markers.

    Parameters
    ----------
    line : xtrack.Line
        Line containing the markers.
    twiss : xtrack.TwissTable
        Twiss table of the line.
    lma_s, lma_delta_neg, lma_delta_pos : array_like
        Local momentum acceptance, already scaled, as a function of ``s``.
    bunch_intensity, gemitt_x, gemitt_y, sigma_z, sigma_delta : float
        Beam parameters.
    n_scattering_events : int
        Number of Monte Carlo scattering events per marker.
    nx, ny, nz : float
        Gaussian sampling cutoffs.
    weight_retention_fraction : float
        Fraction of the scattering weight retained in the samples.
    rng : {'elegant', 'xtrack'}
        Random-number generator.
    seed : int
        Seed of the random-number streams of the markers.
    at : str, sequence of str, or None, optional
        Markers to configure. ``None`` configures all of them. The sections
        are always defined by all the markers of the line.
    verbose : bool, optional
        Whether to print the progress of the configuration.

    Returns
    -------
    None
    """
    generator = TouschekGenerator(line)
    markers = generator._markers()
    generator._check_sections_cover_line()
    selected = markers if at is None else generator._markers(at)

    p0c = line.particle_ref.p0c[0]
    tab = line.get_table()

    def local_rate(name):
        s = twiss['s', name]
        return local_piwinski_rate(
            p0c=p0c,
            delta=twiss['delta', name],
            alfx=twiss['alfx', name], betx=twiss['betx', name],
            alfy=twiss['alfy', name], bety=twiss['bety', name],
            dx=twiss['dx', name], dpx=twiss['dpx', name],
            dy=twiss['dy', name], dpy=twiss['dpy', name],
            delta_neg=np.interp(s, lma_s, lma_delta_neg),
            delta_pos=np.interp(s, lma_s, lma_delta_pos),
            gemitt_x=gemitt_x, gemitt_y=gemitt_y,
            sigma_z=sigma_z, sigma_delta=sigma_delta,
            bunch_intensity=bunch_intensity,
        )

    t0 = time.time()
    integrated = {}
    for nn in ([None] if at is None else selected):
        integrated.update(integrated_piwinski_rates(
            names=tab.name,
            s=tab.s,
            markers=markers,
            local_rate=local_rate,
            line_length=float(twiss.line_length),
            at=nn,
        ))
    if verbose:
        print(f"Computed integrated Piwinski rates in "
              f"{time.time() - t0:.2f} s.")

    for nn in selected:
        s = tab['s', nn]
        delta_neg = np.interp(s, lma_s, lma_delta_neg)
        delta_pos = np.interp(s, lma_s, lma_delta_pos)

        nz_eff = _effective_nz(nz, delta_neg, delta_pos, sigma_delta)
        if verbose:
            print(f'Configuring TouschekMarker {nn}')
            if nz_eff < nz:
                print(f'  longitudinal cutoff reduced to nz = {nz_eff:.2f} '
                      f'(instead of {nz:.2f}) to sample inside the local '
                      'momentum acceptance')

        line.element_dict[nn]._configure(
            p0c=p0c,
            bunch_intensity=bunch_intensity,
            gemitt_x=gemitt_x,
            gemitt_y=gemitt_y,
            alfx=twiss['alfx', nn], betx=twiss['betx', nn],
            alfy=twiss['alfy', nn], bety=twiss['bety', nn],
            dx=twiss['dx', nn], dpx=twiss['dpx', nn],
            dy=twiss['dy', nn], dpy=twiss['dpy', nn],
            x_co=twiss['x', nn], px_co=twiss['px', nn],
            y_co=twiss['y', nn], py_co=twiss['py', nn],
            zeta_co=twiss['zeta', nn], delta_co=twiss['delta', nn],
            delta_neg=delta_neg, delta_pos=delta_pos,
            sigma_z=sigma_z,
            sigma_delta=sigma_delta,
            n_scattering_events=n_scattering_events,
            nx=nx, ny=ny, nz=nz_eff,
            theta_min=_THETA_MIN, theta_max=_THETA_MAX,
            weight_retention_fraction=weight_retention_fraction,
            piwinski_rate=local_rate(nn),
            integrated_piwinski_rate=integrated[nn],
            rng=rng,
            seed=seed,
        )


def _is_sequence(value):
    return isinstance(value, (list, tuple, np.ndarray))


def _per_marker(value, n_markers, name):
    """
    One value per marker: a sequence is checked, a scalar is repeated.
    """
    if _is_sequence(value):
        value = list(value)
        if len(value) != n_markers:
            raise ValueError(
                f"`{name}` has {len(value)} entries, but {n_markers} "
                "markers are installed.")
        return value
    return [value] * n_markers


def _is_free(env, name):
    """
    Whether `name` can be given to a new element of the environment.
    """
    if name in env.elements:
        return False
    try:
        env._check_name_clashes(name)
    except ValueError:
        return False
    return True


def _free_names(env, prefix, n_names):
    """
    The `n_names` names ``f'{prefix}.{i}'`` with the lowest free indices.
    """
    names = []
    ii = 0
    while len(names) < n_names:
        name = f'{prefix}.{ii}'
        if _is_free(env, name):
            names.append(name)
        ii += 1
    return names


class TouschekGenerator:
    """
    Touschek scattering simulation of a line, based on the
    :class:`xfields.TouschekMarker` elements installed in it.

    An instance bound to a line is available as ``line.xfields.touschek``.
    The generator holds no state other than the line: the configuration is
    stored in the markers, so that a line saved to JSON after it has been
    configured generates the same particles once loaded again.

    Parameters
    ----------
    line : xtrack.Line
        Line containing, or receiving, the Touschek markers.

    Examples
    --------
    .. code-block:: python

        tsc = line.xfields.touschek
        tsc.install_markers(at=['mq1', 'mb1@start', 12.3])
        tsc.configure(local_momentum_acceptance=lma, nemitt_x=1e-5,
                      nemitt_y=1e-7, sigma_z=4e-3, sigma_delta=1e-3,
                      bunch_intensity=4e9, n_scattering_events=100_000,
                      seed=1997)

        tab = line.get_table()
        touschek_elements = tab.rows.match(element_type='TouschekMarker').name

        particles_list = []
        for name in touschek_elements:
            part = tsc.generate(at=name)
            line.track(part, ele_start=name, ele_stop=name, num_turns=1000)
            particles_list.append(part)

        result = tsc.analyse(xt.Particles.merge(particles_list))
        print(result.lifetime_tracking)
    """

    def __init__(self, line):
        if not isinstance(line, xt.Line):
            raise TypeError(
                f"`line` must be an xtrack.Line (got {type(line).__name__}).")
        self._line = line

    @property
    def line(self):
        """
        Line of the Touschek simulation.

        Returns
        -------
        line : xtrack.Line
            Line bound to this generator.
        """
        return self._line

    @property
    def marker_names(self):
        """
        Names of the Touschek markers of the line.

        Returns
        -------
        names : list of str
            Names of the :class:`xfields.TouschekMarker` elements, in line
            order.
        """
        element_dict = self.line.element_dict
        names = [nn for nn in self.line.element_names
                 if isinstance(element_dict[nn], TouschekMarker)]
        if len(set(names)) != len(names):
            repeated = sorted({nn for nn in names if names.count(nn) > 1})
            raise ValueError(
                f"Touschek markers {repeated} appear more than once in the "
                "line: each marker must be a distinct element.")
        return names

    def install_markers(self, at, *, from_=None, anchor=None,
                        from_anchor=None, names=None, prefix='tscatter',
                        close_ring=True):
        """
        Install Touschek markers in the line.

        Each marker is a new :class:`xfields.TouschekMarker` element, placed
        with ``env.place`` and inserted with ``line.insert``: ``at``,
        ``from_``, ``anchor`` and ``from_anchor`` have the same meaning as
        there. Installing the markers by hand in the same way is equivalent.

        Each marker represents the section of the line from the previous
        marker (from the start of the line for the first one) to itself, so
        markers are best placed where the optics changes, e.g. in the magnets.

        Parameters
        ----------
        at : float, str, xdeps reference, or sequence of them
            Position of each marker, evaluated at installation: an ``s``
            position [m], an expression of the environment variables, the
            name of an element (to place the marker at its centre), or a
            reference ``'name@anchor'`` with ``anchor`` one of ``'start'``,
            ``'center'`` and ``'end'``. With ``from_``, the offset from the
            reference element.
        from_ : str or sequence of str, optional
            Element from which ``at`` is measured. ``'name@anchor'`` selects
            the anchor of the element.
        anchor : {'start', 'center', 'end'} or sequence of them, optional
            Anchor of the marker placed at ``at``. Markers are thin, so this
            only matters to follow the conventions of ``env.place``.
        from_anchor : {'start', 'center', 'end'} or sequence of them, optional
            Anchor of ``from_`` from which ``at`` is measured.
        names : str or sequence of str, optional
            Names of the new markers. By default ``f'{prefix}.{i}'``, with the
            lowest indices whose names are free in the environment.
        prefix : str, optional
            Prefix of the default names.
        close_ring : bool, optional
            If ``True`` (default), a marker is added at the end of the line
            when, after the insertion, the last marker is not there: otherwise
            the part of the line after it would belong to no section and
            :meth:`configure` refuses the line.

        Returns
        -------
        names : list of str
            Names of the installed markers, the closing one included.
        """
        line = self.line
        env = line.env

        at = list(at) if _is_sequence(at) else [at]
        n_markers = len(at)
        if n_markers == 0:
            raise ValueError("`at` must give at least one position.")
        from_ = _per_marker(from_, n_markers, 'from_')
        anchor = _per_marker(anchor, n_markers, 'anchor')
        from_anchor = _per_marker(from_anchor, n_markers, 'from_anchor')

        if names is None:
            names = _free_names(env, prefix, n_markers)
        else:
            names = _per_marker(
                [names] if isinstance(names, str) else names,
                n_markers, 'names')
            if len(set(names)) != len(names):
                raise ValueError("`names` contains repeated names.")
            for nn in names:
                if not _is_free(env, nn):
                    raise ValueError(
                        f"The name '{nn}' is already used in the environment.")

        places = []
        for nn, aa, ff, an, fa in zip(names, at, from_, anchor, from_anchor):
            env.elements[nn] = TouschekMarker()
            places.append(env.place(
                nn, at=aa, from_=ff, anchor=an, from_anchor=fa))
        line.insert(places)

        installed = list(names)
        if close_ring and not self._last_marker_at_end():
            end_name, = _free_names(env, prefix, 1)
            env.elements[end_name] = TouschekMarker()
            line.insert(end_name, at=line.get_length())
            installed.append(end_name)

        return installed

    def configure(self, *, local_momentum_acceptance=None, sigma_z=None,
                  sigma_delta=None, bunch_intensity=None,
                  n_scattering_events=None, nemitt_x=None, nemitt_y=None,
                  gemitt_x=None, gemitt_y=None,
                  local_momentum_acceptance_scale=0.85,
                  weight_retention_fraction=0.99, nx=3, ny=3, nz=3,
                  seed=None, rng='elegant', twiss=None, method='6d', at=None,
                  verbose=True, **kwargs):
        """
        Configure the Touschek markers of the line.

        Computes the local Piwinski scattering rate at the markers and its
        integral over the section of each marker, and stores on each marker
        the local optics and closed orbit, the local momentum acceptance, the
        beam parameters and the Monte Carlo settings. The section of a marker
        goes from the previous marker to it, and the section of the first
        one from ``s = 0``: the last marker must sit at the end of the line
        (as in ELEGANT), which :meth:`install_markers` ensures.

        The whole configuration is stored on the markers: a line saved to
        JSON after this call generates the same particles once loaded again.

        Parameters
        ----------
        local_momentum_acceptance : xtrack.Table
            Table with ``name``, ``s``, ``delta_neg`` and ``delta_pos``
            columns, e.g. from ``line.get_local_momentum_acceptance(...)``.
            It is interpolated in ``s`` at the markers. It is not modified.
        sigma_z : float
            RMS bunch length [m].
        sigma_delta : float
            RMS relative momentum spread.
        bunch_intensity : float
            Number of particles in the bunch.
        n_scattering_events : int
            Number of Monte Carlo scattering events generated at each marker.
        nemitt_x, nemitt_y : float, optional
            Normalized emittances. Mutually exclusive with ``gemitt_x`` and
            ``gemitt_y``.
        gemitt_x, gemitt_y : float, optional
            Geometric emittances. Mutually exclusive with ``nemitt_x`` and
            ``nemitt_y``.
        local_momentum_acceptance_scale : float, optional
            Factor applied to ``delta_neg`` and ``delta_pos``.
        weight_retention_fraction : float, optional
            Fraction of the generated scattering weight kept in the particle
            samples: the lowest-weight particles beyond it are discarded,
            without renormalizing the others. ``1`` keeps all the particles.
        nx, ny, nz : float, optional
            Cutoffs, in sigmas, of the Gaussian distribution from which the
            particles are drawn. ``nz`` is reduced where needed to stay inside
            the local momentum acceptance.
        seed : int or None, optional
            Seed of the random-number streams of the markers. If ``None``, a
            seed is drawn now and stored, so that the generation stays
            reproducible.
        rng : {'elegant', 'xtrack'}, optional
            Random-number generator: ``'elegant'`` reproduces the sequences of
            ELEGANT, ``'xtrack'`` uses the xtrack accurate uniform generator.
        twiss : xtrack.TwissTable or None, optional
            Twiss table of the line. If ``None``, it is computed with
            ``line.twiss(method=method, **kwargs)``.
        method : {'6d', '4d'}, optional
            Twiss method, used if ``twiss`` is ``None``.
        at : str, sequence of str, or None, optional
            Markers to configure. If ``None`` (default), all the markers. The
            sections are always defined by all the markers of the line.
        verbose : bool, optional
            Whether to print the progress of the configuration.
        **kwargs
            Keyword arguments forwarded to ``line.twiss`` if ``twiss`` is
            ``None``.

        Returns
        -------
        None

        References
        ----------
        .. [1] A. Xiao and M. Borland, "Monte Carlo simulation of Touschek
           effect", Phys. Rev. ST Accel. Beams **13**, 074201 (2010).
           https://doi.org/10.1103/PhysRevSTAB.13.074201
        .. [2] M. Borland, "elegant: A Flexible SDDS-Compliant Code for
           Accelerator Simulation", APS LS-287 (2000).
        """
        line = self.line
        if getattr(line, "particle_ref", None) is None:
            raise ValueError("`line` must have a `particle_ref`.")
        for name, value in (
                ("local_momentum_acceptance", local_momentum_acceptance),
                ("sigma_z", sigma_z),
                ("sigma_delta", sigma_delta),
                ("bunch_intensity", bunch_intensity),
                ("n_scattering_events", n_scattering_events)):
            if value is None:
                raise ValueError(f"`{name}` is required.")
        _check_local_momentum_acceptance(local_momentum_acceptance)

        gemitt_x, gemitt_y = _geometric_emittances(
            line.particle_ref, nemitt_x=nemitt_x, nemitt_y=nemitt_y,
            gemitt_x=gemitt_x, gemitt_y=gemitt_y)
        bunch_intensity = _resolve_bunch_intensity(
            bunch_intensity=bunch_intensity, default=None)
        n_scattering_events = _resolve_n_scattering_events(
            n_scattering_events=n_scattering_events, default=None)
        weight_retention_fraction = _resolve_weight_retention_fraction(
            weight_retention_fraction=weight_retention_fraction,
            default=0.99)
        rng = _rng_source_name(_resolve_rng_source(rng))
        seed = _resolve_seed(seed)

        # Check the markers before the (long) twiss
        self._markers()
        self._check_sections_cover_line()
        if at is not None:
            self._markers(at)

        if twiss is None:
            twiss = line.twiss(method=method, **kwargs)

        lma = local_momentum_acceptance
        _configure_markers(
            line,
            twiss=twiss,
            lma_s=np.asarray(lma.s, dtype=float),
            lma_delta_neg=(local_momentum_acceptance_scale
                           * np.asarray(lma.delta_neg, dtype=float)),
            lma_delta_pos=(local_momentum_acceptance_scale
                           * np.asarray(lma.delta_pos, dtype=float)),
            bunch_intensity=bunch_intensity,
            gemitt_x=gemitt_x,
            gemitt_y=gemitt_y,
            sigma_z=sigma_z,
            sigma_delta=sigma_delta,
            n_scattering_events=n_scattering_events,
            nx=nx, ny=ny, nz=nz,
            weight_retention_fraction=weight_retention_fraction,
            rng=rng,
            seed=seed,
            at=at,
            verbose=verbose,
        )

    def generate(self, at=None, *, replica=0, n_replicas=1, seed=None,
                 rng_stream='per_marker'):
        """
        Generate Touschek-scattered particles at the markers.

        At each marker, scattering events are drawn from the local 6D
        Gaussian beam with the Moller cross section, following the Monte
        Carlo of Xiao and Borland as in ELEGANT. The particles scattered
        beyond the local momentum acceptance are kept, the highest-weight ones
        up to the retained fraction of the weight, and shifted onto the
        closed orbit. The weights of the particles of a marker sum to the
        retained part of the scattering rate of its section [1/s per bunch],
        divided by ``n_replicas``.

        The particles of a marker are tracked from the marker itself, e.g.
        ``line.track(part, ele_start=name, ele_stop=name, num_turns=...)``,
        and the samples of different markers or replicas, generated in the
        same or in different jobs, can then be merged with
        :meth:`xtrack.Particles.merge` and given to :meth:`analyse`.

        Parameters
        ----------
        at : str, sequence of str, or None, optional
            Markers at which particles are generated. If ``None`` (default),
            all the markers of the line.
        replica : int, optional
            Index of the replica: replicas of a marker are independent
            samples of the same section, e.g. generated in different jobs.
            Default ``0``.
        n_replicas : int, optional
            Number of replicas of each marker that are merged. The weights are
            divided by it, so that the merged replicas represent the scattering
            rate once. Default ``1``.
        seed : int or None, optional
            Seed of the random-number streams. If ``None`` (default), the seed
            stored on each marker by :meth:`configure`.
        rng_stream : {'per_marker', 'continuous'}, optional
            With ``'per_marker'`` (default), each marker and replica has its
            own random-number stream, derived from the seed, the name of the
            marker and the replica: the particles of a marker do not depend on
            the other markers that are generated, nor on the order. With
            ``'continuous'``, one stream seeded with the seed runs through the
            selected markers in line order, as in ELEGANT, whose sequence it
            reproduces; only ``n_replicas=1`` is supported then.

        Returns
        -------
        particles : xtrack.Particles or dict
            If ``at`` is a string, the particles generated at that marker.
            Otherwise, a mapping ``{name: xtrack.Particles}`` in line order.
        """
        line = self.line
        names = self._markers(at)

        if (isinstance(n_replicas, (bool, np.bool_))
                or not isinstance(n_replicas, (int, np.integer))
                or n_replicas < 1):
            raise ValueError("`n_replicas` must be a positive integer.")
        if (isinstance(replica, (bool, np.bool_))
                or not isinstance(replica, (int, np.integer))
                or not 0 <= replica < n_replicas):
            raise ValueError(
                f"`replica` must be an integer in [0, n_replicas) "
                f"(got {replica} with n_replicas={n_replicas}).")
        if rng_stream not in ('per_marker', 'continuous'):
            raise ValueError(
                "`rng_stream` must be either 'per_marker' or 'continuous'.")
        if seed is not None:
            seed = _resolve_seed(seed)

        particle_ref = line.particle_ref
        markers = {nn: line.element_dict[nn] for nn in names}
        for nn, marker in markers.items():
            _check_configured(nn, marker, particle_ref)

        tab = line.get_table()
        element_index = {nn: ii for ii, nn in enumerate(line.element_names)}

        if rng_stream == 'continuous':
            if n_replicas != 1:
                raise ValueError(
                    "The continuous random-number stream supports only "
                    "`n_replicas=1`.")
            seeds = {int(mm.seed) for mm in markers.values()}
            rng_sources = {int(mm.rng_source) for mm in markers.values()}
            if len(rng_sources) > 1 or (seed is None and len(seeds) > 1):
                raise ValueError(
                    "The continuous random-number stream needs the same "
                    "`rng` and seed on all the selected markers.")
            rng_state, rng_particle = _new_rng(
                markers[names[0]], seeds.pop() if seed is None else seed)

        particles = {}
        for nn in names:
            marker = markers[nn]
            if rng_stream == 'continuous':
                rng_state, rng_particle = _on_context(
                    rng_state, rng_particle, marker._context)
            else:
                rng_state, rng_particle = _new_rng(
                    marker,
                    _stream_seed(marker.seed if seed is None else seed,
                                 nn, replica))
            try:
                particles[nn] = marker._generate(
                    particle_ref=particle_ref,
                    at_element=element_index[nn],
                    s=tab['s', nn],
                    rng_state=rng_state,
                    rng_particle=rng_particle,
                    weight_scale=1.0 / n_replicas,
                )
            except ValueError as error:
                raise ValueError(f"TouschekMarker '{nn}': {error}") from error

        if isinstance(at, str):
            return particles[at]
        return particles

    def analyse(self, particles=None, *, at=None):
        """
        Analyse the Touschek scattering rates and the tracked particles.

        Without particles, the result contains the scattering rates
        configured on the markers. With the particles generated at the
        markers and tracked, it also contains the loss rate: the total weight
        of the scattered particles that are lost (any ``state <= 0``, e.g. on
        an aperture or in a collimator). Secondaries produced while tracking
        carry the weight of their parent and are not counted.

        ``particles`` can be the sample of one marker, the particles of several
        markers merged with :meth:`xtrack.Particles.merge`, possibly coming
        from separate jobs, or the samples of each marker separately. The
        particles do not record the marker at which they were generated, so
        the per-marker counts are only available for the samples given
        separately, or for the sample of a single marker selected with
        ``at``. The Monte Carlo error of the loss rate is estimated from the
        spread between the replicas of the markers given separately.

        When the analysis covers all the markers of the line, the total
        weight of the scattered particles is compared with the retained
        fraction of the scattering rate: a warning is issued if it differs by
        more than 5 %, which indicates missing or repeated samples, or
        replicas generated with a different ``n_replicas``.

        Parameters
        ----------
        particles : xtrack.Particles, dict, or None, optional
            The particles: ``None`` (only the configured rates), an
            ``xtrack.Particles`` object, or a mapping ``{name: particles}``
            from marker names to the ``xtrack.Particles`` of the marker or a
            list of its replicas.
        at : str, sequence of str, or None, optional
            Markers whose particles are given, when ``particles`` is not a
            mapping. If ``None`` (default), all the markers of the line.

        Returns
        -------
        result : TouschekResult
            Scattering rates and lifetime, and, with tracked particles, the
            loss rate and lifetime.
        """
        line = self.line

        samples = None
        if isinstance(particles, dict):
            if at is not None:
                raise ValueError(
                    "`at` cannot be given with a mapping of particles: its "
                    "keys select the markers.")
            names = self._markers(list(particles))
            samples = {}
            for nn in names:
                replicas = particles[nn]
                if not isinstance(replicas, (list, tuple)):
                    replicas = [replicas]
                if len(replicas) == 0:
                    raise ValueError(f"No particles given for marker '{nn}'.")
                samples[nn] = list(replicas)
            all_particles = [pp for vv in samples.values() for pp in vv]
        else:
            names = self._markers(at)
            all_particles = [] if particles is None else [particles]
        for pp in all_particles:
            if not isinstance(pp, xt.Particles):
                raise TypeError(
                    "The particles must be `xtrack.Particles` objects "
                    f"(got {type(pp).__name__}).")

        markers = {nn: line.element_dict[nn] for nn in names}
        for nn, marker in markers.items():
            _check_configured(nn, marker, line.particle_ref)
        bunch_intensities = {float(mm.bunch_intensity)
                             for mm in markers.values()}
        if len(bunch_intensities) > 1:
            raise ValueError(
                "The markers are configured with different bunch "
                f"intensities: {sorted(bunch_intensities)}.")
        bunch_intensity = bunch_intensities.pop()

        tab = line.get_table()
        data = {
            'name': np.array(names),
            's': np.array([float(tab['s', nn]) for nn in names]),
            'delta_neg': np.array([mm.delta_neg for mm in markers.values()]),
            'delta_pos': np.array([mm.delta_pos for mm in markers.values()]),
        }
        local_momentum_acceptance = xt.Table(dict(data))
        data['piwinski_rate'] = np.array(
            [mm.piwinski_rate for mm in markers.values()])
        data['integrated_piwinski_rate'] = np.array(
            [mm.integrated_piwinski_rate for mm in markers.values()])
        rate_scattering = float(np.sum(data['integrated_piwinski_rate']))

        tracked = any(_tracked(pp) for pp in all_particles)
        rate_generated = None
        rate_tracking = None
        rate_tracking_error = None

        if samples is not None:
            sums = {nn: [_sample_sums(pp) for pp in samples[nn]]
                    for nn in names}

            def per_marker(key):
                return np.array([sum(ss[key] for ss in sums[nn])
                                 for nn in names])

            data['n_replicas'] = np.array([len(samples[nn]) for nn in names])
            data['num_particles'] = per_marker('num_particles')
            data['sum_weight'] = per_marker('sum_weight')
            rate_generated = float(np.sum(data['sum_weight']))
            if tracked:
                data['num_lost_particles'] = per_marker('num_lost_particles')
                data['sum_lost_weight'] = per_marker('sum_lost_weight')
                variance = np.array([
                    _replica_variance(
                        [ss['sum_lost_weight'] for ss in sums[nn]])
                    for nn in names])
                data['sum_lost_weight_error'] = np.sqrt(variance)
                rate_tracking = float(np.sum(data['sum_lost_weight']))
                if np.any(np.isfinite(variance)):
                    rate_tracking_error = float(np.sqrt(np.nansum(variance)))
        elif particles is not None:
            sums = _sample_sums(particles)
            rate_generated = sums['sum_weight']
            if tracked:
                rate_tracking = sums['sum_lost_weight']
            if len(names) == 1:
                data['num_particles'] = np.array([sums['num_particles']])
                data['sum_weight'] = np.array([sums['sum_weight']])
                if tracked:
                    data['num_lost_particles'] = np.array(
                        [sums['num_lost_particles']])
                    data['sum_lost_weight'] = np.array(
                        [sums['sum_lost_weight']])

        if rate_generated is not None and names == self.marker_names:
            expected = float(sum(
                mm.weight_retention_fraction * mm.integrated_piwinski_rate
                for mm in markers.values()))
            if (expected > 0 and abs(rate_generated / expected - 1)
                    > _RATE_GENERATED_RTOL):
                warnings.warn(
                    f"The scattered particles carry a rate of "
                    f"{rate_generated:.6g} 1/s, "
                    f"{100 * rate_generated / expected:.1f} % of the "
                    f"{expected:.6g} 1/s retained from the scattering rate "
                    "of all the markers: samples are missing or repeated, or "
                    "replicas were generated with a different `n_replicas`.",
                    stacklevel=2)

        def lifetime(rate):
            return np.inf if rate == 0 else float(bunch_intensity / rate)

        lifetime_tracking = None
        lifetime_tracking_error = None
        if rate_tracking is not None:
            lifetime_tracking = lifetime(rate_tracking)
            if rate_tracking_error is not None and rate_tracking > 0:
                lifetime_tracking_error = float(
                    lifetime_tracking * rate_tracking_error / rate_tracking)

        return TouschekResult(
            element_names=names,
            local_momentum_acceptance=local_momentum_acceptance,
            local_rates=xt.Table(data),
            rate_scattering=rate_scattering,
            lifetime_scattering=lifetime(rate_scattering),
            rate_tracking=rate_tracking,
            lifetime_tracking=lifetime_tracking,
            tracked=tracked,
            rate_generated=rate_generated,
            rate_tracking_error=rate_tracking_error,
            lifetime_tracking_error=lifetime_tracking_error,
        )

    def run(self, n_turns, *, at=None, n_replicas=1, keep_particles=False,
            with_progress=False, rng_stream='per_marker'):
        """
        Generate, track and analyse the Touschek particles of the markers.

        Convenience helper doing in one call the steps of a Touschek
        simulation: the particles of each marker (and replica) are generated
        with :meth:`generate`, tracked from the marker back to it with
        ``line.track(part, ele_start=name, ele_stop=name, num_turns=n_turns)``
        and analysed with :meth:`analyse`. Doing these steps explicitly gives
        the same result, and allows splitting them over separate jobs or
        refining the loss locations before the analysis.

        Parameters
        ----------
        n_turns : int
            Number of turns to track.
        at : str, sequence of str, or None, optional
            Markers at which particles are generated. If ``None`` (default),
            all the markers of the line.
        n_replicas : int, optional
            Number of replicas of each marker. With more than one, the result
            includes the Monte Carlo error of the loss rate.
        keep_particles : bool, optional
            If ``True``, the result keeps the particles of each marker (its
            replicas merged), all of them merged, and the lost ones (if any).
        with_progress : bool, optional
            Forwarded to ``line.track``.
        rng_stream : {'per_marker', 'continuous'}, optional
            Random-number streams, see :meth:`generate`.

        Returns
        -------
        result : TouschekResult
            Result of :meth:`analyse` for the particles of each marker.
        """
        if (isinstance(n_turns, (bool, np.bool_))
                or not isinstance(n_turns, (int, np.integer)) or n_turns < 1):
            raise ValueError("`n_turns` must be a positive integer.")

        line = self.line
        names = self._markers(at)
        samples = {nn: [] for nn in names}
        for replica in range(n_replicas):
            generated = self.generate(
                at=names, replica=replica, n_replicas=n_replicas,
                rng_stream=rng_stream)
            for nn, part in generated.items():
                line.track(part, ele_start=nn, ele_stop=nn,
                           num_turns=n_turns, with_progress=with_progress)
                samples[nn].append(part)

        result = self.analyse(samples)
        if keep_particles:
            result.particles_by_element = {
                nn: vv[0] if len(vv) == 1 else xt.Particles.merge(vv)
                for nn, vv in samples.items()}
            result.particles = xt.Particles.merge(
                list(result.particles_by_element.values()))
            lost = _lost(result.particles)
            if np.any(lost):
                result.lost_particles = result.particles.filter(lost)
        return result

    def _check_sections_cover_line(self):
        """
        Check that every part of the line belongs to the section of a marker.

        The section of a marker goes from the previous one to it, and the
        section of the first one from ``s = 0``. So the last marker must sit
        at the end of the line: otherwise the part of the line after it is
        not integrated, and its Touschek rate is silently missing from the
        weights and from the scattering rate. (ELEGANT imposes the same for
        its TSCATTER elements.)

        Returns
        -------
        None
        """
        if self._last_marker_at_end():
            return
        last = self._markers()[-1]
        tab = self.line.get_table()
        raise ValueError(
            f"The last TouschekMarker ('{last}') is at "
            f"s = {float(tab['s', last]):.6g} m, but the line ends at "
            f"s = {float(tab.s[-1]):.6g} m: the last TouschekMarker must sit "
            "at the end of the line, otherwise the part of the line after "
            "it is not accounted for in the integrated Touschek rate. "
            "`line.xfields.touschek.install_markers(...)` adds it, or place "
            "a marker at `at=line.get_length()`.")

    def _last_marker_at_end(self):
        """
        Whether the last marker sits at the end of the line.

        Returns
        -------
        at_end : bool
            ``True`` if the line has markers and the last one is at the end
            of the line.
        """
        names = self.marker_names
        if len(names) == 0:
            return False
        tab = self.line.get_table()
        return abs(float(tab['s', names[-1]]) - float(tab.s[-1])) <= _S_TOL

    def _markers(self, at=None):
        """
        Validated names of the selected markers, in line order.

        Parameters
        ----------
        at : str, sequence of str, or None
            Selected markers. ``None`` selects all the markers of the line.

        Returns
        -------
        names : list of str
            Names of the selected markers, in line order.
        """
        names = self.marker_names
        if at is None:
            if len(names) == 0:
                raise ValueError(
                    "The line does not contain any TouschekMarker. Install "
                    "them with `line.xfields.touschek.install_markers(...)`.")
            return names

        if isinstance(at, str):
            selected = [at]
        elif hasattr(at, '__iter__'):
            selected = list(at)
        else:
            raise TypeError(
                "Touschek markers are selected by name "
                f"(got {type(at).__name__}).")
        if len(selected) == 0:
            raise ValueError("No Touschek marker selected.")
        position = {nn: ii for ii, nn in enumerate(names)}
        for nn in selected:
            if not isinstance(nn, str):
                raise TypeError(
                    "Touschek markers are selected by name "
                    f"(got {type(nn).__name__}).")
            if nn in position:
                continue
            if nn in set(self.line.element_names):
                raise TypeError(
                    f"Element '{nn}' is not a TouschekMarker "
                    f"(got {type(self.line.element_dict[nn]).__name__}).")
            raise ValueError(f"Element '{nn}' is not present in the line.")

        return sorted(set(selected), key=position.__getitem__)
