# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import time

import numpy as np

import xtrack as xt

from ..beam_elements.touschek import (
    TouschekMarker, _make_random_seed, _resolve_bunch_intensity,
    _resolve_n_scattering_events, _resolve_rng_source,
    _resolve_weight_retention_fraction, _rng_source_name)
from .piwinski import integrated_piwinski_rates, local_piwinski_rate

# Tolerance on the position of the last marker, at the end of the line [m]
_S_TOL = 1e-6

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
                       weight_retention_fraction, rng, seed=None, at=None,
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
    seed : int or None, optional
        Seed stored on the markers. If ``None``, the stored seeds are not
        changed.
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

    particle_ref = line.particle_ref
    p0c = particle_ref.p0c[0]
    tab = line.get_table()
    element_index = {nn: ii for ii, nn in enumerate(line.element_names)}

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

        marker = line.element_dict[nn]
        marker._configure(
            s=s,
            particle_ref=particle_ref,
            element_index=element_index[nn],
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
        )
        if seed is not None:
            marker.seed = seed


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
    """

    def __init__(self, line):
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

        selected = [at] if isinstance(at, str) else list(at)
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
