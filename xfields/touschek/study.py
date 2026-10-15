# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xtrack as xt

import numpy as np

from ..beam_elements.touschek import (
    TouschekRNGState, TouschekMarker, _resolve_weight_retention_fraction,
    _resolve_n_scattering_events, _resolve_bunch_intensity,
    _resolve_rng_source, _rng_source_name, _make_xtrack_rng_seed,
    _TOUSCHEK_RNG_ELEGANT, _TOUSCHEK_RNG_XTRACK)
from .analysis import TouschekResult, _generated, _lost
from .generator import _configure_markers
from .piwinski import local_piwinski_rate, piwinski_b2, piwinski_integral


class TouschekStudy:
    def __init__(self, line=None, elements=None, twiss=None,
                 local_momentum_acceptance=None,
                 nemitt_x=None, nemitt_y=None,
                 sigma_z=None, sigma_delta=None,
                 bunch_intensity=None,
                 n_scattering_events=None,
                 gemitt_x=None, gemitt_y=None,
                 local_momentum_acceptance_scale=0.85,
                 weight_retention_fraction=None,
                 seed=None, rng='elegant', nx=3, ny=3, nz=3, **kwargs):
        """
        Build a Touschek study and validate the supplied line, optics, beam
        parameters, and local momentum acceptance table.

        The configured study computes the local Piwinski scattering rate at
        each :class:`TouschekMarker` element, integrates those rates over
        the lattice sections represented by the elements, and configures the
        elements with the local optics and beam parameters needed to generate
        weighted Monte Carlo particles.

        In user code, the :class:`TouschekStudy` instance for a line is
        typically obtained with ``line.xfields.touschek_configure(...)``.

        After construction, call :meth:`run` to obtain a
        :class:`TouschekResult`. With tracking disabled, :meth:`run` returns
        the integrated scattering rate, the corresponding Touschek lifetime,
        and a per-element diagnostics table. With tracking enabled, it also
        returns the tracked loss rate and the corresponding lifetime. The
        generated and lost particle samples are returned only when
        ``keep_particles=True`` is passed to :meth:`run`.

        Parameters
        ----------
        line : xtrack.Line
            Line containing the :class:`TouschekMarker` elements to
            configure.
        elements : str, sequence of str, or None, optional
            Touschek scattering elements included in the study. If ``None``,
            all :class:`TouschekMarker` elements in the line are used.
        twiss : xtrack.TwissTable or None, optional
            Twiss table used for optics-dependent rates. If ``None``, it is
            computed when :meth:`initialise_touschek` is called.
        local_momentum_acceptance : xtrack.Table
            Table with ``name``, ``s``, ``delta_neg``, and ``delta_pos``
            columns.
        nemitt_x, nemitt_y : float, optional
            Normalized horizontal and vertical emittances. Mutually exclusive
            with ``gemitt_x`` and ``gemitt_y``.
        sigma_z : float
            RMS bunch length in meters.
        sigma_delta : float
            RMS relative momentum spread.
        bunch_intensity : float
            Number of real particles in the bunch.
        n_scattering_events : int
            Number of Monte Carlo scattering events generated at each
            scattering element.
        gemitt_x, gemitt_y : float, optional
            Geometric horizontal and vertical emittances. Mutually exclusive
            with ``nemitt_x`` and ``nemitt_y``.
        local_momentum_acceptance_scale : float, optional
            Multiplicative factor applied to ``delta_neg`` and ``delta_pos``.
        weight_retention_fraction : float, optional
            Fraction of the generated scattering weight retained in the
            generated particle sample.
        seed : int or None, optional
            Seed for the random-number generator. If provided, a reproducible
            Touschek Monte Carlo sequence is used. If ``None`` (default), a
            seed is drawn with :mod:`numpy.random` when particles are
            generated.
        rng : {'elegant', 'xtrack'}, optional
            Random-number generator used for Monte Carlo particle generation.
            ``'elegant'`` reproduces the Elegant/SDDS generator sequence;
            ``'xtrack'`` uses the xtrack accurate uniform generator.
        nx, ny, nz : float, optional
            Gaussian sampling cutoffs in the horizontal, vertical, and
            longitudinal planes.
        **kwargs
            Additional keyword arguments forwarded to ``line.twiss()`` when a
            Twiss table is computed internally.

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

        # Input validation
        if line is None:
            raise ValueError("`line` is required.")
        if getattr(line, "particle_ref", None) is None:
            raise ValueError("`line` must have a `particle_ref`.")
        if local_momentum_acceptance is None:
            raise ValueError("`local_momentum_acceptance` is required.")
        if sigma_z is None:
            raise ValueError("`sigma_z` is required.")
        if sigma_delta is None:
            raise ValueError("`sigma_delta` is required.")
        if bunch_intensity is None:
            raise ValueError("`bunch_intensity` is required.")
        if n_scattering_events is None:
            raise ValueError("`n_scattering_events` is required.")

        # Local momentum acceptnace validation
        required_cols = {"name", "s", "delta_neg", "delta_pos"}
        if not isinstance(local_momentum_acceptance, xt.Table):
            raise TypeError("`local_momentum_acceptance` must be an `xt.Table` object.")
        missing = required_cols - set(local_momentum_acceptance._col_names)
        if missing:
            raise ValueError(f"`local_momentum_acceptance` missing columns: {sorted(missing)}")

        for col in ("s", "delta_neg", "delta_pos"):
            try:
                vals = np.asarray(local_momentum_acceptance[col], dtype=float)
            except Exception:
                raise TypeError(f"`{col}` column must be numeric (cannot coerce to float).")
            nan_mask = np.isnan(vals)
            if nan_mask.any():
                bad = list(np.where(nan_mask)[0][:5])
                raise ValueError(f"`{col}` contains NaN at indices {bad}.")
            inf_mask = np.isinf(vals)
            if inf_mask.any():
                bad = list(np.where(inf_mask)[0][:5])
                raise ValueError(f"`{col}` contains inf at indices {bad}.")

        self.line = line
        self.particle_ref = line.particle_ref
        self.twiss = twiss

        # Check that the line contains TouschekMarkers
        tab = line.get_table()
        try:
            has = "TouschekMarker" in set(np.unique(tab.element_type))
        except Exception:
            has = "TouschekMarker" in set(getattr(tab, "element_type", []))
        if not has:
            raise ValueError("The line does not contain any TouschekMarker. "
                             "Please add them before initializing the TouschekStudy.")

        if elements is None:
            elements = [
                nn for nn in tab.name[:-1]
                if isinstance(line[nn], TouschekMarker)
            ]
        elif isinstance(elements, str):
            elements = [elements]
        else:
            elements = list(elements)

        if len(elements) == 0:
            raise ValueError(
                "No TouschekMarker elements selected for this study."
            )

        for nn in elements:
            if nn not in line.element_names:
                raise ValueError(f"Element '{nn}' is not present in the line.")
            if not isinstance(line[nn], TouschekMarker):
                raise TypeError(
                    f"Element '{nn}' is not a TouschekMarker "
                    f"(got {type(line[nn]).__name__})."
                )

        self.elements = elements

        # Local momentum acceptance
        local_momentum_acceptance.delta_neg *= local_momentum_acceptance_scale
        local_momentum_acceptance.delta_pos *= local_momentum_acceptance_scale
        self.local_momentum_acceptance = local_momentum_acceptance

        self.sigma_z = sigma_z
        self.sigma_delta = sigma_delta
        self.bunch_intensity = _resolve_bunch_intensity(
            bunch_intensity=bunch_intensity,
            default=None)
        self.n_scattering_events = _resolve_n_scattering_events(
            n_scattering_events=n_scattering_events,
            default=None)
        self.weight_retention_fraction = _resolve_weight_retention_fraction(
            weight_retention_fraction=weight_retention_fraction,
            default=0.99)
        self.seed = seed
        self.rng_source = _resolve_rng_source(rng)
        self.rng = _rng_source_name(self.rng_source)
        self.nx = nx
        self.ny = ny
        self.nz = nz

        # Limits from ELEGANT
        self._theta_min = 0.00005*np.pi
        self._theta_max = 0.99995*np.pi

        # Emittance validation
        nemitt_given = nemitt_x is not None and nemitt_y is not None
        gemitt_given = gemitt_x is not None and gemitt_y is not None

        if nemitt_given and gemitt_given:
            raise ValueError("Provide either normalized emittances (nemitt_x, nemitt_y) "
                             "OR geometric emittances (gemitt_x, gemitt_y), not both.")
        if not (nemitt_given or gemitt_given):
            raise ValueError("You must provide either both normalized emittances (nemitt_x, nemitt_y) "
                             "OR both geometric emittances (gemitt_x, gemitt_y).")

        if nemitt_given:
            beta0 = line.particle_ref.beta0[0]
            gamma0 = line.particle_ref.gamma0[0]
            self.gemitt_x = nemitt_x / (beta0 * gamma0)
            self.gemitt_y = nemitt_y / (beta0 * gamma0)
        else:
            self.gemitt_x = gemitt_x
            self.gemitt_y = gemitt_y

        self.kwargs = kwargs
        self._rng_state = None
        self._rng_particle = None

    def _new_rng(self):
        context = self.line[self.elements[0]]._context
        if self.rng_source == _TOUSCHEK_RNG_ELEGANT:
            self._rng_state = TouschekRNGState(
                seed=self.seed, _context=context)
            self._rng_particle = None
        elif self.rng_source == _TOUSCHEK_RNG_XTRACK:
            self._rng_state = None
            self._rng_particle = xt.Particles(_context=context)
            self._rng_particle._init_random_number_generator(
                seeds=_make_xtrack_rng_seed(self.seed))
        else:
            raise ValueError(
                f"Unsupported Touschek RNG source: {self.rng_source}")

    def _rng_state_for_element(self, element_name):
        if self.rng_source != _TOUSCHEK_RNG_ELEGANT:
            return None
        context = self.line[element_name]._context
        if self._rng_state is None:
            self._rng_state = TouschekRNGState(
                seed=self.seed, _context=context)
        elif self._rng_state._context is not context:
            self._rng_state = self._rng_state.copy(_context=context)
        return self._rng_state

    def _rng_particle_for_element(self, element_name):
        if self.rng_source != _TOUSCHEK_RNG_XTRACK:
            return None
        context = self.line[element_name]._context
        if self._rng_particle is None:
            self._rng_particle = xt.Particles(_context=context)
            self._rng_particle._init_random_number_generator(
                seeds=_make_xtrack_rng_seed(self.seed))
        elif self._rng_particle._context is not context:
            self._rng_particle = self._rng_particle.copy(_context=context)
        return self._rng_particle

    def run(self, *, track=False, n_turns=None, generate_particles=None,
            keep_particles=False, with_progress=False):
        """
        Run the configured Touschek rate or loss study.

        The scattering rate reported in the result is the integrated Piwinski
        rate configured on the Touschek elements. If ``track`` is true,
        particles are generated and tracked from each scattering element back
        to itself to estimate the tracking-derived loss rate.

        Parameters
        ----------
        track : bool, optional
            If ``True``, track generated particles and compute the loss rate
            from particles lost during tracking. If ``False``, only the
            configured scattering rate is reported.
        n_turns : int or None, optional
            Number of turns to track. Required when ``track`` is ``True``.
        generate_particles : bool or None, optional
            If ``True``, generate weighted scattered particles even when
            tracking is disabled. If ``None``, particles are generated only
            when ``track`` is ``True``.
        keep_particles : bool, optional
            If ``True``, store the generated particle samples in the returned
            :class:`TouschekResult`. If ``False`` (default), particles are
            generated and tracked as needed but are not retained in the result.
            When tracking is disabled, setting ``keep_particles=True`` also
            requests particle generation.
        with_progress : bool, optional
            Forwarded to :meth:`xtrack.Line.track` when tracking is enabled.

        Returns
        -------
        result : TouschekResult
            Study result containing local rates, scattering rate and lifetime,
            optional tracking-derived rate and lifetime, and optionally the
            generated and lost particle samples.
        """
        if keep_particles and generate_particles is False:
            raise ValueError(
                "`keep_particles=True` is incompatible with "
                "`generate_particles=False`."
            )

        if track:
            if generate_particles is False:
                raise ValueError(
                    "`generate_particles=False` is incompatible with "
                    "`track=True`."
                )
            if n_turns is None:
                raise ValueError("`n_turns` is required when `track=True`.")
            generate_particles = True
        elif generate_particles is None:
            generate_particles = keep_particles

        particles_by_element = {}
        merged_particles = None
        lost_particles = None
        rate_scattering = float(sum(
            getattr(self.line[nn], "integrated_piwinski_rate")
            for nn in self.elements
        ))

        if generate_particles:
            self._new_rng()
            for nn in self.elements:
                rng_state = self._rng_state_for_element(nn)
                rng_particle = self._rng_particle_for_element(nn)
                particles = self.line[nn].scatter(
                    _rng_state=rng_state,
                    _rng_particle=rng_particle,
                )
                if track:
                    self.line.track(
                        particles,
                        ele_start=nn,
                        ele_stop=nn,
                        num_turns=n_turns,
                        with_progress=with_progress,
                    )
                particles_by_element[nn] = particles

            merged_particles = xt.Particles.merge(
                list(particles_by_element.values()))

        rate_tracking = None
        lifetime_tracking = None
        if track:
            lost = _lost(merged_particles)
            lost_particles = merged_particles.filter(lost)
            # Only the Touschek-scattered particles define the loss rate
            # (secondaries carry the weight of their parent)
            rate_tracking = float(np.sum(
                merged_particles.weight[lost & _generated(merged_particles)]))
            if rate_tracking == 0:
                lifetime_tracking = np.inf
            else:
                lifetime_tracking = float(
                    self.bunch_intensity / rate_tracking)

        if rate_scattering == 0:
            lifetime_scattering = np.inf
        else:
            lifetime_scattering = float(self.bunch_intensity / rate_scattering)

        local_rates = self.local_rates(
            particles_by_element=(
                particles_by_element if generate_particles else None),
            include_tracking=track,
        )

        if not keep_particles:
            particles_by_element = None
            merged_particles = None
            lost_particles = None

        return TouschekResult(
            element_names=list(self.elements),
            local_momentum_acceptance=self.local_momentum_acceptance,
            local_rates=local_rates,
            rate_scattering=rate_scattering,
            lifetime_scattering=lifetime_scattering,
            rate_tracking=rate_tracking,
            lifetime_tracking=lifetime_tracking,
            particles_by_element=particles_by_element,
            particles=merged_particles,
            lost_particles=lost_particles,
            tracked=track,
        )

    _compute_piwinski_integral = staticmethod(piwinski_integral)
    _compute_piwinski_b2 = staticmethod(piwinski_b2)

    def _compute_piwinski_scattering_rate(self, element):
        """
        Compute Piwinski Touschek scattering rate.
        """
        local_momentum_acceptance = self.local_momentum_acceptance
        twiss = self.twiss

        try:
            s = twiss.rows[element].s[0]
        except Exception:
            s = self.line.get_s_position(element)

        return local_piwinski_rate(
            p0c=self.particle_ref.p0c[0],
            delta=twiss['delta', element],
            alfx=twiss['alfx', element], betx=twiss['betx', element],
            alfy=twiss['alfy', element], bety=twiss['bety', element],
            dx=twiss['dx', element], dpx=twiss['dpx', element],
            dy=twiss['dy', element], dpy=twiss['dpy', element],
            delta_neg=np.interp(
                s, local_momentum_acceptance.s,
                local_momentum_acceptance.delta_neg),
            delta_pos=np.interp(
                s, local_momentum_acceptance.s,
                local_momentum_acceptance.delta_pos),
            gemitt_x=self.gemitt_x, gemitt_y=self.gemitt_y,
            sigma_z=self.sigma_z, sigma_delta=self.sigma_delta,
            bunch_intensity=self.bunch_intensity,
        )

    def initialise_touschek(self, element=None):
        '''
        Compute and configure all Piwinski rates in the lattice.

        For each :class:`TouschekMarker` element this method:

        1. Evaluates the local Piwinski scattering rate using the Twiss
        parameters, beam emittances, and the local momentum acceptance.
        2. Integrates the rate over the preceding lattice section using the
        trapezoidal rule to obtain the per-bunch, per-turn loss probability.
        3. Stores the integrated rate and all local optics parameters on the
        element via :meth:`TouschekMarker._configure` so that
        :meth:`TouschekMarker.scatter` can weight the Monte Carlo
        macro-particles correctly.

        The section of a :class:`TouschekMarker` goes from the previous
        one to it, and the section of the first one from ``s = 0``. The last
        :class:`TouschekMarker` must therefore sit at the end of the
        line (as in ELEGANT); otherwise a ``ValueError`` is raised, since the
        part of the line after it would not be accounted for.

        Parameters
        ----------
        element : str or None, optional
            If ``None`` (default), all :class:`TouschekMarker` elements in
            the line are initialised in one pass.  If a string, only the named
            element is (re-)initialised; the Piwinski rate is integrated only
            over the lattice section between that element and the preceding
            scattering centre.

        Returns
        -------
        None
        '''
        line = self.line
        tab = line.get_table()

        if element is not None:
            if not isinstance(element, str):
                raise TypeError(f"`element` must be a string (got {type(element).__name__}).")
            if element not in set(tab.name):
                raise ValueError(
                    f"`element='{element}'` is not present in the line provided to the TouschekStudy."
                )
            if not isinstance(line[element], TouschekMarker):
                raise TypeError(
                    f"`line['{element}']` is not a TouschekMarker (got {type(line[element]).__name__})."
                )

        local_momentum_acceptance = self.local_momentum_acceptance

        if self.twiss is None:
            twiss_method = self.kwargs.get("method", "6d")
            self.twiss = self.line.twiss(method=twiss_method)

        _configure_markers(
            line,
            twiss=self.twiss,
            lma_s=local_momentum_acceptance.s,
            lma_delta_neg=local_momentum_acceptance.delta_neg,
            lma_delta_pos=local_momentum_acceptance.delta_pos,
            bunch_intensity=self.bunch_intensity,
            gemitt_x=self.gemitt_x,
            gemitt_y=self.gemitt_y,
            sigma_z=self.sigma_z,
            sigma_delta=self.sigma_delta,
            n_scattering_events=self.n_scattering_events,
            nx=self.nx, ny=self.ny, nz=self.nz,
            weight_retention_fraction=self.weight_retention_fraction,
            rng=self.rng,
            at=element,
        )

    def local_rates(self, *, particles_by_element=None, include_tracking=False):
        """
        Return an ``xt.Table`` with per-scattering-element diagnostics.

        Parameters
        ----------
        particles_by_element : dict or None, optional
            Mapping from element name to the particles generated at that
            element. If provided, Monte Carlo particle counts and weight sums
            are included in the table.
        include_tracking : bool, optional
            If ``True``, include loss-count and lost-weight columns computed
            from tracked particle states. This option is meaningful only when
            ``particles_by_element`` is provided.

        Returns
        -------
        table : xtrack.Table
            Per-element diagnostics table. It always contains ``name``, ``s``,
            ``delta_neg``, ``delta_pos``, ``piwinski_rate``, and
            ``integrated_piwinski_rate``. If particles are provided, it also
            contains ``total_mc_rate``, ``ignored_rate``, ``num_particles``,
            and ``sum_weight``. If ``include_tracking`` is ``True``, it
            additionally contains ``num_lost_particles`` and
            ``sum_lost_weight``.
        """
        data = {
            "name": [],
            "s": [],
            "delta_neg": [],
            "delta_pos": [],
            "piwinski_rate": [],
            "integrated_piwinski_rate": [],
        }
        include_particles = particles_by_element is not None
        if include_particles:
            data.update({
                "total_mc_rate": [],
                "ignored_rate": [],
                "num_particles": [],
                "sum_weight": [],
            })
        if include_tracking:
            data.update({
                "num_lost_particles": [],
                "sum_lost_weight": [],
            })

        tab = self.line.get_table()
        for nn in self.elements:
            elem = self.line[nn]
            data["name"].append(nn)
            if hasattr(elem, "s"):
                s = elem.s
            else:
                s = tab["s", nn]
            data["s"].append(float(s))
            data["delta_neg"].append(float(getattr(elem, "delta_neg", np.nan)))
            data["delta_pos"].append(float(getattr(elem, "delta_pos", np.nan)))
            data["piwinski_rate"].append(
                float(getattr(elem, "piwinski_rate", np.nan)))
            data["integrated_piwinski_rate"].append(
                float(getattr(elem, "integrated_piwinski_rate", np.nan)))

            particles = None
            if particles_by_element is not None:
                particles = particles_by_element.get(nn)

            if include_particles:
                data["total_mc_rate"].append(
                    float(getattr(elem, "total_mc_rate", np.nan)))
                data["ignored_rate"].append(
                    float(getattr(elem, "ignored_rate", np.nan)))
                if particles is None:
                    data["num_particles"].append(0)
                    data["sum_weight"].append(np.nan)
                else:
                    generated = _generated(particles)
                    data["num_particles"].append(int(np.sum(generated)))
                    data["sum_weight"].append(
                        float(np.sum(particles.weight[generated])))

            if include_tracking:
                if particles is None:
                    data["num_lost_particles"].append(0)
                    data["sum_lost_weight"].append(np.nan)
                else:
                    lost_mask = _lost(particles) & _generated(particles)
                    data["num_lost_particles"].append(int(np.sum(lost_mask)))
                    data["sum_lost_weight"].append(
                        float(np.sum(particles.weight[lost_mask])))

        for kk, vv in data.items():
            data[kk] = np.array(vv)

        return xt.Table(data)

    def generate_particles(self):
        """
        Generate Touschek-scattered particles at the configured elements.

        Parameters
        ----------
        None

        Returns
        -------
        particles_by_element : dict
            Mapping ``{element_name: xt.Particles}``.
        """
        self._new_rng()
        return {
            nn: self.line[nn].scatter(
                _rng_state=self._rng_state_for_element(nn),
                _rng_particle=self._rng_particle_for_element(nn))
            for nn in self.elements
        }
