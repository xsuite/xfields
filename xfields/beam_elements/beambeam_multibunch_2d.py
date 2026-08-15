# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np

import xobjects as xo
import xtrack as xt


class BeamBeamBiGaussianMultibunch2D(xt.BeamElement):

    """
    2D (transverse) beam-beam element for multi-bunch beams in the
    soft-Gaussian approximation.

    The opposing beam is described as a set of bunches, each one represented
    by a single macroparticle holding the bunch centroid (``x``, ``y``), its
    longitudinal position (``zeta``), its population (number of real charges)
    and its transverse sizes (``other_beam_sigma_x``, ``other_beam_sigma_y``).

    With ``coherent=False`` (incoherent, weak-strong) the kick is the field
    of a Gaussian charge distribution with the opposing bunch's own sizes.
    With ``coherent=True`` (rigid-bunch dipole model) the effective Gaussian
    size is the CONVOLUTION of the pair, ``sqrt(sigma_own**2 +
    sigma_other**2)``, computed from this beam's own sizes at the element
    (``sigma_x``, ``sigma_y``, required in this mode).

    During tracking, a particle (bunch) of this beam located at ``zeta``
    interacts with the opposing bunch located at ``zeta + zeta_offset``. The
    matching opposing bunch is the one whose ``zeta`` is closest to
    ``zeta + zeta_offset`` within ``zeta_match_tol``; if none is found the
    particle receives no kick.
    """

    _xofields = {

        'scale_strength': xo.Float64,

        'zeta_offset': xo.Float64,
        'zeta_match_tol': xo.Float64,
        'zeta_period': xo.Float64,

        'other_beam_q0': xo.Float64,
        'other_beam_beta0': xo.Float64,

        'coherent': xo.Int64,

        # This (the tracked/OWN) beam's per-bunch transverse sizes, indexed by
        # the OWN beam bunches (like the tracked particles' own populations).
        # The kernel matches each tracked particle to its own bunch on
        # `own_beam_zeta`; a single own bunch means a uniform size (index 0).
        'num_own_bunches': xo.Int64,
        'own_beam_zeta': xo.Float64[:],
        'sigma_x': xo.Float64[:],
        'sigma_y': xo.Float64[:],

        'min_sigma_diff': xo.Float64,

        # Per-bunch description of the opposing beam
        'num_other_bunches': xo.Int64,
        'other_beam_zeta': xo.Float64[:],
        'other_beam_x': xo.Float64[:],
        'other_beam_y': xo.Float64[:],
        'other_beam_num_particles': xo.Float64[:],
        'other_beam_sigma_x': xo.Float64[:],
        'other_beam_sigma_y': xo.Float64[:],

    }

    _extra_c_sources = [
        '#include "xfields/beam_elements/beambeam_src/beambeam_multibunch_2d.h"',
    ]

    def __init__(self,
                    num_bunches=None,

                    scale_strength=1.,

                    zeta_offset=0.,
                    zeta_match_tol=1e-3,
                    zeta_period=0.,

                    other_beam_q0=0,
                    other_beam_beta0=1,

                    coherent=False,
                    num_own_bunches=None,
                    own_beam_zeta=None,
                    sigma_x=None,
                    sigma_y=None,

                    other_particles=None,

                    other_beam_sigma_x=None,
                    other_beam_sigma_y=None,

                    min_sigma_diff=1e-10,

                    **kwargs):

        """
        Args:
            num_bunches (int): Maximum number of bunches of the opposing beam.
                Used to allocate the internal arrays. Inferred from
                ``other_particles`` if not given.
            scale_strength (float): Used to scale the beam-beam force strength.
                Scales ``other_beam_q0``.
            zeta_offset (float): A particle of this beam at ``zeta`` interacts
                with the opposing bunch located at ``zeta + zeta_offset``.
            zeta_match_tol (float): Maximum allowed distance in ``zeta`` between
                a particle's encounter position (``zeta + zeta_offset``) and the
                centroid of an opposing bunch for them to interact.
            zeta_period (float): Periodicity of the ``zeta`` bunch-label axis
                (e.g. ``n_slots * slot_spacing`` for a circular machine). If
                larger than zero, the encounter distance is evaluated modulo
                this period, so encounter offsets that wrap around the ring
                still find their partner. Zero (default) disables wrapping.
            other_beam_q0 (float): Charge sign of the opposing beam. -1 for
                electrons, +1 for protons or positrons.
            other_beam_beta0 (float): Relativistic beta of the opposing beam.
            coherent (bool): If False (default, incoherent weak-strong) the
                kick uses each opposing bunch's own sizes and ``sigma_x``/
                ``sigma_y`` are ignored. If True (coherent rigid-bunch
                model) the effective size is the convolution
                ``sqrt(sigma_own**2 + sigma_other**2)`` and ``sigma_x``/
                ``sigma_y`` are required.
            num_own_bunches (int): Number of bunches of THIS (the tracked) beam,
                to allocate the own per-bunch arrays. Inferred from
                ``own_beam_zeta`` (or 1) if not given.
            own_beam_zeta (float array): Longitudinal positions (bunch labels)
                of this beam's bunches, one per bunch, used by the kernel to
                match each tracked particle to its own bunch (and hence its own
                size) -- the OWN-beam analogue of ``other_beam_zeta``. Required
                for per-bunch own sizes; omit it (single own bunch) for a
                uniform own size.
            sigma_x, sigma_y (float or float array): Transverse sizes of THIS
                (the tracked) beam at the element, used only with
                ``coherent=True``. Indexed by the OWN beam bunches (aligned with
                ``own_beam_zeta``), just like the tracked particles carry their
                own populations; a scalar is broadcast (uniform own size). The
                opposing sizes are ``other_beam_sigma_x``/``other_beam_sigma_y``
                (indexed by the OTHER beam); the kernel convolves the matched
                pair.
            other_particles (xpart.Particles): Particles object of the opposing
                beam in which each active macroparticle represents one bunch.
                Its centroids (``x``, ``y``), longitudinal positions (``zeta``)
                and populations (``weight``) are loaded into the element (as by
                :meth:`update_from_other_beam`). Also used to infer
                ``num_bunches`` when not given explicitly.
            other_beam_sigma_x, other_beam_sigma_y (float or float array):
                Transverse sizes of each opposing bunch (aligned with the
                active particles of ``other_particles``). A scalar is
                broadcast to all bunches.
            min_sigma_diff (float): Round-beam kick (~2x faster) is used instead
                of the elliptical kick if
                ``fabs(sigma_x - sigma_y) < min_sigma_diff``.
        """

        if '_xobject' in kwargs.keys():
            self.xoinitialize(**kwargs)
            return

        # Determine the number of active bunches in the opposing beam
        n_active = 0
        if other_particles is not None:
            state = other_particles._context.nparray_from_context_array(
                other_particles.state)
            n_active = int((state > 0).sum())

        if num_bunches is None:
            num_bunches = n_active
        if num_bunches == 0:
            raise ValueError(
                'Specify `num_bunches` or `other_particles` to allocate the '
                'element.')
        assert num_bunches >= n_active, (
            '`num_bunches` must be >= the number of bunches in `other_particles`')

        # Own beam allocation (this beam's bunches: own zeta grid + own sizes)
        if num_own_bunches is None:
            num_own_bunches = (len(np.atleast_1d(own_beam_zeta))
                               if own_beam_zeta is not None else 1)
        num_own_bunches = max(int(num_own_bunches), 1)

        self.xoinitialize(
            own_beam_zeta=num_own_bunches,
            sigma_x=num_own_bunches,
            sigma_y=num_own_bunches,
            other_beam_zeta=num_bunches,
            other_beam_x=num_bunches,
            other_beam_y=num_bunches,
            other_beam_num_particles=num_bunches,
            other_beam_sigma_x=num_bunches,
            other_beam_sigma_y=num_bunches,
            **kwargs)

        self.scale_strength = scale_strength

        self.zeta_offset = zeta_offset
        self.zeta_match_tol = zeta_match_tol
        self.zeta_period = zeta_period

        self.other_beam_q0 = other_beam_q0
        self.other_beam_beta0 = other_beam_beta0

        if coherent and (sigma_x is None or sigma_y is None):
            raise ValueError(
                '`sigma_x` and `sigma_y` (own beam sizes) are required for '
                'the coherent (rigid-bunch) mode.')
        self.coherent = bool(coherent)
        # Own per-bunch sizes are indexed by THIS beam. With an explicit own
        # zeta grid the kernel matches the tracked particle to its bunch; else a
        # single (uniform) own size broadcast over the one own bunch.
        self.num_own_bunches = 1
        if own_beam_zeta is not None:
            self.update_from_own_beam(
                own_beam_zeta,
                sigma_x=0. if sigma_x is None else sigma_x,
                sigma_y=0. if sigma_y is None else sigma_y)
        else:
            self._set_per_bunch('sigma_x', 0. if sigma_x is None else sigma_x,
                                num_own_bunches)
            self._set_per_bunch('sigma_y', 0. if sigma_y is None else sigma_y,
                                num_own_bunches)

        self.min_sigma_diff = min_sigma_diff

        self.num_other_bunches = 0
        if other_particles is not None:
            self.update_from_other_beam(
                other_particles,
                other_beam_sigma_x=other_beam_sigma_x,
                other_beam_sigma_y=other_beam_sigma_y)
        else:
            # sizes stored now, loaded bunches later (update_from_other_beam)
            if other_beam_sigma_x is not None:
                self._set_per_bunch('other_beam_sigma_x', other_beam_sigma_x,
                                    num_bunches)
            if other_beam_sigma_y is not None:
                self._set_per_bunch('other_beam_sigma_y', other_beam_sigma_y,
                                    num_bunches)

    def _set_per_bunch(self, name, value, num_bunches):
        value = np.atleast_1d(np.asarray(value, dtype=float))
        if value.size == 1:
            value = np.full(num_bunches, value[0])
        assert value.size <= num_bunches, (
            f'`{name}` has {value.size} entries but the element was allocated '
            f'for {num_bunches} bunches.')
        getattr(self, name)[:value.size] = self._arr2ctx(value)

    def update_from_own_beam(self, zeta=None, sigma_x=None, sigma_y=None):
        """Set THIS (the tracked) beam's per-bunch data. With ``zeta`` given, set
        the per-bunch zeta grid ``own_beam_zeta`` (used by the kernel to match
        each tracked particle to its own bunch) and, optionally, the own sizes
        ``sigma_x``/``sigma_y`` -- the three are sorted together along ``zeta``
        (the kernel partner search is a binary search). With ``zeta=None`` only
        the sizes are updated, for the already-registered own bunches (the first
        ``num_own_bunches`` entries, in ``own_beam_zeta`` order), e.g. to feed the
        dynamic-beta sizes each iteration. A scalar size is broadcast. The
        OWN-beam analogue of :meth:`update_from_other_beam`; here x/y/population
        come from the tracked particles, so only zeta and the sizes are stored.
        Writes to a prefix, so it is robust to the array capacity exceeding the
        active bunch count (a setup sized for a larger filling than the solved
        one)."""
        if zeta is not None:
            zeta = np.atleast_1d(np.asarray(zeta, dtype=float))
            n = len(zeta)
            capacity = len(self.own_beam_zeta)
            if n > capacity:
                raise ValueError(
                    f'This beam has {n} bunches but the element was allocated '
                    f'for {capacity}. Increase `num_own_bunches`.')
            order = np.argsort(zeta, kind='stable')
            self.num_own_bunches = n
            self.own_beam_zeta[:n] = self._arr2ctx(zeta[order])
        else:
            n = int(self.num_own_bunches)
            order = np.arange(n)   # keep the existing own_beam_zeta order
        for name, value in (('sigma_x', sigma_x), ('sigma_y', sigma_y)):
            if value is None:
                continue
            value = np.atleast_1d(np.asarray(value, dtype=float))
            if value.size == 1:
                value = np.full(n, value[0])
            assert value.size == n, (
                f'`{name}` has {value.size} entries but this beam has {n} '
                f'bunches.')
            getattr(self, name)[:n] = self._arr2ctx(value[order])

    def update_from_other_beam(self, other_particles,
                               other_beam_sigma_x=None,
                               other_beam_sigma_y=None):

        """
        Load the centroid, longitudinal position and population of the bunches
        of the opposing beam from a :class:`xpart.Particles` object in which
        each (active) macroparticle represents one bunch, optionally together
        with the per-bunch transverse sizes (scalar or array aligned with the
        active particles).

        Should be called before tracking either beam through the beam-beam
        elements so that both kicks are computed from the bunch positions at the
        same turn (strong-strong simultaneity).

        Note: if ``other_beam_sigma_x``/``other_beam_sigma_y`` are not
        given the stored sizes are kept -- only valid if the set (and zeta
        ordering) of bunches is unchanged since the sizes were last set.
        """

        ctx2np = self._buffer.context.nparray_from_context_array

        state = ctx2np(other_particles.state)
        mask = state > 0

        x = ctx2np(other_particles.x)[mask]
        y = ctx2np(other_particles.y)[mask]
        zeta = ctx2np(other_particles.zeta)[mask]
        weight = ctx2np(other_particles.weight)[mask]

        n = len(x)
        capacity = len(self.other_beam_zeta)
        if n > capacity:
            raise ValueError(
                f'The opposing beam has {n} bunches but the element was '
                f'allocated for {capacity}. Increase `num_bunches`.')

        # The tracking kernel finds the encounter partner by binary search, so
        # the bunches are stored sorted in zeta.
        order = np.argsort(zeta, kind='stable')

        self.num_other_bunches = n
        self.other_beam_zeta[:n] = self._arr2ctx(zeta[order])
        self.other_beam_x[:n] = self._arr2ctx(x[order])
        self.other_beam_y[:n] = self._arr2ctx(y[order])
        self.other_beam_num_particles[:n] = self._arr2ctx(weight[order])

        for name, value in (('other_beam_sigma_x', other_beam_sigma_x),
                            ('other_beam_sigma_y', other_beam_sigma_y)):
            if value is None:
                continue
            value = np.atleast_1d(np.asarray(value, dtype=float))
            if value.size == 1:
                value = np.full(n, value[0])
            assert value.size == n, (
                f'`{name}` has {value.size} entries but the opposing beam has '
                f'{n} bunches.')
            getattr(self, name)[:n] = self._arr2ctx(value[order])
