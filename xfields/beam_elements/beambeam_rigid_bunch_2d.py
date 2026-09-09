# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import warnings

import numpy as np

import xobjects as xo
import xtrack as xt


def _warn_if_large_transverse_coupling(
        Sigma_11, Sigma_13, Sigma_33, max_correlation=1e-2):
    """Warn when an unvalidated coupled covariance is significant."""
    if Sigma_13 is None:
        return
    Sigma_13, Sigma_11, Sigma_33 = np.broadcast_arrays(
        np.asarray(Sigma_13), np.asarray(Sigma_11), np.asarray(Sigma_33))
    scale = np.sqrt(np.abs(Sigma_11 * Sigma_33))
    correlation = np.divide(
        np.abs(Sigma_13), scale,
        out=np.zeros_like(scale, dtype=float),
        where=scale > 0)
    correlation[(scale == 0) & (Sigma_13 != 0)] = np.inf
    max_observed = np.max(correlation)
    if max_observed > max_correlation:
        warnings.warn(
            'Rigid-bunch transverse coupling has not yet been validated; '
            'maximum normalized transverse correlation '
            f'|Sigma_13|/sqrt(Sigma_11*Sigma_33) is {max_observed:.3e}.',
            RuntimeWarning, stacklevel=2)


def _resolve_covariance_inputs(
        *, Sigma_11, Sigma_13, Sigma_33, sigma_x, sigma_y,
        beam_name, required=False, default=None):
    """Return canonical covariance inputs from covariance or RMS-size data."""
    has_covariance = any(value is not None for value in
                         (Sigma_11, Sigma_13, Sigma_33))
    has_sigmas = sigma_x is not None or sigma_y is not None
    if has_covariance and has_sigmas:
        raise ValueError(
            f'Provide either `{beam_name}_Sigma_11/13/33` or '
            f'`{beam_name}_sigma_x/y`, not both.')

    if has_sigmas:
        if sigma_x is None or sigma_y is None:
            raise ValueError(
                f'`{beam_name}_sigma_x` and `{beam_name}_sigma_y` must be '
                'provided together.')
        Sigma_11 = np.asarray(sigma_x, dtype=float) ** 2
        Sigma_13 = 0.
        Sigma_33 = np.asarray(sigma_y, dtype=float) ** 2
    elif has_covariance:
        if Sigma_11 is None or Sigma_33 is None:
            raise ValueError(
                f'`{beam_name}_Sigma_11` and `{beam_name}_Sigma_33` are '
                'required when covariance inputs are used.')
        if Sigma_13 is None:
            Sigma_13 = 0.
    elif default is not None:
        Sigma_11, Sigma_13, Sigma_33 = default
    elif required:
        raise ValueError(
            f'`{beam_name}_Sigma_11/33` or `{beam_name}_sigma_x/y` are '
            'required.')
    else:
        return None, None, None

    _warn_if_large_transverse_coupling(Sigma_11, Sigma_13, Sigma_33)
    return Sigma_11, Sigma_13, Sigma_33


class BeamBeamBiGaussianRigidBunch2D(xt.BeamElement):

    """
    2D (transverse) rigid-bunch beam-beam element in the soft-Gaussian
    approximation.

    The opposing beam is described as a set of bunches, each one represented
    by a single macroparticle holding the bunch centroid (``x``, ``y``), its
    longitudinal position (``zeta``), its population (number of real charges)
    and its transverse covariance (``other_beam_Sigma_11``,
    ``other_beam_Sigma_13``, ``other_beam_Sigma_33``).

    With ``coherent=False`` (incoherent, weak-strong) the kick is the field
    of a Gaussian charge distribution with the opposing bunch's own sizes.
    With ``coherent=True`` (rigid-bunch dipole model) the effective Gaussian
    covariance is the CONVOLUTION of the pair, ``Sigma_own + Sigma_other``,
    computed from this beam's own covariance at the element. Transverse
    coupling is represented by ``Sigma_13`` in the API and serialized storage,
    but is deliberately ignored by the kick until coupled rigid-bunch operation
    has been validated. A warning is emitted when the normalized transverse
    correlation exceeds 1e-2.

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

        # This (the tracked/OWN) beam's per-bunch transverse covariance, indexed by
        # the OWN beam bunches (like the tracked particles' own populations).
        # The kernel matches each tracked particle to its own bunch on
        # `own_beam_zeta`; one own bunch means uniform covariance (index 0).
        'num_own_bunches': xo.Int64,
        'own_beam_zeta': xo.Float64[:],
        'own_beam_Sigma_11': xo.Float64[:],
        'own_beam_Sigma_13': xo.Float64[:],
        'own_beam_Sigma_33': xo.Float64[:],

        'min_sigma_diff': xo.Float64,

        # Per-bunch description of the opposing beam
        'num_other_bunches': xo.Int64,
        'other_beam_zeta': xo.Float64[:],
        'other_beam_shift_x': xo.Float64[:],
        'other_beam_shift_y': xo.Float64[:],
        'other_beam_num_particles': xo.Float64[:],
        'other_beam_Sigma_11': xo.Float64[:],
        'other_beam_Sigma_13': xo.Float64[:],
        'other_beam_Sigma_33': xo.Float64[:],

    }

    _extra_c_sources = [
        '#include "xfields/beam_elements/beambeam_src/beambeam_rigid_bunch_2d.h"',
    ]

    def __init__(self,
                    scale_strength=1.,

                    zeta_offset=0.,
                    zeta_match_tol=1e-3,
                    zeta_period=0.,

                    other_beam_q0=0,
                    other_beam_beta0=1,

                    coherent=False,
                    own_beam_zeta=None,
                    own_beam_Sigma_11=None,
                    own_beam_Sigma_13=None,
                    own_beam_Sigma_33=None,

                    other_particles=None,

                    other_beam_Sigma_11=None,
                    other_beam_Sigma_13=None,
                    other_beam_Sigma_33=None,

                    own_beam_sigma_x=None,
                    own_beam_sigma_y=None,
                    other_beam_sigma_x=None,
                    other_beam_sigma_y=None,

                    min_sigma_diff=1e-10,

                    **kwargs):

        """
        Parameters
        ----------
        scale_strength : float, optional
                Used to scale the beam-beam force strength.
                Scales ``other_beam_q0``.
        zeta_offset : float, optional
                A particle of this beam at ``zeta`` interacts
                with the opposing bunch located at ``zeta + zeta_offset``.
        zeta_match_tol : float, optional
                Maximum allowed distance in ``zeta`` between
                a particle's encounter position (``zeta + zeta_offset``) and the
                centroid of an opposing bunch for them to interact.
        zeta_period : float, optional
                Periodicity of the ``zeta`` bunch-label axis
                (e.g. ``n_slots * slot_spacing`` for a circular machine). If
                larger than zero, the encounter distance is evaluated modulo
                this period, so encounter offsets that wrap around the ring
                still find their partner. Zero (default) disables wrapping.
        other_beam_q0 : float, optional
                Charge sign of the opposing beam. -1 for
                electrons, +1 for protons or positrons.
        other_beam_beta0 : float, optional
                Relativistic beta of the opposing beam.
        coherent : bool, optional
                If False (default, incoherent weak-strong) the
                kick uses each opposing bunch's covariance and the own-beam
                covariance is ignored. If True (coherent rigid-bunch model),
                the effective covariance is the sum of the matched own- and
                opposing-beam covariances; own-beam covariance inputs are then
                required.
        own_beam_zeta : array-like, optional
                Longitudinal positions (bunch labels)
                of this beam's bunches, one per bunch, used by the kernel to
                match each tracked particle to its own bunch (and hence its own
                size) -- the OWN-beam analogue of ``other_beam_zeta``. Required
                for per-bunch own sizes; omit it (single own bunch) for a
                uniform own size.
        own_beam_Sigma_11, own_beam_Sigma_13, own_beam_Sigma_33 : array-like, optional
                Transverse covariance components of THIS (the tracked) beam,
                used only with ``coherent=True``. Values are indexed by
                ``own_beam_zeta`` and scalars are broadcast. ``Sigma_13`` is
                stored but currently ignored by the kick.
        other_particles : xpart.Particles, optional
                Particles object of the opposing
                beam in which each active macroparticle represents one bunch.
                Its centroids (``x``, ``y``), longitudinal positions (``zeta``)
                and populations (``weight``) are loaded into the element (as by
                :meth:`update_from_other_beam`). The active particles determine
                the exact lengths of the opposing-bunch arrays.
        other_beam_Sigma_11, other_beam_Sigma_13, other_beam_Sigma_33 : array-like, optional
                Transverse covariance components of each opposing bunch,
                aligned with the active particles of ``other_particles``.
                Scalars are broadcast. The names and meaning match
                :class:`BeamBeamBiGaussian2D`, although ``Sigma_13`` is stored
                but currently ignored by the rigid-bunch kick.
        own_beam_sigma_x, own_beam_sigma_y : float or array-like, optional
                Convenience alternative to the own-beam covariance inputs.
                Both must be supplied and are converted to diagonal covariance.
        other_beam_sigma_x, other_beam_sigma_y : float or array-like, optional
                Convenience alternative to the opposing-beam covariance inputs.
                Both must be supplied and are converted to diagonal covariance.
        min_sigma_diff : float, optional
                Round-beam kick (~2x faster) is used instead
                of the elliptical kick if
                ``fabs(sigma_x - sigma_y) < min_sigma_diff``.
        **kwargs
                Xobject construction arguments used for deserialization and
                context/buffer placement.
        """

        if '_xobject' in kwargs.keys():
            self.xoinitialize(**kwargs)
            return

        # Dictionary/JSON restoration already contains the exactly sized
        # arrays and active counts, so no representative Particles is needed.
        if other_particles is None and 'other_beam_zeta' in kwargs:
            _warn_if_large_transverse_coupling(
                own_beam_Sigma_11, own_beam_Sigma_13, own_beam_Sigma_33)
            _warn_if_large_transverse_coupling(
                other_beam_Sigma_11, other_beam_Sigma_13,
                other_beam_Sigma_33)
            self.xoinitialize(
                scale_strength=scale_strength,
                zeta_offset=zeta_offset,
                zeta_match_tol=zeta_match_tol,
                zeta_period=zeta_period,
                other_beam_q0=other_beam_q0,
                other_beam_beta0=other_beam_beta0,
                coherent=coherent,
                own_beam_zeta=own_beam_zeta,
                own_beam_Sigma_11=own_beam_Sigma_11,
                own_beam_Sigma_13=own_beam_Sigma_13,
                own_beam_Sigma_33=own_beam_Sigma_33,
                other_beam_Sigma_11=other_beam_Sigma_11,
                other_beam_Sigma_13=other_beam_Sigma_13,
                other_beam_Sigma_33=other_beam_Sigma_33,
                min_sigma_diff=min_sigma_diff,
                **kwargs)
            return

        if other_particles is None:
            raise ValueError(
                '`other_particles` is required to infer the opposing filling '
                'and allocate its arrays.')
        state = other_particles._context.nparray_from_context_array(
            other_particles.state)
        num_bunches = int((state > 0).sum())
        if num_bunches == 0:
            raise ValueError(
                '`other_particles` must contain at least one active bunch.')

        # The own filling determines the exact own-array lengths. With no
        # explicit grid, one entry represents a size common to all own bunches.
        num_own_bunches = (len(np.atleast_1d(own_beam_zeta))
                           if own_beam_zeta is not None else 1)

        self.xoinitialize(
            own_beam_zeta=num_own_bunches,
            own_beam_Sigma_11=num_own_bunches,
            own_beam_Sigma_13=num_own_bunches,
            own_beam_Sigma_33=num_own_bunches,
            other_beam_zeta=num_bunches,
            other_beam_shift_x=num_bunches,
            other_beam_shift_y=num_bunches,
            other_beam_num_particles=num_bunches,
            other_beam_Sigma_11=num_bunches,
            other_beam_Sigma_13=num_bunches,
            other_beam_Sigma_33=num_bunches,
            **kwargs)

        self.scale_strength = scale_strength

        self.zeta_offset = zeta_offset
        self.zeta_match_tol = zeta_match_tol
        self.zeta_period = zeta_period

        self.other_beam_q0 = other_beam_q0
        self.other_beam_beta0 = other_beam_beta0

        own_covariance = _resolve_covariance_inputs(
            Sigma_11=own_beam_Sigma_11,
            Sigma_13=own_beam_Sigma_13,
            Sigma_33=own_beam_Sigma_33,
            sigma_x=own_beam_sigma_x,
            sigma_y=own_beam_sigma_y,
            beam_name='own_beam', required=coherent,
            default=None if coherent else (0., 0., 0.))
        other_covariance = _resolve_covariance_inputs(
            Sigma_11=other_beam_Sigma_11,
            Sigma_13=other_beam_Sigma_13,
            Sigma_33=other_beam_Sigma_33,
            sigma_x=other_beam_sigma_x,
            sigma_y=other_beam_sigma_y,
            beam_name='other_beam', default=(1., 0., 1.))
        self.coherent = bool(coherent)
        # Own per-bunch covariances are indexed by THIS beam. With an explicit
        # zeta grid the kernel matches the tracked particle to its bunch; else a
        # single covariance is broadcast over the one own bunch.
        self.num_own_bunches = 1
        if own_beam_zeta is not None:
            self.update_from_own_beam(
                own_beam_zeta,
                own_beam_Sigma_11=own_covariance[0],
                own_beam_Sigma_13=own_covariance[1],
                own_beam_Sigma_33=own_covariance[2])
        else:
            for name, value in zip(
                    ('own_beam_Sigma_11', 'own_beam_Sigma_13',
                     'own_beam_Sigma_33'), own_covariance):
                self._set_per_bunch(name, value, num_own_bunches)

        self.min_sigma_diff = min_sigma_diff

        self.num_other_bunches = num_bunches
        self.update_from_other_beam(
            other_particles,
            other_beam_Sigma_11=other_covariance[0],
            other_beam_Sigma_13=other_covariance[1],
            other_beam_Sigma_33=other_covariance[2])

    def _set_per_bunch(self, name, value, num_bunches):
        value = np.atleast_1d(np.asarray(value, dtype=float))
        if value.size == 1:
            value = np.full(num_bunches, value[0])
        assert value.size == num_bunches, (
            f'`{name}` has {value.size} entries but the element was allocated '
            f'for {num_bunches} bunches.')
        getattr(self, name)[:] = self._arr2ctx(value)

    def update_from_own_beam(
            self, zeta=None,
            own_beam_Sigma_11=None, own_beam_Sigma_13=None,
            own_beam_Sigma_33=None,
            own_beam_sigma_x=None, own_beam_sigma_y=None):
        """Set THIS (the tracked) beam's per-bunch data. With ``zeta`` given, set
        the per-bunch zeta grid ``own_beam_zeta`` and, optionally, the own
        covariance. Covariance or convenience sigma inputs are sorted together
        with ``zeta``. With ``zeta=None`` only the covariance is updated for the
        already-registered own bunches. Scalars are broadcast. A change in the
        number of bunches requires element reconfiguration.

        ``own_beam_Sigma_13`` is stored and serialized but is currently ignored
        by the rigid-bunch kick. A :class:`RuntimeWarning` is emitted if
        ``abs(Sigma_13) / sqrt(Sigma_11 * Sigma_33)`` exceeds ``1e-2``.

        Parameters
        ----------
        zeta : array-like, optional
            Longitudinal positions of this beam's bunches.
        own_beam_Sigma_11, own_beam_Sigma_13, own_beam_Sigma_33 : array-like, optional
            Per-bunch transverse covariance components. Scalars are broadcast.
        own_beam_sigma_x, own_beam_sigma_y : float or array-like, optional
            RMS-size alternative to the covariance components. Both are
            required when this representation is used.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If inputs are incomplete or incompatible with the allocated bunch
            capacity.
        """
        covariance = _resolve_covariance_inputs(
            Sigma_11=own_beam_Sigma_11,
            Sigma_13=own_beam_Sigma_13,
            Sigma_33=own_beam_Sigma_33,
            sigma_x=own_beam_sigma_x,
            sigma_y=own_beam_sigma_y,
            beam_name='own_beam')
        if zeta is not None:
            zeta = np.atleast_1d(np.asarray(zeta, dtype=float))
            n = len(zeta)
            capacity = len(self.own_beam_zeta)
            if n != capacity:
                raise ValueError(
                    f'This beam has {n} bunches but the element has arrays for '
                    f'{capacity}; reconfigure the element when the filling '
                    'changes.')
            order = np.argsort(zeta, kind='stable')
            self.num_own_bunches = n
            self.own_beam_zeta[:n] = self._arr2ctx(zeta[order])
        else:
            n = int(self.num_own_bunches)
            order = np.arange(n)   # keep the existing own_beam_zeta order
        for name, value in zip(
                ('own_beam_Sigma_11', 'own_beam_Sigma_13',
                 'own_beam_Sigma_33'), covariance):
            if value is None:
                continue
            value = np.atleast_1d(np.asarray(value, dtype=float))
            if value.size == 1:
                value = np.full(n, value[0])
            assert value.size == n, (
                f'`{name}` has {value.size} entries but this beam has {n} '
                f'bunches.')
            getattr(self, name)[:n] = self._arr2ctx(value[order])

    def update_from_other_beam(
            self, other_particles,
            other_beam_Sigma_11=None, other_beam_Sigma_13=None,
            other_beam_Sigma_33=None,
            other_beam_sigma_x=None, other_beam_sigma_y=None):

        """
        Load the centroid, longitudinal position and population of the bunches
        of the opposing beam from a :class:`xpart.Particles` object in which
        each (active) macroparticle represents one bunch, optionally together
        with the per-bunch transverse covariance (scalar or array aligned with
        the active particles). RMS sizes can be supplied as a convenience
        instead of covariance components.

        Should be called before tracking either beam through the beam-beam
        elements so that both kicks are computed from the bunch positions at the
        same turn (strong-strong simultaneity).

        If no covariance or sigma inputs are given, the stored covariance is
        kept. This is valid only if the set and zeta ordering of bunches is
        unchanged since the covariance was last set.

        ``other_beam_Sigma_13`` is stored and serialized but is currently
        ignored by the rigid-bunch kick. A :class:`RuntimeWarning` is emitted
        if ``abs(Sigma_13) / sqrt(Sigma_11 * Sigma_33)`` exceeds ``1e-2``.

        Parameters
        ----------
        other_particles : xpart.Particles
            Opposing-beam bunch centroids, longitudinal positions, populations,
            and active-particle mask.
        other_beam_Sigma_11, other_beam_Sigma_13, other_beam_Sigma_33 : array-like, optional
            Opposing-beam transverse covariance components. Scalars are
            broadcast.
        other_beam_sigma_x, other_beam_sigma_y : float or array-like, optional
            RMS-size alternative to the covariance components. Both are
            required when this representation is used.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If inputs are incomplete or the active bunch count differs from
            the element's allocated capacity.
        """

        covariance = _resolve_covariance_inputs(
            Sigma_11=other_beam_Sigma_11,
            Sigma_13=other_beam_Sigma_13,
            Sigma_33=other_beam_Sigma_33,
            sigma_x=other_beam_sigma_x,
            sigma_y=other_beam_sigma_y,
            beam_name='other_beam')

        ctx2np = self._buffer.context.nparray_from_context_array

        state = ctx2np(other_particles.state)
        mask = state > 0

        x = ctx2np(other_particles.x)[mask]
        y = ctx2np(other_particles.y)[mask]
        zeta = ctx2np(other_particles.zeta)[mask]
        weight = ctx2np(other_particles.weight)[mask]

        n = len(x)
        capacity = len(self.other_beam_zeta)
        if n != capacity:
            raise ValueError(
                f'The opposing beam has {n} bunches but the element has '
                f'arrays for {capacity}; reconfigure the element when the '
                'filling changes.')

        # The tracking kernel finds the encounter partner by binary search, so
        # the bunches are stored sorted in zeta.
        order = np.argsort(zeta, kind='stable')

        self.num_other_bunches = n
        self.other_beam_zeta[:n] = self._arr2ctx(zeta[order])
        self.other_beam_shift_x[:n] = self._arr2ctx(x[order])
        self.other_beam_shift_y[:n] = self._arr2ctx(y[order])
        self.other_beam_num_particles[:n] = self._arr2ctx(weight[order])

        for name, value in zip(
                ('other_beam_Sigma_11', 'other_beam_Sigma_13',
                 'other_beam_Sigma_33'), covariance):
            if value is None:
                continue
            value = np.atleast_1d(np.asarray(value, dtype=float))
            if value.size == 1:
                value = np.full(n, value[0])
            assert value.size == n, (
                f'`{name}` has {value.size} entries but the opposing beam has '
                f'{n} bunches.')
            getattr(self, name)[:n] = self._arr2ctx(value[order])
