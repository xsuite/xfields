# copyright ############################### #
# This file is part of the Xfields Package. #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Generic (machine-independent) rigid-bunch beam-beam mode.

Install coherent (rigid-bunch) 2D beam-beam elements
(:class:`xfields.BeamBeamBiGaussianRigidBunch2D`) for the head-on and
long-range (LR) encounters at an arbitrary set of interaction points (IPs) of
two counter-rotating rings, and find the per-bunch self-consistent closed orbit
of the two multi-bunch beams by iterating the multi-bunch twiss.

The public workflow uses the standard beam-beam install/configure entry points
and returns a small state object:

    env.xfields.install_beambeam_interactions(
        clockwise_line, anticlockwise_line, ip_names=[...],
        num_long_range_encounters_per_side=..., harmonic_number=...,
        bunch_spacing_buckets=..., mode='rigid_bunch')
    study = env.xfields.configure_beambeam_interactions(
        num_particles=..., nemitt_x=..., nemitt_y=...,
        filling_pattern_cw=..., filling_pattern_acw=...)
    solution = study.solve()

Installation places one beam-beam element per encounter DIRECTLY on the two
lines (the element is its own twiss/survey observation point -- there are no
separate markers), with arrays covering every RF slot. Configuration loads the
populations and optional filling, computes the encounter geometry (per-encounter
bunch-pairing offset, convolved sizes, survey separation) and returns a
:class:`BeamBeamRigidBunchStudy`. All further operations are methods on that object:

* :meth:`BeamBeamRigidBunchStudy.solve` -- self-consistent per-bunch closed orbit;
* :meth:`BeamBeamRigidBunchStudy.twiss` -- per-bunch optics for the currently loaded
  opposing-beam state;
* :meth:`BeamBeamRigidBunchStudy.second_order_maps` -- a fast sector-map copy: the arcs
  between the encounters are replaced by second-order maps (splitting the lines
  at the beam-beam elements, which stay exact) and a NEW study on the reduced
  lines is returned; solving it is orders of magnitude faster and gives the same
  per-bunch orbit and tunes;
* :meth:`BeamBeamRigidBunchStudy.load_solution` -- load a converged solution (from a
  reduced-model solve) onto this study's lattice, e.g. to compute footprints on
  the full thick lattice;
* :meth:`BeamBeamRigidBunchStudy.apply_filling_pattern` -- change the per-beam
  bunch filling.

Nothing is LHC specific: the IPs (a ``{ip: offset}`` mapping, or a list of IP
element names for which the head-on offsets are derived from the ring geometry
as ``round(2 * (s_ip - s_ref) / bunch_spacing_zeta)``), the RF harmonic number
and the bunch spacing (in RF buckets) are all inputs. The number of slots is
``n_slots = harmonic_number / bunch_spacing_buckets`` and the positive physical
slot spacing is ``bunch_spacing_zeta = circumference / n_slots``. Physical slot
``i`` is centred at ``zeta = -i * bunch_spacing_zeta``, consistently with the
Xpart, Xwakes and :class:`BeamStatsMonitor` bunch-pattern APIs.

The two lines are the usual xsuite two-ring setup: the ``clockwise_line`` runs
in ``+s`` and the ``anticlockwise_line`` is the *reversed* line (also running in
``+s``); a given encounter element name is the same physical point in both
beams, mirrored on the reversed line.

The full transverse covariance is retained in the configured elements, but the
rigid-bunch kick currently uses only ``Sigma_11`` and ``Sigma_33``.
``Sigma_13`` is stored and serialized but ignored by the kick until coupled
rigid-bunch operation is validated. A :class:`RuntimeWarning` is emitted when
``abs(Sigma_13) / sqrt(Sigma_11 * Sigma_33)`` exceeds ``1e-2``.
"""

from dataclasses import dataclass

import numpy as np

from xtrack.general import _print

from ...beam_elements.beambeam_rigid_bunch_2d import (
    _warn_if_large_transverse_coupling,
)

from .config_tools import (
    BEAMBEAM_CONFIG_KEY,
    BEAMBEAM_CONFIG_VERSION,
    BEAMBEAM_ELEMENT_EXTRA_KEY,
    BEAMBEAM_ELEMENT_EXTRA_VERSION,
    _beambeam_element_name,
    compute_beambeam_geometry,
    compute_twiss_and_madpoints_at_bb,
)


_BEAMBEAM_EXTRA_KEY = BEAMBEAM_ELEMENT_EXTRA_KEY
_BEAMBEAM_EXTRA_VERSION = BEAMBEAM_ELEMENT_EXTRA_VERSION


@dataclass
class _RigidBunchInstallation:
    elements: dict
    line_names: dict
    ip_names: list
    config: dict


def _encounter_specs(ip_names, num_long_range_encounters_per_side):
    """Yield ``(base_name, ip, signed_n)``; ``signed_n == 0`` is the head-on
    encounter."""
    if isinstance(num_long_range_encounters_per_side, dict):
        num_lr_by_ip = [num_long_range_encounters_per_side[ip]
                        for ip in ip_names]
    elif np.ndim(num_long_range_encounters_per_side) == 0:
        num_lr_by_ip = [num_long_range_encounters_per_side] * len(ip_names)
    else:
        num_lr_by_ip = list(num_long_range_encounters_per_side)
        if len(num_lr_by_ip) != len(ip_names):
            raise ValueError(
                '`num_long_range_encounters_per_side` must have one entry '
                'per IP.')

    for ip, num_lr_value in zip(ip_names, num_lr_by_ip):
        num_lr = int(num_lr_value)
        if num_lr != num_lr_value or num_lr < 0:
            raise ValueError(
                '`num_long_range_encounters_per_side` entries must be '
                'non-negative integers.')
        yield f'bb_{ip}_ho', ip, 0
        for identifier in range(1, num_lr + 1):
            yield f'bb_{ip}_r{identifier:02d}', ip, identifier
            yield f'bb_{ip}_l{identifier:02d}', ip, -identifier


def _gamma0(line):
    return float(line.particle_ref.gamma0[0])


def _beta0(line):
    return float(line.particle_ref.beta0[0])


def _bind_beambeam_scale(line, bb_names):
    """Bind all beam-beam elements to the environment-wide scale knob."""
    env = line.env
    if 'beambeam_scale' not in env.vars:
        env.vars['beambeam_scale'] = 1.0
    for name in bb_names:
        line.element_refs[name].scale_strength = env.vars['beambeam_scale']


def _representative_other_beam(
        line, other_line, n_slots, bunch_spacing_zeta):
    """One inactive-kick representative per opposing RF slot."""
    import xtrack as xt
    slots = np.arange(n_slots)
    return xt.Particles(
        _context=line._context,
        p0c=other_line.particle_ref.p0c[0],
        mass0=other_line.particle_ref.mass0,
        q0=other_line.particle_ref.q0,
        x=np.zeros(len(slots)),
        y=np.zeros(len(slots)),
        zeta=-slots * bunch_spacing_zeta,
        weight=np.zeros(len(slots)))


def _new_beambeam_element(line, other_line, n_slots, bunch_spacing_zeta):
    import xfields as xf
    own_zeta = -np.arange(n_slots) * bunch_spacing_zeta
    return xf.BeamBeamBiGaussianRigidBunch2D(
        other_particles=_representative_other_beam(
            line=line, other_line=other_line, n_slots=n_slots,
            bunch_spacing_zeta=bunch_spacing_zeta),
        own_beam_zeta=own_zeta,
        zeta_offset=0.0,
        zeta_match_tol=0.1 * bunch_spacing_zeta,
        zeta_period=n_slots * bunch_spacing_zeta,
        other_beam_q0=float(other_line.particle_ref.q0),
        other_beam_beta0=_beta0(other_line),
        coherent=True,
        own_beam_Sigma_11=1.0,
        own_beam_Sigma_13=0.0,
        own_beam_Sigma_33=1.0,
        other_beam_Sigma_11=1.0,
        other_beam_Sigma_13=0.0,
        other_beam_Sigma_33=1.0,
        _context=line._context)


def _install_elements(
        env, line_name, other_line_name, encounter_specs,
        n_slots, bunch_spacing_zeta, mirror):
    line = env.lines[line_name]
    other_line = env.lines[other_line_name]
    length = line.get_length()
    s_ip = {
        ip_name: float(line.get_table()['s', ip_name])
        for ip_name in dict.fromkeys(
            ip_name for _, ip_name, _ in encounter_specs)}
    placements = []
    element_names = []
    position_sign = -1 if mirror else 1
    for encounter_name, ip_name, identifier in encounter_specs:
        displacement = (
            position_sign * identifier * bunch_spacing_zeta / 2)
        at = (s_ip[ip_name] + displacement + 1e-6) % length
        label = 'bb_ho' if identifier == 0 else 'bb_lr'
        beam_name = 'b2' if mirror else 'b1'
        other_beam_name = 'b1' if mirror else 'b2'
        element_name = _beambeam_element_name(
            label, ip_name, beam_name, identifier)
        other_element_name = _beambeam_element_name(
            label, ip_name, other_beam_name, identifier)
        element = _new_beambeam_element(
            line=line, other_line=other_line, n_slots=n_slots,
            bunch_spacing_zeta=bunch_spacing_zeta)
        if not hasattr(element, 'extra') or element.extra is None:
            element.extra = {}
        element.extra[_BEAMBEAM_EXTRA_KEY] = {
            'version': _BEAMBEAM_EXTRA_VERSION,
            'mode': 'rigid_bunch',
            'ip_name': ip_name,
            'label': label,
            'identifier': identifier,
            'encounter_name': encounter_name,
            'other_element_name': other_element_name,
        }
        placements.append(env.place(element_name, element, at=at))
        element_names.append(element_name)
    line.insert(placements)
    _bind_beambeam_scale(line, element_names)


def _normalize_filling(filling_pattern, num_particles, n_slots, beam_name):
    """Normalize one beam's occupancy and return compact filled-bunch data."""
    pattern = np.asarray(filling_pattern)
    if pattern.ndim != 1 or len(pattern) != n_slots:
        raise ValueError(
            f'`filling_pattern_{beam_name}` must be a one-dimensional array '
            f'of length n_slots={n_slots}.')
    if not np.all(np.isfinite(pattern)):
        raise ValueError(f'`filling_pattern_{beam_name}` must be finite.')
    pattern = (pattern != 0).astype(np.int64)
    filled_slots = np.nonzero(pattern)[0].astype(np.int64)
    if len(filled_slots) == 0:
        raise ValueError(
            f'`filling_pattern_{beam_name}` must contain at least one filled '
            'slot.')

    intensity = np.asarray(num_particles, dtype=float)
    if intensity.ndim == 0:
        intensity = np.full(len(filled_slots), float(intensity))
    elif intensity.ndim == 1 and len(intensity) == n_slots:
        intensity = intensity[filled_slots]
    else:
        raise ValueError(
            f'`num_particles["{beam_name}"]` must be a scalar or a '
            f'one-dimensional slot-indexed array of length n_slots={n_slots}.')
    if not np.all(np.isfinite(intensity)) or np.any(intensity <= 0):
        raise ValueError(
            f'`num_particles["{beam_name}"]` must be finite and '
            'strictly positive at every filled slot.')
    return pattern, filled_slots, intensity


def _num_particles_by_orientation(num_particles):
    if isinstance(num_particles, dict):
        missing = [orientation for orientation in ('cw', 'acw')
                   if orientation not in num_particles]
        if missing:
            raise ValueError(
                '`num_particles` is missing rigid-bunch entries: '
                + ', '.join(missing))
        return {orientation: num_particles[orientation]
                for orientation in ('cw', 'acw')}
    return {'cw': num_particles, 'acw': num_particles}


class BeamBeamRigidBunchStudy:
    """State and operations of one rigid-bunch beam-beam problem.

    Returned by
    :meth:`xfields.XfieldsEnvironmentAPI.configure_beambeam_interactions` in
    rigid-bunch mode. Holds the two lines, the encounter geometry
    (per-encounter pairing offset, beta
    functions and survey separation), the installed beam-beam elements
    (``bb_cw`` / ``bb_acw``, keyed by encounter base name) and the per-beam bunch
    filling. Per-bunch optics, the self-consistent solve, sector-map reduction
    and solution transfer are methods (:meth:`twiss`, :meth:`solve`,
    :meth:`second_order_maps`, :meth:`load_solution`,
    :meth:`apply_filling_pattern`).

    Beam-beam elements use the same names as the particles-mode infrastructure,
    e.g. ``bb_ho.c1b1_00`` and ``bb_ho.c1b2_00``. The element itself is the
    observation point used for the geometry and the orbit feedback.

    The configured elements retain ``Sigma_13`` from the bare optics, but the
    rigid-bunch kick ignores it and uses the diagonal covariance only. A
    :class:`RuntimeWarning` reports a normalized transverse correlation above
    ``1e-2``.
    """

    def __init__(self, clockwise_line, anticlockwise_line, ips,
                 num_long_range_encounters_per_side,
                 harmonic_number, bunch_spacing_buckets,
                 nemitt_x=None, nemitt_y=None, num_particles=None):
        self.cw_line = clockwise_line
        self.acw_line = anticlockwise_line
        self.ips = ips                          # dict {ip: offset} or list
        self.ip_names = list(ips)
        self.ip_offsets = None                  # resolved by _compute_geometry
        self.num_long_range_encounters_per_side = \
            num_long_range_encounters_per_side
        self.harmonic_number = int(harmonic_number)
        self.bunch_spacing_buckets = int(bunch_spacing_buckets)
        self.n_slots = int(harmonic_number) // int(bunch_spacing_buckets)
        self.bunch_spacing_zeta = clockwise_line.get_length() / self.n_slots
        self.b_h_dist = self.bunch_spacing_zeta / 2.0
        self.nemitt_x = nemitt_x
        self.nemitt_y = nemitt_y
        self._num_particles = (
            None if num_particles is None
            else _num_particles_by_orientation(num_particles))
        self.enc_specs = list(_encounter_specs(
            self.ip_names, num_long_range_encounters_per_side))
        self.enc_names = [b for b, _, _ in self.enc_specs]
        self._bb_names = {'cw': {}, 'acw': {}}
        for encounter_name, ip_name, identifier in self.enc_specs:
            label = 'bb_ho' if identifier == 0 else 'bb_lr'
            self._bb_names['cw'][encounter_name] = _beambeam_element_name(
                label, ip_name, 'b1', identifier)
            self._bb_names['acw'][encounter_name] = _beambeam_element_name(
                label, ip_name, 'b2', identifier)
        self.bb_names_cw = [
            self._bb_names['cw'][name] for name in self.enc_names]
        self.bb_names_acw = [
            self._bb_names['acw'][name] for name in self.enc_names]

        self.geom = {}               # base_name -> geometry dict
        self.meta = {}
        self.bb_cw = {}              # base_name -> element (in cw line)
        self.bb_acw = {}             # base_name -> element (in acw line)
        # Occupancy stays slot-indexed; populations are compact arrays aligned
        # with the corresponding physical filled-slot arrays.
        self.filling_pattern_cw = None
        self.filling_pattern_acw = None
        self.filled_slots_cw = None
        self.filled_slots_acw = None
        self.num_particles_cw = None
        self.num_particles_acw = None

    # ------------------------------------------------------------------
    # Naming / bookkeeping
    # ------------------------------------------------------------------
    def bb_name(self, base, mirror):
        """Beam-beam element name of one beam (``mirror=True`` -> acw)."""
        orientation = 'acw' if mirror else 'cw'
        return self._bb_names[orientation][base]

    def bunch_zeta(self, mirror):
        """Bunch centres in ascending physical-slot order."""
        slots = self.filled_slots_acw if mirror else self.filled_slots_cw
        return -np.asarray(slots) * self.bunch_spacing_zeta

    def __repr__(self):
        n_cw = 0 if self.filled_slots_cw is None else len(self.filled_slots_cw)
        n_acw = (0 if self.filled_slots_acw is None
                 else len(self.filled_slots_acw))
        return (f'BeamBeamRigidBunchStudy({len(self.enc_names)} encounters, '
                f'n_slots={self.n_slots}, CW={n_cw} ACW={n_acw} bunches)')

    def apply_filling_pattern(
            self, filling_pattern_cw, filling_pattern_acw):
        """Apply the two occupancy patterns to the configured populations.

        Each filling pattern is a slot-indexed occupancy array of length
        ``n_slots``. The configured ``num_particles`` value for each beam is
        either a scalar, applied uniformly to all filled slots, or a
        slot-indexed array of the same length. Derived physical slot identifiers
        are exposed as ``filled_slots_cw`` and ``filled_slots_acw``.

        The installed elements have one entry per RF slot, so any filling
        change updates their slot-indexed data in place without reallocating
        the Xobjects."""
        if self._num_particles is None:
            raise RuntimeError(
                '`num_particles` was not provided when the rigid-bunch study '
                'was created.')
        normalized_cw = _normalize_filling(
            filling_pattern_cw, self._num_particles['cw'],
            self.n_slots, 'cw')
        normalized_acw = _normalize_filling(
            filling_pattern_acw, self._num_particles['acw'],
            self.n_slots, 'acw')
        (self.filling_pattern_cw, self.filled_slots_cw,
         self.num_particles_cw) = normalized_cw
        (self.filling_pattern_acw, self.filled_slots_acw,
         self.num_particles_acw) = normalized_acw

        # Skipped before both elements and geometry exist. Once configured,
        # filling changes reset the slot-indexed opposing state in place.
        if self.bb_cw and self.geom:
            self._configure_bb()

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def _representative_other_beam(self, line, mirror):
        """One inactive-kick representative per opposing RF slot.

        All representatives remain active so their count allocates full-slot
        storage, while zero weight preserves the bare-lattice cold start until
        a solution update loads the physical bunch populations.
        """
        other_line = self.cw_line if mirror else self.acw_line
        return _representative_other_beam(
            line=line, other_line=other_line, n_slots=self.n_slots,
            bunch_spacing_zeta=self.bunch_spacing_zeta)

    def _resolve_ip_offsets(self, tw_cw):
        """Head-on pairing offset (in slots) of each IP: from ``self.ips`` if a
        mapping, else from the ring geometry (first IP as the reference),
        ``round(2 * (s_ip - s_ref) / bunch_spacing_zeta)``."""
        if isinstance(self.ips, dict):
            return {ip: int(v) % self.n_slots for ip, v in self.ips.items()}
        ref = self.ip_names[0]
        s_ref = tw_cw['s', self.bb_name(f'bb_{ref}_ho', False)]
        return {ip: int(round(2 * (tw_cw['s', self.bb_name(f'bb_{ip}_ho',
                                                           False)] - s_ref)
                              / self.bunch_spacing_zeta)) % self.n_slots
                for ip in self.ip_names}

    def _compute_geometry(self):
        """Fill ``self.geom`` from the shared Xfields geometry description.

        The beam-beam elements are the observation points. They must already
        be placed and inactive, so the shared Twiss and covariance calculation
        sees the bare optics. All three transverse covariance components are
        retained; ``Sigma_13`` is ignored by the kick, with a warning when its
        normalized correlation exceeds ``1e-2``.
        """
        names_by_ip = {'cw': {}, 'acw': {}}
        for base, ip, _ in self.enc_specs:
            names_by_ip['cw'].setdefault(ip, []).append(
                self.bb_name(base, False))
            names_by_ip['acw'].setdefault(ip, []).append(
                self.bb_name(base, True))
        twiss_and_madpoints = compute_twiss_and_madpoints_at_bb(
            line_cw=self.cw_line, line_acw=self.acw_line,
            element_names_by_ip=names_by_ip,
            nemitt_x=self.nemitt_x, nemitt_y=self.nemitt_y,
            survey_separation=True)
        tw_cw = twiss_and_madpoints['twiss']['cw']
        tw_acw = twiss_and_madpoints['twiss']['acw']
        n_slots = self.n_slots
        self.ip_offsets = self._resolve_ip_offsets(tw_cw)

        geom = {}
        covariance_components = []
        for base, ip, signed_identifier in self.enc_specs:
            offset = (self.ip_offsets[ip] + signed_identifier) % n_slots
            encounter = compute_beambeam_geometry(
                twiss_and_madpoints=twiss_and_madpoints,
                element_name_cw=self.bb_name(base, False),
                element_name_acw=self.bb_name(base, True))
            cw = encounter['cw']
            acw = encounter['acw']
            covariance_components.extend(
                (beam['sigma'][11], beam['sigma'][13], beam['sigma'][33])
                for beam in (cw, acw))
            geom[base] = dict(
                ip=ip, offset=offset, signed_n=signed_identifier,
                betx_cw=cw['betx'], bety_cw=cw['bety'],
                betx_acw=acw['betx'], bety_acw=acw['bety'],
                Sigma_11_cw=cw['sigma'][11],
                Sigma_13_cw=cw['sigma'][13],
                Sigma_33_cw=cw['sigma'][33],
                Sigma_11_acw=acw['sigma'][11],
                Sigma_13_acw=acw['sigma'][13],
                Sigma_33_acw=acw['sigma'][33],
                sep_x=encounter['separation_x'],
                sep_y=encounter['separation_y'],
            )
        # Report significant coupling once for the complete set of encounters.
        Sigma_11, Sigma_13, Sigma_33 = map(np.asarray,
                                           zip(*covariance_components))
        _warn_if_large_transverse_coupling(Sigma_11, Sigma_13, Sigma_33)
        self.geom = geom
        self.meta = dict(
            qx_cw=float(tw_cw.qx), qy_cw=float(tw_cw.qy),
            qx_acw=float(tw_acw.qx), qy_acw=float(tw_acw.qy))
        self._configure_bb()

    def _configure_bb(self):
        """Set the pairing offset and static opposing-beam covariance on the
        placed beam-beam elements from the computed geometry, then register the
        own bunch grid and covariance (:meth:`_register_own_covariance`). With
        the design optics, the covariance is the same for all bunches and is
        broadcast over the slot-indexed arrays."""
        for mirror, bb_dict in ((False, self.bb_cw), (True, self.bb_acw)):
            oth = 'cw' if mirror else 'acw'
            for base in self.enc_names:
                e = self.geom[base]
                bb = bb_dict[base]
                # Public bunch centres use zeta = -slot * spacing. For the CW
                # beam, an opposing slot ``own + offset`` is therefore at
                # ``zeta_own - offset * spacing``; ACW uses the inverse map.
                bb.zeta_offset = (e['offset'] if mirror else -e['offset']) \
                    * self.bunch_spacing_zeta
                for component in (11, 13, 33):
                    setattr(
                        bb, f'other_beam_Sigma_{component}',
                        np.full(self.n_slots,
                                e[f'Sigma_{component}_{oth}']))
        self._register_own_covariance()
        for line, mirror, bb_dict in (
                (self.cw_line, False, self.bb_cw),
                (self.acw_line, True, self.bb_acw)):
            particles = self._representative_other_beam(line, mirror)
            for bb in bb_dict.values():
                bb.update_from_other_beam(particles)

    def _register_own_covariance(self):
        """(Re)register each element's OWN bunch grid (``own_beam_zeta``) and
        static design covariance, indexed by THIS beam, for every RF slot.
        Uses the bare-optics covariance cached in ``self.geom``."""
        for mirror, bb_dict in ((False, self.bb_cw), (True, self.bb_acw)):
            own = 'acw' if mirror else 'cw'
            own_zeta = -np.arange(self.n_slots) * self.bunch_spacing_zeta
            for base in self.enc_names:
                e = self.geom[base]
                bb_dict[base].update_from_own_beam(
                    own_zeta,
                    own_beam_Sigma_11=e[f'Sigma_11_{own}'],
                    own_beam_Sigma_13=e[f'Sigma_13_{own}'],
                    own_beam_Sigma_33=e[f'Sigma_33_{own}'])

    # ------------------------------------------------------------------
    # Sector-map reduction
    # ------------------------------------------------------------------
    def second_order_maps(self, keep_extra_cw=None, keep_extra_acw=None,
                          context=None):
        """Return a NEW :class:`BeamBeamRigidBunchStudy` on second-order-map copies of
        the two lines: the arcs between the encounters are replaced by
        second-order maps (the beam-beam elements, kept as split points, stay
        exact), so solving the returned study is much faster and gives the same
        per-bunch orbit and tunes. This study (the full lattice) is left
        untouched; transfer a converged reduced solution back with
        :meth:`load_solution`.

        ``keep_extra_cw`` / ``keep_extra_acw`` are extra element names to
        preserve exactly (e.g. lattice octupoles for amplitude-detuning
        studies). ``context`` selects the CPU context for the reduced trackers
        (default: the clockwise line's context).
        """
        if context is None:
            context = self.cw_line._context
        method = self.cw_line.twiss_default.get('method', '4d')
        split_cw = self.bb_names_cw + list(keep_extra_cw or [])
        split_acw = self.bb_names_acw + list(keep_extra_acw or [])
        red_cw = self.cw_line.get_line_with_second_order_maps(split_at=split_cw)
        red_acw = self.acw_line.get_line_with_second_order_maps(
            split_at=split_acw)
        for rl in (red_cw, red_acw):
            rl.twiss_default['method'] = method
            rl.build_tracker(_context=context)

        new = BeamBeamRigidBunchStudy(
            red_cw, red_acw, self.ips,
            self.num_long_range_encounters_per_side, self.harmonic_number,
            self.bunch_spacing_buckets, self.nemitt_x, self.nemitt_y,
            self._num_particles)
        new.geom = self.geom
        new.meta = self.meta
        new.ip_offsets = self.ip_offsets
        new.filling_pattern_cw = self.filling_pattern_cw
        new.filling_pattern_acw = self.filling_pattern_acw
        new.filled_slots_cw = self.filled_slots_cw
        new.filled_slots_acw = self.filled_slots_acw
        new.num_particles_cw = self.num_particles_cw
        new.num_particles_acw = self.num_particles_acw
        new.bb_cw = {b: red_cw[new.bb_name(b, False)] for b in new.enc_names}
        new.bb_acw = {b: red_acw[new.bb_name(b, True)] for b in new.enc_names}
        # the reduced lines have their own env: re-create the beambeam_scale knob
        _bind_beambeam_scale(red_cw, new.bb_names_cw)
        _bind_beambeam_scale(red_acw, new.bb_names_acw)
        return new

    # ------------------------------------------------------------------
    # Solve / solution feed-in
    # ------------------------------------------------------------------
    def _compute_covariances(self, mbtw, bb_names, gamma0):
        """Diagonal covariance from live per-bunch beta functions."""
        Sigma_11 = mbtw['betx', bb_names] * self.nemitt_x / gamma0
        Sigma_33 = mbtw['bety', bb_names] * self.nemitt_y / gamma0
        Sigma_13 = np.zeros_like(Sigma_11)
        return Sigma_11, Sigma_13, Sigma_33

    def _sigma_vector(self, bb_dict, mirror):
        """Own-beam per-bunch sizes laid out to match :func:`_orbit_vector` (x
        then y, each the ``(n_bunches, n_enc)`` array raveled).
        :meth:`apply_filling_pattern` keeps every element's own arrays in sync
        with all RF slots. The element stores increasing zeta, while Twiss
        follows increasing physical slot number (decreasing zeta), so the
        filled slots are selected and mapped back to public order before
        stacking."""
        zeta = self.bunch_zeta(mirror)

        def active(bb):
            n = int(bb.num_own_bunches)
            stored_zeta = np.asarray(bb.own_beam_zeta)[:n]
            indices = np.searchsorted(stored_zeta, zeta)
            return (np.sqrt(np.asarray(bb.own_beam_Sigma_11)[:n][indices]),
                    np.sqrt(np.asarray(bb.own_beam_Sigma_33)[:n][indices]))
        cols = [active(bb_dict[b]) for b in self.enc_names]
        sx = np.stack([c[0] for c in cols], axis=1)   # (n_bunches, n_enc)
        sy = np.stack([c[1] for c in cols], axis=1)
        return np.concatenate([sx.ravel(), sy.ravel()])

    def _update_opposing(self, bb_dict, mbtw_other, slots_other, slots_own,
                         bb_names_other, num_particles_other,
                         covariances_other=None, covariances_own=None):
        """Write the opposing beam's per-bunch orbit (+ survey separation) into
        the beam-beam elements ``bb_dict`` (optionally also the dynamic-beta
        covariance). Between the two (opposite-parity) beam lines x flips and y does
        not; matching TRAIN/pytrain and the reversed-line x-flip, the survey
        separation enters as ``-sep_x`` in x for BOTH beams.

        The opposing covariance is indexed by the OTHER beam; the own
        covariance is indexed by THIS beam."""
        import xtrack as xt
        xs = -mbtw_other['x', bb_names_other]
        ys = mbtw_other['y', bb_names_other]
        slots_other = np.asarray(slots_other, dtype=np.int64)
        slots_own = np.asarray(slots_own, dtype=np.int64)
        all_slots = np.arange(self.n_slots)
        zeta_all = -all_slots * self.bunch_spacing_zeta
        weight_all = np.zeros(self.n_slots)
        weight_all[slots_other] = num_particles_other
        ref = self.cw_line.particle_ref
        p = xt.Particles(
            p0c=ref.p0c[0], mass0=ref.mass0, q0=ref.q0,
            x=np.zeros(self.n_slots), y=np.zeros(self.n_slots),
            zeta=zeta_all, weight=weight_all)
        other = 'acw' if bb_dict is self.bb_cw else 'cw'
        own = 'cw' if bb_dict is self.bb_cw else 'acw'
        for j, base in enumerate(self.enc_names):
            x_all = np.zeros(self.n_slots)
            y_all = np.zeros(self.n_slots)
            x_all[slots_other] = xs[:, j] - self.geom[base]['sep_x']
            y_all[slots_other] = ys[:, j] - self.geom[base]['sep_y']
            p.x[:] = x_all
            p.y[:] = y_all
            kw = {}
            if covariances_other is not None:
                covariance_other = []
                for component, values in zip(
                        (11, 13, 33), covariances_other):
                    value_all = np.full(
                        self.n_slots,
                        self.geom[base][f'Sigma_{component}_{other}'])
                    value_all[slots_other] = values[:, j]
                    covariance_other.append(value_all)
                kw = dict(
                    other_beam_Sigma_11=covariance_other[0],
                    other_beam_Sigma_13=covariance_other[1],
                    other_beam_Sigma_33=covariance_other[2])
            bb = bb_dict[base]
            bb.update_from_other_beam(p, **kw)
            if covariances_own is not None:
                covariance_own = []
                for component, values in zip((11, 13, 33), covariances_own):
                    value_all = np.full(
                        self.n_slots,
                        self.geom[base][f'Sigma_{component}_{own}'])
                    value_all[slots_own] = values[:, j]
                    covariance_own.append(value_all)
                bb.update_from_own_beam(
                    zeta=zeta_all,
                    own_beam_Sigma_11=covariance_own[0],
                    own_beam_Sigma_13=covariance_own[1],
                    own_beam_Sigma_33=covariance_own[2])

    def load_solution(self, rigid_bunch_twiss, dynamic_beta=False):
        """Load a converged per-bunch solution (e.g. from a reduced-model
        :meth:`solve`) into this study's beam-beam elements, so a subsequent
        :meth:`twiss` / footprint on this study's lattice
        reproduces it. ``rigid_bunch_twiss`` contains the two beams' Twiss
        results; their orbits are read at the beam-beam elements. With
        ``dynamic_beta`` the per-bunch diagonal covariance is taken from the
        live beta functions of the solution; its ``Sigma_13`` is set to zero.
        This does not change the kick, which currently ignores ``Sigma_13``.
        """
        mbtw_clockwise = rigid_bunch_twiss.cw
        mbtw_anticlockwise = rigid_bunch_twiss.acw
        covariances_cw = covariances_acw = None
        if dynamic_beta:
            covariances_cw = self._compute_covariances(
                mbtw_clockwise, self.bb_names_cw, _gamma0(self.cw_line))
            covariances_acw = self._compute_covariances(
                mbtw_anticlockwise, self.bb_names_acw,
                _gamma0(self.acw_line))
        self._update_opposing(
            self.bb_cw, mbtw_anticlockwise,
            self.filled_slots_acw, self.filled_slots_cw,
            self.bb_names_acw, self.num_particles_acw,
            covariances_other=covariances_acw,
            covariances_own=covariances_cw)
        self._update_opposing(
            self.bb_acw, mbtw_clockwise,
            self.filled_slots_cw, self.filled_slots_acw,
            self.bb_names_cw, self.num_particles_cw,
            covariances_other=covariances_cw,
            covariances_own=covariances_acw)

    def twiss(self, method='4d', mode='fast', show_progress=True, **kwargs):
        """Compute per-bunch optics for both beams using the opposing-beam
        state currently loaded in the rigid-bunch elements.

        Unlike :meth:`solve`, this does not iterate the beams to
        self-consistency. Bunch positions and labels come directly from the
        filling patterns stored in this study.

        Returns
        -------
        RigidBunchTwiss
            Two-beam result with ``cw`` (clockwise) and ``acw``
            (anticlockwise) bunch Twiss data.
        """
        if self.filled_slots_cw is None or self.filled_slots_acw is None:
            raise RuntimeError(
                'bunch filling not set; call apply_filling_pattern first')

        from .rigid_bunch_twiss import (
            RigidBunchTwiss, _twiss_rigid_bunch_line)

        common = dict(method=method, mode=mode,
                      show_progress=show_progress, **kwargs)
        twiss_cw = _twiss_rigid_bunch_line(
            self.cw_line,
            zeta_bunches=self.bunch_zeta(mirror=False),
            bunch_names=[f'slot_{slot}' for slot in self.filled_slots_cw],
            **common)
        twiss_acw = _twiss_rigid_bunch_line(
            self.acw_line,
            zeta_bunches=self.bunch_zeta(mirror=True),
            bunch_names=[f'slot_{slot}' for slot in self.filled_slots_acw],
            **common)
        return RigidBunchTwiss(cw=twiss_cw, acw=twiss_acw)

    def solve(self, max_iterations=5, tol_sigma=1e-4, dynamic_beta=False,
              method='4d', chrom=False, twiss_mode=None, show_progress=True,
              continue_on_closed_orbit_error=False,
              require_convergence=True):
        """Find the per-bunch self-consistent closed orbit: iterate the
        rigid-bunch Twiss on both beams, feeding each beam's per-bunch closed
        orbit (plus the survey separation) into the other beam's elements, until
        the closed orbit at every beam-beam element stops changing.

        The elements are left holding the final opposing-beam state. After a
        successful solve, a subsequent :meth:`twiss` (or plain ``line.twiss()``
        for one bunch) reproduces the solution without re-iterating. If the
        solve does not converge, the last iterate remains loaded even when the
        default ``RuntimeError`` is raised.

        Parameters
        ----------
        max_iterations : int
            Maximum number of iterations (default 5).
        tol_sigma : float
            Convergence tolerance, in units of the local beam size: stop once
            the maximum change of the x/y closed orbit at all beam-beam elements
            (over all bunches of both beams) between two successive iterations,
            each normalised by that element's own-beam transverse size, is below
            this (default 1e-4).
        dynamic_beta : bool
            If True, recompute the per-bunch effective (convolved) sizes from the
            live per-bunch beta functions at each iteration. Forces the
            optics-carrying twiss. Default False.
        method : str
            Twiss method, ``'4d'`` (default) or ``'6d'``.
        chrom : bool
            Whether to compute chromatic properties in the multi-bunch twiss.
        twiss_mode : str, optional
            ``'fast_orbit'`` (orbit only, the default when ``dynamic_beta`` is
            False), ``'fast'`` (adds per-bunch optics, forced when
            ``dynamic_beta`` is True) or ``'full'``.
        show_progress : bool
            Print per-iteration convergence information (default True).
        continue_on_closed_orbit_error : bool
            If True, the closed-orbit search of the INTERMEDIATE iterations may
            return its last iterate instead of raising
            :class:`ClosedOrbitSearchError` (same meaning as in
            :meth:`Line.twiss`); a FINAL iteration is then always run without
            it, and it must converge. That last orbit is the one returned and
            the one left on the elements, so the result is exactly as strict as
            without this option -- only the intermediate rounds are relaxed.
            Use it on lattices where the search cannot reach ``co_tol`` from a
            cold start but does once the opposing beam has settled. Default
            False (every iteration strict).
        require_convergence : bool
            If True (default), raise :class:`RuntimeError` when the maximum
            number of iterations is reached without satisfying ``tol_sigma``.
            If False, return the last iterate with ``converged=False`` and its
            diagnostics attached. This is intended for deliberate
            fixed-iteration studies; callers must inspect the convergence
            metadata before treating the result as a solution.

        Returns
        -------
        RigidBunchTwiss
            Two-beam result with ``cw`` (clockwise) and ``acw``
            (anticlockwise) bunch Twiss data, plus convergence metadata. With
            ``require_convergence=False``, this can be a non-converged
            last iterate rather than a solution.

        Raises
        ------
        RuntimeError
            If the solve does not converge and
            ``require_convergence`` is True.
        """
        if self.filled_slots_cw is None or self.filled_slots_acw is None:
            raise RuntimeError(
                'bunch filling not set; call apply_filling_pattern first')
        if max_iterations < 1:
            raise ValueError('`max_iterations` must be at least one.')
        if twiss_mode is None:
            twiss_mode = 'fast' if dynamic_beta else 'fast_orbit'
        if dynamic_beta and twiss_mode == 'fast_orbit':
            twiss_mode = 'fast'

        # The intermediate rounds may keep going on a closed-orbit error: their
        # orbit is only an input to the next round, so a few bunches short of
        # `co_tol` cost nothing, and on some machines the search cannot reach
        # it from a cold start at all. The FINAL round below is always strict.
        co_kwargs = (dict(continue_on_closed_orbit_error=True)
                     if continue_on_closed_orbit_error else {})

        result = None
        prev = None
        err = np.inf
        for it in range(max_iterations):
            result = self.twiss(
                method=method, chrom=chrom, mode=twiss_mode,
                show_progress=show_progress, **co_kwargs)

            cur = np.concatenate([
                _orbit_vector(result.cw, self.bb_names_cw),
                _orbit_vector(result.acw, self.bb_names_acw)])
            sig = np.concatenate([self._sigma_vector(self.bb_cw, mirror=False),
                                  self._sigma_vector(self.bb_acw, mirror=True)])
            err = (np.inf if prev is None
                   else float(np.max(np.abs(cur - prev) / sig)))
            prev = cur

            self.load_solution(result, dynamic_beta=dynamic_beta)

            if show_progress:
                _print(f'  rigid-bunch orbit iteration {it}: '
                       f'max orbit change = {err:.2e} sigma')
            if err < tol_sigma:
                if show_progress:
                    _print(f'  converged after {it + 1} iterations '
                           f'(< {tol_sigma:.1e} sigma)')
                break
        else:
            if show_progress:
                _print(f'  reached max_iterations={max_iterations} '
                       f'(last change {err:.2e} sigma)')

        if continue_on_closed_orbit_error:
            # One last pass that must converge, so what is returned -- and what
            # the elements are left holding -- is a genuine closed orbit of the
            # converged state. If this raises, the solution is NOT usable and
            # the caller must know.
            if show_progress:
                _print('  final closed-orbit pass (strict)')
            result = self.twiss(
                method=method, chrom=chrom, mode=twiss_mode,
                show_progress=show_progress)
            self.load_solution(result, dynamic_beta=dynamic_beta)

        result.converged = err < tol_sigma
        result.num_iterations = it + 1
        result.max_orbit_change = err
        if not result.converged and require_convergence:
            raise RuntimeError(
                'Rigid-bunch beam-beam solve did not converge after '
                f'{result.num_iterations} iterations: maximum orbit change '
                f'is {result.max_orbit_change:.3e} sigma, requested tolerance '
                f'is {tol_sigma:.3e}. The last iterate remains loaded in the '
                'beam-beam elements. Pass '
                '`require_convergence=False` to return it explicitly.')
        return result


def _orbit_vector(mbtw, bb_names):
    """Flat (x then y) per-bunch orbit at all elements, for convergence."""
    x = mbtw['x', bb_names]
    y = mbtw['y', bb_names]
    return np.concatenate([np.asarray(x).ravel(), np.asarray(y).ravel()])


def _metadata(element):
    extra = getattr(element, 'extra', None)
    if not isinstance(extra, dict):
        return None
    metadata = extra.get(_BEAMBEAM_EXTRA_KEY)
    if metadata is None:
        return None
    if metadata.get('mode') != 'rigid_bunch':
        return None
    if metadata.get('version') != _BEAMBEAM_EXTRA_VERSION:
        raise RuntimeError(
            'Unsupported rigid-bunch beam-beam element metadata version.')
    return metadata


def _discover_installation(env):
    config = env.extra_config.get(BEAMBEAM_CONFIG_KEY)
    if config is None:
        raise RuntimeError(
            'No rigid-bunch beam-beam configuration was found. Call '
            '`install_beambeam_interactions(...)` first.')
    if config.get('version') != BEAMBEAM_CONFIG_VERSION:
        raise RuntimeError(
            'Unsupported rigid-bunch beam-beam configuration version.')
    if config.get('mode') != 'rigid_bunch':
        raise RuntimeError(
            'The environment beam-beam configuration is not rigid-bunch.')

    line_names = {
        'cw': config['clockwise_line'],
        'acw': config['anticlockwise_line'],
    }
    ip_names = list(config['ip_names'])
    elements = {'cw': {}, 'acw': {}}
    for orientation, line_name in line_names.items():
        if line_name not in env.lines:
            raise RuntimeError(
                f'Configured beam-beam line `{line_name}` is not present in '
                'the environment.')
        line = env.lines[line_name]
        for element_name in dict.fromkeys(line.element_names):
            metadata = _metadata(line[element_name])
            if metadata is None:
                continue
            if metadata['ip_name'] not in ip_names:
                raise RuntimeError(
                    f'Tagged beam-beam element `{element_name}` refers to '
                    f'unknown IP `{metadata["ip_name"]}`.')
            elements[orientation][element_name] = metadata
        if not elements[orientation]:
            raise RuntimeError(
                f'No tagged rigid-bunch beam-beam elements were found in '
                f'configured line `{line_name}`.')
        installed_ips = {
            metadata['ip_name']
            for metadata in elements[orientation].values()}
        if installed_ips != set(ip_names):
            raise RuntimeError(
                f'Rigid-bunch elements in line `{line_name}` do not cover '
                'the configured interaction points.')
        elements[orientation] = dict(sorted(elements[orientation].items()))

    for orientation, other_orientation in (('cw', 'acw'), ('acw', 'cw')):
        for element_name, metadata in elements[orientation].items():
            other_name = metadata['other_element_name']
            if other_name not in elements[other_orientation]:
                raise RuntimeError(
                    f'Rigid-bunch element `{element_name}` refers to missing '
                    f'opposing element `{other_name}`.')

    return _RigidBunchInstallation(
        elements=elements, line_names=line_names,
        ip_names=ip_names, config=config)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def install_rigid_bunch_beambeam(
        env, clockwise_line, anticlockwise_line, ip_names,
        num_long_range_encounters_per_side, harmonic_number,
        bunch_spacing_buckets, delay_at_ips_slots=None):
    """Install full-slot rigid-bunch elements and store their configuration."""
    if harmonic_number is None or bunch_spacing_buckets is None:
        raise ValueError(
            '`harmonic_number` and `bunch_spacing_buckets` are required '
            "when mode='rigid_bunch'.")
    if clockwise_line is None or anticlockwise_line is None:
        raise ValueError("mode='rigid_bunch' requires both beam lines.")

    def line_name(line, argument_name):
        if isinstance(line, str):
            if line not in env.lines:
                raise KeyError(f'Line `{line}` not found in environment.')
            return line
        matches = [name for name, candidate in env.lines.items()
                   if candidate is line]
        if len(matches) != 1:
            raise ValueError(
                f'`{argument_name}` must be a line in this environment or '
                'its name.')
        return matches[0]

    ip_names = list(ip_names)
    if delay_at_ips_slots is None:
        ips = ip_names
    elif isinstance(delay_at_ips_slots, dict):
        ips = {ip: int(delay_at_ips_slots[ip]) for ip in ip_names}
    else:
        delays = list(delay_at_ips_slots)
        if len(delays) != len(ip_names):
            raise ValueError('`delay_at_ips_slots` must have one entry per IP.')
        ips = {ip: int(delay) for ip, delay in zip(ip_names, delays)}

    if isinstance(num_long_range_encounters_per_side, dict):
        num_lr_values = [num_long_range_encounters_per_side[ip]
                         for ip in ip_names]
        num_lr_is_scalar = False
    elif np.ndim(num_long_range_encounters_per_side) == 0:
        num_lr_values = [num_long_range_encounters_per_side]
        num_lr_is_scalar = True
    else:
        num_lr_values = list(num_long_range_encounters_per_side)
        num_lr_is_scalar = False
        if len(num_lr_values) != len(ip_names):
            raise ValueError(
                '`num_long_range_encounters_per_side` must have one entry '
                'per IP.')
    normalized_num_lr = []
    for value in num_lr_values:
        normalized = int(value)
        if normalized != value or normalized < 0:
            raise ValueError(
                '`num_long_range_encounters_per_side` entries must be '
                'non-negative integers.')
        normalized_num_lr.append(normalized)
    if num_lr_is_scalar:
        num_lr = normalized_num_lr[0]
    else:
        num_lr = dict(zip(ip_names, normalized_num_lr))

    cw_name = line_name(clockwise_line, 'clockwise_line')
    acw_name = line_name(anticlockwise_line, 'anticlockwise_line')
    n_slots_float = harmonic_number / bunch_spacing_buckets
    n_slots = int(n_slots_float)
    if n_slots != n_slots_float:
        raise ValueError(
            '`harmonic_number` must be divisible by '
            '`bunch_spacing_buckets`.')
    circumference_cw = env.lines[cw_name].get_length()
    circumference_acw = env.lines[acw_name].get_length()
    if not np.isclose(
            circumference_cw, circumference_acw, atol=1e-4, rtol=0):
        raise ValueError(
            'The clockwise and anticlockwise lines must have the same '
            'circumference.')
    bunch_spacing_zeta = circumference_cw / n_slots
    encounter_specs = list(_encounter_specs(ip_names, num_lr))

    for line in env.lines.values():
        line.discard_tracker()
    _install_elements(
        env=env, line_name=cw_name, other_line_name=acw_name,
        encounter_specs=encounter_specs,
        n_slots=n_slots, bunch_spacing_zeta=bunch_spacing_zeta,
        mirror=False)
    _install_elements(
        env=env, line_name=acw_name, other_line_name=cw_name,
        encounter_specs=encounter_specs,
        n_slots=n_slots, bunch_spacing_zeta=bunch_spacing_zeta,
        mirror=True)

    env.extra_config[BEAMBEAM_CONFIG_KEY] = {
        'version': BEAMBEAM_CONFIG_VERSION,
        'mode': 'rigid_bunch',
        'clockwise_line': cw_name,
        'anticlockwise_line': acw_name,
        'ip_names': ip_names,
        'ips': ips,
        'num_long_range_encounters_per_side': num_lr,
        'harmonic_number': int(harmonic_number),
        'bunch_spacing_buckets': int(bunch_spacing_buckets),
    }


def configure_rigid_bunch_beambeam(
        env, num_particles, nemitt_x, nemitt_y,
        filling_pattern_cw=None, filling_pattern_acw=None):
    """Populate installed rigid-bunch elements and return their study.

    The elements retain all transverse covariance components computed from the
    bare optics. ``Sigma_13`` is stored and serialized but currently ignored by
    the rigid-bunch kick. A :class:`RuntimeWarning` is emitted when its
    normalized correlation exceeds ``1e-2``.
    """
    if (filling_pattern_cw is None) != (filling_pattern_acw is None):
        raise ValueError(
            '`filling_pattern_cw` and `filling_pattern_acw` must be provided '
            'together.')
    installation = _discover_installation(env)
    config = installation.config
    cw = env[installation.line_names['cw']]
    acw = env[installation.line_names['acw']]
    study = BeamBeamRigidBunchStudy(
        cw, acw, config['ips'],
        config['num_long_range_encounters_per_side'],
        config['harmonic_number'], config['bunch_spacing_buckets'],
        nemitt_x=nemitt_x, nemitt_y=nemitt_y,
        num_particles=num_particles)
    elements_by_encounter = {
        orientation: {
            metadata['encounter_name']: line[element_name]
            for element_name, metadata
            in installation.elements[orientation].items()}
        for orientation, line in (('cw', cw), ('acw', acw))
    }
    study.bb_cw = elements_by_encounter['cw']
    study.bb_acw = elements_by_encounter['acw']

    # Reanalyse the bare lines even when configuration is repeated. Preserve
    # the user's knob value or expression, including when geometry fails.
    previous_beambeam_scale = env.ref['beambeam_scale'].xdeps.expr
    if previous_beambeam_scale is None:
        previous_beambeam_scale = env['beambeam_scale']
    try:
        env['beambeam_scale'] = 0.0
        study._compute_geometry()
    finally:
        env['beambeam_scale'] = previous_beambeam_scale
    if filling_pattern_cw is not None:
        study.apply_filling_pattern(
            filling_pattern_cw=filling_pattern_cw,
            filling_pattern_acw=filling_pattern_acw)
    return study
