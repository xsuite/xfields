# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Xfields-specific helpers associated with an Xtrack environment."""


class XfieldsEnvironmentAPI:
    """Accessor exposed as ``env.xfields``."""

    def __init__(self, env):
        self.env = env

    def install_beambeam_interactions(
            self, clockwise_line, anticlockwise_line, ip_names,
            num_long_range_encounters_per_side, num_slices_head_on=None,
            harmonic_number=None, bunch_spacing_buckets=None, sigmaz=None,
            delay_at_ips_slots=None, mode=None, survey_separation=True,
            bb_suffix_cw='_cw', bb_suffix_acw='_acw'):
        """Install conventional or rigid-bunch beam-beam interactions.

        ``mode=None`` (or ``'conventional'``) preserves the sliced head-on and
        scalar long-range workflow, including optional pipeline operation.
        ``mode='rigid_bunch'`` installs coherent train elements with one array
        entry per RF slot. Call :meth:`configure_beambeam_interactions`
        afterwards to load filling-dependent populations and geometry.

        Parameters
        ----------
        clockwise_line, anticlockwise_line : str or xtrack.Line
            The two counter-rotating beam lines, or their environment names.
        ip_names : sequence of str
            Interaction-point element names.
        num_long_range_encounters_per_side : int or sequence of int
            Number of long-range encounters on each side of every IP.
        num_slices_head_on : int, optional
            Longitudinal slices per head-on encounter in conventional mode.
        harmonic_number : int
            RF harmonic number.
        bunch_spacing_buckets : int
            Bunch spacing in RF buckets.
        sigmaz : float, optional
            RMS bunch length required in conventional mode.
        delay_at_ips_slots : sequence or mapping, optional
            Head-on bunch-pairing offsets in physical RF slots.
        mode : {None, 'conventional', 'rigid_bunch'}, optional
            Beam-beam model; the default preserves conventional behavior.
        survey_separation : bool, optional
            Include geometric survey separation in rigid-bunch mode.
        bb_suffix_cw, bb_suffix_acw : str, optional
            Beam-specific element-name suffixes in rigid-bunch mode.
        """
        if mode not in (None, 'conventional', 'rigid_bunch'):
            raise ValueError(
                "`mode` must be None, 'conventional' or 'rigid_bunch'.")

        if mode != 'rigid_bunch':
            missing = [name for name, value in (
                ('num_slices_head_on', num_slices_head_on),
                ('harmonic_number', harmonic_number),
                ('bunch_spacing_buckets', bunch_spacing_buckets),
                ('sigmaz', sigmaz),
            ) if value is None]
            if missing:
                raise ValueError(
                    'Missing conventional beam-beam installation arguments: '
                    + ', '.join(missing))

            from xtrack.environment import MultilineLegacy

            return MultilineLegacy.install_beambeam_interactions(
                self.env,
                clockwise_line=clockwise_line,
                anticlockwise_line=anticlockwise_line,
                ip_names=ip_names,
                num_long_range_encounters_per_side=
                    num_long_range_encounters_per_side,
                num_slices_head_on=num_slices_head_on,
                harmonic_number=harmonic_number,
                bunch_spacing_buckets=bunch_spacing_buckets,
                sigmaz=sigmaz,
                delay_at_ips_slots=delay_at_ips_slots)

        if num_slices_head_on is not None or sigmaz is not None:
            raise ValueError(
                '`num_slices_head_on` and `sigmaz` do not apply when '
                "mode='rigid_bunch'.")

        from .config_tools.beambeam_config_tools.rigid_bunch import (
            install_rigid_bunch_beambeam,
        )

        return install_rigid_bunch_beambeam(
            self.env,
            clockwise_line=clockwise_line,
            anticlockwise_line=anticlockwise_line,
            ip_names=ip_names,
            num_long_range_encounters_per_side=
                num_long_range_encounters_per_side,
            harmonic_number=harmonic_number,
            bunch_spacing_buckets=bunch_spacing_buckets,
            delay_at_ips_slots=delay_at_ips_slots,
            survey_separation=survey_separation,
            bb_suffix_cw=bb_suffix_cw,
            bb_suffix_acw=bb_suffix_acw)

    def configure_beambeam_interactions(
            self, num_particles=None, nemitt_x=None, nemitt_y=None,
            crab_strong_beam=True, use_antisymmetry=False,
            separation_bumps=None, filling_scheme_cw=None,
            filling_scheme_acw=None, bunch_intensity_particles_cw=None,
            bunch_intensity_particles_acw=None):
        """Configure conventional or rigid-bunch beam-beam interactions.

        Conventional mode uses ``num_particles`` and the two emittances.
        Rigid-bunch mode instead requires two slot-indexed filling schemes and
        their scalar or slot-indexed bunch populations; it returns the
        :class:`BeamBeamRigidBunchStudy` that owns subsequent optics and solve
        operations.

        Parameters
        ----------
        num_particles : float, optional
            Uniform bunch population in conventional mode.
        nemitt_x, nemitt_y : float
            Normalized transverse emittances.
        crab_strong_beam, use_antisymmetry, separation_bumps
            Conventional beam-beam configuration options.
        filling_scheme_cw, filling_scheme_acw : array_like, optional
            Slot-indexed occupancy patterns required in rigid-bunch mode.
        bunch_intensity_particles_cw, bunch_intensity_particles_acw
            Scalar uniform populations or slot-indexed population arrays.

        Returns
        -------
        BeamBeamRigidBunchStudy or None
            The rigid-bunch study, or the conventional helper's result.
        """
        config = getattr(self.env, '_bb_config', None) or {}
        mode = config.get('mode', 'conventional')

        if mode != 'rigid_bunch':
            if num_particles is None or nemitt_x is None or nemitt_y is None:
                raise ValueError(
                    '`num_particles`, `nemitt_x` and `nemitt_y` are required '
                    'for conventional beam-beam configuration.')

            from xtrack.environment import MultilineLegacy

            return MultilineLegacy.configure_beambeam_interactions(
                self.env,
                num_particles=num_particles,
                nemitt_x=nemitt_x,
                nemitt_y=nemitt_y,
                crab_strong_beam=crab_strong_beam,
                use_antisymmetry=use_antisymmetry,
                separation_bumps=separation_bumps)

        if num_particles is not None:
            raise ValueError(
                '`num_particles` is replaced by the two per-beam intensity '
                "inputs when mode='rigid_bunch'.")
        if use_antisymmetry or separation_bumps is not None:
            raise ValueError(
                '`use_antisymmetry` and `separation_bumps` do not apply when '
                "mode='rigid_bunch'.")

        required = {
            'nemitt_x': nemitt_x,
            'nemitt_y': nemitt_y,
            'filling_scheme_cw': filling_scheme_cw,
            'filling_scheme_acw': filling_scheme_acw,
            'bunch_intensity_particles_cw':
                bunch_intensity_particles_cw,
            'bunch_intensity_particles_acw':
                bunch_intensity_particles_acw,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                'Missing rigid-bunch configuration arguments: '
                + ', '.join(missing))

        from .config_tools.beambeam_config_tools.rigid_bunch import (
            configure_rigid_bunch_beambeam,
        )

        return configure_rigid_bunch_beambeam(
            self.env,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y,
            filling_scheme_cw=filling_scheme_cw,
            filling_scheme_acw=filling_scheme_acw,
            bunch_intensity_particles_cw=bunch_intensity_particles_cw,
            bunch_intensity_particles_acw=bunch_intensity_particles_acw)

    def apply_filling_pattern(self, filling_pattern_cw, filling_pattern_acw,
                              i_bunch_cw, i_bunch_acw):
        """Select one bunch from conventional beam-beam filling patterns.

        This established helper belongs to the conventional, potentially
        pipeline-enabled workflow. Rigid-bunch fillings are changed with
        :meth:`BeamBeamRigidBunchStudy.set_filling`.
        """
        from xtrack.environment import MultilineLegacy

        return MultilineLegacy.apply_filling_pattern(
            self.env,
            filling_pattern_cw=filling_pattern_cw,
            filling_pattern_acw=filling_pattern_acw,
            i_bunch_cw=i_bunch_cw,
            i_bunch_acw=i_bunch_acw)
