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
            delay_at_ips_slots=None, mode=None):
        """Install particles or rigid-bunch beam-beam interactions.

        ``mode=None`` (or ``'particles'``) selects the sliced head-on and
        scalar long-range workflow, including optional pipeline operation.
        ``mode='rigid_bunch'`` installs coherent train elements with one array
        entry per RF slot. Call :meth:`configure_beambeam_interactions`
        afterwards to load populations and geometry, then apply the filling
        pattern on the returned study.

        Parameters
        ----------
        clockwise_line, anticlockwise_line : str or xtrack.Line
            The two counter-rotating beam lines, or their environment names.
        ip_names : sequence of str
            Interaction-point element names.
        num_long_range_encounters_per_side : int or sequence of int
            Number of long-range encounters on each side of every IP.
        num_slices_head_on : int, optional
            Longitudinal slices per head-on encounter in particles mode.
        harmonic_number : int
            RF harmonic number.
        bunch_spacing_buckets : int
            Bunch spacing in RF buckets.
        sigmaz : float, optional
            RMS bunch length required in particles mode.
        delay_at_ips_slots : sequence or mapping, optional
            Head-on bunch-pairing offsets in physical RF slots.
        mode : {None, 'particles', 'rigid_bunch'}, optional
            Beam representation used by the beam-beam model. The default is
            ``'particles'``.
        """
        if mode not in (None, 'particles', 'rigid_bunch'):
            raise ValueError(
                "`mode` must be None, 'particles' or 'rigid_bunch'.")

        if mode != 'rigid_bunch':
            missing = [name for name, value in (
                ('num_slices_head_on', num_slices_head_on),
                ('harmonic_number', harmonic_number),
                ('bunch_spacing_buckets', bunch_spacing_buckets),
                ('sigmaz', sigmaz),
            ) if value is None]
            if missing:
                raise ValueError(
                    'Missing `particles`-mode installation arguments: '
                    + ', '.join(missing))

            from .config_tools.beambeam_config_tools.particles_mode import (
                install_beambeam_interactions,
            )

            return install_beambeam_interactions(
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

        from .config_tools.beambeam_config_tools.rigid_bunch_mode import (
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
            delay_at_ips_slots=delay_at_ips_slots)

    def configure_beambeam_interactions(
            self, num_particles=None, nemitt_x=None, nemitt_y=None,
            crab_strong_beam=True, use_antisymmetry=False,
            separation_bumps=None):
        """Configure particles or rigid-bunch beam-beam interactions.

        Particles mode uses ``num_particles`` and the two emittances.
        Rigid-bunch mode accepts either a common scalar population or a
        ``{'cw': ..., 'acw': ...}`` mapping whose values are scalar or
        slot-indexed. It returns the :class:`BeamBeamRigidBunchStudy` that owns
        filling-pattern application and subsequent optics and solve operations.

        Parameters
        ----------
        num_particles : float or mapping
            Uniform bunch population in particles mode. In rigid-bunch
            mode, either a common population or a ``'cw'`` / ``'acw'`` mapping
            of scalar or slot-indexed populations.
        nemitt_x, nemitt_y : float
            Normalized transverse emittances.
        crab_strong_beam, use_antisymmetry, separation_bumps
            Particles-mode beam-beam configuration options.

        Returns
        -------
        BeamBeamRigidBunchStudy or None
            The rigid-bunch study, or the particles-mode helper's result.
        """
        config = self.env.extra_config.get('xfields_beambeam', {})
        mode = config.get('mode', 'particles')

        if mode != 'rigid_bunch':
            if num_particles is None or nemitt_x is None or nemitt_y is None:
                raise ValueError(
                    '`num_particles`, `nemitt_x` and `nemitt_y` are required '
                    'for `particles`-mode beam-beam configuration.')

            from .config_tools.beambeam_config_tools.particles_mode import (
                configure_beambeam_interactions,
            )

            return configure_beambeam_interactions(
                self.env,
                num_particles=num_particles,
                nemitt_x=nemitt_x,
                nemitt_y=nemitt_y,
                crab_strong_beam=crab_strong_beam,
                use_antisymmetry=use_antisymmetry,
                separation_bumps=separation_bumps)

        if use_antisymmetry or separation_bumps is not None:
            raise ValueError(
                '`use_antisymmetry` and `separation_bumps` do not apply when '
                "mode='rigid_bunch'.")

        required = {
            'num_particles': num_particles,
            'nemitt_x': nemitt_x,
            'nemitt_y': nemitt_y,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                'Missing rigid-bunch configuration arguments: '
                + ', '.join(missing))

        from .config_tools.beambeam_config_tools.rigid_bunch_mode import (
            configure_rigid_bunch_beambeam,
        )

        return configure_rigid_bunch_beambeam(
            self.env,
            num_particles=num_particles,
            nemitt_x=nemitt_x,
            nemitt_y=nemitt_y)

    def apply_filling_pattern(self, filling_pattern_cw, filling_pattern_acw,
                              i_bunch_cw, i_bunch_acw):
        """Select one bunch from particles-mode beam-beam filling patterns.

        This helper belongs to the particle-based, potentially
        pipeline-enabled workflow. Rigid-bunch fillings are changed with
        :meth:`BeamBeamRigidBunchStudy.apply_filling_pattern`.
        """
        from .config_tools.beambeam_config_tools.particles_mode import (
            apply_filling_pattern,
        )

        return apply_filling_pattern(
            self.env,
            filling_pattern_cw=filling_pattern_cw,
            filling_pattern_acw=filling_pattern_acw,
            i_bunch_cw=i_bunch_cw,
            i_bunch_acw=i_bunch_acw)
