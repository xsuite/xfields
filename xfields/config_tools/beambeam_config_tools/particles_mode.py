# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Environment-level setup of particle-based beam--beam interactions.

Installation-wide state is stored in ``Environment.extra_config``. Each
installed element carries only its encounter-local description in the generic,
serializable ``BeamElement.extra`` container.
"""

from dataclasses import dataclass

import numpy as np
import xobjects as xo
from xtrack._filling_pattern import _FillingPattern

import xfields as xf

from .config_tools import (
    BEAMBEAM_CONFIG_KEY,
    BEAMBEAM_CONFIG_VERSION,
    BEAMBEAM_ELEMENT_EXTRA_KEY,
    BEAMBEAM_ELEMENT_EXTRA_VERSION,
    _beambeam_element_name,
    compute_beambeam_geometry,
    compute_twiss_and_madpoints_at_bb,
    find_alpha_and_phi,
)
_BEAMBEAM_EXTRA_KEY = BEAMBEAM_ELEMENT_EXTRA_KEY
_BEAMBEAM_EXTRA_VERSION = BEAMBEAM_ELEMENT_EXTRA_VERSION
_BEAMBEAM_CONFIG_KEY = BEAMBEAM_CONFIG_KEY
_BEAMBEAM_CONFIG_VERSION = BEAMBEAM_CONFIG_VERSION


def install_beambeam_interactions(
        env, clockwise_line, anticlockwise_line, ip_names,
        num_long_range_encounters_per_side, num_slices_head_on,
        harmonic_number, bunch_spacing_buckets, sigmaz,
        delay_at_ips_slots=None):
    """Install inactive, tagged particle-based beam-beam elements."""
    cw_name = _line_name(env, clockwise_line, 'clockwise_line')
    acw_name = _line_name(env, anticlockwise_line, 'anticlockwise_line')
    ip_names = list(ip_names)
    num_lr = _normalize_num_long_range(
        ip_names, num_long_range_encounters_per_side)
    delays_by_ip = _normalize_delays(ip_names, delay_at_ips_slots)

    n_slots_float = harmonic_number / bunch_spacing_buckets
    n_slots = int(n_slots_float)
    if n_slots != n_slots_float:
        raise ValueError(
            '`harmonic_number` must be divisible by '
            '`bunch_spacing_buckets`.')

    for line in env.lines.values():
        line.discard_tracker()
    if cw_name is not None and acw_name is not None:
        circumference_cw = env.lines[cw_name].get_length()
        circumference_acw = env.lines[acw_name].get_length()
        if not np.isclose(circumference_cw, circumference_acw,
                          atol=1e-4, rtol=0):
            raise ValueError(
                'The clockwise and anticlockwise lines must have the same '
                'circumference.')

    for orientation, line_name, beam_name, other_beam_name in (
            ('clockwise', cw_name, 'b1', 'b2'),
            ('anticlockwise', acw_name, 'b2', 'b1')):
        if line_name is None:
            continue
        line = env.lines[line_name]
        specs = _element_specs(
            circumference=line.get_length(), ip_names=ip_names,
            num_long_range_encounters_per_side=num_lr,
            num_slices_head_on=num_slices_head_on,
            harmonic_number=harmonic_number,
            bunch_spacing_buckets=bunch_spacing_buckets, sigmaz=sigmaz,
            orientation=orientation, beam_name=beam_name,
            other_beam_name=other_beam_name)
        _install_elements(env, line_name, specs)

    env.extra_config[_BEAMBEAM_CONFIG_KEY] = {
        'version': _BEAMBEAM_CONFIG_VERSION,
        'mode': 'particles',
        'clockwise_line': cw_name,
        'anticlockwise_line': acw_name,
        'ip_names': ip_names,
        'n_slots': n_slots,
        'delay_at_ips_slots': delays_by_ip,
    }


def configure_beambeam_interactions(
        env, num_particles, nemitt_x, nemitt_y, crab_strong_beam=True,
        use_antisymmetry=False, separation_bumps=None,
        filling_pattern_cw=None, filling_pattern_acw=None,
        i_bunch_cw=None, i_bunch_acw=None,
        filled_slots_cw=None, filled_slots_acw=None):
    """Reanalyse the live lines and configure their tagged BB elements."""
    has_filling_cw = (
        filling_pattern_cw is not None or filled_slots_cw is not None)
    has_filling_acw = (
        filling_pattern_acw is not None or filled_slots_acw is not None)
    num_filling_arguments = sum((
        has_filling_cw, has_filling_acw,
        i_bunch_cw is not None, i_bunch_acw is not None))
    if num_filling_arguments not in (0, 4):
        raise ValueError(
            'Particles-mode filling requires both filling patterns and both '
            'selected bunch indices.')

    installation = _discover_installation(env)
    if has_filling_cw:
        filling_pattern_cw = _FillingPattern.from_inputs(
            filling_pattern=filling_pattern_cw,
            filled_slots=filled_slots_cw,
            num_slots=installation.n_slots,
            allow_none=False).filling_pattern
        filling_pattern_acw = _FillingPattern.from_inputs(
            filling_pattern=filling_pattern_acw,
            filled_slots=filled_slots_acw,
            num_slots=installation.n_slots,
            allow_none=False).filling_pattern

    for orientation in ('clockwise', 'anticlockwise'):
        line_name = installation.line_names[orientation]
        if line_name is None:
            continue
        line = env.lines[line_name]
        if not line._has_valid_tracker():
            line.build_tracker()
        if not isinstance(line.tracker._context, xo.ContextCpu):
            raise ValueError(
                'The trackers need to be built on CPU before configuring the '
                'beam-beam elements.')

        # Disable any previous configuration while analysing the bare lines.
        for element_name in installation.elements[orientation]:
            env[element_name].scale_strength = 0.0

    line_cw = env.lines.get(installation.line_names['clockwise'])
    line_acw = env.lines.get(installation.line_names['anticlockwise'])
    if use_antisymmetry:
        if line_cw is not None and line_acw is not None:
            raise ValueError(
                'Antisymmetry configuration requires exactly one beam line.')
    else:
        if line_cw is None or line_acw is None:
            raise ValueError(
                'Both beam lines are required when antisymmetry is disabled.')

    lines = {'cw': line_cw, 'acw': line_acw}
    elements = {
        'cw': installation.elements['clockwise'],
        'acw': installation.elements['anticlockwise'],
    }
    names_by_ip = {'cw': {}, 'acw': {}}
    crab_s_by_element = {} if crab_strong_beam else None
    antisymmetry_elements = None
    for orientation in ('cw', 'acw'):
        if lines[orientation] is None:
            continue
        names_by_ip[orientation] = _element_names_by_ip(
            elements[orientation])
        if crab_strong_beam:
            crab_s_by_element[orientation] = {
                element_name: metadata['s_crab']
                for element_name, metadata
                in elements[orientation].items()}
        if use_antisymmetry:
            antisymmetry_elements = elements[orientation]
    twiss_and_madpoints = compute_twiss_and_madpoints_at_bb(
        line_cw=line_cw, line_acw=line_acw,
        element_names_by_ip=names_by_ip,
        nemitt_x=nemitt_x, nemitt_y=nemitt_y,
        survey_separation=True,
        acw_is_reversed=True,
        crab_s_by_element=crab_s_by_element,
        antisymmetry_elements=antisymmetry_elements,
        separation_bumps=separation_bumps)
    geometry_by_pair = {}

    for weak_orientation in ('cw', 'acw'):
        if lines[weak_orientation] is None:
            continue
        for element_name, metadata in elements[weak_orientation].items():
            strong_line, strong_metadata = _resolve_strong_beam(
                weak_orientation=weak_orientation,
                element_name=element_name,
                metadata=metadata,
                lines=lines,
                elements=elements,
                twiss_and_madpoints=twiss_and_madpoints)
            if weak_orientation == 'cw':
                pair = (element_name, metadata['other_element_name'])
            else:
                pair = (metadata['other_element_name'], element_name)
            if pair not in geometry_by_pair:
                geometry_by_pair[pair] = compute_beambeam_geometry(
                    twiss_and_madpoints=twiss_and_madpoints,
                    element_name_cw=pair[0],
                    element_name_acw=pair[1])
            _configure_element(
                line=lines[weak_orientation], element_name=element_name,
                strong_line=strong_line, strong_metadata=strong_metadata,
                geometry=geometry_by_pair[pair],
                weak_orientation=weak_orientation,
                num_particles=num_particles)

        _store_self_orbit_and_dipolar_kick(
            line=lines[weak_orientation],
            particle_on_co=(
                twiss_and_madpoints['particle_on_co'][weak_orientation]))

    env['beambeam_scale'] = 1.0
    for orientation in ('clockwise', 'anticlockwise'):
        line_name = installation.line_names[orientation]
        if line_name is None:
            continue
        line = env.lines[line_name]
        for element_name in installation.elements[orientation]:
            variable_name = f'{element_name}_scale_strength'
            env[variable_name] = env.ref['beambeam_scale']
            env[element_name].scale_strength = env.ref[
                variable_name]

    if filling_pattern_cw is not None:
        apply_filling_pattern(
            env,
            filling_pattern_cw=filling_pattern_cw,
            filling_pattern_acw=filling_pattern_acw,
            i_bunch_cw=i_bunch_cw,
            i_bunch_acw=i_bunch_acw)


@dataclass
class _ParticlesInstallation:
    elements: dict
    line_names: dict
    ip_names: list
    n_slots: int
    delay_at_ips_slots: dict | None


def _line_name(env, line, argument_name):
    if line is None:
        return None
    if isinstance(line, str):
        if line not in env.lines:
            raise KeyError(f'Line `{line}` not found in environment.')
        return line
    matches = [
        name for name, candidate in env.lines.items() if candidate is line]
    if len(matches) != 1:
        raise ValueError(
            f'`{argument_name}` must be a line in this environment or its '
            'name.')
    return matches[0]


def _normalize_num_long_range(ip_names, values):
    if isinstance(values, dict):
        values = [values[ip_name] for ip_name in ip_names]
    elif np.ndim(values) == 0:
        values = [values] * len(ip_names)
    else:
        values = list(values)
        if len(values) != len(ip_names):
            raise ValueError(
                '`num_long_range_encounters_per_side` must have one entry '
                'per IP.')

    normalized = []
    for value in values:
        integer_value = int(value)
        if integer_value != value or integer_value < 0:
            raise ValueError(
                '`num_long_range_encounters_per_side` entries must be '
                'non-negative integers.')
        normalized.append(integer_value)
    return normalized


def _normalize_delays(ip_names, delay_at_ips_slots):
    if delay_at_ips_slots is None:
        return None
    if isinstance(delay_at_ips_slots, dict):
        return {ip_name: int(delay_at_ips_slots[ip_name])
                for ip_name in ip_names}
    delays = list(delay_at_ips_slots)
    if len(delays) != len(ip_names):
        raise ValueError('`delay_at_ips_slots` must have one entry per IP.')
    return {ip_name: int(delay) for ip_name, delay in zip(ip_names, delays)}


def _head_on_centroids(sigmaz, num_slices):
    integer_num_slices = int(num_slices)
    if (integer_num_slices != num_slices or integer_num_slices < 1
            or integer_num_slices % 2 != 1):
        raise ValueError('`num_slices_head_on` must be a positive odd integer.')
    num_slices = integer_num_slices
    if num_slices == 1:
        return np.array([0.])

    slice_charge_boundaries = np.arange(1, num_slices) / num_slices
    from scipy.special import erfinv
    cuts = (np.sqrt(2) * sigmaz / 2
            * erfinv(2 * slice_charge_boundaries - 1))
    edges = np.concatenate(([-np.inf], cuts, [np.inf]))
    exponentials = np.exp(-edges**2 / (2 * (sigmaz / 2)**2))
    centroids = (-sigmaz / 2 / np.sqrt(2 * np.pi)
                 * np.diff(exponentials) * num_slices)
    centroids[num_slices // 2] = 0.
    return centroids


def _element_specs(
        circumference, ip_names, num_long_range_encounters_per_side,
        num_slices_head_on, harmonic_number, bunch_spacing_buckets, sigmaz,
        orientation, beam_name, other_beam_name):
    bunch_spacing = (
        circumference / harmonic_number * bunch_spacing_buckets)
    centroids = _head_on_centroids(sigmaz, num_slices_head_on)
    num_slices_head_on = int(num_slices_head_on)
    slice_identifiers = range(-(num_slices_head_on // 2),
                              num_slices_head_on // 2 + 1)

    specs = []
    for ip_name, num_lr in zip(
            ip_names, num_long_range_encounters_per_side):
        for identifier, centroid in zip(slice_identifiers, centroids):
            specs.append({
                'ip_name': ip_name,
                'label': 'bb_ho',
                'identifier': identifier,
                'at_position': float(centroid),
                's_crab': float(centroid),
                'self_frac_of_bunch': 1 / num_slices_head_on,
            })
        for identifier in range(1, num_lr + 1):
            for signed_identifier in (identifier, -identifier):
                specs.append({
                    'ip_name': ip_name,
                    'label': 'bb_lr',
                    'identifier': signed_identifier,
                    'at_position': signed_identifier * bunch_spacing / 2,
                    's_crab': 0.,
                    'self_frac_of_bunch': 1.,
                })

    position_sign = 1 if orientation == 'clockwise' else -1
    for spec in specs:
        identifier = spec['identifier']
        spec['element_name'] = _beambeam_element_name(
            spec['label'], spec['ip_name'], beam_name, identifier)
        spec['other_element_name'] = _beambeam_element_name(
            spec['label'], spec['ip_name'], other_beam_name, identifier)
        spec['at_position'] *= position_sign
    return sorted(specs, key=lambda spec: spec['element_name'])


def _new_beambeam_element(label):
    if label == 'bb_ho':
        return xf.BeamBeamBiGaussian3D(
            phi=0, alpha=0, other_beam_q0=0.,
            slices_other_beam_num_particles=[0],
            slices_other_beam_zeta_center=[0],
            slices_other_beam_Sigma_11=[1],
            slices_other_beam_Sigma_12=[0],
            slices_other_beam_Sigma_22=[0],
            slices_other_beam_Sigma_33=[1],
            slices_other_beam_Sigma_34=[0],
            slices_other_beam_Sigma_44=[0])
    return xf.BeamBeamBiGaussian2D(
        other_beam_beta0=1., other_beam_q0=0,
        other_beam_num_particles=0., other_beam_Sigma_11=1,
        other_beam_Sigma_33=1)


def _install_elements(env, line_name, specs):
    if line_name is None:
        return
    line = env.lines[line_name]
    placements = []
    ip_positions = {
        ip_name: line.get_table()['s', ip_name]
        for ip_name in dict.fromkeys(spec['ip_name'] for spec in specs)}

    for spec in specs:
        element_name = spec['element_name']
        print(f'Insert: {element_name}     ', end='\r', flush=True)
        element = _new_beambeam_element(spec['label'])
        metadata = {
            'version': _BEAMBEAM_EXTRA_VERSION,
            'mode': 'particles',
            'ip_name': spec['ip_name'],
            'label': spec['label'],
            'identifier': spec['identifier'],
            'other_element_name': spec['other_element_name'],
            'self_frac_of_bunch': spec['self_frac_of_bunch'],
            's_crab': spec['s_crab'],
        }
        if not hasattr(element, 'extra') or element.extra is None:
            element.extra = {}
        element.extra[_BEAMBEAM_EXTRA_KEY] = metadata
        env.elements[element_name] = element
        placements.append(env.place(
            element_name,
            at=ip_positions[spec['ip_name']] + spec['at_position']))
    line.insert(placements)


def _metadata(element):
    extra = getattr(element, 'extra', None)
    if not isinstance(extra, dict):
        return None
    metadata = extra.get(_BEAMBEAM_EXTRA_KEY)
    if metadata is None:
        return None
    if metadata.get('mode') != 'particles':
        return None
    if metadata.get('version') != _BEAMBEAM_EXTRA_VERSION:
        raise RuntimeError(
            'Unsupported `particles`-mode element metadata version '
            f'for element `{getattr(element, "env_name", "<unknown>")}`.')
    return metadata


def _discover_installation(env):
    elements = {'clockwise': {}, 'anticlockwise': {}}
    config = env.extra_config.get(_BEAMBEAM_CONFIG_KEY)
    if config is None:
        raise RuntimeError(
            'No `particles`-mode beam-beam configuration was found. '
            'Call `install_beambeam_interactions(...)` first.')
    if config.get('version') != _BEAMBEAM_CONFIG_VERSION:
        raise RuntimeError(
            'Unsupported `particles`-mode beam-beam configuration version.')
    if config.get('mode') != 'particles':
        raise RuntimeError(
            'The environment beam-beam configuration is not in `particles` '
            'mode.')

    line_names = {
        'clockwise': config['clockwise_line'],
        'anticlockwise': config['anticlockwise_line'],
    }
    ip_names = list(config['ip_names'])
    for orientation, line_name in line_names.items():
        if line_name is None:
            continue
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
                f'No tagged `particles`-mode beam-beam elements were found in '
                f'configured line `{line_name}`.')
        installed_ips = {
            metadata['ip_name']
            for metadata in elements[orientation].values()}
        if installed_ips != set(ip_names):
            raise RuntimeError(
                f'Beam-beam elements in line `{line_name}` do not cover the '
                'configured interaction points.')

    if all(line_name is None for line_name in line_names.values()):
        raise RuntimeError(
            'The `particles`-mode beam-beam configuration contains no lines.')
    for orientation in elements:
        elements[orientation] = dict(sorted(elements[orientation].items()))

    return _ParticlesInstallation(
        elements=elements, line_names=line_names, ip_names=ip_names,
        n_slots=int(config['n_slots']),
        delay_at_ips_slots=config.get('delay_at_ips_slots'))


def _resolve_strong_beam(
        weak_orientation, element_name, metadata, lines, elements,
        twiss_and_madpoints):
    strong_orientation = 'acw' if weak_orientation == 'cw' else 'cw'
    if lines[strong_orientation] is not None:
        strong_metadata = elements[strong_orientation][
            metadata['other_element_name']]
        return lines[strong_orientation], strong_metadata

    partner_name = twiss_and_madpoints[
        'antisymmetric_partner_names'][weak_orientation][element_name]
    return (lines[weak_orientation],
            elements[weak_orientation][partner_name])


def _element_names_by_ip(elements):
    names = {}
    for element_name, metadata in elements.items():
        names.setdefault(metadata['ip_name'], []).append(element_name)
    return names


def _configure_element(
        line, element_name, strong_line, strong_metadata,
        geometry, weak_orientation, num_particles):
    strong_orientation = 'acw' if weak_orientation == 'cw' else 'cw'
    weak_geometry = geometry[weak_orientation]
    strong_geometry = geometry[strong_orientation]
    separation_x = weak_geometry['separation_x']
    separation_y = weak_geometry['separation_y']

    sigma = strong_geometry['sigma']
    dpx = weak_geometry['dpx']
    dpy = weak_geometry['dpy']
    if weak_orientation == 'acw':
        separation_x = -separation_x
        sigma = _to_stored_acw_sigma(sigma)
        dpy = -dpy

    alpha, phi = find_alpha_and_phi(dpx, dpy)
    other_num_particles = (
        num_particles * strong_metadata['self_frac_of_bunch'])
    other_particle_charge = float(strong_line.particle_ref.q0)
    other_relativistic_beta = float(strong_line.particle_ref.beta0[0])
    element = line[element_name]
    if isinstance(element, xf.BeamBeamBiGaussian2D):
        element.other_beam_num_particles = other_num_particles
        element.other_beam_q0 = other_particle_charge
        element.other_beam_Sigma_11 = sigma[11]
        element.other_beam_Sigma_33 = sigma[33]
        element.other_beam_beta0 = other_relativistic_beta
        element.other_beam_shift_x = separation_x
        element.other_beam_shift_y = separation_y
    elif isinstance(element, xf.BeamBeamBiGaussian3D):
        params = {
            'phi': phi,
            'alpha': alpha,
            'other_beam_shift_x': separation_x,
            'other_beam_shift_y': separation_y,
            'slices_other_beam_num_particles': [other_num_particles],
            'other_beam_q0': other_particle_charge,
            'slices_other_beam_zeta_center': [0.0],
        }
        for sigma_name, value in sigma.items():
            params[f'slices_other_beam_Sigma_{sigma_name}'] = [value]
        for sigma_name in (13, 14, 23, 24):
            params[f'slices_other_beam_Sigma_{sigma_name}'] = [0.0]

        new_element = xf.BeamBeamBiGaussian3D(**params)
        if new_element._xobject._size != element._xobject._size:
            raise RuntimeError(
                'The configured 3D beam-beam element changed allocation size.')
        new_element.move(_buffer=element._buffer, _offset=element._offset)
    else:
        raise TypeError(
            f'Tagged element `{element_name}` is not a supported '
            '`particles`-mode beam-beam element.')
    line.env[element_name].scale_strength = 1.0


def _store_self_orbit_and_dipolar_kick(line, particle_on_co):
    """Store the self-beam orbit and subtract its dipolar beam--beam kick."""
    temp_particles = particle_on_co.copy()
    for ii, element in enumerate(line.elements):
        if element.__class__.__name__ == 'BeamBeamBiGaussian2D':
            px_0 = temp_particles.px[0]
            py_0 = temp_particles.py[0]

            element.post_subtract_px = 0
            element.post_subtract_py = 0

            element.ref_shift_x = temp_particles.x[0]
            element.ref_shift_y = temp_particles.y[0]

            element.track(temp_particles)

            element.post_subtract_px = temp_particles.px[0] - px_0
            element.post_subtract_py = temp_particles.py[0] - py_0

            temp_particles.px -= element.post_subtract_px
            temp_particles.py -= element.post_subtract_py

        elif element.__class__.__name__ == 'BeamBeamBiGaussian3D':
            element.ref_shift_x = temp_particles.x[0]
            element.ref_shift_px = temp_particles.px[0]
            element.ref_shift_y = temp_particles.y[0]
            element.ref_shift_py = temp_particles.py[0]
            element.ref_shift_zeta = temp_particles.zeta[0]
            # The element assumes beta0=1 anyhow.
            element.ref_shift_pzeta = temp_particles.delta[0]

            element.post_subtract_x = 0
            element.post_subtract_px = 0
            element.post_subtract_y = 0
            element.post_subtract_py = 0
            element.post_subtract_zeta = 0
            element.post_subtract_pzeta = 0

            element.track(temp_particles)

            element.post_subtract_x = (
                temp_particles.x[0] - element.ref_shift_x)
            element.post_subtract_px = (
                temp_particles.px[0] - element.ref_shift_px)
            element.post_subtract_y = (
                temp_particles.y[0] - element.ref_shift_y)
            element.post_subtract_py = (
                temp_particles.py[0] - element.ref_shift_py)
            element.post_subtract_zeta = (
                temp_particles.zeta[0] - element.ref_shift_zeta)
            element.post_subtract_pzeta = (
                temp_particles.delta[0] - element.ref_shift_pzeta)

            temp_particles.x[0] = element.ref_shift_x
            temp_particles.px[0] = element.ref_shift_px
            temp_particles.y[0] = element.ref_shift_y
            temp_particles.py[0] = element.ref_shift_py
            temp_particles.zeta[0] = element.ref_shift_zeta
            # The element assumes beta0=1 anyhow.
            temp_particles.delta[0] = element.ref_shift_pzeta
        elif element.__class__.__name__ == 'Wire':
            px_0 = temp_particles.px[0]
            py_0 = temp_particles.py[0]

            element.post_subtract_px = 0
            element.post_subtract_py = 0

            element.track(temp_particles)

            element.post_subtract_px = temp_particles.px[0] - px_0
            element.post_subtract_py = temp_particles.py[0] - py_0

            temp_particles.px -= element.post_subtract_px
            temp_particles.py -= element.post_subtract_py
        else:
            line.track(temp_particles, ele_start=ii, num_elements=1)


# Public compatibility name used by standalone Xtrack examples.
configure_orbit_dependent_parameters_for_bb = (
    _store_self_orbit_and_dipolar_kick)


def _to_stored_acw_sigma(sigma):
    signs = {
        11: 1, 12: -1, 13: -1, 14: 1, 22: 1,
        23: 1, 24: -1, 33: 1, 34: -1, 44: 1,
    }
    return {name: signs[name] * value for name, value in sigma.items()}


def apply_filling_pattern(
        env, filling_pattern_cw=None, filling_pattern_acw=None,
        i_bunch_cw=None, i_bunch_acw=None, *,
        filled_slots_cw=None, filled_slots_acw=None):
    """Enable tagged encounters having a filled opposing partner slot."""
    if i_bunch_cw is None or i_bunch_acw is None:
        raise ValueError(
            'Both selected bunch indices are required with beam fillings.')
    installation = _discover_installation(env)
    filling_patterns = {
        'clockwise': _FillingPattern.from_inputs(
            filling_pattern=filling_pattern_cw,
            filled_slots=filled_slots_cw,
            num_slots=installation.n_slots,
            allow_none=False).filling_pattern,
        'anticlockwise': _FillingPattern.from_inputs(
            filling_pattern=filling_pattern_acw,
            filled_slots=filled_slots_acw,
            num_slots=installation.n_slots,
            allow_none=False).filling_pattern,
    }
    selected_bunches = {
        'clockwise': i_bunch_cw,
        'anticlockwise': i_bunch_acw,
    }

    for orientation, pattern in filling_patterns.items():
        if pattern.ndim != 1 or len(pattern) != installation.n_slots:
            suffix = 'cw' if orientation == 'clockwise' else 'acw'
            raise ValueError(
                f'`filling_pattern_{suffix}` must have length '
                f'{installation.n_slots}.')
        if not set(pattern.tolist()).issubset({0, 1}):
            raise ValueError('Filling patterns can contain only zero and one.')
        if pattern[selected_bunches[orientation]] != 1:
            raise ValueError(
                f'The selected {orientation} bunch is not in its filling '
                'pattern.')

    for orientation in ('clockwise', 'anticlockwise'):
        elements = installation.elements[orientation]
        if not elements:
            continue
        if installation.delay_at_ips_slots is None:
            raise RuntimeError(
                'Filling-pattern selection requires `delay_at_ips_slots` at '
                'beam-beam installation time.')
        other_orientation = (
            'anticlockwise' if orientation == 'clockwise' else 'clockwise')
        for element_name, metadata in elements.items():
            delay = _delay_in_slots(installation, orientation, metadata)
            partner_slot = (
                delay + selected_bunches[orientation]) % installation.n_slots
            is_active = filling_patterns[other_orientation][partner_slot] == 1
            variable_name = f'{element_name}_scale_strength'
            env[variable_name] = (
                env.ref['beambeam_scale'] if is_active else 0)


def _delay_in_slots(installation, orientation, metadata):
    """Return the opposing bunch-slot offset for one installed encounter."""
    delay_at_ip = installation.delay_at_ips_slots[
        metadata['ip_name']]
    encounter_identifier = (
        metadata['identifier']
        if metadata['label'] == 'bb_lr' else 0)
    if orientation == 'clockwise':
        return delay_at_ip + encounter_identifier
    if orientation == 'anticlockwise':
        return ((installation.n_slots - delay_at_ip)
                % installation.n_slots - encounter_identifier)
    raise ValueError(f'Unknown beam-beam orientation `{orientation}`.')
