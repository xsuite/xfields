# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Environment-level setup of weak--strong beam--beam interactions.

The installed elements are the persistent source of truth. Metadata which is
not part of the tracking model is stored in the generic, serializable
``BeamElement.extra`` container.
"""

from dataclasses import dataclass

import numpy as np
import xobjects as xo

import xfields as xf

from .config_tools import _configure_beam_beam_elements


_BEAMBEAM_EXTRA_KEY = '_xfields_weak_strong_beambeam'
_BEAMBEAM_EXTRA_VERSION = 1


@dataclass(frozen=True)
class _InstalledBeamBeamElement:
    name: str
    metadata: dict


@dataclass
class _WeakStrongInstallation:
    elements: dict
    line_names: dict
    ip_names: list
    n_slots: int


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


def _element_name(label, ip_name, beam_name, identifier):
    side = '.r' if identifier > 0 else '.l' if identifier < 0 else '.c'
    ip_identifier = ip_name.replace('ip', '')
    return (f'{label}{side}{ip_identifier}{beam_name}_'
            f'{abs(identifier):02d}')


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
        orientation, beam_name, other_beam_name, delays_by_ip, n_slots):
    bunch_spacing = (
        circumference / harmonic_number * bunch_spacing_buckets)
    centroids = _head_on_centroids(sigmaz, num_slices_head_on)
    slice_identifiers = range(-(num_slices_head_on // 2),
                              num_slices_head_on // 2 + 1)

    specs = []
    for ip_index, (ip_name, num_lr) in enumerate(zip(
            ip_names, num_long_range_encounters_per_side)):
        for identifier, centroid in zip(slice_identifiers, centroids):
            specs.append({
                'ip_name': ip_name,
                'ip_index': ip_index,
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
                    'ip_index': ip_index,
                    'label': 'bb_lr',
                    'identifier': signed_identifier,
                    'at_position': signed_identifier * bunch_spacing / 2,
                    's_crab': 0.,
                    'self_frac_of_bunch': 1.,
                })

    position_sign = 1 if orientation == 'clockwise' else -1
    for spec in specs:
        identifier = spec['identifier']
        spec['element_name'] = _element_name(
            spec['label'], spec['ip_name'], beam_name, identifier)
        spec['other_element_name'] = _element_name(
            spec['label'], spec['ip_name'], other_beam_name, identifier)
        spec['at_position'] *= position_sign
        if delays_by_ip is not None:
            encounter_identifier = (
                identifier if spec['label'] == 'bb_lr' else 0)
            ip_delay = delays_by_ip[spec['ip_name']]
            if orientation == 'clockwise':
                spec['delay_in_slots'] = ip_delay + encounter_identifier
            else:
                spec['delay_in_slots'] = (
                    (n_slots - ip_delay) % n_slots - encounter_identifier)
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


def _install_elements(
        env, line_name, other_line_name, orientation, specs, n_slots):
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
            'orientation': orientation,
            'line_name': line_name,
            'other_line_name': other_line_name,
            'ip_name': spec['ip_name'],
            'ip_index': spec['ip_index'],
            'label': spec['label'],
            'identifier': spec['identifier'],
            'other_element_name': spec['other_element_name'],
            'self_frac_of_bunch': spec['self_frac_of_bunch'],
            's_crab': spec['s_crab'],
            'n_slots': n_slots,
        }
        if 'delay_in_slots' in spec:
            metadata['delay_in_slots'] = spec['delay_in_slots']
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
    if metadata.get('version') != _BEAMBEAM_EXTRA_VERSION:
        raise RuntimeError(
            'Unsupported weak--strong beam-beam element metadata version '
            f'for element `{getattr(element, "env_name", "<unknown>")}`.')
    return metadata


def _discover_installation(env):
    elements = {'clockwise': [], 'anticlockwise': []}
    line_names = {'clockwise': None, 'anticlockwise': None}
    ip_indices = {}
    n_slots_values = set()

    for actual_line_name, line in env.lines.items():
        for element_name in dict.fromkeys(line.element_names):
            metadata = _metadata(line[element_name])
            if metadata is None:
                continue
            orientation = metadata['orientation']
            if orientation not in elements:
                raise RuntimeError(
                    f'Invalid beam-beam orientation `{orientation}` on '
                    f'element `{element_name}`.')
            if metadata['line_name'] != actual_line_name:
                continue
            if (line_names[orientation] is not None
                    and line_names[orientation] != actual_line_name):
                raise RuntimeError(
                    'More than one weak--strong beam-beam installation was '
                    f'found for the {orientation} orientation.')
            line_names[orientation] = actual_line_name
            ip_indices[metadata['ip_name']] = metadata['ip_index']
            n_slots_values.add(int(metadata['n_slots']))
            elements[orientation].append(
                _InstalledBeamBeamElement(element_name, metadata))

    if not elements['clockwise'] and not elements['anticlockwise']:
        raise RuntimeError(
            'No new-style weak--strong beam-beam interactions were found. '
            'Call `install_beambeam_interactions(...)` first.')
    if len(n_slots_values) != 1:
        raise RuntimeError(
            'Installed weak--strong beam-beam elements have inconsistent '
            '`n_slots` metadata.')
    for orientation in elements:
        elements[orientation].sort(key=lambda record: record.name)

    ip_names = [
        name for name, _ in sorted(ip_indices.items(), key=lambda item: item[1])]
    return _WeakStrongInstallation(
        elements=elements, line_names=line_names, ip_names=ip_names,
        n_slots=n_slots_values.pop())


def install_beambeam_interactions(
        env, clockwise_line, anticlockwise_line, ip_names,
        num_long_range_encounters_per_side, num_slices_head_on,
        harmonic_number, bunch_spacing_buckets, sigmaz,
        delay_at_ips_slots=None):
    """Install inactive, self-describing weak--strong beam-beam elements."""
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

    for orientation, line_name, other_line_name, beam_name, other_beam_name in (
            ('clockwise', cw_name, acw_name, 'b1', 'b2'),
            ('anticlockwise', acw_name, cw_name, 'b2', 'b1')):
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
            other_beam_name=other_beam_name, delays_by_ip=delays_by_ip,
            n_slots=n_slots)
        _install_elements(
            env, line_name, other_line_name, orientation, specs, n_slots)

    # A rigid-bunch installation uses this attribute to select its configure
    # path. Installing weak--strong elements replaces that mode explicitly.
    if hasattr(env, '_bb_config'):
        del env._bb_config


def configure_beambeam_interactions(
        env, num_particles, nemitt_x, nemitt_y, crab_strong_beam=True,
        use_antisymmetry=False, separation_bumps=None):
    """Reanalyse the live lines and configure their tagged BB elements."""
    installation = _discover_installation(env)

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

        for record in installation.elements[orientation]:
            line.element_refs[record.name].scale_strength = 1.0
            element = line[record.name]
            element.other_beam_q0 = 0.0
            for field_name in (
                    'post_subtract_x', 'post_subtract_px',
                    'post_subtract_y', 'post_subtract_py',
                    'post_subtract_zeta', 'post_subtract_pzeta'):
                if hasattr(element, field_name):
                    setattr(element, field_name, 0.0)

    _configure_beam_beam_elements(
        elements_cw=installation.elements['clockwise'],
        elements_acw=installation.elements['anticlockwise'],
        line_cw=env.lines.get(installation.line_names['clockwise'], None),
        line_acw=env.lines.get(
            installation.line_names['anticlockwise'], None),
        num_particles=num_particles, nemitt_x=nemitt_x, nemitt_y=nemitt_y,
        crab_strong_beam=crab_strong_beam, ip_names=installation.ip_names,
        use_antisymmetry=use_antisymmetry,
        separation_bumps=separation_bumps)

    env.vars['beambeam_scale'] = 1.0
    for orientation in ('clockwise', 'anticlockwise'):
        line_name = installation.line_names[orientation]
        if line_name is None:
            continue
        line = env.lines[line_name]
        for record in installation.elements[orientation]:
            variable_name = f'{record.name}_scale_strength'
            env.vars[variable_name] = env.vars['beambeam_scale']
            line.element_refs[record.name].scale_strength = env.vars[
                variable_name]


def apply_filling_pattern(env, filling_pattern_cw, filling_pattern_acw,
                          i_bunch_cw, i_bunch_acw):
    """Enable tagged encounters having a filled opposing partner slot."""
    installation = _discover_installation(env)
    filling_patterns = {
        'clockwise': np.asarray(filling_pattern_cw, dtype=int),
        'anticlockwise': np.asarray(filling_pattern_acw, dtype=int),
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
        records = installation.elements[orientation]
        if not records:
            continue
        if any('delay_in_slots' not in record.metadata for record in records):
            raise RuntimeError(
                'Filling-pattern selection requires `delay_at_ips_slots` at '
                'beam-beam installation time.')
        other_orientation = (
            'anticlockwise' if orientation == 'clockwise' else 'clockwise')
        for record in records:
            partner_slot = (
                record.metadata['delay_in_slots']
                + selected_bunches[orientation]) % installation.n_slots
            is_active = filling_patterns[other_orientation][partner_slot] == 1
            variable_name = f'{record.name}_scale_strength'
            env.vars[variable_name] = (
                env.vars['beambeam_scale'] if is_active else 0)
