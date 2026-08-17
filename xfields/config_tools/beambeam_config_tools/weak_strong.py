# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

"""Environment-level setup of weak--strong beam--beam interactions.

Installation-wide state is stored in ``Environment.extra_config``. Each
installed element carries only its encounter-local description in the generic,
serializable ``BeamElement.extra`` container.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xobjects as xo

import xfields as xf

from .config_tools import (
    compute_beambeam_geometry,
    compute_dpx_dpy,
    compute_geometry_and_optics,
    compute_local_crossing_angle_and_plane,
    find_bb_separations,
    get_counter_rotating,
    get_partner_position_and_optics,
    get_partner_position_and_optics_antisymmetry,
)
from .orbit_dependent_configuration_tools import (
    configure_orbit_dependent_parameters_for_bb,
)


_BEAMBEAM_EXTRA_KEY = '_xfields_weak_strong_beambeam'
_BEAMBEAM_EXTRA_VERSION = 1
_BEAMBEAM_CONFIG_KEY = 'xfields_beambeam'
_BEAMBEAM_CONFIG_VERSION = 1


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
        spec['element_name'] = _element_name(
            spec['label'], spec['ip_name'], beam_name, identifier)
        spec['other_element_name'] = _element_name(
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
    if metadata.get('version') != _BEAMBEAM_EXTRA_VERSION:
        raise RuntimeError(
            'Unsupported weak--strong beam-beam element metadata version '
            f'for element `{getattr(element, "env_name", "<unknown>")}`.')
    return metadata


def _discover_installation(env):
    elements = {'clockwise': [], 'anticlockwise': []}
    config = env.extra_config.get(_BEAMBEAM_CONFIG_KEY)
    if config is None:
        raise RuntimeError(
            'No new-style weak--strong beam-beam configuration was found. '
            'Call `install_beambeam_interactions(...)` first.')
    if config.get('version') != _BEAMBEAM_CONFIG_VERSION:
        raise RuntimeError(
            'Unsupported weak--strong beam-beam configuration version.')
    if config.get('mode') != 'weak_strong':
        raise RuntimeError(
            'The environment beam-beam configuration is not weak--strong.')

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
            elements[orientation].append(
                _InstalledBeamBeamElement(element_name, metadata))
        if not elements[orientation]:
            raise RuntimeError(
                f'No tagged weak--strong beam-beam elements were found in '
                f'configured line `{line_name}`.')
        installed_ips = {
            record.metadata['ip_name'] for record in elements[orientation]}
        if installed_ips != set(ip_names):
            raise RuntimeError(
                f'Beam-beam elements in line `{line_name}` do not cover the '
                'configured interaction points.')

    if all(line_name is None for line_name in line_names.values()):
        raise RuntimeError(
            'The weak--strong beam-beam configuration contains no lines.')
    for orientation in elements:
        elements[orientation].sort(key=lambda record: record.name)

    return _WeakStrongInstallation(
        elements=elements, line_names=line_names, ip_names=ip_names,
        n_slots=int(config['n_slots']),
        delay_at_ips_slots=config.get('delay_at_ips_slots'))


def install_beambeam_interactions(
        env, clockwise_line, anticlockwise_line, ip_names,
        num_long_range_encounters_per_side, num_slices_head_on,
        harmonic_number, bunch_spacing_buckets, sigmaz,
        delay_at_ips_slots=None):
    """Install inactive, tagged weak--strong beam-beam elements."""
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
        'mode': 'weak_strong',
        'clockwise_line': cw_name,
        'anticlockwise_line': acw_name,
        'ip_names': ip_names,
        'n_slots': n_slots,
        'delay_at_ips_slots': delays_by_ip,
    }

    # A rigid-bunch installation uses this attribute to select its configure
    # path. Installing weak--strong elements replaces that mode explicitly.
    if hasattr(env, '_bb_config'):
        del env._bb_config


def _build_configuration_table(elements, line, beam, other_beam):
    if not elements:
        return None
    rows = []
    for record in elements:
        metadata = record.metadata
        rows.append({
            'beam': beam,
            'other_beam': other_beam,
            'ip_name': metadata['ip_name'],
            'elementName': record.name,
            'other_elementName': metadata['other_element_name'],
            'label': metadata['label'],
            'self_particle_charge': float(line.particle_ref.q0),
            'self_relativistic_beta': float(line.particle_ref.beta0[0]),
            'self_frac_of_bunch': metadata['self_frac_of_bunch'],
            'identifier': metadata['identifier'],
            's_crab': metadata['s_crab'],
        })
    dataframe = pd.DataFrame(rows).set_index('elementName', drop=False)
    for which_beam in ('self', 'other'):
        for coordinate in ('x', 'px', 'y', 'py'):
            dataframe[f'{which_beam}_{coordinate}_crab'] = 0.
    return dataframe.sort_index()


def _analyse_and_configure_elements(
        installation, line_cw, line_acw, num_particles,
        nemitt_x, nemitt_y, crab_strong_beam,
        use_antisymmetry=False, separation_bumps=None):

    bb_df_cw = _build_configuration_table(
        installation.elements['clockwise'], line_cw, 'b1', 'b2')
    bb_df_acw = _build_configuration_table(
        installation.elements['anticlockwise'], line_acw, 'b2', 'b1')

    if line_cw is None or line_acw is None:
        assert use_antisymmetry is True, (
            'If you are not using antisymmetry, you need to provide both beams')
    else:
        assert use_antisymmetry is False, (
            'If you are using antisymmetry, you need to provide only one beam'
            ' (for now...).')

    shared_geometry = None
    twisses = {}
    if not use_antisymmetry:
        # CW follows the stored line. The stored B4 line is reversed to the
        # physical ACW convention while its beam-beam elements are analysed.
        twiss_cw = line_cw.twiss(reverse=False)
        twiss_acw = line_acw.twiss(reverse=False).reverse()
        encounter_instances = bb_df_cw[[
            'ip_name', 'label', 'identifier']].rename(
                columns={'label': 'encounter_type'}).reset_index(drop=True)
        shared_geometry, twisses = compute_beambeam_geometry(
            encounter_table=encounter_instances,
            line_cw=line_cw, line_acw=line_acw,
            element_names_cw=bb_df_cw.index,
            element_names_acw=bb_df_cw['other_elementName'],
            nemitt_x=nemitt_x, nemitt_y=nemitt_y,
            survey_separation=True,
            twiss_cw=twiss_cw, twiss_acw=twiss_acw,
            acw_is_reversed=True)

    for bb_df, line, orientation in zip(
            [bb_df_cw, bb_df_acw], [line_cw, line_acw], ['cw', 'acw']):
        if bb_df is None:
            continue

        if shared_geometry is not None:
            twiss = twisses[orientation]
        else:
            twiss = line.twiss(reverse=False)
            if orientation == 'acw':
                twiss = twiss.reverse()
            twisses[orientation] = twiss

        surveys = None
        if shared_geometry is None:
            surveys = {}
            for ip_name in installation.ip_names:
                survey = line.survey(element0=ip_name, reverse=False)
                if orientation == 'acw':
                    survey = survey.reverse()
                surveys[ip_name] = survey
                assert survey['X', ip_name] == 0
                assert survey['Y', ip_name] == 0
                assert survey['Z', ip_name] == 0

        sigmas = None
        if shared_geometry is None:
            sigmas = twiss.get_beam_covariance(
                nemitt_x=nemitt_x, nemitt_y=nemitt_y)

        bb_df['self_num_particles'] = (
            num_particles * bb_df['self_frac_of_bunch'])
        compute_geometry_and_optics(
            bb_df=bb_df, xsuite_twiss=twiss, xsuite_survey=surveys,
            xsuite_sigmas=sigmas, shared_geometry=shared_geometry,
            orientation=orientation)

        if crab_strong_beam:
            _measure_crabbing(line, bb_df, reverse=orientation == 'acw')

    if not use_antisymmetry:
        get_partner_position_and_optics(
            bb_df_cw, bb_df_acw, crab_strong_beam=crab_strong_beam,
            include_lab_positions=shared_geometry is None)
    elif line_cw is not None:
        get_partner_position_and_optics_antisymmetry(
            bb_df_cw, crab_strong_beam=crab_strong_beam,
            separation_bumps=separation_bumps)
    else:
        get_partner_position_and_optics_antisymmetry(
            bb_df_acw, crab_strong_beam=crab_strong_beam,
            separation_bumps=separation_bumps)

    for bb_df, orientation in zip([bb_df_cw, bb_df_acw], ['cw', 'acw']):
        if bb_df is None:
            continue
        if shared_geometry is not None:
            shared_by_element = shared_geometry.set_index(
                f'element_name_{orientation}')
            for field in ('separation_x', 'separation_y', 'dpx', 'dpy'):
                bb_df[field] = [
                    shared_by_element.loc[name][f'{field}_{orientation}']
                    for name in bb_df.index]
        else:
            bb_df['separation_x'], bb_df['separation_y'] = \
                find_bb_separations(
                    points_weak=bb_df['self_lab_position'].values,
                    points_strong=bb_df['other_lab_position'].values,
                    names=bb_df.index.values)
            compute_dpx_dpy(bb_df)
        compute_local_crossing_angle_and_plane(bb_df)

        if crab_strong_beam:
            bb_df['separation_x_no_crab'] = bb_df['separation_x']
            bb_df['separation_y_no_crab'] = bb_df['separation_y']
            bb_df['separation_x'] += bb_df['other_x_crab']
            bb_df['separation_y'] += bb_df['other_y_crab']

    if line_cw is not None:
        _configure_elements_in_line(line_cw, bb_df_cw)
        configure_orbit_dependent_parameters_for_bb(
            line=line_cw, particle_on_co=twisses['cw'].particle_on_co)
    if line_acw is not None:
        bb_df_b4 = get_counter_rotating(bb_df_acw)
        _configure_elements_in_line(line_acw, bb_df_b4)
        configure_orbit_dependent_parameters_for_bb(
            line=line_acw,
            particle_on_co=twisses['acw'].reverse().particle_on_co)


def _configure_elements_in_line(line, bb_df):
    for element_name in bb_df.index:
        element = line[element_name]
        if isinstance(element, xf.BeamBeamBiGaussian2D):
            element.other_beam_num_particles = bb_df.loc[
                element_name, 'other_num_particles']
            element.other_beam_q0 = bb_df.loc[
                element_name, 'other_particle_charge']
            element.other_beam_Sigma_11 = bb_df.loc[
                element_name, 'other_Sigma_11']
            element.other_beam_Sigma_33 = bb_df.loc[
                element_name, 'other_Sigma_33']
            element.other_beam_beta0 = bb_df.loc[
                element_name, 'other_relativistic_beta']
            element.other_beam_shift_x = bb_df.loc[
                element_name, 'separation_x']
            element.other_beam_shift_y = bb_df.loc[
                element_name, 'separation_y']
        elif isinstance(element, xf.BeamBeamBiGaussian3D):
            params = {
                'phi': bb_df.loc[element_name, 'phi'],
                'alpha': bb_df.loc[element_name, 'alpha'],
                'other_beam_shift_x': bb_df.loc[element_name, 'separation_x'],
                'other_beam_shift_y': bb_df.loc[element_name, 'separation_y'],
                'slices_other_beam_num_particles': [bb_df.loc[
                    element_name, 'other_num_particles']],
                'other_beam_q0': bb_df.loc[
                    element_name, 'other_particle_charge'],
                'slices_other_beam_zeta_center': [0.0],
            }
            for sigma_name in (11, 12, 13, 14, 22, 23, 24, 33, 34, 44):
                params[f'slices_other_beam_Sigma_{sigma_name}'] = [
                    bb_df.loc[element_name, f'other_Sigma_{sigma_name}']]
            for sigma_name in (13, 14, 23, 24):
                params[f'slices_other_beam_Sigma_{sigma_name}'] = [0.0]

            new_element = xf.BeamBeamBiGaussian3D(**params)
            assert new_element._xobject._size == element._xobject._size
            new_element.move(
                _buffer=element._buffer, _offset=element._offset)
        else:
            raise TypeError(
                f'Tagged element `{element_name}` is not a supported '
                'weak--strong beam-beam element.')


def _measure_crabbing(line, bb_df, reverse):
    twiss = line.twiss(reverse=False)
    if reverse:
        twiss = twiss.reverse()

    for element_name in bb_df.index:
        s_crab = bb_df.loc[element_name, 's_crab']
        if s_crab != 0.0:
            print(f'Crabbing at {element_name}     ', end='\r', flush=True)
            zeta0 = -2 * s_crab if reverse else 2 * s_crab
            twiss_crab = line.twiss(
                method='4d', zeta0=zeta0, reverse=False)
            if reverse:
                twiss_crab = twiss_crab.reverse()
            element_index = np.where(
                np.array(twiss.name) == element_name)[0][0]
            for coordinate in ('x', 'px', 'y', 'py'):
                bb_df.loc[element_name, f'self_{coordinate}_crab'] = (
                    twiss_crab[coordinate][element_index]
                    - twiss[coordinate][element_index])


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

    _analyse_and_configure_elements(
        installation=installation,
        line_cw=env.lines.get(installation.line_names['clockwise']),
        line_acw=env.lines.get(installation.line_names['anticlockwise']),
        num_particles=num_particles, nemitt_x=nemitt_x, nemitt_y=nemitt_y,
        crab_strong_beam=crab_strong_beam,
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
        if installation.delay_at_ips_slots is None:
            raise RuntimeError(
                'Filling-pattern selection requires `delay_at_ips_slots` at '
                'beam-beam installation time.')
        other_orientation = (
            'anticlockwise' if orientation == 'clockwise' else 'clockwise')
        for record in records:
            delay = _delay_in_slots(installation, orientation, record)
            partner_slot = (
                delay + selected_bunches[orientation]) % installation.n_slots
            is_active = filling_patterns[other_orientation][partner_slot] == 1
            variable_name = f'{record.name}_scale_strength'
            env.vars[variable_name] = (
                env.vars['beambeam_scale'] if is_active else 0)


def _delay_in_slots(installation, orientation, record):
    """Return the opposing bunch-slot offset for one installed encounter."""
    delay_at_ip = installation.delay_at_ips_slots[
        record.metadata['ip_name']]
    encounter_identifier = (
        record.metadata['identifier']
        if record.metadata['label'] == 'bb_lr' else 0)
    if orientation == 'clockwise':
        return delay_at_ip + encounter_identifier
    if orientation == 'anticlockwise':
        return ((installation.n_slots - delay_at_ip)
                % installation.n_slots - encounter_identifier)
    raise ValueError(f'Unknown beam-beam orientation `{orientation}`.')
