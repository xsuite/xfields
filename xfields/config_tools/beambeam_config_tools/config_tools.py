# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

"""Shared Twiss and geometry data for beam--beam configuration."""

import copy
import numpy as np

from ._madpoint import MadPoint


_SIGMA_NAMES = (11, 12, 13, 14, 22, 23, 24, 33, 34, 44)
BEAMBEAM_CONFIG_KEY = 'xfields_beambeam'
BEAMBEAM_CONFIG_VERSION = 1
BEAMBEAM_ELEMENT_EXTRA_KEY = '_xfields_beambeam'
BEAMBEAM_ELEMENT_EXTRA_VERSION = 1


def _beambeam_element_name(label, ip_name, beam_name, identifier):
    side = '.r' if identifier > 0 else '.l' if identifier < 0 else '.c'
    ip_identifier = ip_name.replace('ip', '')
    return (f'{label}{side}{ip_identifier}{beam_name}_'
            f'{abs(identifier):02d}')


def compute_twiss_and_madpoints_at_bb(
        line_cw, line_acw, element_names_by_ip, nemitt_x, nemitt_y,
        survey_separation=True, acw_is_reversed=False,
        crab_s_by_element=None, antisymmetry_elements=None,
        separation_bumps=None):
    """Compute reduced Twiss tables and MadPoints at beam--beam elements.

    ``element_names_by_ip`` is indexed first by ``'cw'`` / ``'acw'`` and
    then by IP name. ``crab_s_by_element`` optionally provides the
    longitudinal offset used to measure crabbing at each element. When
    ``antisymmetry_elements`` is provided, the missing beam is synthesized
    from mirrored elements of the available beam.
    """
    lines = {'cw': line_cw, 'acw': line_acw}
    lines = {orientation: line for orientation, line in lines.items()
             if line is not None}
    element_names_by_ip = {
        orientation: {
            ip_name: list(names) for ip_name, names in by_ip.items()}
        for orientation, by_ip in element_names_by_ip.items()
    }
    particle_on_co = {}
    twiss = {
        orientation: line.twiss(reverse=False)
        for orientation, line in lines.items()
    }
    particle_on_co.update({
        orientation: table.particle_on_co
        for orientation, table in twiss.items()
    })
    if acw_is_reversed and 'acw' in twiss:
        twiss['acw'] = twiss['acw'].reverse()
    covariance = {
        orientation: table.get_beam_covariance(
            nemitt_x=nemitt_x, nemitt_y=nemitt_y)
        for orientation, table in twiss.items()
    }
    points = {'cw': {}, 'acw': {}}
    for orientation, line in lines.items():
        for ip_name, names in element_names_by_ip[orientation].items():
            region_survey = None
            if survey_separation:
                line_survey = line.survey(element0=ip_name)
                region_survey = line_survey.rows[[ip_name, *names]]
                region_span = np.ptp(region_survey.s)
                assert 2 * region_span <= line_survey.s[-1], (
                    f'The beam-beam region around {ip_name!r} wraps across '
                    'the line boundary, which is not supported.')

            for element_name in names:
                point = MadPoint(
                    element_name, use_twiss=True,
                    use_survey=survey_separation,
                    xsuite_twiss=twiss[orientation],
                    xsuite_survey=region_survey)
                if survey_separation and orientation == 'acw':
                    _transform_acw_point(
                        point, reverse_local=acw_is_reversed)
                points[orientation][element_name] = point

    crab_offsets = {'cw': {}, 'acw': {}}
    if crab_s_by_element is not None:
        for orientation, line in lines.items():
            crab_offsets[orientation] = _measure_crabbing(
                line=line,
                crab_s_by_element=crab_s_by_element.get(orientation, {}),
                twiss=twiss[orientation],
                reverse=orientation == 'acw' and acw_is_reversed)

    antisymmetric_partner_names = {'cw': {}, 'acw': {}}
    if antisymmetry_elements is not None:
        if len(lines) != 1:
            raise ValueError(
                'Antisymmetry requires exactly one beam line.')
        orientation = next(iter(lines))
        missing_orientation = 'acw' if orientation == 'cw' else 'cw'
        (twiss[missing_orientation], covariance[missing_orientation],
         points[missing_orientation],
         crab_offsets[missing_orientation],
         antisymmetric_partner_names[orientation]) = (
            _synthesize_antisymmetric_beam(
                elements=antisymmetry_elements,
                twiss=twiss[orientation],
                covariance=covariance[orientation],
                points=points[orientation],
                crab_offsets=crab_offsets[orientation],
                separation_bumps=separation_bumps))
        for metadata in antisymmetry_elements.values():
            element_names_by_ip[missing_orientation].setdefault(
                metadata['ip_name'], []).append(
                    metadata['other_element_name'])

    for orientation in ('cw', 'acw'):
        if orientation not in twiss:
            continue
        names = [name
                 for names_at_ip in element_names_by_ip[orientation].values()
                 for name in names_at_ip]
        twiss[orientation] = twiss[orientation].rows[names]
        covariance[orientation] = covariance[orientation].rows[names]

    return {
        'twiss': twiss,
        'covariance': covariance,
        'points': points,
        'crab_offsets': crab_offsets,
        'particle_on_co': particle_on_co,
        'survey_separation': bool(survey_separation),
        'antisymmetric_partner_names': antisymmetric_partner_names,
    }


def _transform_acw_point(point, reverse_local):
    reverse_global = np.diag([-1., 1., -1.])
    local_transform = (
        np.diag([-1., 1., -1.]) if reverse_local else np.eye(3))
    frame = np.column_stack((point.ex, point.ey, point.ez))
    frame = reverse_global @ frame @ local_transform
    point.sp = reverse_global @ point.sp
    point.sx, point.sy, point.sz = point.sp
    point.ex, point.ey, point.ez = frame.T
    point.p = point.sp + point.ex * point.tx + point.ey * point.ty


def _synthesize_antisymmetric_beam(
        elements, twiss, covariance, points,
        crab_offsets, separation_bumps):
    element_names = list(elements)
    positions = np.array([
        twiss['s', element_name] for element_name in element_names])
    sigma_sign = {
        11: 1, 12: -1, 13: 1, 14: -1, 22: 1,
        23: -1, 24: 1, 33: 1, 34: -1, 44: 1,
    }
    source_names = []
    target_names = []
    virtual_points = {}
    virtual_crab_offsets = {}
    partner_names = {}

    for element_name, metadata in elements.items():
        ip_name = metadata['ip_name']
        mirrored_s = 2 * twiss['s', ip_name] - twiss['s', element_name]
        partner_index = int(np.argmin(np.abs(positions - mirrored_s)))
        partner_name = element_names[partner_index]
        if not np.isclose(
                positions[partner_index], mirrored_s, rtol=0, atol=1e-5):
            raise ValueError(
                f'No antisymmetric beam-beam partner found for '
                f'`{element_name}`.')

        target_name = metadata['other_element_name']
        weak_point = points[element_name]
        strong_point = copy.deepcopy(points[partner_name])
        strong_point.name = target_name
        strong_point.sz = weak_point.sz
        strong_point.p[2] = weak_point.p[2]
        strong_point.tpx *= -1
        strong_point.tpy *= -1
        if separation_bumps is not None and ip_name in separation_bumps:
            plane = separation_bumps[ip_name]
            setattr(
                strong_point, f't{plane}',
                -getattr(strong_point, f't{plane}'))
            setattr(
                strong_point, f'tp{plane}',
                -getattr(strong_point, f'tp{plane}'))
            strong_point.p[{'x': 0, 'y': 1}[plane]] += (
                2 * getattr(strong_point, f't{plane}'))

        source_names.append(partner_name)
        target_names.append(target_name)
        virtual_points[target_name] = strong_point
        virtual_crab_offsets[target_name] = crab_offsets.get(
            partner_name, {'x': 0., 'y': 0.})
        partner_names[element_name] = partner_name

    virtual_twiss = twiss.rows[source_names].cols[
        's betx bety x px y py']
    virtual_twiss.name = np.array(target_names, dtype=object)
    virtual_twiss.x = np.array([
        virtual_points[name].tx for name in target_names])
    virtual_twiss.px = np.array([
        virtual_points[name].tpx for name in target_names])
    virtual_twiss.y = np.array([
        virtual_points[name].ty for name in target_names])
    virtual_twiss.py = np.array([
        virtual_points[name].tpy for name in target_names])

    sigma_columns = [f'Sigma{name}' for name in _SIGMA_NAMES]
    virtual_covariance = covariance.rows[source_names].cols[sigma_columns]
    virtual_covariance.name = np.array(target_names, dtype=object)
    for sigma_name, sign in sigma_sign.items():
        virtual_covariance[f'Sigma{sigma_name}'] *= sign

    return (virtual_twiss, virtual_covariance, virtual_points,
            virtual_crab_offsets, partner_names)


def compute_beambeam_geometry(
        twiss_and_madpoints, element_name_cw, element_name_acw):
    """Compute the optics and geometry of one paired encounter."""
    beam = {
        orientation: _beam_data(
            twiss_and_madpoints, orientation, element_name)
        for orientation, element_name in (
            ('cw', element_name_cw), ('acw', element_name_acw))
    }
    point_cw = twiss_and_madpoints['points']['cw'][element_name_cw]
    point_acw = twiss_and_madpoints['points']['acw'][element_name_acw]
    strong_minus_weak_cw = point_acw.p - point_cw.p
    separation_cw = np.array([
        strong_minus_weak_cw @ point_cw.ex,
        strong_minus_weak_cw @ point_cw.ey,
    ])
    strong_minus_weak_acw = -strong_minus_weak_cw
    separation_acw = np.array([
        strong_minus_weak_acw @ point_acw.ex,
        strong_minus_weak_acw @ point_acw.ey,
    ])
    reference_separation = np.zeros(2)
    if twiss_and_madpoints['survey_separation']:
        reference_delta = point_cw.sp - point_acw.sp
        reference_separation = np.array([
            reference_delta @ point_cw.ex,
            reference_delta @ point_cw.ey,
        ])

    dpx_cw = point_cw.tpx - point_acw.tpx
    dpy_cw = point_cw.tpy - point_acw.tpy
    beam['cw'].update(
        separation_x=float(separation_cw[0]),
        separation_y=float(separation_cw[1]),
        dpx=float(dpx_cw), dpy=float(dpy_cw))
    beam['acw'].update(
        separation_x=float(separation_acw[0]),
        separation_y=float(separation_acw[1]),
        dpx=float(-dpx_cw), dpy=float(-dpy_cw))

    crab_offsets = twiss_and_madpoints['crab_offsets']
    for weak_orientation, strong_orientation, strong_element_name in (
            ('cw', 'acw', element_name_acw),
            ('acw', 'cw', element_name_cw)):
        strong_crab = crab_offsets[strong_orientation].get(
            strong_element_name)
        if strong_crab is not None:
            beam[weak_orientation]['separation_x'] += strong_crab['x']
            beam[weak_orientation]['separation_y'] += strong_crab['y']

    return {
        'cw': beam['cw'],
        'acw': beam['acw'],
        'separation_x': float(reference_separation[0]),
        'separation_y': float(reference_separation[1]),
    }


def _beam_data(twiss_and_madpoints, orientation, element_name):
    twiss = twiss_and_madpoints['twiss'][orientation]
    covariance = twiss_and_madpoints['covariance'][orientation]
    data = {
        coordinate: float(twiss[coordinate, element_name])
        for coordinate in ('betx', 'bety', 'x', 'px', 'y', 'py')
    }
    data['sigma'] = {
        sigma_name: float(covariance[f'Sigma{sigma_name}', element_name])
        for sigma_name in _SIGMA_NAMES
    }
    return data


def _measure_crabbing(line, crab_s_by_element, twiss, reverse):
    offsets = {}
    for element_name, s_crab in crab_s_by_element.items():
        if s_crab == 0.0:
            offsets[element_name] = {'x': 0.0, 'y': 0.0}
            continue

        print(f'Crabbing at {element_name}     ', end='\r', flush=True)
        zeta0 = -2 * s_crab if reverse else 2 * s_crab
        twiss_crab = line.twiss(method='4d', zeta0=zeta0, reverse=False)
        if reverse:
            twiss_crab = twiss_crab.reverse()
        offsets[element_name] = {
            coordinate: (twiss_crab[coordinate, element_name]
                         - twiss[coordinate, element_name])
            for coordinate in ('x', 'y')
        }
    return offsets


def find_alpha_and_phi(dpx, dpy):
    absphi = np.sqrt(dpx ** 2 + dpy ** 2) / 2.0
    if absphi < 1e-20:
        return 0.0, absphi

    if dpy >= 0.:
        if dpx >= 0:
            phi = absphi
            alpha = (np.arctan(dpy / dpx) if np.abs(dpx) >= np.abs(dpy)
                     else 0.5 * np.pi - np.arctan(dpx / dpy))
        elif np.abs(dpx) < np.abs(dpy):
            phi = absphi
            alpha = 0.5 * np.pi - np.arctan(dpx / dpy)
        else:
            phi = -absphi
            alpha = np.arctan(dpy / dpx)
    elif dpx <= 0:
        phi = -absphi
        alpha = (np.arctan(dpy / dpx) if np.abs(dpx) >= np.abs(dpy)
                 else 0.5 * np.pi - np.arctan(dpx / dpy))
    elif np.abs(dpx) <= np.abs(dpy):
        phi = -absphi
        alpha = 0.5 * np.pi - np.arctan(dpx / dpy)
    else:
        phi = absphi
        alpha = np.arctan(dpy / dpx)
    return alpha, phi


def find_bb_separations(points_weak, points_strong, names=None):
    if names is None:
        names = [f'bb_{ii}' for ii in range(len(points_weak))]

    separation_x = []
    separation_y = []
    for name, weak, strong in zip(names, points_weak, points_strong):
        delta = strong.p - weak.p
        frame_differences = [
            np.linalg.norm(weak.ex - strong.ex),
            np.linalg.norm(weak.ey - strong.ey),
            np.linalg.norm(weak.ez - strong.ez),
        ]
        if any(difference >= 1e-10 for difference in frame_differences):
            print(name, 'Reference systems are not parallel')
            frame_difference = np.linalg.norm(frame_differences)
            if frame_difference < 5e-3:
                print('Smaller that 5e-3, tolerated.')
            else:
                raise ValueError('Too large! Stopping.')
        if np.abs(np.dot(delta, weak.ez)) >= 1e-4:
            print(name, 'The beams are longitudinally shifted')
        separation_x.append(np.dot(delta, weak.ex))
        separation_y.append(np.dot(delta, weak.ey))
    return separation_x, separation_y
