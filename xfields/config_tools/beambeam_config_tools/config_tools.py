# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

"""Shared optics and survey analysis for beam--beam configuration."""

import numpy as np


_SIGMA_NAMES = (11, 12, 13, 14, 22, 23, 24, 33, 34, 44)
BEAMBEAM_CONFIG_KEY = 'xfields_beambeam'
BEAMBEAM_CONFIG_VERSION = 1


def prepare_beambeam_analysis(
        line_cw, line_acw, element_names_by_ip, nemitt_x, nemitt_y,
        survey_separation=True, twiss_cw=None, twiss_acw=None,
        acw_is_reversed=False):
    """Prepare the reusable tables needed to analyse individual encounters.

    ``element_names_by_ip`` is indexed first by ``'cw'`` / ``'acw'`` and
    then by IP name. No per-encounter results are assembled or retained here.
    """
    twiss = {
        'cw': line_cw.twiss() if twiss_cw is None else twiss_cw,
        'acw': line_acw.twiss() if twiss_acw is None else twiss_acw,
    }
    covariance = {
        orientation: table.get_beam_covariance(
            nemitt_x=nemitt_x, nemitt_y=nemitt_y)
        for orientation, table in twiss.items()
    }
    survey = None
    if survey_separation:
        survey = {'cw': {}, 'acw': {}}
        for orientation, line in (('cw', line_cw), ('acw', line_acw)):
            for ip_name, names in element_names_by_ip[orientation].items():
                line_survey = line.survey(element0=ip_name)
                region_survey = line_survey.rows[[ip_name, *names]]
                region_span = np.ptp(region_survey.s)
                assert 2 * region_span <= line_survey.s[-1], (
                    f'The beam-beam region around {ip_name!r} wraps across '
                    'the line boundary, which is not supported.')
                survey[orientation][ip_name] = (
                    region_survey.rows[names].cols['XYZ E_matrix'])

    return {
        'twiss': twiss,
        'covariance': covariance,
        'survey': survey,
        'acw_is_reversed': bool(acw_is_reversed),
    }


def compute_beambeam_geometry(
        analysis, ip_name, element_name_cw, element_name_acw):
    """Compute the optics and geometry of one paired encounter."""
    beam = {
        orientation: _beam_data(analysis, orientation, element_name)
        for orientation, element_name in (
            ('cw', element_name_cw), ('acw', element_name_acw))
    }

    # Weak--strong convention: strong minus weak, expressed in the weak
    # beam's frame. This is sufficient when survey geometry is disabled.
    separation_cw = np.array([
        beam['acw']['x'] - beam['cw']['x'],
        beam['acw']['y'] - beam['cw']['y'],
    ])
    separation_acw = -separation_cw
    reference_separation = np.zeros(2)

    if analysis['survey'] is not None:
        survey_cw = analysis['survey']['cw'][ip_name]
        survey_acw = analysis['survey']['acw'][ip_name]
        position_cw = survey_cw['XYZ', element_name_cw]
        frame_cw = survey_cw['E_matrix', element_name_cw]
        position_acw_stored = survey_acw['XYZ', element_name_acw]
        frame_acw_stored = survey_acw['E_matrix', element_name_acw]

        reverse_global = np.diag([-1., 1., -1.])
        reverse_local = (np.diag([-1., 1., -1.])
                         if analysis['acw_is_reversed'] else np.eye(3))
        position_acw = reverse_global @ position_acw_stored
        frame_acw = reverse_global @ frame_acw_stored @ reverse_local

        reference_delta = position_cw - position_acw
        reference_separation = np.array([
            reference_delta @ frame_cw[:, 0],
            reference_delta @ frame_cw[:, 1],
        ])

        orbit_cw = (frame_cw[:, 0] * beam['cw']['x']
                    + frame_cw[:, 1] * beam['cw']['y'])
        orbit_acw = (frame_acw[:, 0] * beam['acw']['x']
                     + frame_acw[:, 1] * beam['acw']['y'])
        strong_minus_weak_cw = (
            position_acw + orbit_acw - position_cw - orbit_cw)
        separation_cw = np.array([
            strong_minus_weak_cw @ frame_cw[:, 0],
            strong_minus_weak_cw @ frame_cw[:, 1],
        ])
        strong_minus_weak_acw = -strong_minus_weak_cw
        separation_acw = np.array([
            strong_minus_weak_acw @ frame_acw[:, 0],
            strong_minus_weak_acw @ frame_acw[:, 1],
        ])

    dpx_cw = beam['cw']['px'] - beam['acw']['px']
    dpy_cw = beam['cw']['py'] - beam['acw']['py']
    beam['cw'].update(
        separation_x=float(separation_cw[0]),
        separation_y=float(separation_cw[1]),
        dpx=float(dpx_cw), dpy=float(dpy_cw))
    beam['acw'].update(
        separation_x=float(separation_acw[0]),
        separation_y=float(separation_acw[1]),
        dpx=float(-dpx_cw), dpy=float(-dpy_cw))
    return {
        'cw': beam['cw'],
        'acw': beam['acw'],
        'separation_x': float(reference_separation[0]),
        'separation_y': float(reference_separation[1]),
    }


def _beam_data(analysis, orientation, element_name):
    twiss = analysis['twiss'][orientation]
    covariance = analysis['covariance'][orientation]
    data = {
        coordinate: float(twiss[coordinate, element_name])
        for coordinate in ('betx', 'bety', 'x', 'px', 'y', 'py')
    }
    data['sigma'] = {
        sigma_name: float(covariance[f'Sigma{sigma_name}', element_name])
        for sigma_name in _SIGMA_NAMES
    }
    return data


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
