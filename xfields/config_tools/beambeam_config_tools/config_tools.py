# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

"""Beam--beam encounter tables and geometry calculations.

This module contains no environment orchestration and does not populate beam
elements. The weak--strong lifecycle lives in :mod:`.weak_strong`.
"""

import copy

import numpy as np
import pandas as pd

from ._madpoint import MadPoint


_sigma_names = [11, 12, 13, 14, 22, 23, 24, 33, 34, 44]


def generate_beambeam_encounter_table(
        ip_names, num_long_range_encounters_per_side,
        bunch_spacing_zeta=None, delay_at_ips_slots=None, n_slots=None):
    """Build the logical head-on and long-range encounter description.

    The returned table is independent of the beam-beam element type and of
    head-on slicing. Each row describes one physical encounter through its IP,
    type and signed long-range index. The head-on encounter has index zero;
    positive and negative indices identify the right and left encounters.

    ``num_long_range_encounters_per_side`` can be a scalar, a sequence aligned
    with ``ip_names``, or a mapping keyed by IP name. If
    ``bunch_spacing_zeta`` is supplied, the table also contains the signed
    displacement from the IP in each line orientation. If
    ``delay_at_ips_slots`` and ``n_slots`` are supplied, it contains the bunch
    pairing offsets in both orientations.
    """
    ip_names = list(ip_names)
    if isinstance(num_long_range_encounters_per_side, dict):
        n_lr_by_ip = [num_long_range_encounters_per_side[ip]
                      for ip in ip_names]
    elif np.ndim(num_long_range_encounters_per_side) == 0:
        n_lr_by_ip = [num_long_range_encounters_per_side] * len(ip_names)
    else:
        n_lr_by_ip = list(num_long_range_encounters_per_side)
        if len(n_lr_by_ip) != len(ip_names):
            raise ValueError(
                '`num_long_range_encounters_per_side` must have one entry '
                'per IP.')

    encounters = []
    for ip_name, n_lr_value in zip(ip_names, n_lr_by_ip):
        n_lr = int(n_lr_value)
        if n_lr != n_lr_value or n_lr < 0:
            raise ValueError(
                '`num_long_range_encounters_per_side` entries must be '
                'non-negative integers.')
        encounters.append({
            'ip_name': ip_name,
            'encounter_type': 'head_on',
            'identifier': 0,
        })
        for identifier in range(1, n_lr + 1):
            encounters.append({
                'ip_name': ip_name,
                'encounter_type': 'long_range',
                'identifier': identifier,
            })
            encounters.append({
                'ip_name': ip_name,
                'encounter_type': 'long_range',
                'identifier': -identifier,
            })

    table = pd.DataFrame(encounters, columns=(
        'ip_name', 'encounter_type', 'identifier'))

    if bunch_spacing_zeta is not None:
        displacement = table['identifier'] * bunch_spacing_zeta / 2
        table['s_from_ip_cw'] = displacement
        table['s_from_ip_acw'] = -displacement

    if delay_at_ips_slots is not None:
        if n_slots is None:
            raise ValueError(
                '`n_slots` is required with `delay_at_ips_slots`.')
        table = _add_beambeam_pairing_offsets(
            table, ip_names, delay_at_ips_slots, n_slots)

    return table


def _add_beambeam_pairing_offsets(
        encounter_table, ip_names, delay_at_ips_slots, n_slots):
    if isinstance(delay_at_ips_slots, dict):
        delay_by_ip = {ip: delay_at_ips_slots[ip] for ip in ip_names}
    else:
        delay_at_ips_slots = list(delay_at_ips_slots)
        if len(delay_at_ips_slots) != len(ip_names):
            raise ValueError(
                '`delay_at_ips_slots` must have one entry per IP.')
        delay_by_ip = dict(zip(ip_names, delay_at_ips_slots))

    table = encounter_table.copy()
    delay_cw = np.array([
        delay_by_ip[ip] + identifier
        for ip, identifier in zip(table['ip_name'], table['identifier'])])
    delay_acw = np.array([
        np.mod(n_slots - delay_by_ip[ip], n_slots) - identifier
        for ip, identifier in zip(table['ip_name'], table['identifier'])])
    table['delay_in_slots_cw'] = delay_cw.astype(int)
    table['delay_in_slots_acw'] = delay_acw.astype(int)
    return table

def get_counter_rotating(bb_df):

    c_bb_df = pd.DataFrame(index=bb_df.index)

    c_bb_df['beam'] = bb_df['beam']
    c_bb_df['other_beam'] = bb_df['other_beam']
    c_bb_df['ip_name'] = bb_df['ip_name']
    c_bb_df['label'] = bb_df['label']
    c_bb_df['identifier'] = bb_df['identifier']
    if 'elementClass' in bb_df.columns:
        c_bb_df['elementClass'] = bb_df['elementClass']
    c_bb_df['elementName'] = bb_df['elementName']
    c_bb_df['self_num_particles'] = bb_df['self_num_particles']
    c_bb_df['other_num_particles'] = bb_df['other_num_particles']
    c_bb_df['self_particle_charge'] = bb_df['self_particle_charge']
    c_bb_df['other_particle_charge'] = bb_df['other_particle_charge']
    c_bb_df['other_elementName'] = bb_df['other_elementName']

    if 'atPosition' in bb_df.columns:
        c_bb_df['atPosition'] = bb_df['atPosition'] * (-1.)

    c_bb_df['elementDefinition'] = np.nan
    c_bb_df['elementInstallation'] = np.nan

    c_bb_df['self_lab_position'] = np.nan
    c_bb_df['other_lab_position'] = np.nan

    c_bb_df['self_Sigma_11'] = bb_df['self_Sigma_11'] * (-1.) * (-1.)                  # x * x
    c_bb_df['self_Sigma_12'] = bb_df['self_Sigma_12'] * (-1.) * (-1.) * (-1.)          # x * dx / ds
    c_bb_df['self_Sigma_13'] = bb_df['self_Sigma_13'] * (-1.)                          # x * y
    c_bb_df['self_Sigma_14'] = bb_df['self_Sigma_14'] * (-1.) * (-1.)                  # x * dy / ds
    c_bb_df['self_Sigma_22'] = bb_df['self_Sigma_22'] * (-1.) * (-1.) * (-1.) * (-1.)  # dx / ds * dx / ds
    c_bb_df['self_Sigma_23'] = bb_df['self_Sigma_23'] * (-1.) * (-1.)                  # dx / ds * y
    c_bb_df['self_Sigma_24'] = bb_df['self_Sigma_24'] * (-1.) * (-1.) * (-1.)          # dx / ds * dy / ds
    c_bb_df['self_Sigma_33'] = bb_df['self_Sigma_33']                                  # y * y
    c_bb_df['self_Sigma_34'] = bb_df['self_Sigma_34'] * (-1.)                          # y * dy / ds
    c_bb_df['self_Sigma_44'] = bb_df['self_Sigma_44'] * (-1.) * (-1.)                  # dy / ds * dy / ds

    c_bb_df['other_Sigma_11'] = bb_df['other_Sigma_11'] * (-1.) * (-1.)
    c_bb_df['other_Sigma_12'] = bb_df['other_Sigma_12'] * (-1.) * (-1.) * (-1.)
    c_bb_df['other_Sigma_13'] = bb_df['other_Sigma_13'] * (-1.)
    c_bb_df['other_Sigma_14'] = bb_df['other_Sigma_14'] * (-1.) * (-1.)
    c_bb_df['other_Sigma_22'] = bb_df['other_Sigma_22'] * (-1.) * (-1.) * (-1.) * (-1.)
    c_bb_df['other_Sigma_23'] = bb_df['other_Sigma_23'] * (-1.) * (-1.)
    c_bb_df['other_Sigma_24'] = bb_df['other_Sigma_24'] * (-1.) * (-1.) * (-1.)
    c_bb_df['other_Sigma_33'] = bb_df['other_Sigma_33']
    c_bb_df['other_Sigma_34'] = bb_df['other_Sigma_34'] * (-1.)
    c_bb_df['other_Sigma_44'] = bb_df['other_Sigma_44'] * (-1.) * (-1.)

    c_bb_df['other_relativistic_beta']=bb_df['other_relativistic_beta']
    c_bb_df['separation_x'] = bb_df['separation_x'] * (-1.)
    c_bb_df['separation_y'] = bb_df['separation_y']

    c_bb_df['dpx'] = bb_df['dpx'] * (-1.) * (-1.)
    c_bb_df['dpy'] = bb_df['dpy'] * (-1.)

    if 'self_x_crab' in c_bb_df.columns:
        # Crab cavities are accounted for
        c_bb_df['separation_x_no_crab'] = bb_df['separation_x_no_crab'] * (-1.)
        c_bb_df['separation_y_no_crab'] = bb_df['separation_y_no_crab']
        for ww in ['self', 'other']:
            c_bb_df[f'{ww}_x_crab'] = bb_df[f'{ww}_x_crab'] * (-1)
            c_bb_df[f'{ww}_px_crab'] = bb_df[f'{ww}_px_crab'] * (-1) * (-1)
            c_bb_df[f'{ww}_y_crab'] = bb_df[f'{ww}_y_crab']
            c_bb_df[f'{ww}_py_crab'] = bb_df[f'{ww}_py_crab'] * (-1)

    # Compute phi and alpha from dpx and dpy
    compute_local_crossing_angle_and_plane(c_bb_df)

    return c_bb_df

def compute_geometry_and_optics(
        bb_df=None, xsuite_twiss=None, xsuite_survey=None,
        xsuite_sigmas=None, shared_geometry=None, orientation=None):


    # Get positions of the bb encounters (absolute from survey), closed orbit
    # and orientation of the local reference system (MadPoint objects)

    # Add empty columns to dataframe
    bb_df['self_lab_position'] = None
    bb_df['s'] = None
    bb_df['s_ip'] = None
    bb_df['self_Sigma_11'] = None
    bb_df['self_Sigma_12'] = None
    bb_df['self_Sigma_13'] = None
    bb_df['self_Sigma_14'] = None
    bb_df['self_Sigma_22'] = None
    bb_df['self_Sigma_23'] = None
    bb_df['self_Sigma_24'] = None
    bb_df['self_Sigma_33'] = None
    bb_df['self_Sigma_34'] = None
    bb_df['self_Sigma_44'] = None

    if shared_geometry is not None:
        assert orientation in ('cw', 'acw')
        shared_by_element = shared_geometry.set_index(
            f'element_name_{orientation}')

    for ele_name in bb_df.index.values:
        ip_name = bb_df['ip_name'][ele_name]

        if shared_geometry is None:
            bb_df.loc[ele_name, 'self_lab_position'] = MadPoint(
                ele_name, None, use_twiss=True, use_survey=True,
                xsuite_survey=xsuite_survey[ip_name],
                xsuite_twiss=xsuite_twiss)

        bb_df.loc[ele_name, 's'] = xsuite_twiss['s', ele_name]
        bb_df.loc[ele_name, 's_ip'] = xsuite_twiss['s', ip_name]

        # Get the sigmas for the element. The standard two-beam path consumes
        # the shared element-independent geometry; antisymmetry retains its
        # established single-line covariance path for now.
        if shared_geometry is not None:
            for ss in _sigma_names:
                bb_df.loc[ele_name, f'self_Sigma_{ss}'] = \
                    shared_by_element.loc[ele_name][
                        f'Sigma_{ss}_{orientation}']
        else:
            i_sigma = np.where(np.array(xsuite_sigmas.name) == ele_name)[0][0]
            for ss in _sigma_names:
                bb_df.loc[ele_name, f'self_Sigma_{ss}'] = xsuite_sigmas[
                                                        f'Sigma{ss}'][i_sigma]


def compute_beambeam_geometry(
        encounter_table, line_cw, line_acw,
        element_names_cw, element_names_acw,
        nemitt_x, nemitt_y, survey_separation=True,
        twiss_cw=None, twiss_acw=None, acw_is_reversed=False):
    """Compute element-independent optics and survey encounter geometry.

    ``encounter_table`` has one row per paired observation instance. A
    rigid-bunch row represents a physical encounter; a conventional caller can
    expand one logical head-on encounter into several slice rows. The two
    element-name sequences identify the observation element representing each
    row in the clockwise and anticlockwise lines. Element classes and beam-beam
    kick conventions deliberately remain outside this helper.

    Precomputed Twiss tables can be supplied when a caller needs explicit
    orientation conventions. Set ``acw_is_reversed`` when ``twiss_acw`` has
    been reversed from the stored B4 line to the physical ACW convention.
    Returns a copy of the encounter table augmented with element names,
    closed-orbit coordinates, Twiss beta functions, transverse beam
    covariances and separation geometry, together with the two Twiss tables.
    """
    geometry = encounter_table.reset_index(drop=True).copy()
    element_names_cw = list(element_names_cw)
    element_names_acw = list(element_names_acw)
    n_encounters = len(geometry)
    if (len(element_names_cw) != n_encounters
            or len(element_names_acw) != n_encounters):
        raise ValueError(
            'The CW and ACW element-name sequences must contain one entry '
            'per encounter.')

    geometry['element_name_cw'] = element_names_cw
    geometry['element_name_acw'] = element_names_acw

    twisses = {
        'cw': line_cw.twiss() if twiss_cw is None else twiss_cw,
        'acw': line_acw.twiss() if twiss_acw is None else twiss_acw,
    }
    covariances = {
        orientation: twiss.get_beam_covariance(
            nemitt_x=nemitt_x, nemitt_y=nemitt_y)
        for orientation, twiss in twisses.items()
    }
    for orientation, names in (
            ('cw', element_names_cw), ('acw', element_names_acw)):
        twiss = twisses[orientation]
        covariance = covariances[orientation]
        geometry[f'betx_{orientation}'] = [
            float(twiss['betx', name]) for name in names]
        geometry[f'bety_{orientation}'] = [
            float(twiss['bety', name]) for name in names]
        for coordinate in ('x', 'px', 'y', 'py'):
            geometry[f'{coordinate}_{orientation}'] = [
                float(twiss[coordinate, name]) for name in names]
        for sigma_name in _sigma_names:
            geometry[f'Sigma_{sigma_name}_{orientation}'] = [
                float(covariance[f'Sigma{sigma_name}', name])
                for name in names]

    # Rigid-bunch convention: CW reference trajectory minus ACW reference
    # trajectory, expressed in the CW encounter frame.
    separation_x = np.zeros(n_encounters)
    separation_y = np.zeros(n_encounters)

    # Conventional weak-strong convention: strong beam minus weak beam,
    # including the design closed orbits, expressed in each weak-beam frame.
    separation_x_cw = (
        geometry['x_acw'] - geometry['x_cw']).to_numpy(copy=True)
    separation_y_cw = (
        geometry['y_acw'] - geometry['y_cw']).to_numpy(copy=True)
    separation_x_acw = -separation_x_cw
    separation_y_acw = -separation_y_cw
    geometry['dpx_cw'] = geometry['px_cw'] - geometry['px_acw']
    geometry['dpy_cw'] = geometry['py_cw'] - geometry['py_acw']
    geometry['dpx_acw'] = -geometry['dpx_cw']
    geometry['dpy_acw'] = -geometry['dpy_cw']

    if survey_separation:
        reverse_global = np.diag([-1., 1., -1.])
        reverse_local = (np.diag([-1., 1., -1.])
                         if acw_is_reversed else np.eye(3))
        for ip_name in geometry['ip_name'].unique():
            at_ip = geometry['ip_name'] == ip_name
            row_indices = np.flatnonzero(at_ip.to_numpy())
            names_cw = geometry.loc[at_ip, 'element_name_cw'].tolist()
            names_acw = geometry.loc[at_ip, 'element_name_acw'].tolist()
            frames_cw = _local_survey(line_cw, ip_name, names_cw)
            frames_acw = _local_survey(line_acw, ip_name, names_acw)

            for row_index, name_cw, name_acw in zip(
                    row_indices, names_cw, names_acw):
                position_cw, frame_cw = frames_cw[name_cw]
                position_acw_stored, frame_acw_stored = frames_acw[name_acw]
                position_acw = reverse_global @ position_acw_stored
                frame_acw = (
                    reverse_global @ frame_acw_stored @ reverse_local)

                reference_delta = position_cw - position_acw
                separation_x[row_index] = reference_delta @ frame_cw[:, 0]
                separation_y[row_index] = reference_delta @ frame_cw[:, 1]

                orbit_cw = (frame_cw[:, 0] * geometry.at[row_index, 'x_cw']
                            + frame_cw[:, 1]
                            * geometry.at[row_index, 'y_cw'])
                orbit_acw = (
                    frame_acw[:, 0] * geometry.at[row_index, 'x_acw']
                    + frame_acw[:, 1] * geometry.at[row_index, 'y_acw'])
                total_cw = position_cw + orbit_cw
                total_acw = position_acw + orbit_acw

                strong_minus_weak_cw = total_acw - total_cw
                separation_x_cw[row_index] = (
                    strong_minus_weak_cw @ frame_cw[:, 0])
                separation_y_cw[row_index] = (
                    strong_minus_weak_cw @ frame_cw[:, 1])
                strong_minus_weak_acw = -strong_minus_weak_cw
                separation_x_acw[row_index] = (
                    strong_minus_weak_acw @ frame_acw[:, 0])
                separation_y_acw[row_index] = (
                    strong_minus_weak_acw @ frame_acw[:, 1])

    geometry['separation_x'] = separation_x
    geometry['separation_y'] = separation_y
    geometry['separation_x_cw'] = separation_x_cw
    geometry['separation_y_cw'] = separation_y_cw
    geometry['separation_x_acw'] = separation_x_acw
    geometry['separation_y_acw'] = separation_y_acw
    return geometry, twisses


def _local_survey(line, ref_name, names):
    """Survey nearby elements in ``ref_name``'s local frame.

    Rolling the sequence around the reference avoids crossing the line seam
    and importing a whole-ring closure defect into local encounter geometry.
    """
    from xtrack.survey import get_survey

    table = line.get_table(attr=True)
    elements = list(line._elements)
    n_elements = len(elements)
    length = np.array(table.length[:n_elements], dtype=float)
    length[~np.array(table.isthick[:n_elements], dtype=bool)] = 0.
    angle = np.array(table.angle[:n_elements], dtype=float)
    tilt = np.array(table.rot_s_rad[:n_elements], dtype=float)

    index = {name: ii for ii, name in enumerate(line.element_names)}
    shift = (index[ref_name] - n_elements // 2) % n_elements
    rolled = {
        name: (index[name] - shift) % n_elements
        for name in list(names) + [ref_name]
    }
    lo, hi = min(rolled.values()), max(rolled.values())

    def window(array):
        return np.concatenate([array[shift:], array[:shift]])[lo:hi + 1]

    ordered_elements = elements[shift:] + elements[:shift]
    positions, frames = get_survey(
        elements=ordered_elements[lo:hi + 1],
        X0=0., Y0=0., Z0=0., theta0=0., phi0=0., psi0=0.,
        drift_length=window(length), angle=window(angle), tilt=window(tilt),
        element0=rolled[ref_name] - lo)

    return {
        name: (np.array(positions[rolled[name] - lo]),
               np.array(frames[rolled[name] - lo]))
        for name in names
    }


def get_partner_position_and_optics(
        bb_df_b1, bb_df_b2, crab_strong_beam,
        include_lab_positions=True):

    dict_dfs = {'b1': bb_df_b1, 'b2': bb_df_b2}

    for self_beam_nn in ['b1', 'b2']:

        self_df = dict_dfs[self_beam_nn]
        self_df['other_num_particles'] = None
        self_df['other_particle_charge'] = None
        self_df['other_relativistic_beta'] = None
        for ee in self_df.index:
            other_beam_nn = self_df.loc[ee, 'other_beam']
            other_df = dict_dfs[other_beam_nn]
            other_ee = self_df.loc[ee, 'other_elementName']

            if include_lab_positions:
                # Get position of the other beam in its own survey
                other_lab_position = copy.deepcopy(
                    other_df.loc[other_ee, 'self_lab_position'])
                self_df.loc[ee, 'other_lab_position'] = other_lab_position

            # Get sigmas of the other beam in its own survey
            for ss in _sigma_names:
                self_df.loc[ee, f'other_Sigma_{ss}'] = other_df.loc[other_ee, f'self_Sigma_{ss}']
            # Get charge of other beam
            self_df.loc[ee, 'other_num_particles'] = other_df.loc[other_ee, 'self_num_particles']
            self_df.loc[ee, 'other_particle_charge'] = other_df.loc[other_ee, 'self_particle_charge']
            self_df.loc[ee, 'other_relativistic_beta'] = other_df.loc[other_ee, 'self_relativistic_beta']

            if crab_strong_beam:
                for coord in ['x', 'y']:
                    self_df.loc[ee, f'other_{coord}_crab'] = other_df.loc[
                        other_ee, f'self_{coord}_crab']

def get_partner_position_and_optics_antisymmetry(bb_df, crab_strong_beam,
                    separation_bumps=None):

    bb_df['other_num_particles'] = None
    bb_df['other_particle_charge'] = None
    bb_df['other_relativistic_beta'] = None
    for ee in bb_df.index:

        ds = bb_df.loc[ee, 's'] - bb_df.loc[ee, 's_ip']
        s_antisim = bb_df.loc[ee, 's_ip'] - ds
        i_antisym = np.argmin(np.abs(bb_df.s - s_antisim))
        other_ee = bb_df.index[i_antisym]

        assert np.isclose(
            bb_df.loc[other_ee, 's'], s_antisim, rtol=0, atol=1e-5)

        position_ee = bb_df.loc[ee, 'self_lab_position']
        position_other_ee = copy.deepcopy(
            bb_df.loc[other_ee, 'self_lab_position'])
        # Assuming survey has been made starting from the IP and neglecting
        # angle between the two surveys
        position_other_ee.sz = position_ee.sz # longitudinal component
        position_other_ee.p[2] = position_ee.p[2] # longitudinal component
        position_other_ee.tpx *= -1 # anti-symmetry
        position_other_ee.tpy *= -1 # anti-symmetry

        if separation_bumps is not None:
            if bb_df.loc[ee, 'ip_name'] in separation_bumps:
                sep_plane = separation_bumps[bb_df.loc[ee, 'ip_name']]
                setattr(position_other_ee, f't{sep_plane}',
                    -getattr(position_other_ee, f't{sep_plane}'))
                setattr(position_other_ee, f'tp{sep_plane}',
                    -getattr(position_other_ee, f'tp{sep_plane}'))
                position_other_ee.p[{'x': 0, 'y': 1}[sep_plane]] += (
                    2 * getattr(position_other_ee, f't{sep_plane}')
                    # 2 is to compensate for the fact that the orbit was already
                    # added with the wrong sign
                )

        # Store positions
        bb_df.loc[ee, 'other_lab_position'] = position_other_ee

        # Get sigmas of the other beam (signs come from anti-symmetry)
        bb_df.loc[ee, 'other_Sigma_11'] = bb_df.loc[other_ee, 'self_Sigma_11']
        bb_df.loc[ee, 'other_Sigma_12'] = -bb_df.loc[other_ee, 'self_Sigma_12']
        bb_df.loc[ee, 'other_Sigma_13'] = bb_df.loc[other_ee, 'self_Sigma_13']
        bb_df.loc[ee, 'other_Sigma_14'] = -bb_df.loc[other_ee, 'self_Sigma_14']
        bb_df.loc[ee, 'other_Sigma_22'] = bb_df.loc[other_ee, 'self_Sigma_22']
        bb_df.loc[ee, 'other_Sigma_23'] = -bb_df.loc[other_ee, 'self_Sigma_23']
        bb_df.loc[ee, 'other_Sigma_24'] = bb_df.loc[other_ee, 'self_Sigma_24']
        bb_df.loc[ee, 'other_Sigma_33'] = bb_df.loc[other_ee, 'self_Sigma_33']
        bb_df.loc[ee, 'other_Sigma_34'] = -bb_df.loc[other_ee, 'self_Sigma_34']
        bb_df.loc[ee, 'other_Sigma_44'] = bb_df.loc[other_ee, 'self_Sigma_44']

        # Get charge of other beam
        bb_df.loc[ee, 'other_num_particles'] = bb_df.loc[other_ee, 'self_num_particles']
        bb_df.loc[ee, 'other_particle_charge'] = bb_df.loc[other_ee, 'self_particle_charge']
        bb_df.loc[ee, 'other_relativistic_beta'] = bb_df.loc[other_ee, 'self_relativistic_beta']

        if crab_strong_beam:
            for coord in ['x', 'y']:
                bb_df.loc[ee, f'other_{coord}_crab'] = bb_df.loc[
                    other_ee, f'self_{coord}_crab']

def compute_dpx_dpy(bb_df):
    # Defined as (weak) - (strong)
    for ee in bb_df.index:
        dpx = (bb_df.loc[ee, 'self_lab_position'].tpx
                - bb_df.loc[ee, 'other_lab_position'].tpx)
        dpy = (bb_df.loc[ee, 'self_lab_position'].tpy
                - bb_df.loc[ee, 'other_lab_position'].tpy)

        bb_df.loc[ee, 'dpx'] = dpx
        bb_df.loc[ee, 'dpy'] = dpy

def compute_local_crossing_angle_and_plane(bb_df):

    for ee in bb_df.index:
        alpha, phi = find_alpha_and_phi(
                bb_df.loc[ee, 'dpx'], bb_df.loc[ee, 'dpy'])

        bb_df.loc[ee, 'alpha'] = alpha
        bb_df.loc[ee, 'phi'] = phi

def find_alpha_and_phi(dpx, dpy):

    absphi = np.sqrt(dpx ** 2 + dpy ** 2) / 2.0

    if absphi < 1e-20:
        phi = absphi
        alpha = 0.0
    else:
        if dpy>=0.:
            if dpx>=0:
                # First quadrant
                if np.abs(dpx) >= np.abs(dpy):
                    # First octant
                    phi = absphi
                    alpha = np.arctan(dpy/dpx)
                else:
                    # Second octant
                    phi = absphi
                    alpha = 0.5*np.pi - np.arctan(dpx/dpy)
            else: #dpx<0
                # Second quadrant
                if np.abs(dpx) <  np.abs(dpy):
                    # Third octant
                    phi = absphi
                    alpha = 0.5*np.pi - np.arctan(dpx/dpy)
                else:
                    # Forth  octant
                    phi = -absphi
                    alpha = np.arctan(dpy/dpx)
        else: #dpy<0
            if dpx<=0:
                # Third quadrant
                if np.abs(dpx) >= np.abs(dpy):
                    # Fifth octant
                    phi = -absphi
                    alpha = np.arctan(dpy/dpx)
                else:
                    # Sixth octant
                    phi = -absphi
                    alpha = 0.5*np.pi - np.arctan(dpx/dpy)
            else: #dpx>0
                # Forth quadrant
                if np.abs(dpx) <= np.abs(dpy):
                    # Seventh octant
                    phi = -absphi
                    alpha = 0.5*np.pi - np.arctan(dpx/dpy)
                else:
                    # Eighth octant
                    phi = absphi
                    alpha = np.arctan(dpy/dpx)

    return alpha, phi


def find_bb_separations(points_weak, points_strong, names=None):

    if names is None:
        names = ["bb_%d" % ii for ii in range(len(points_weak))]

    sep_x = []
    sep_y = []
    for i_bb, name_bb in enumerate(names):

        pbw = points_weak[i_bb]
        pbs = points_strong[i_bb]

        # Find vws
        vbb_ws = points_strong[i_bb].p - points_weak[i_bb].p

        # Check that the two reference system are parallel
        try:
            assert np.linalg.norm(pbw.ex - pbs.ex) < 1e-10
            assert np.linalg.norm(pbw.ey - pbs.ey) < 1e-10
            assert np.linalg.norm(pbw.ez - pbs.ez) < 1e-10
        except AssertionError:
            print(name_bb, "Reference systems are not parallel")
            if (
                np.sqrt(
                    np.linalg.norm(pbw.ex - pbs.ex) ** 2
                    + np.linalg.norm(pbw.ey - pbs.ey) ** 2
                    + np.linalg.norm(pbw.ez - pbs.ez) ** 2
                )
                < 5e-3
            ):
                print("Smaller that 5e-3, tolerated.")
            else:
                raise ValueError("Too large! Stopping.")

        # Check that there is no longitudinal separation
        try:
            assert np.abs(np.dot(vbb_ws, pbw.ez)) < 1e-4
        except AssertionError:
            print(name_bb, "The beams are longitudinally shifted")

        # Find separations
        sep_x.append(np.dot(vbb_ws, pbw.ex))
        sep_y.append(np.dot(vbb_ws, pbw.ey))

    return sep_x, sep_y
