# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np

from ..beam_elements.electroncloud import ElectronCloud
from ..fieldmaps.tricubicinterpolated import TriCubicInterpolatedFieldMap

import xpart as xp

from xobjects.general import _print
from xtrack.progress_indicator import progress

def get_electroncloud_fieldmap_from_h5(
        filename, zeta_max=None, buffer=None, ecloud_name="e-cloud"):
    assert buffer is not None
    import h5py
    ff = h5py.File(filename, "r")

    nx = len(ff["grid/xg"][()])
    ix1 = 0
    ix2 = nx

    ny = len(ff["grid/yg"][()])
    iy1 = 0
    iy2 = ny

    # Select a subset of slices (from -zeta_max to +zeta_max)
    if zeta_max is None:
        nz = len(ff["grid/zg"][()])
        iz1 = 0
        iz2 = nz
    else:
        zg = ff["grid/zg"][()]
        nz = len(zg)
        # add one index to make sure range is included
        min_index = np.argmin(np.abs(zg + np.abs(zeta_max))) - 1
        max_index = np.argmin(np.abs(zg - np.abs(zeta_max))) + 1

        if min_index < 0 or max_index > nz:
            raise Exception(
                f"Range ({-np.abs(zeta_max):.4f}, {np.abs(zeta_max):.4f}) not in maximum range of z_grid: ({zg[0]:.4f},{zg[-1]:.3f}) of file: {filename}")

        iz1 = min_index
        iz2 = max_index

    x_grid = ff["grid/xg"][ix1:ix2]
    y_grid = ff["grid/yg"][iy1:iy2]
    z_grid = ff["grid/zg"][iz1:iz2]

    mirror2D = ff["settings/symmetric2D"][()]
    # (in GB), 8 bytes per double-precision number
    memory_estimate = (ix2 - ix1) * (iy2 - iy1) * (iz2 - iz1) * 8 * 8 * 1.e-9
    fieldmap = TriCubicInterpolatedFieldMap(x_grid=x_grid, y_grid=y_grid, z_grid=z_grid,
                                               mirror_x=mirror2D, mirror_y=mirror2D, mirror_z=0, _buffer=buffer)

    scale = [1., fieldmap.dx, fieldmap.dy, fieldmap.dz,
             fieldmap.dx * fieldmap.dy, fieldmap.dx *
             fieldmap.dz, fieldmap.dy * fieldmap.dz,
             fieldmap.dx * fieldmap.dy * fieldmap.dz]

    ####### Optimized version of the loop in the block below. ################
    _prog = progress(range(iz1, iz2), desc=f'Reading ecloud {ecloud_name} (~ {memory_estimate:.2f} GB)')
    for iz in _prog:
        phi_slice = ff[f"slices/slice{iz}/phi"][ix1:ix2,
                                                iy1:iy2, :].transpose(1, 0, 2)
        for ll in range(8):
            phi_slice[:, :, ll] *= scale[ll]
        index_offset = 8 * nx * ny * (iz - iz1)
        len_slice = phi_slice.shape[0] * \
            phi_slice.shape[1] * phi_slice.shape[2]
        fieldmap._phi_taylor[index_offset:index_offset +
                             len_slice] = fieldmap._context.nparray_to_context_array(phi_slice.flatten())
    ##########################################################################
    # for iz in range(iz1, iz2):
    #     if (iz-iz1)/nz > kk:
    #         while (iz-iz1)/nz > kk:
    #             kk += 0.1
    #         print(f"{int(np.round(100*kk)):d}%..")
    #     phi_slice = ff[f"slices/slice{iz}/phi"][ix1:ix2, iy1:iy2,:]
    #     for iy in range(ny):
    #         for ix in range(nx):
    #             for ll in range(8):
    #                 index = ll + 8 * ix + 8 * nx * iy + 8 * nx * ny * (iz - iz1)
    #                 fieldmap._phi_taylor[index] = phi_slice[ix, iy, ll] * scale[ll]
    ##########################################################################

    return fieldmap


def insert_electronclouds(eclouds, fieldmap=None, line=None):
    assert line is not None
    insertions = []
    env = line.env
    for name in eclouds.keys():
        s = eclouds[name]["s"]
        length = 0.
        element=ElectronCloud(
                length=length,
                fieldmap=fieldmap,
                _buffer=fieldmap._buffer)
        env.elements[name] = element
        insertions.append(env.place(name, at=s))

    line.insert(insertions)

def config_electronclouds(line, twiss=None, ecloud_info=None, shift_to_closed_orbit=False,
                          subtract_dipolar_kicks=False, fieldmaps=None):
    assert twiss is not None
    assert ecloud_info is not None
    # if subtract_dipolar_kicks:
    #     dipolar_kicks = {}
    #     assert fieldmaps is not None
    #     for key in fieldmaps.keys():
    #         fieldmap = fieldmaps[key]
    #         dipolar_kicks[key] = electroncloud_dipolar_kicks_of_fieldmap(
    #             fieldmap=fieldmap, p0c=line.particle_ref.p0c)

    length_factor = line.vars['ecloud_strength'] / \
        (line.particle_ref.p0c[0] * line.particle_ref.beta0[0])
    for ii, el_name in enumerate(twiss["name"]):
        if 'ecloud' in el_name:
            # naming format is "ecloud.ecloud_type.sector.index_in_sector",
            # e.g.  ecloud.mb.78.38
            ecloud_type = el_name.split(".")[1]
            length = ecloud_info[ecloud_type][el_name]["length"] * \
                length_factor
            line.element_refs[el_name].length = length
            # line.elements[ii].length = length
            assert el_name == line.element_names[ii]

            if shift_to_closed_orbit:
                line.elements[ii].x_shift = twiss["x"][ii]
                line.elements[ii].y_shift = twiss["y"][ii]
                line.elements[ii].zeta_shift = twiss["zeta"][ii]

            if subtract_dipolar_kicks:
                temp_part = line.particle_ref.copy(_context=line._context)
                temp_part.x = twiss["x"][ii]
                temp_part.y = twiss["y"][ii]
                temp_part.zeta = twiss["zeta"][ii]
                temp_part.px = 0
                temp_part.py = 0
                temp_part.ptau = 0 #pzeta has no setter
                line.elements[ii].track(temp_part)

                ctx_to_np = line._context.nparray_from_context_array
                line.element_refs[el_name].dipolar_px_kick = ctx_to_np(temp_part.px)[0] * line.vars['ecloud_strength'] / line.vars['ecloud_strength']._value
                line.element_refs[el_name].dipolar_py_kick = ctx_to_np(temp_part.py)[0] * line.vars['ecloud_strength'] / line.vars['ecloud_strength']._value
                line.element_refs[el_name].dipolar_pzeta_kick = ctx_to_np(temp_part.pzeta)[0] * line.vars['ecloud_strength'] / line.vars['ecloud_strength']._value


def electroncloud_dipolar_kicks_of_fieldmap(fieldmap=None, p0c=None):

    assert p0c is not None
    assert fieldmap is not None

    part = xp.Particles(_context=fieldmap._context, p0c=p0c)
    ecloud = ElectronCloud(
        length=1,
        fieldmap=fieldmap,
        _buffer=fieldmap._buffer)
    ecloud.track(part)
    px = part.px[0]
    py = part.py[0]
    pzeta = part.pzeta[0]
    return [px, py, pzeta]


def full_electroncloud_setup(line=None, ecloud_info=None, filenames=None, context=None,
                             zeta_max=None, subtract_dipolar_kicks=True, shift_to_closed_orbit=True,
                             steps_r_matrix=None):

    line.build_tracker(compile=False) # To move all elements to the same buffer
    buffer = line._buffer
    line.discard_tracker()
    fieldmaps = {
        ecloud_type: get_electroncloud_fieldmap_from_h5(
            filename=filename,
            buffer=buffer,
            zeta_max=zeta_max,
            ecloud_name=ecloud_type) for (
            ecloud_type,
            filename) in filenames.items()}

    line.vars['ecloud_strength'] = 1

    for ecloud_type, fieldmap in fieldmaps.items():
        _print(f"Inserting \"{ecloud_type}\" electron clouds...")
        insert_electronclouds(
            ecloud_info[ecloud_type],
            fieldmap=fieldmap,
            line=line)

    line.build_tracker(_buffer=buffer)

    twiss_without_ecloud = line.twiss()
    config_electronclouds(
        line,
        twiss=twiss_without_ecloud,
        ecloud_info=ecloud_info,
        subtract_dipolar_kicks=subtract_dipolar_kicks,
        shift_to_closed_orbit=shift_to_closed_orbit,
        fieldmaps=fieldmaps
        )
    twiss_with_ecloud = line.twiss(steps_r_matrix=steps_r_matrix)

    return twiss_without_ecloud, twiss_with_ecloud
