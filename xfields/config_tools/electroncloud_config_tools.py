import numpy as np

import xfields as xf
import xpart as xp
import xtrack as xt


def get_electroncloud_fieldmap_from_h5(
        filename, tau_max=None, buffer=None, ecloud_name="e-cloud"):
    assert buffer is not None
    import h5py
    ff = h5py.File(filename, "r")

    nx = len(ff["grid/xg"][()])
    ix1 = 0
    ix2 = nx

    ny = len(ff["grid/yg"][()])
    iy1 = 0
    iy2 = ny

    # Select a subset of slices (from -tau_max to +tau_max)
    if tau_max is None:
        nz = len(ff["grid/zg"][()])
        iz1 = 0
        iz2 = nz
    else:
        zg = ff["grid/zg"][()]
        nz = len(zg)
        # add one index to make sure range is included
        min_index = np.argmin(np.abs(zg + np.abs(tau_max))) - 1
        max_index = np.argmin(np.abs(zg - np.abs(tau_max))) + 1

        if min_index < 0 or max_index > nz:
            raise Exception(
                f"Range ({-np.abs(tau_max):.4f}, {np.abs(tau_max):.4f}) not in maximum range of z_grid: ({zg[0]:.4f},{zg[-1:]:.3f}) of file: {filename}")

        iz1 = min_index
        iz2 = max_index

    x_grid = ff["grid/xg"][ix1:ix2]
    y_grid = ff["grid/yg"][iy1:iy2]
    z_grid = ff["grid/zg"][iz1:iz2]

    mirror2D = ff["settings/symmetric2D"][()]
    # (in GB), 8 bytes per double-precision number
    memory_estimate = (ix2 - ix1) * (iy2 - iy1) * (iz2 - iz1) * 8 * 8 * 1.e-9
    print(f"Creating fieldmap... (Memory estimate = {memory_estimate:.2f} GB)")
    fieldmap = xf.TriCubicInterpolatedFieldMap(x_grid=x_grid, y_grid=y_grid, z_grid=z_grid,
                                               mirror_x=mirror2D, mirror_y=mirror2D, mirror_z=0, _buffer=buffer)
    print(f"Reading {ecloud_name}: ")
    kk = 0.
    scale = [1., fieldmap.dx, fieldmap.dy, fieldmap.dz,
             fieldmap.dx * fieldmap.dy, fieldmap.dx *
             fieldmap.dz, fieldmap.dy * fieldmap.dz,
             fieldmap.dx * fieldmap.dy * fieldmap.dz]

    ####### Optimized version of the loop in the block below. ################
    for iz in range(iz1, iz2):
        if (iz - iz1) / (iz2 - iz1) > kk:
            while (iz - iz1) / (iz2 - iz1) > kk:
                kk += 0.2
            print(f"{int(np.round(100*kk)):d}%..")
        phi_slice = ff[f"slices/slice{iz}/phi"][ix1:ix2,
                                                iy1:iy2, :].transpose(1, 0, 2)
        for ll in range(8):
            phi_slice[:, :, ll] *= scale[ll]
        index_offset = 8 * nx * ny * (iz - iz1)
        len_slice = phi_slice.shape[0] * \
            phi_slice.shape[1] * phi_slice.shape[2]
        fieldmap._phi_taylor[index_offset:index_offset +
                             len_slice] = phi_slice.flatten()
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
    for name in eclouds.keys():
        s = eclouds[name]["s"]
        length = 0.
        line.insert_element(
            element=xf.ElectronCloud(
                length=length,
                fieldmap=fieldmap,
                _buffer=fieldmap._buffer),
            name=name,
            at_s=s)


def config_electronclouds(line, twiss=None, ecloud_info=None, shift_to_closed_orbit=False,
                          subtract_dipolar_kicks=False, fieldmaps=None, ecloud_strength=1.):
    assert twiss is not None
    assert ecloud_info is not None
    if subtract_dipolar_kicks:
        dipolar_kicks = {}
        assert fieldmaps is not None
        for key in fieldmaps.keys():
            fieldmap = fieldmaps[key]
            dipolar_kicks[key] = electroncloud_dipolar_kicks_of_fieldmap(
                fieldmap=fieldmap, p0c=line.particle_ref.p0c)

    length_factor = ecloud_strength / \
        (line.particle_ref.p0c * line.particle_ref.beta0)
    part = twiss["particle_on_co"].copy()
    for ii, el_name in enumerate(twiss["name"]):
        if 'ecloud' in el_name:
            # naming format is "ecloud.ecloud_type.sector.index_in_sector",
            # e.g.  ecloud.mb.78.38
            ecloud_type = el_name.split(".")[1]
            length = ecloud_info[ecloud_type][el_name]["length"] * \
                length_factor
            line.elements[ii].length = length
            assert el_name == line.element_names[ii]

            if shift_to_closed_orbit:
                line.elements[ii].x_shift = twiss["x"][ii]
                line.elements[ii].y_shift = twiss["y"][ii]
                part.delta = twiss["delta"][ii]
                part.zeta = twiss["zeta"][ii]
                line.elements[ii].tau_shift = part.zeta[0] / \
                    (part.beta0[0] * part.rvv[0])

            if subtract_dipolar_kicks:
                line.elements[ii].dipolar_px_kick = dipolar_kicks[ecloud_type][0] * length
                line.elements[ii].dipolar_py_kick = dipolar_kicks[ecloud_type][1] * length
                line.elements[ii].dipolar_ptau_kick = dipolar_kicks[ecloud_type][2] * length


def electroncloud_dipolar_kicks_of_fieldmap(fieldmap=None, p0c=None):

    assert p0c is not None
    assert fieldmap is not None

    part = xp.Particles(p0c=p0c)
    ecloud = xf.ElectronCloud(
        length=1,
        fieldmap=fieldmap,
        _buffer=fieldmap._buffer)
    ecloud.track(part)
    px = part.px[0]
    py = part.py[0]
    ptau = part.ptau[0]
    return [px, py, ptau]


def full_electroncloud_setup(line=None, ecloud_info=None, filenames=None, context=None,
                             tau_max=None, subtract_dipolar_kicks=True, shift_to_closed_orbit=True):

    buffer = context.new_buffer()
    fieldmaps = {
        ecloud_type: get_electroncloud_fieldmap_from_h5(
            filename=filename,
            buffer=buffer,
            tau_max=tau_max,
            ecloud_name=ecloud_type) for (
            ecloud_type,
            filename) in filenames.items()}

    for ecloud_type, fieldmap in fieldmaps.items():
        print(f"Inserting \"{ecloud_type}\" electron clouds...")
        insert_electronclouds(
            ecloud_info[ecloud_type],
            fieldmap=fieldmap,
            line=line)

    tracker = xt.Tracker(_context=context, line=line, _buffer=buffer)
    twiss_without_ecloud = tracker.twiss()
    config_electronclouds(
        line,
        twiss=twiss_without_ecloud,
        ecloud_info=ecloud_info,
        subtract_dipolar_kicks=subtract_dipolar_kicks,
        shift_to_closed_orbit=shift_to_closed_orbit,
        fieldmaps=fieldmaps,
        ecloud_strength=1)
    twiss_with_ecloud = tracker.twiss()

    return tracker, twiss_without_ecloud, twiss_with_ecloud
