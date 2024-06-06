# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
import pandas as pd

from xtrack.progress_indicator import progress

from ..beam_elements.spacecharge import SpaceChargeBiGaussian
from ..beam_elements.spacecharge import SpaceCharge3D

import xpart as xp
import xobjects as xo

def install_spacecharge_frozen(line=None, _buffer=None,
                               particle_ref=None,
                               longitudinal_profile=None,
                               nemitt_x=None, nemitt_y=None, sigma_z=None,
                               num_spacecharge_interactions=None,
                               tol_spacecharge_position=None,
                               s_spacecharge=None,
                               delta_rms=None):

    '''
    Install spacecharge elements (frozen modeling) in a xtrack.Line object.

    Parameters
    ----------
    line : xtrack.Line
        Line in which the spacecharge elements are installed.
    particle_ref : xpart.Particles (optional)
        Reference particle for the spacecharge elements.
    longitudinal_profile : str
        Longitudinal profile for the spacecharge elements.
    nemitt_x : float
        Normalized emittance in the horizontal plane (in m rad).
    nemitt_y : float
        Normalized emittance in the vertical plane (in m rad).
    sigma_z : float
        RMS bunch length in meters.
    num_spacecharge_interactions : int
        Number of spacecharge interactions to be installed.
    tol_spacecharge_position : float
        Tolerance for the spacecharge position.
    s_spacecharge : np.ndarray (optional)
        Position of the spacecharge elements.
    delta_rms : float
        Matched momentum spread. If None, it is computed from a matched gaussian bunch.

    Returns
    -------
    spacecharge_elements : list
        List of spacecharge elements.
    '''

    if _buffer is None:
        if not line._has_valid_tracker():
            line.build_tracker(compile=False) # Put everything in the same buffer
        _buffer = line._buffer

    line.discard_tracker() # as we will be changing element types

    if tol_spacecharge_position is not None:
        raise NotImplementedError('tol_spacecharge_position not implemented')

    if particle_ref is None:
        particle_ref = line.particle_ref
        assert particle_ref is not None

    line_no_sc = line.copy(_context=xo.ContextCpu())
    line_no_sc.build_tracker()

    if delta_rms is None:
        # Make a matched bunch just to get the matched momentum spread
        bunch = xp.generate_matched_gaussian_bunch(
                num_particles=int(2e6), total_intensity_particles=1.,
                nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
                particle_ref=particle_ref, line=line_no_sc)
        delta_rms = np.std(bunch.delta)

    # Generate spacecharge positions
    if s_spacecharge is None:
        s_spacecharge = np.linspace(0, line.get_length(),
                                    num_spacecharge_interactions+1)[:-1]

    # Create spacecharge elements (dummy)
    sc_elements = []
    sc_names = []
    insertions = []
    for ii in progress(range(len(s_spacecharge)),
                           desc='Creating spacecharge elements'):

        ss = s_spacecharge[ii]

        sc_elements.append(SpaceChargeBiGaussian(
            _buffer=_buffer,
            length=-9999,
            apply_z_kick=False,
            longitudinal_profile=longitudinal_profile,
            mean_x=0.,
            mean_y=0.,
            sigma_x=1.,
            sigma_y=1.))
        sc_names.append(f'spacecharge_{ii}')

        insertions.append((ss, [(sc_names[-1], sc_elements[-1])]))

    # Insert spacecharge elements
    line._insert_thin_elements_at_s(insertions)

    actual_s_spch = line.get_s_position(sc_names)

    sc_lengths = 0*s_spacecharge
    sc_lengths[:-1] = np.diff(actual_s_spch)
    sc_lengths[-1] = line.get_length() - np.sum(sc_lengths[:-1])

    # Twiss at spacecharge
    line_sc_off = line.copy(_context=xo.ContextCpu()).filter_elements(
                                           exclude_types_starting_with='SpaceCh')

    line_sc_off.build_tracker()
    tw_at_sc = line_sc_off.twiss(particle_ref=particle_ref, at_elements=sc_names, method='4d')

    # Configure lenses
    for ii, sc in enumerate(sc_elements):
        sc.mean_x = tw_at_sc['x'][ii]
        sc.mean_y = tw_at_sc['y'][ii]
        sc.sigma_x = np.sqrt(tw_at_sc['betx'][ii]*nemitt_x
                               /particle_ref.beta0/particle_ref.gamma0
                             + (tw_at_sc['dx'][ii]*delta_rms)**2)
        sc.sigma_y = np.sqrt(tw_at_sc['bety'][ii]*nemitt_y
                               /particle_ref.beta0/particle_ref.gamma0
                             + (tw_at_sc['dy'][ii]*delta_rms)**2)
        sc.length = sc_lengths[ii]


def replace_spacecharge_with_quasi_frozen(
                        line, _buffer=None,
                        update_mean_x_on_track=True,
                        update_mean_y_on_track=True,
                        update_sigma_x_on_track=True,
                        update_sigma_y_on_track=True):

    '''
    Replace spacecharge elements with quasi-frozen spacecharge elements.

    Parameters
    ----------
    line : xtrack.Line
        Line in which the spacecharge elements are replaced.
    _buffer : xtrack.Buffer
        Buffer used allocate the spacecharge elements.
    update_mean_x_on_track : bool (optional)
        Update the mean x position on track.
    update_mean_y_on_track : bool (optional)
        Update the mean y position on track.
    update_sigma_x_on_track : bool (optional)
        Update the sigma x position on track.
    update_sigma_y_on_track : bool (optional)
        Update the sigma y position on track.

    Returns
    -------
    spacecharge_elements : list
        List of spacecharge elements.
    '''

    if _buffer is None:
        if not line._has_valid_tracker():
            line.build_tracker(compile=False) # Put everything in the same buffer
        _buffer = line._buffer

    line.discard_tracker() # as we will be changing element types

    spch_elements = []
    for ii, ee in enumerate(line.elements):
        if isinstance(ee, SpaceChargeBiGaussian):
            ee.move(_buffer=_buffer)
            ee.update_mean_x_on_track = update_mean_x_on_track
            ee.update_mean_y_on_track = update_mean_y_on_track
            ee.update_sigma_x_on_track = update_sigma_x_on_track
            ee.update_sigma_y_on_track = update_sigma_y_on_track
            ee.iscollective = True
            spch_elements.append(ee)

    return spch_elements


class PICCollection:

    def __init__(self,
                 nx_grid,
                 ny_grid,
                 nz_grid,
                 x_lim_min,
                 x_lim_max,
                 y_lim_min,
                 y_lim_max,
                 z_range,
                 n_lims_x,
                 n_lims_y,
                 solver='FFTSolver2p5D',
                 apply_z_kick=False,
                 gamma0 = None,
                 _context=None,
                 _buffer=None,
                     ):

        self._context = _context
        self._buffer = _buffer

        self.nx_grid = nx_grid
        self.ny_grid = ny_grid
        self.nz_grid = nz_grid

        self.z_range = z_range
        self.solver = solver
        self.apply_z_kick = apply_z_kick
        self.gamma0 = gamma0

        self.x_lims = np.linspace(x_lim_min, x_lim_max, n_lims_x)
        self.y_lims = np.linspace(y_lim_min, y_lim_max, n_lims_y)

        self._existing_pics = {}
        self._fftplan = None


    def get_pic(self, x_lim, y_lim):

        assert x_lim < self.x_lims[-1]
        assert x_lim > self.x_lims[0]
        assert y_lim < self.y_lims[-1]
        assert y_lim > self.y_lims[0]

        ix = np.argmin(np.abs(x_lim - self.x_lims))
        iy = np.argmin(np.abs(y_lim - self.y_lims))

        if (ix, iy) not in self._existing_pics.keys():
            print(f'Creating PIC ({ix}, {iy})')
            xlim_pic = self.x_lims[ix]
            ylim_pic = self.y_lims[iy]
            new_pic = SpaceCharge3D(
                _context=self._context,
                _buffer=self._buffer,
                length=0.,
                apply_z_kick=self.apply_z_kick,
                x_range=(-xlim_pic, xlim_pic),
                y_range=(-ylim_pic, ylim_pic),
                z_range=self.z_range,
                nx=self.nx_grid, ny=self.ny_grid, nz=self.nz_grid,
                solver=self.solver,
                gamma0=self.gamma0,
                fftplan=self._fftplan)
            new_pic._buffer.grow(10*1024**2) # Add 10 MB for sc copies
            if self._fftplan is None:
                self._fftplan = new_pic.fieldmap.solver.fftplan
            self._existing_pics[ix, iy] = new_pic

        return self._existing_pics[ix, iy]


def replace_spacecharge_with_PIC(
        line,
        n_sigmas_range_pic_x, n_sigmas_range_pic_y,
        nx_grid, ny_grid, nz_grid, n_lims_x, n_lims_y, z_range,
        solver='FFTSolver2p5D',
        _context=None,
        _buffer=None):

    '''
    Replace spacecharge elements with Particle In Cell (PIC) elements.

    Parameters
    ----------
    line : xtrack.Line
        Line in which the spacecharge elements are replaced.
    n_sigmas_range_pic_x : float
        Extent of the PIC grid in the horizontal direction in units beam sigmas.
    n_sigmas_range_pic_y : float
        Extent of the PIC grid in the vertical direction in units beam sigmas.
    nx_grid : int
        Number of grid points in the horizontal direction.
    ny_grid : int
        Number of grid points in the vertical direction.
    nz_grid : int
        Number of grid points in the longitudinal direction.
    n_lims_x : int
        Number different limits in x for which PIC need to be generated.
    n_lims_y : int
        Number different limits in y for which PIC need to be generated.
    z_range : float
        Range of the longitudinal grid.
    _context : xtrack.Context (optional)
        Context in which the PIC elements are created.
    _buffer : xtrack.Buffer (optional)
        Buffer in which the PIC elements are created.

    Returns
    -------
    pic_collection : xfields.PICCollection
        Collection of PIC elements.
    all_pics: list
        List of all PIC elements.
    '''
    if _buffer is None and _context is None:
        if not line._has_valid_tracker():
            line.build_tracker(compile=False) # Put everything in the same buffer
        _buffer = line._buffer

    line.discard_tracker()

    all_sc_elems = []
    name_sc_elems = []
    all_sigma_x = []
    all_sigma_y = []
    for ii, ee in enumerate(line.elements):
        if isinstance(ee, SpaceChargeBiGaussian):
            all_sc_elems.append(ee)
            name_sc_elems.append(line.element_names[ii])
            all_sigma_x.append(ee.sigma_x)
            all_sigma_y.append(ee.sigma_y)

    x_lim_min = np.min(all_sigma_x) * (n_sigmas_range_pic_x - 0.5)
    x_lim_max = np.max(all_sigma_x) * (n_sigmas_range_pic_x + 0.5)
    y_lim_min = np.min(all_sigma_y) * (n_sigmas_range_pic_y - 0.5)
    y_lim_max = np.max(all_sigma_y) * (n_sigmas_range_pic_y + 0.5)

    pic_collection = PICCollection(
        _context=_context,
        _buffer=_buffer,
        nx_grid=nx_grid, ny_grid=ny_grid, nz_grid=nz_grid,
        x_lim_min=x_lim_min, x_lim_max=x_lim_max, n_lims_x=n_lims_x,
        y_lim_min=y_lim_min, y_lim_max=y_lim_max, n_lims_y=n_lims_y,
        z_range=z_range,
        solver=solver,
        gamma0=line.particle_ref.gamma0[0])

    all_pics = []
    for nn, ee in zip(name_sc_elems, all_sc_elems):
        xlim = n_sigmas_range_pic_x*ee.sigma_x
        ylim = n_sigmas_range_pic_y*ee.sigma_y
        base_sc = pic_collection.get_pic(xlim, ylim)
        sc = base_sc.copy(_buffer=base_sc._buffer)
        sc.length = ee.length
        line.element_dict[nn] = sc
        all_pics.append(sc)

    return pic_collection, all_pics
