import numpy as np
import pandas as pd

from ..beam_elements.spacecharge import SpaceChargeBiGaussian
from ..beam_elements.spacecharge import SpaceCharge3D

import xpart as xp
import xtrack as xt
from xtrack.line import _is_thick, _is_drift

def install_spacecharge_frozen(line, particle_ref, longitudinal_profile,
                               nemitt_x, nemitt_y, sigma_z,
                               num_spacecharge_interactions,
                               tol_spacecharge_position,
                               s_spacecharge=None):

    tracker_no_sc = xt.Tracker(line=line.copy())

    # Make a matched bunch just to get the matched momentum spread
    bunch = xp.generate_matched_gaussian_bunch(
             num_particles=int(2e6), total_intensity_particles=1.,
             nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
             particle_ref=particle_ref, tracker=tracker_no_sc)
    delta_rms = np.std(bunch.delta)

    # Generate spacecharge positions
    if s_spacecharge is None:
        s_spacecharge = np.linspace(0, line.get_length(),
                                    num_spacecharge_interactions+1)[:-1]

    # Create spacecharge elements (dummy)
    sc_elements = []
    sc_names = []
    for ii, ss in enumerate(s_spacecharge):
        sc_elements.append(SpaceChargeBiGaussian(
            length=-9999,
            apply_z_kick=False,
            longitudinal_profile=longitudinal_profile,
            mean_x=0.,
            mean_y=0.,
            sigma_x=1.,
            sigma_y=1.))
        sc_names.append(f'spacecharge_{ii}')

        #TODO Replace loop with single insert_element when available in xtrack
        line.insert_element(name=sc_names[-1], element=sc_elements[-1],
                            at_s=ss, s_tol=tol_spacecharge_position)


    actual_s_spch = line.get_s_position(sc_names)

    sc_lengths = 0*s_spacecharge
    sc_lengths[:-1] = np.diff(actual_s_spch)
    sc_lengths[-1] = line.get_length() - np.sum(sc_lengths[:-1]) 

    # Twiss at spacecharge
    line_sc_off = line.filter_elements(exclude_types_starting_with='SpaceCh')
    tracker_sc_off = xt.Tracker(line=line_sc_off,
            element_classes=tracker_no_sc.element_classes,
            track_kernel=tracker_no_sc.track_kernel)
    tw_at_sc = tracker_sc_off.twiss(particle_ref=particle_ref, at_elements=sc_names)

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
                        line, _buffer,
                        update_mean_x_on_track=True,
                        update_mean_y_on_track=True,
                        update_sigma_x_on_track=True,
                        update_sigma_y_on_track=True):

    spch_elements = []
    for ii, ee in enumerate(line.elements):
        if isinstance(ee, SpaceChargeBiGaussian):
            ee._move_to(_buffer=_buffer)
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
        _context=None,
        _buffer=None):

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
        z_range=z_range)

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
