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
                               tol_spacecharge_position):
    line_no_sc = line

    tracker_no_sc = xt.Tracker(line=line_no_sc)

    # Make a matched bunch just to get the matched momentum spread
    bunch = xp.generate_matched_gaussian_bunch(
             num_particles=int(2e6), total_intensity_particles=1.,
             nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
             particle_ref=particle_ref, tracker=tracker_no_sc)
    delta_rms = np.std(bunch.delta)

    # Remove all drifts
    s_no_drifts = []
    e_no_drifts = []
    n_no_drifts = []
    for ss, ee, nn in zip(line_no_sc.get_s_elements(), line_no_sc.elements,
                          line_no_sc.element_names):
        if not _is_drift(ee):
            assert not _is_thick(ee)
            s_no_drifts.append(ss)
            e_no_drifts.append(ee)
            n_no_drifts.append(nn)

    s_no_drifts = np.array(s_no_drifts)

    # Generate spacecharge positions
    s_spacecharge = np.linspace(0, line_no_sc.get_length(),
                                num_spacecharge_interactions+1)[:-1]

    # Adjust spacecharge positions where possible
    for ii, ss in enumerate(s_spacecharge):
        s_closest = np.argmin(np.abs(ss-s_no_drifts))
        if np.abs(ss - s_closest) < tol_spacecharge_position:
            s_spacecharge[ii] = s_closest

    sc_lengths = 0*s_spacecharge
    sc_lengths[:-1] = np.diff(s_spacecharge)
    sc_lengths[-1] = line_no_sc.get_length() - s_spacecharge[-1]

    # Create spacecharge elements (dummy)
    sc_elements = []
    sc_names = []
    for ii, ll in enumerate(sc_lengths):
        sc_elements.append(SpaceChargeBiGaussian(
            length=ll,
            apply_z_kick=False,
            longitudinal_profile=longitudinal_profile,
            mean_x=0.,
            mean_y=0.,
            sigma_x=1.,
            sigma_y=1.))
        sc_names.append(f'spacecharge_{ii}')

    # Merge lattice and spacecharge elements
    df_lattice = pd.DataFrame({'s': s_no_drifts, 'elements': e_no_drifts,
                               'element_names': n_no_drifts})
    df_spacecharge = pd.DataFrame({'s': s_spacecharge, 'elements': sc_elements,
                                   'element_names': sc_names})
    df_elements = pd.concat([df_lattice, df_spacecharge]).sort_values('s')

    # Build new line with drifts
    new_elements = []
    new_names = []
    s_curr = 0
    i_drift = 0
    for ss, ee, nn, in zip(df_elements['s'].values,
                           df_elements['elements'].values,
                           df_elements['element_names'].values):

        if ss > s_curr + 1e-10:
            new_elements.append(xt.Drift(length=(ss-s_curr)))
            new_names.append(f'drift_{i_drift}')
            s_curr = ss
            i_drift += 1
        new_elements.append(ee)
        new_names.append(nn)

    if s_curr < line_no_sc.get_length():
        new_elements.append(xt.Drift(length=line_no_sc.get_length() - s_curr))
        new_names.append(f'drift_{i_drift}')
    line = xt.Line(elements=new_elements, element_names=new_names)
    assert np.isclose(line.get_length(), line_no_sc.get_length(), rtol=0, atol=1e-10)

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

    return line

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
