import numpy as np

from .beam_elements.spacecharge import SpaceChargeBiGaussian
from .beam_elements.spacecharge import SpaceCharge3D

def replace_spaceharge_with_quasi_frozen(
                        line, _buffer,
                        update_mean_x_on_track=True,
                        update_mean_y_on_track=True,
                        update_sigma_x_on_track=True,
                        update_sigma_y_on_track=True):

    spch_elements = []
    for ii, ee in enumerate(line.elements):
        if ee.__class__.__name__ == 'SCQGaussProfile':
            newee = SpaceChargeBiGaussian.from_xline(ee, _buffer=_buffer)
            newee.update_mean_x_on_track = update_mean_x_on_track
            newee.update_mean_y_on_track = update_mean_y_on_track
            newee.update_sigma_x_on_track = update_sigma_x_on_track
            newee.update_sigma_y_on_track = update_sigma_y_on_track
            newee.iscollective = True
            line.elements[ii] = newee
            spch_elements.append(newee)

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
            if self._fftplan is None:
                self._fftplan = new_pic.fieldmap.solver.fftplan
            self._existing_pics[ix, iy] = new_pic

        return self._existing_pics[ix, iy]

class DerivedElement:

    def __init__(self, base_element, changes=None,
                 restore_base=False):
        self.base_element = base_element
        self.changes = changes

    def track(self, particles):
        for nn, vv in self.changes.items():
            setattr(self.base_element, nn, vv)
        self.base_element.track(particles)


def replace_spaceharge_with_PIC(
        sequence,
        n_sigmas_range_pic_x, n_sigmas_range_pic_y,
        nx_grid, ny_grid, nz_grid, n_lims_x, n_lims_y, z_range,
        _context=None,
        _buffer=None):

    all_sc_elems = []
    ind_sc_elems = []
    all_sigma_x = []
    all_sigma_y = []
    for ii, ee in enumerate(sequence.elements):
        if ee.__class__.__name__ == 'SCQGaussProfile':
            all_sc_elems.append(ee)
            ind_sc_elems.append(ii)
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
    for ii, ee in zip(ind_sc_elems, all_sc_elems):
        xlim = n_sigmas_range_pic_x*ee.sigma_x
        ylim = n_sigmas_range_pic_y*ee.sigma_y
        base_sc = pic_collection.get_pic(xlim, ylim)
        sc = DerivedElement(base_sc, changes={'length': ee.length})
        sc.iscollective = True
        sequence.elements[ii] = sc
        all_pics.append(sc)

    return pic_collection, all_pics
