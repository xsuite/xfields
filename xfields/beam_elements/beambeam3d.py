from scipy.constants import e as qe
import xobjects as xo
import xtrack as xt

from ..general import _pkg_root

class BoostParameters(xo.Struct):
    sphi = xo.Float64
    cphi = xo.Float64
    tphi = xo.Float64
    salpha = xo.Float64
    calpha = xo.Float64

class Sigmas(xo.Struct):
    Sig_11 = xo.Float64
    Sig_12 = xo.Float64
    Sig_13 = xo.Float64
    Sig_14 = xo.Float64
    Sig_22 = xo.Float64
    Sig_23 = xo.Float64
    Sig_24 = xo.Float64
    Sig_33 = xo.Float64
    Sig_34 = xo.Float64
    Sig_44 = xo.Float64


class BeamBeamBiGaussian3D(xt.BeamElement):

    _xofields = {
        'q0': xo.Float64,
        'boost_parameters': BoostParameters,
        'Sigmas_0_star': Sigmas,
        'min_sigma_diff': xo.Float64,
        'threshold_singular': xo.Float64,
        'num_slices': xo.Int64,
        'delta_x': xo.Float64,
        'delta_y': xo.Float64,
        'x_CO': xo.Float64,
        'px_CO': xo.Float64,
        'y_CO': xo.Float64,
        'py_CO': xo.Float64,
        'sigma_CO': xo.Float64,
        'delta_CO': xo.Float64,
        'Dx_sub': xo.Float64,
        'Dpx_sub': xo.Float64,
        'Dy_sub': xo.Float64,
        'Dpy_sub': xo.Float64,
        'Dsigma_sub': xo.Float64,
        'Ddelta_sub': xo.Float64,
        'N_part_per_slice': xo.Float64[:],
        'x_slices_star': xo.Float64[:],
        'y_slices_star': xo.Float64[:],
        'sigma_slices_star': xo.Float64[:],
    }

    _input_param_names = [
        'phi',
        'alpha',
        'zeta_slices',
        'Sig_11_0',
        'Sig_12_0',
        'Sig_13_0',
        'Sig_14_0',
        'Sig_22_0',
        'Sig_23_0',
        'Sig_24_0',
        'Sig_33_0',
        'Sig_34_0',
        'Sig_44_0',
        ]

    def to_dict(self):

        dct = super().to_dict()
        for nn in self._input_param_names:
            dct[nn] = getattr(self, nn)
        # For compatibility with ducktrack:
        dct['x_bb_co'] = self.delta_x
        dct['y_bb_co'] = self.delta_y
        dct['x_co'] = self.x_CO
        dct['px_co'] = self.px_CO
        dct['y_co'] = self.y_CO
        dct['py_co'] = self.py_CO
        dct['zeta_co'] = self.sigma_CO
        dct['delta_co'] = self.delta_CO
        dct['d_x'] = self.Dx_sub
        dct['d_px'] = self.Dpx_sub
        dct['d_y'] = self.Dy_sub
        dct['d_py'] = self.Dpy_sub
        dct['d_zeta'] = self.Dsigma_sub
        dct['d_delta'] = self.Ddelta_sub
        dct['charge_slices'] = self.N_part_per_slice
        dct['sigma_11'] = self.Sig_11_0
        dct['sigma_12'] = self.Sig_12_0
        dct['sigma_13'] = self.Sig_13_0
        dct['sigma_14'] = self.Sig_14_0
        dct['sigma_22'] = self.Sig_22_0
        dct['sigma_23'] = self.Sig_23_0
        dct['sigma_24'] = self.Sig_24_0
        dct['sigma_33'] = self.Sig_33_0
        dct['sigma_34'] = self.Sig_34_0
        dct['sigma_44'] = self.Sig_44_0
        return dct


    def __init__(self, **kwargs):

        if 'old_interface' in kwargs:
            params=kwargs['old_interface']
            n_slices=len(params["charge_slices"])
            super().__init__(
                N_part_per_slice=n_slices,
                x_slices_star=n_slices,
                y_slices_star=n_slices,
                sigma_slices_star=n_slices)
            self._from_oldinterface(params)
        else:
            super().__init__(**kwargs)
            for nn in self._input_param_names:
                if nn in kwargs.keys():
                    setattr(self, nn, kwargs[nn])

    def _from_oldinterface(self, params):
        import ducktrack

        from scipy.constants import e as qe
        bb6d_data = ducktrack.BB6Ddata.BB6D_init(
                q_part=qe, # the ducktrack input has the charge
                phi=params["phi"],
                alpha=params["alpha"],
                delta_x=params["x_bb_co"],
                delta_y=params["y_bb_co"],
                N_part_per_slice=params["charge_slices"], #
                z_slices=params["zeta_slices"],
                Sig_11_0=params["sigma_11"],
                Sig_12_0=params["sigma_12"],
                Sig_13_0=params["sigma_13"],
                Sig_14_0=params["sigma_14"],
                Sig_22_0=params["sigma_22"],
                Sig_23_0=params["sigma_23"],
                Sig_24_0=params["sigma_24"],
                Sig_33_0=params["sigma_33"],
                Sig_34_0=params["sigma_34"],
                Sig_44_0=params["sigma_44"],
                x_CO=params["x_co"],
                px_CO=params["px_co"],
                y_CO=params["y_co"],
                py_CO=params["py_co"],
                sigma_CO=params["zeta_co"],
                delta_CO=params["delta_co"],
                min_sigma_diff=1e-10,
                threshold_singular=1e-28,
                Dx_sub=params["d_x"],
                Dpx_sub=params["d_px"],
                Dy_sub=params["d_y"],
                Dpy_sub=params["d_py"],
                Dsigma_sub=params["d_zeta"],
                Ddelta_sub=params["d_delta"],
                enabled=1,
            )
        assert(
            len(bb6d_data.N_part_per_slice) ==
            len(bb6d_data.x_slices_star) ==
            len(bb6d_data.y_slices_star) ==
            len(bb6d_data.sigma_slices_star))

        self.q0 = bb6d_data.q_part/qe, # ducktrack uses coulomb
        self.boost_parameters = {
                 'sphi': bb6d_data.parboost.sphi,
                 'cphi': bb6d_data.parboost.cphi,
                 'tphi': bb6d_data.parboost.tphi,
                 'salpha': bb6d_data.parboost.salpha,
                 'calpha': bb6d_data.parboost.calpha}
        self.Sigmas_0_star = {
                 'Sig_11': bb6d_data.Sigmas_0_star.Sig_11_0,
                 'Sig_12': bb6d_data.Sigmas_0_star.Sig_12_0,
                 'Sig_13': bb6d_data.Sigmas_0_star.Sig_13_0,
                 'Sig_14': bb6d_data.Sigmas_0_star.Sig_14_0,
                 'Sig_22': bb6d_data.Sigmas_0_star.Sig_22_0,
                 'Sig_23': bb6d_data.Sigmas_0_star.Sig_23_0,
                 'Sig_24': bb6d_data.Sigmas_0_star.Sig_24_0,
                 'Sig_33': bb6d_data.Sigmas_0_star.Sig_33_0,
                 'Sig_34': bb6d_data.Sigmas_0_star.Sig_34_0,
                 'Sig_44': bb6d_data.Sigmas_0_star.Sig_44_0}
        self.min_sigma_diff = bb6d_data.min_sigma_diff
        self.threshold_singular = bb6d_data.threshold_singular
        self.delta_x = bb6d_data.delta_x
        self.delta_y = bb6d_data.delta_y
        self.x_CO = bb6d_data.x_CO
        self.px_CO = bb6d_data.px_CO
        self.y_CO = bb6d_data.y_CO
        self.py_CO = bb6d_data.py_CO
        self.sigma_CO = bb6d_data.sigma_CO
        self.delta_CO = bb6d_data.delta_CO
        self.Dx_sub = bb6d_data.Dx_sub
        self.Dpx_sub = bb6d_data.Dpx_sub
        self.Dy_sub = bb6d_data.Dy_sub
        self.Dpy_sub = bb6d_data.Dpy_sub
        self.Dsigma_sub = bb6d_data.Dsigma_sub
        self.Ddelta_sub = bb6d_data.Ddelta_sub
        self.num_slices = len(bb6d_data.N_part_per_slice)
        self.N_part_per_slice = bb6d_data.N_part_per_slice
        self.x_slices_star = bb6d_data.x_slices_star
        self.y_slices_star = bb6d_data.y_slices_star
        self.sigma_slices_star = bb6d_data.sigma_slices_star

        # input_params
        self.phi = params['phi']
        self.alpha = params['alpha']
        self.zeta_slices = params['zeta_slices']
        self.Sig_11_0 = params["sigma_11"]
        self.Sig_12_0 = params["sigma_12"]
        self.Sig_13_0 = params["sigma_13"]
        self.Sig_14_0 = params["sigma_14"]
        self.Sig_22_0 = params["sigma_22"]
        self.Sig_23_0 = params["sigma_23"]
        self.Sig_24_0 = params["sigma_24"]
        self.Sig_33_0 = params["sigma_33"]
        self.Sig_34_0 = params["sigma_34"]
        self.Sig_44_0 = params["sigma_44"]

srcs = []
srcs.append(_pkg_root.joinpath('headers/constants.h'))
srcs.append(_pkg_root.joinpath('headers/sincos.h'))
srcs.append(_pkg_root.joinpath('headers/power_n.h'))
srcs.append(_pkg_root.joinpath('fieldmaps/bigaussian_src/complex_error_function.h'))
srcs.append('#define NOFIELDMAP') #TODO Remove this workaound
srcs.append(_pkg_root.joinpath('fieldmaps/bigaussian_src/bigaussian.h'))
srcs.append(_pkg_root.joinpath('beam_elements/beambeam_src/beambeam3d.h'))

BeamBeamBiGaussian3D.XoStruct.extra_sources = srcs
