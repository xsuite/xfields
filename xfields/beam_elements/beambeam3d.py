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

    @classmethod
    def from_xline(cls, xline_beambeam=None,
            _context=None, _buffer=None, _offset=None):

        params = xline_beambeam.to_dict(keepextra=True)
        import xline
        bb6d_data = xline.BB6Ddata.BB6D_init(
                q_part=qe*float(params['enabled']), # the xline input has the charge
                                                    # of the slices in elementary charges 
                phi=params["phi"],
                alpha=params["alpha"],
                delta_x=params["x_bb_co"],
                delta_y=params["y_bb_co"],
                N_part_per_slice=params["charge_slices"],
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
                min_sigma_diff=params["min_sigma_diff"],
                threshold_singular=params["threshold_singular"],
                Dx_sub=params["d_x"],
                Dpx_sub=params["d_px"],
                Dy_sub=params["d_y"],
                Dpy_sub=params["d_py"],
                Dsigma_sub=params["d_zeta"],
                Ddelta_sub=params["d_delta"],
                enabled=params["enabled"],
            )
        assert(
            len(bb6d_data.N_part_per_slice) ==
            len(bb6d_data.x_slices_star) ==
            len(bb6d_data.y_slices_star) ==
            len(bb6d_data.sigma_slices_star))

        if _buffer is not None:
            ctx = _buffer.context
        elif _context is not None:
            ctx = _context

        bb = cls(
            _context=_context,
            _buffer=_buffer,
            _offset=_offset,
            q0 = bb6d_data.q_part/qe, # xline uses coulomb
            boost_parameters = {
                'sphi': bb6d_data.parboost.sphi,
                'cphi': bb6d_data.parboost.cphi,
                'tphi': bb6d_data.parboost.tphi,
                'salpha': bb6d_data.parboost.salpha,
                'calpha': bb6d_data.parboost.calpha},
            Sigmas_0_star = {
                'Sig_11': bb6d_data.Sigmas_0_star.Sig_11_0,
                'Sig_12': bb6d_data.Sigmas_0_star.Sig_12_0,
                'Sig_13': bb6d_data.Sigmas_0_star.Sig_13_0,
                'Sig_14': bb6d_data.Sigmas_0_star.Sig_14_0,
                'Sig_22': bb6d_data.Sigmas_0_star.Sig_22_0,
                'Sig_23': bb6d_data.Sigmas_0_star.Sig_23_0,
                'Sig_24': bb6d_data.Sigmas_0_star.Sig_24_0,
                'Sig_33': bb6d_data.Sigmas_0_star.Sig_33_0,
                'Sig_34': bb6d_data.Sigmas_0_star.Sig_34_0,
                'Sig_44': bb6d_data.Sigmas_0_star.Sig_44_0},
            min_sigma_diff = bb6d_data.min_sigma_diff,
            threshold_singular = bb6d_data.threshold_singular,
            delta_x = bb6d_data.delta_x,
            delta_y = bb6d_data.delta_y,
            x_CO = bb6d_data.x_CO,
            px_CO = bb6d_data.px_CO,
            y_CO = bb6d_data.y_CO,
            py_CO = bb6d_data.py_CO,
            sigma_CO = bb6d_data.sigma_CO,
            delta_CO = bb6d_data.delta_CO,
            Dx_sub = bb6d_data.Dx_sub,
            Dpx_sub = bb6d_data.Dpx_sub,
            Dy_sub = bb6d_data.Dy_sub,
            Dpy_sub = bb6d_data.Dpy_sub,
            Dsigma_sub = bb6d_data.Dsigma_sub,
            Ddelta_sub = bb6d_data.Ddelta_sub,
            num_slices = len(bb6d_data.N_part_per_slice),
            N_part_per_slice = ctx.nparray_to_context_array(
                                          bb6d_data.N_part_per_slice),
            x_slices_star = ctx.nparray_to_context_array(bb6d_data.x_slices_star),
            y_slices_star = ctx.nparray_to_context_array(bb6d_data.y_slices_star),
            sigma_slices_star = ctx.nparray_to_context_array(
                                          bb6d_data.sigma_slices_star),
            )

        return bb

srcs = []
srcs.append(_pkg_root.joinpath('headers/constants.h'))
srcs.append(_pkg_root.joinpath('fieldmaps/bigaussian_src/complex_error_function.h'))
srcs.append('#define NOFIELDMAP') #TODO Remove this workaound
srcs.append(_pkg_root.joinpath('fieldmaps/bigaussian_src/bigaussian.h'))
srcs.append(_pkg_root.joinpath('beam_elements/beambeam_src/beambeam3d.h'))

BeamBeamBiGaussian3D.XoStruct.extra_sources = srcs
