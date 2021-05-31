import xobjects as xo
import xtrack as xt

from ..general import _pkg_root

api_conf = {'prepointer': ' /*gpuglmem*/ '}

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

class BeamBeamBiGaussian3DData(xo.Struct):
    q0 = xo.Float64
    boost_parameters = BoostParameters
    Sigmas_0_star = Sigmas
    min_sigma_diff = xo.Float64
    threshold_singular = xo.Float64
    num_slices = xo.Int64
    delta_x = xo.Float64
    delta_y = xo.Float64
    x_CO = xo.Float64
    px_CO = xo.Float64
    y_CO = xo.Float64
    py_CO = xo.Float64
    sigma_CO = xo.Float64
    delta_CO = xo.Float64
    Dx_sub = xo.Float64
    Dpx_sub = xo.Float64
    Dy_sub = xo.Float64
    Dpy_sub = xo.Float64
    Dsigma_sub = xo.Float64
    Ddelta_sub = xo.Float64
    N_part_per_slice = xo.Float64[:]
    x_slices_star = xo.Float64[:]
    y_slices_star = xo.Float64[:]
    sigma_slices_star = xo.Float64[:]

srcs = []
srcs.append(_pkg_root.joinpath('headers/constants.h'))
srcs.append(_pkg_root.joinpath('fieldmaps/bigaussian_src/complex_error_function.h'))
srcs.append('#define NOFIELDMAP') #TODO Remove this workaound
srcs.append(_pkg_root.joinpath('fieldmaps/bigaussian_src/bigaussian.h'))
srcs.append(Sigmas._gen_c_api(api_conf)[0]) #TODO This shouldnt be needed
srcs.append(BoostParameters._gen_c_api(api_conf)[0]) #TODO This shouldnt be needed
srcs.append(_pkg_root.joinpath('beam_elements/beambeam_src/beambeam3d.h'))
BeamBeamBiGaussian3DData.extra_sources = srcs

class BeamBeamBiGaussian3D(xt.dress_element(BeamBeamBiGaussian3DData)):
        pass
