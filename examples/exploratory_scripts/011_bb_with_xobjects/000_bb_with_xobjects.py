import xobjects as xo
from xfields.beam_elements.beambeam import BeamBeamBiGaussian2DData
from xfields.fieldmaps.bigaussian import BiGaussianFieldMapData
import xfields

context = xo.ContextCpu()

sources = []

sources.append(xfields._pkg_root.joinpath('src/constants.h'))
sources.append(xfields._pkg_root.joinpath('src/complex_error_function.h'))
sources.append(xfields._pkg_root.joinpath('src/fields_bigaussian.h'))

api_conf = {'prepointer': ' /*gpuglmem*/ '}

source_bb, kernels_bb, cdefs_bb = BeamBeamBiGaussian2DData._gen_c_api(
        conf=api_conf)
sources.append(source_bb)
source_fm, kernels_fm, cdefs_fm = BiGaussianFieldMapData._gen_c_api(
        conf=api_conf)
sources.append(source_fm)
sources.append(xfields._pkg_root.joinpath('fieldmaps/bigaussian_src/bigaussian.h'))

context.add_kernels(sources, kernels_bb, extra_cdef=cdefs_bb,
                            save_source_as='test.c',
                            specialize=True)
