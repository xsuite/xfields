import xobjects as xo
from xfields.beam_elements.beambeam import BeamBeamBiGaussian2DData
from xfields.fieldmaps.bigaussian import BiGaussianFieldMapData
import xfields
from xtrack.particles import gen_local_particle_api, Particles

context = xo.ContextCpu()

sources = []
sources.append("#include <stdint.h>\n")

sources.append(xfields._pkg_root.joinpath('src/constants.h'))
sources.append(xfields._pkg_root.joinpath('src/complex_error_function.h'))
sources.append(xfields._pkg_root.joinpath('src/fields_bigaussian.h'))

sources.append(Particles.XoStruct._gen_c_api()[0])
sources.append(gen_local_particle_api())
source_bb, kernels_bb, cdefs_bb = BeamBeamBiGaussian2DData._gen_c_api()
sources.append(source_bb)
source_fm, kernels_fm, cdefs_fm = BiGaussianFieldMapData._gen_c_api()
sources.append(source_fm)
sources.append(xfields._pkg_root.joinpath('fieldmaps/bigaussian_src/bigaussian.h'))
sources.append(xfields._pkg_root.joinpath('beam_elements/beambeam_src/beambeam.h'))

context.add_kernels(sources, kernels_bb, extra_cdef=cdefs_bb,
                            save_source_as='test.c',
                            specialize=True)
