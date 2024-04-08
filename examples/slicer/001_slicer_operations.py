import xtrack as xt
import xfields as xf

p1 = xt.Particles(p0c=6500e9, zeta=0, x=1e-3)
p2 = xt.Particles(p0c=6500e9, zeta=10, x=1e-3)

slicer_1 = xf.UniformBinSlicer(zeta_range=(-1, 1), num_slices=10,
                               bunch_spacing_zeta=10, num_bunches=2)
slicer_2 = slicer_1.copy()

slicer_1.slice(p1)
slicer_2.slice(p2)

# buffer from slicer
buf1 = slicer_1._to_npbuffer()

# slicer from buffer
slicer_1_from_buf = xf.UniformBinSlicer._from_npbuffer(buf1)

# sum two slicers
slicer_3 = slicer_1 + slicer_2

# sum a list of slicers
slicer_4 = sum([slicer_1, slicer_2, slicer_3])

# add in place to a slicer
slicer_1 += slicer_2


