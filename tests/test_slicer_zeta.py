import numpy as np
import xfields as xf
from xobjects.test_helpers import for_all_test_contexts

@for_all_test_contexts
def test_slicer_zeta(test_context):
    zeta_range = (-1.0, 1.0)
    num_slices = 10
    dzeta = (zeta_range[1]-zeta_range[0])/num_slices
    zeta_slice_edges = np.linspace(zeta_range[0],zeta_range[1],num_slices+1)
    zeta_centers = zeta_slice_edges[:-1]+dzeta/2
    slicer_0 = xf.UniformBinSlicer(_context=test_context,zeta_range=zeta_range, num_slices=num_slices)
    slicer_1 = xf.UniformBinSlicer(_context=test_context,zeta_range=zeta_range, dzeta=dzeta)
    slicer_2 = xf.UniformBinSlicer(_context=test_context,zeta_slice_edges=zeta_slice_edges)
    assert np.allclose(slicer_0.zeta_centers,zeta_centers)
    assert np.allclose(slicer_1.zeta_centers,zeta_centers)
    assert np.allclose(slicer_2.zeta_centers,zeta_centers)
