# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np

import xfields as xf

from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_mean_and_std(test_context):
    n_x = 100
    a_host = np.array(np.random.rand(n_x))
    a_dev = test_context.nparray_to_context_array(a_host)

    mm, ss = xf.mean_and_std(a_dev)
    assert np.isclose(mm, np.mean(a_host))
    assert np.isclose(ss, np.std(a_host))

    weights_host = np.zeros_like(a_host)+.2
    weights_dev = test_context.nparray_to_context_array(weights_host)
    mm, ss = xf.mean_and_std(a_dev, weights=weights_dev)
    assert np.isclose(mm, np.mean(a_host))
    assert np.isclose(ss, np.std(a_host))
