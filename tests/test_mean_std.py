# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np

import xobjects as xo
import xfields as xf

def test_mean_and_std():

    for ctx in xo.context.get_test_contexts():
        print(f"Test {ctx.__class__}")

        n_x=100
        a_host = np.array(np.random.rand(n_x))
        a_dev = ctx.nparray_to_context_array(a_host)

        mm, ss = xf.mean_and_std(a_dev)
        assert np.isclose(mm, np.mean(a_host))
        assert np.isclose(ss, np.std(a_host))

        weights_host = np.zeros_like(a_host)+.2
        weights_dev = ctx.nparray_to_context_array(weights_host)
        mm, ss = xf.mean_and_std(a_dev, weights=weights_dev)
        assert np.isclose(mm, np.mean(a_host))
        assert np.isclose(ss, np.std(a_host))
