# copyright ############################### #
# This file is part of the Xfields package. #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

import numpy as np


XFIELDS_ELEMENTS_INIT_DEFAULTS = {
    'BeamBeamBiGaussian2D': {
        'other_beam_Sigma_11': 1.,
        'other_beam_Sigma_33': 1.,
        'other_beam_num_particles': 0.,
        'other_beam_q0': 1.,
        'other_beam_beta0': 1.,
    },
    'BeamBeamBiGaussian3D': {
        'slices_other_beam_zeta_center': np.array([0]),
        'slices_other_beam_num_particles': np.array([0]),
        'phi': 0.,
        'alpha': 0,
        'other_beam_q0': 1.,
        'slices_other_beam_Sigma_11': np.array([1]),
        'slices_other_beam_Sigma_12': np.array([0]),
        'slices_other_beam_Sigma_22': np.array([0]),
        'slices_other_beam_Sigma_33': np.array([1]),
        'slices_other_beam_Sigma_34': np.array([0]),
        'slices_other_beam_Sigma_44': np.array([0]),
    },
    'SpaceChargeBiGaussian': {
        'longitudinal_profile': {
            '__class__': 'LongitudinalProfileQGaussian',
            'number_of_particles': 1,
            'sigma_z': 0,
        }
    }
}
