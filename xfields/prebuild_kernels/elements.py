# copyright ############################### #
# This file is part of the Xfields package. #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..beam_elements.beambeam2d import BeamBeamBiGaussian2D
from ..beam_elements.beambeam3d import BeamBeamBiGaussian3D
from ..beam_elements.spacecharge import SpaceChargeBiGaussian


DEFAULT_XFIELDS_ELEMENTS = [
    BeamBeamBiGaussian2D,
    BeamBeamBiGaussian3D,
    SpaceChargeBiGaussian,
]

