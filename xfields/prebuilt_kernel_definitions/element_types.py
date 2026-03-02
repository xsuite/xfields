# copyright ############################### #
# This file is part of the Xfields package. #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

import xfields as xf

DEFAULT_XFIELDS_ELEMENTS = [
    xf.BeamBeamBiGaussian2D,
    xf.BeamBeamBiGaussian3D,
    xf.SpaceChargeBiGaussian,
    xf.BeamBeamPIC3D,
    xf.SpaceCharge3D,
    xf.ElectronCloud,
    xf.ElectronLensInterpolated,
]

NON_TRACKING_ELEMENTS = [
    xf.LongitudinalProfileQGaussian,
    xf.beam_elements.temp_slicer.TempSlicer,
    xf.TriLinearInterpolatedFieldMap,
    xf.TriCubicInterpolatedFieldMap,
    xf.UniformBinSlicer,
    xf.slicers.CompressedProfile,
]