# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2022.                   #
# ########################################### #
import xtrack as xt
import xfields as xf
from xtrack.prebuild_kernels import precompile_single_kernel, DEFAULT_CONFIG

PREBUILT_KERNELS = {
    'xfields.lib.default_kernel': (
        ['BeamBeamBiGaussian2D', 'BeamBeamBiGaussian3D', 'Cavity', 'DipoleEdge',
         'Drift', 'Elens', 'LinearTransferMatrix', 'Multipole',
         'ParticlesMonitor', 'RFMultipole', 'ReferenceEnergyIncrease',
         'SRotation', 'SimpleThinBend', 'SimpleThinQuadrupole',
         'SpaceChargeBiGaussian', 'Wire', 'XYShift'],
        DEFAULT_CONFIG,
    ),
}


def precompile_kernels():
    elements = [
        xt.Drift(length=1.0),
        xt.Multipole(knl=[0]),
        xt.ReferenceEnergyIncrease(),
        xt.Cavity(),
        xt.XYShift(),
        xt.Elens(),
        xt.Wire(),
        xt.SRotation(),
        xt.RFMultipole(knl=[0], pn=[0]),
        xt.DipoleEdge(),
        xt.LinearTransferMatrix(),
        xt.SimpleThinBend(knl=[0]),
        xt.SimpleThinQuadrupole(knl=[0, 0]),
        xf.BeamBeamBiGaussian2D(
            other_beam_Sigma_11=1.,
            other_beam_Sigma_33=1.,
            other_beam_num_particles=0.,
            other_beam_q0=1.,
            other_beam_beta0=1.,
        ),
        xf.BeamBeamBiGaussian3D(
            slices_other_beam_zeta_center=[0],
            slices_other_beam_num_particles=[0],
            phi=0.,
            alpha=0,
            other_beam_q0=1.,
            slices_other_beam_Sigma_11=[1],
            slices_other_beam_Sigma_12=[0],
            slices_other_beam_Sigma_22=[0],
            slices_other_beam_Sigma_33=[1],
            slices_other_beam_Sigma_34=[0],
            slices_other_beam_Sigma_44=[0],

        ),
        xf.SpaceChargeBiGaussian(
            longitudinal_profile=xf.LongitudinalProfileQGaussian(
                number_of_particles=0, sigma_z=1)
        ),
    ]
    precompile_single_kernel(
        name='xfields.lib.default_kernel',
        elements=elements,
        config={},
    )
