# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xtrack as xt


def configure_orbit_dependent_parameters_for_bb(line, particle_on_co):

    """
    Assumes that the separation is saved in mean_x, mean_y for the 4D
    bb lenses.
    """

    temp_particles = particle_on_co.copy()
    for ii, ee in enumerate(line.elements):
        if ee.__class__.__name__ == 'BeamBeamBiGaussian2D':
            px_0 = temp_particles.px[0]
            py_0 = temp_particles.py[0]

            ee.post_subtract_px = 0
            ee.post_subtract_py = 0

            ee.ref_shift_x = temp_particles.x[0]
            ee.ref_shift_y = temp_particles.y[0]

            ee.track(temp_particles)

            ee.post_subtract_px = temp_particles.px[0] - px_0
            ee.post_subtract_py = temp_particles.py[0] - py_0

            temp_particles.px -= ee.post_subtract_px
            temp_particles.py -= ee.post_subtract_py

        elif ee.__class__.__name__ == 'BeamBeamBiGaussian3D':
            ee.ref_shift_x = temp_particles.x[0]
            ee.ref_shift_px = temp_particles.px[0]
            ee.ref_shift_y = temp_particles.y[0]
            ee.ref_shift_py = temp_particles.py[0]
            ee.ref_shift_zeta = temp_particles.zeta[0]
            ee.ref_shift_pzeta = temp_particles.delta[0] # Element assumes beta0=1 anyhow

            ee.post_subtract_x = 0
            ee.post_subtract_px = 0
            ee.post_subtract_y = 0
            ee.post_subtract_py = 0
            ee.post_subtract_zeta = 0
            ee.post_subtract_pzeta = 0

            ee.track(temp_particles)

            ee.post_subtract_x = temp_particles.x[0] - ee.ref_shift_x
            ee.post_subtract_px = temp_particles.px[0] - ee.ref_shift_px
            ee.post_subtract_y = temp_particles.y[0] - ee.ref_shift_y
            ee.post_subtract_py = temp_particles.py[0] - ee.ref_shift_py
            ee.post_subtract_zeta = temp_particles.zeta[0] - ee.ref_shift_zeta
            ee.post_subtract_pzeta = temp_particles.delta[0] - ee.ref_shift_pzeta

            temp_particles.x[0] = ee.ref_shift_x
            temp_particles.px[0] = ee.ref_shift_px
            temp_particles.y[0] = ee.ref_shift_y
            temp_particles.py[0] = ee.ref_shift_py
            temp_particles.zeta[0] = ee.ref_shift_zeta
            temp_particles.delta[0] = ee.ref_shift_pzeta # Element assumes beta0=1 anyhow
        elif ee.__class__.__name__ == 'Wire':
            px_0 = temp_particles.px[0]
            py_0 = temp_particles.py[0]

            ee.post_subtract_px = 0
            ee.post_subtract_py = 0

            ee.track(temp_particles)

            ee.post_subtract_px = temp_particles.px[0] - px_0
            ee.post_subtract_py = temp_particles.py[0] - py_0

            temp_particles.px -= ee.post_subtract_px
            temp_particles.py -= ee.post_subtract_py
        else:
            ee.track(temp_particles)
