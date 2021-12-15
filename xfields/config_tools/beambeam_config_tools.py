import xtrack as xt


def configure_orbit_dependent_parameters_for_bb(tracker, particle_on_co):

    """
    Assumes that the separation is saved in mean_x, mean_y for the 4D
    bb lenses.
    """

    temp_particles = particle_on_co.copy()
    for ii, ee in enumerate(tracker.line.elements):
        if ee.__class__.__name__ == 'BeamBeamBiGaussian2D':
              px_0 = temp_particles.px[0]
              py_0 = temp_particles.py[0]

              # Separation of 4D is so far set w.r.t. the closes orbit
              # (to be able to compare against sixtrack)
              # Here we set the right quantities (coordinates of the strong beam)
              ee.mean_x += temp_particles.x[0]
              ee.mean_y += temp_particles.y[0]

              ee.track(temp_particles)

              ee.d_px = temp_particles.px[0] - px_0
              ee.d_py = temp_particles.py[0] - py_0

              temp_particles.px -= ee.d_px
              temp_particles.py -= ee.d_py

        elif ee.__class__.__name__ == 'BeamBeamBiGaussian3D':
            ee.x_CO = temp_particles.x[0]
            ee.px_CO = temp_particles.px[0]
            ee.y_CO = temp_particles.y[0]
            ee.py_CO = temp_particles.py[0]
            ee.sigma_CO = temp_particles.zeta[0]
            ee.delta_CO = temp_particles.delta[0]

            ee.track(temp_particles)

            ee.Dx_sub = temp_particles.x[0] - ee.x_CO
            ee.Dpx_sub = temp_particles.px[0] - ee.px_CO
            ee.Dy_sub = temp_particles.y[0] - ee.y_CO
            ee.Dpy_sub = temp_particles.py[0] - ee.py_CO
            ee.Dsigma_sub = temp_particles.zeta[0] - ee.sigma_CO
            ee.Ddelta_sub = temp_particles.delta[0] - ee.delta_CO

            temp_particles.x[0] = ee.x_CO
            temp_particles.px[0] = ee.px_CO
            temp_particles.y[0] = ee.y_CO
            temp_particles.py[0] = ee.py_CO
            temp_particles.zeta[0] = ee.sigma_CO
            temp_particles.delta[0] = ee.delta_CO

        else:
            ee.track(temp_particles)
