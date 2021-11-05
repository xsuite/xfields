import xtrack as xt


def configure_orbit_dependent_parameters_for_bb(tracker, particle_on_co, xline=None):

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
              print(ii,temp_particles.x[0], temp_particles.y[0],
                      ee.mean_x, ee.mean_y,
                    ee.d_px, ee.d_py)
              if xline is not None:
                  xline.elements[ii].x_bb = ee.mean_x
                  xline.elements[ii].y_bb = ee.mean_y
                  xline.elements[ii].d_px = ee.d_px
                  xline.elements[ii].d_py = ee.d_py

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

            if xline is not None:
                xline.elements[ii].x_co = ee.x_CO
                xline.elements[ii].px_co = ee.px_CO
                xline.elements[ii].y_co = ee.y_CO
                xline.elements[ii].py_co = ee.py_CO
                xline.elements[ii].zeta_co = ee.sigma_CO
                xline.elements[ii].delta_co = ee.delta_CO

                xline.elements[ii].d_x = ee.Dx_sub
                xline.elements[ii].d_px = ee.Dpx_sub
                xline.elements[ii].d_y = ee.Dy_sub
                xline.elements[ii].d_py = ee.Dpy_sub
                xline.elements[ii].d_zeta = ee.Dsigma_sub
                xline.elements[ii].d_delta = ee.Ddelta_sub
        else:
            ee.track(temp_particles)
