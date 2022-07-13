import xobjects as xo
import xtrack as xt



class BeamBeam3D(xt.BeamElement):

    _xofields = {
        'q0': xo.Float64,

        # Angles used to move particles to the computation reference frame:
        'alpha': xo.Float64,   # Orientation of the crossing plane
        'phi': xo.Float64,     # Half crossing angle

        # Shifts used to move the particles to the computation reference frame:
        'ref_shift_x': xo.Float64,
        'ref_shift_px': xo.Float64,
        'ref_shift_y': xo.Float64,
        'ref_shift_py': xo.Float64,
        'ref_shift_zeta': xo.Float64,
        'ref_shift_delta_shift': xo.Float64,

        # Number of slices (self and other)
        'num_slices_self': xo.Int64,
        'num_slices_other': xo.Int64,

        #### Slice self parameters
        # Binning
        'slices_self_z_star_center': xo.Float64[:],
        'slices_self_z_star_width': xo.Float64[:],
        # Binning
        'slices_other_z_star_center': xo.Float64[:],
        'slices_other_z_star_width': xo.Float64[:],
        # Intensity
        'slices_self_num_particles': xo.Float64[:],
        # First order momenta in the computation reference frame (star):
        'slices_self_x_star': xo.Float64[:],
        'slices_self_px_star': xo.Float64[:],
        'slices_self_y_star': xo.Float64[:],
        'slices_self_py_star': xo.Float64[:],
        # Second order momenta in the computation reference frame (star):
        'slices_self_Sigma_11_star': xo.Float64,
        'slices_self_Sigma_12_star': xo.Float64,
        'slices_self_Sigma_13_star': xo.Float64,
        'slices_self_Sigma_14_star': xo.Float64,
        'slices_self_Sigma_22_star': xo.Float64,
        'slices_self_Sigma_23_star': xo.Float64,
        'slices_self_Sigma_24_star': xo.Float64,
        'slices_self_Sigma_33_star': xo.Float64,
        'slices_self_Sigma_34_star': xo.Float64,
        'slices_self_Sigma_44_star': xo.Float64,

        #### Slice other parameters
        # Binning
        'slices_other_z_star_center': xo.Float64[:],
        'slices_other_z_star_width': xo.Float64[:],
        # Binning
        'slices_other_z_star_center': xo.Float64[:],
        'slices_other_z_star_width': xo.Float64[:],
        # Intensity
        'slices_other_num_particles': xo.Float64[:],
        # First order momenta in the computation reference frame (star):
        'slices_other_x_star': xo.Float64[:],
        'slices_other_px_star': xo.Float64[:],
        'slices_other_y_star': xo.Float64[:],
        'slices_other_py_star': xo.Float64[:],
        # Second order momenta in the computation reference frame (star):
        'slices_other_Sigma_11_star': xo.Float64,
        'slices_other_Sigma_12_star': xo.Float64,
        'slices_other_Sigma_13_star': xo.Float64,
        'slices_other_Sigma_14_star': xo.Float64,
        'slices_other_Sigma_22_star': xo.Float64,
        'slices_other_Sigma_23_star': xo.Float64,
        'slices_other_Sigma_24_star': xo.Float64,
        'slices_other_Sigma_33_star': xo.Float64,
        'slices_other_Sigma_34_star': xo.Float64,
        'slices_other_Sigma_44_star': xo.Float64,

        #### Kick  after the computation
        #### (e.g. to artificially remove dipolar effects)
        'subtract_x': xo.Float64,
        'subtract_px': xo.Float64,
        'subtract_y': xo.Float64,
        'subtract_py': xo.Float64,
        'subtract_zeta': xo.Float64,
        'subtract_delta': xo.Float64,

        #### Tolerances for special cases
        'min_sigma_diff': xo.Float64,
        'threshold_singular': xo.Float64,
    }

    def track(self, particles): # For collective mode

        #### Move particles to the computation reference frame
        
        # Shift
        particles.x += self.ref_shift_x
        particles.px += self.ref_shift_px
        particles.y += self.ref_shift_y
        particles.py += self.ref_shift_py
        particles.zeta += self.ref_shift_zeta
        particles.delta += self.ref_shift_delta_shift

        # Rotate and boost
        self._boost_particles(particles)



