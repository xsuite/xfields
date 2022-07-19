# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from tkinter import N
import numpy as np

import xobjects as xo
import xtrack as xt

from ..general import _pkg_root

class BeamBeamBiGaussian3D(xt.BeamElement):

    _xofields = {

        '_sin_phi': xo.Float64,
        '_cos_phi': xo.Float64,
        '_tan_phi': xo.Float64,
        '_sin_alpha': xo.Float64,
        '_cos_alpha': xo.Float64,

        'ref_shift_x': xo.Float64,
        'ref_shift_px': xo.Float64,
        'ref_shift_y': xo.Float64,
        'ref_shift_py': xo.Float64,
        'ref_shift_zeta': xo.Float64,
        'ref_shift_pzeta': xo.Float64,

        'other_beam_shift_x': xo.Float64,
        'other_beam_shift_px': xo.Float64,
        'other_beam_shift_y': xo.Float64,
        'other_beam_shift_py': xo.Float64,
        'other_beam_shift_zeta': xo.Float64,
        'other_beam_shift_pzeta': xo.Float64,

        'post_subtract_x': xo.Float64,
        'post_subtract_px': xo.Float64,
        'post_subtract_y': xo.Float64,
        'post_subtract_py': xo.Float64,
        'post_subtract_zeta': xo.Float64,
        'post_subtract_pzeta': xo.Float64,

        'q0_other_beam': xo.Float64,

        'num_slices_other_beam': xo.Int64,

        'slices_other_beam_num_particles': xo.Float64[:],

        'slices_other_beam_x_center_star': xo.Float64[:],
        'slices_other_beam_px_center_star': xo.Float64[:],
        'slices_other_beam_y_center_star': xo.Float64[:],
        'slices_other_beam_py_center_star': xo.Float64[:],
        'slices_other_beam_zeta_center_star': xo.Float64[:],
        'slices_other_beam_pzeta_center_star': xo.Float64[:],

        'slices_other_beam_Sigma_11_star': xo.Float64[:],
        'slices_other_beam_Sigma_12_star': xo.Float64[:],
        'slices_other_beam_Sigma_13_star': xo.Float64[:],
        'slices_other_beam_Sigma_14_star': xo.Float64[:],
        'slices_other_beam_Sigma_22_star': xo.Float64[:],
        'slices_other_beam_Sigma_23_star': xo.Float64[:],
        'slices_other_beam_Sigma_24_star': xo.Float64[:],
        'slices_other_beam_Sigma_33_star': xo.Float64[:],
        'slices_other_beam_Sigma_34_star': xo.Float64[:],
        'slices_other_beam_Sigma_44_star': xo.Float64[:],

        'min_sigma_diff': xo.Float64,
        'threshold_singular': xo.Float64,

    }

    extra_sources= [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('headers/sincos.h'),
        _pkg_root.joinpath('headers/power_n.h'),
        _pkg_root.joinpath('fieldmaps/bigaussian_src/complex_error_function.h'),
        '#define NOFIELDMAP', #TODO Remove this workaround
        _pkg_root.joinpath('fieldmaps/bigaussian_src/bigaussian.h'),
        _pkg_root.joinpath('beam_elements/beambeam_src/beambeam3d_transport_sigmas.h'),
        _pkg_root.joinpath('beam_elements/beambeam_src/beambeam3d_ref_frame_changes.h'),
        _pkg_root.joinpath('beam_elements/beambeam_src/beambeam3d.h'),
        _pkg_root.joinpath('beam_elements/beambeam_src/beambeam3d_methods_for_strongstrong.h'),
    ]

    per_particle_kernels={
        'synchro_beam_kick': xo.Kernel(
            c_name='BeamBeam3D_selective_apply_synchrobeam_kick_local_particle',
            args=[
                xo.Arg(xo.Int64, pointer=True, name='i_slice_for_particles')
            ]),
        'change_ref_frame': xo.Kernel(
            c_name='BeamBeamBiGaussian3D_change_ref_frame_local_particle',
            args=[]),
        'change_back_ref_frame_and_subtract_dipolar': xo.Kernel(
            c_name='BeamBeamBiGaussian3D_change_back_ref_frame_and_subtract_dipolar_local_particle',
            args=[]),
    }


    def __init__(self, **kwargs):

        if 'slicer' in kwargs.keys():
            self.slicer = kwargs['slicer']
            del kwargs['slicer']

        # Verify that slices are properly sorted (including Sigmas...)

        if 'old_interface' in kwargs:
            params=kwargs['old_interface']
            n_slices=len(params["charge_slices"])
            super().__init__(

                slices_other_beam_Sigma_11_star=n_slices,
                slices_other_beam_Sigma_12_star=n_slices,
                slices_other_beam_Sigma_13_star=n_slices,
                slices_other_beam_Sigma_14_star=n_slices,
                slices_other_beam_Sigma_22_star=n_slices,
                slices_other_beam_Sigma_23_star=n_slices,
                slices_other_beam_Sigma_24_star=n_slices,
                slices_other_beam_Sigma_33_star=n_slices,
                slices_other_beam_Sigma_34_star=n_slices,
                slices_other_beam_Sigma_44_star=n_slices,

                slices_other_beam_num_particles=n_slices,
                slices_other_beam_x_center_star=n_slices,
                slices_other_beam_px_center_star=n_slices,
                slices_other_beam_y_center_star=n_slices,
                slices_other_beam_py_center_star=n_slices,
                slices_other_beam_zeta_center_star=n_slices,
                slices_other_beam_pzeta_center_star=n_slices,

                )
            self._from_oldinterface(params)
        else:
            super().__init__(**kwargs)

    def _from_oldinterface(self, params):

        self.q0_other_beam = 1., # TODO: handle ions

        phi = params["phi"]
        alpha = params["alpha"]
        self._sin_phi = np.sin(phi)
        self._cos_phi = np.cos(phi)
        self._tan_phi = np.tan(phi)
        self._sin_alpha = np.sin(alpha)
        self._cos_alpha = np.cos(alpha)

        self.slices_other_beam_Sigma_11 = params['sigma_11']
        self.slices_other_beam_Sigma_12 = params['sigma_12']
        self.slices_other_beam_Sigma_13 = params['sigma_13']
        self.slices_other_beam_Sigma_14 = params['sigma_14']
        self.slices_other_beam_Sigma_22 = params['sigma_22']
        self.slices_other_beam_Sigma_23 = params['sigma_23']
        self.slices_other_beam_Sigma_24 = params['sigma_24']
        self.slices_other_beam_Sigma_33 = params['sigma_33']
        self.slices_other_beam_Sigma_34 = params['sigma_34']
        self.slices_other_beam_Sigma_44 = params['sigma_44']

        # Sort according to z, head at the first position in the arrays
        z_slices = np.array(params["zeta_slices"]).copy()
        N_part_per_slice = params["charge_slices"].copy()
        ind_sorted = np.argsort(z_slices)[::-1]
        z_slices = np.take(z_slices, ind_sorted)
        N_part_per_slice = np.take(N_part_per_slice, ind_sorted)

        (
        x_slices_star,
        px_slices_star,
        y_slices_star,
        py_slices_star,
        zeta_slices_star,
        pzeta_slices_star,
        ) = _python_boost(
            x=0 * z_slices,
            px=0 * z_slices,
            y=0 * z_slices,
            py=0 * z_slices,
            sigma=z_slices,
            delta=0 * z_slices,
            sphi = self.sin_phi,
            cphi = self.cos_phi,
            tphi = self.tan_phi,
            salpha = self.sin_alpha,
            calpha = self.cos_alpha,
        )

        self.slices_other_beam_num_particles = N_part_per_slice

        self.slices_other_beam_x_center_star = x_slices_star
        self.slices_other_beam_px_center_star = px_slices_star
        self.slices_other_beam_y_center_star = y_slices_star
        self.slices_other_beam_py_center_star = py_slices_star
        self.slices_other_beam_zeta_center_star = zeta_slices_star
        self.slices_other_beam_pzeta_center_star = pzeta_slices_star

        self.ref_shift_x = params['x_co']
        self.ref_shift_px = params['px_co']
        self.ref_shift_y = params['y_co']
        self.ref_shift_py = params['py_co']
        self.ref_shift_zeta = params['zeta_co']
        self.ref_shift_pzeta = params['delta_co']

        self.other_beam_shift_x =  params["x_bb_co"]
        self.other_beam_shift_px = 0
        self.other_beam_shift_y = params["y_bb_co"]
        self.other_beam_shift_py = 0
        self.other_beam_shift_zeta = 0
        self.other_beam_shift_pzeta = 0

        self.post_subtract_x = params['d_x']
        self.post_subtract_px = params['d_px']
        self.post_subtract_y = params['d_y']
        self.post_subtract_py = params['d_py']
        self.post_subtract_zeta = params['d_zeta']
        self.post_subtract_pzeta = params['d_delta']

        self.min_sigma_diff = 1e-10
        self.threshold_singular = 1e-28

        self.num_slices_other_beam = len(params["charge_slices"])

    def _track_dev(self, particles):

        particles_slice_indices = self.slicer.get_slice_indeces(particles)
        n_slices_self_beam = self.slicer.num_slices

        self.change_ref_frame(particles)

        i_slice_for_particles = np.zeros_like(particles.zeta, dtype=np.int64)
        for ii in range(self.num_slices_other_beam + n_slices_self_beam):
            i_slice_for_particles[:] = ii - particles_slice_indices
            i_slice_for_particles[particles_slice_indices < 0] = -1
            self.synchro_beam_kick(particles,
                                   i_slice_for_particles=i_slice_for_particles)

        self.change_back_ref_frame_and_subtract_dipolar(particles)

    @property
    def sin_phi(self):
        return self._sin_phi

    @property
    def cos_phi(self):
        return self._cos_phi

    @property
    def tan_phi(self):
        return self._tan_phi

    @property
    def sin_alpha(self):
        return self._sin_alpha

    @property
    def cos_alpha(self):
        return self._cos_alpha

    @property
    def phi(self):
        return np.arctan2(self.sin_phi, self.cos_phi)

    @phi.setter
    def phi(self, value):
        raise NotImplementedError("Setting phi is not implemented yet")

    @property
    def alpha(self):
        return np.arctan2(self.sin_alpha, self.cos_alpha)

    @alpha.setter
    def alpha(self, value):
        raise NotImplementedError("Setting alpha is not implemented yet")

    def _inv_boost_slice_centers(self):

        x_star_slices = self.slices_other_beam_x_center_star
        px_star_slices = self.slices_other_beam_px_center_star
        y_star_slices = self.slices_other_beam_y_center_star
        py_star_slices = self.slices_other_beam_py_center_star
        zeta_star_slices = self.slices_other_beam_zeta_center_star
        pzeta_star_slices = self.slices_other_beam_pzeta_center_star

        (
            x_slices,
            px_slices,
            y_slices,
            py_slices,
            zeta_slices,
            pzeta_slices,
        ) = _python_inv_boost(
            x_st=x_star_slices,
            px_st=px_star_slices,
            y_st=y_star_slices,
            py_st=py_star_slices,
            sigma_st=zeta_star_slices,
            delta_st=pzeta_star_slices,
            sphi=self.sin_phi,
            cphi=self.cos_phi,
            tphi=self.tan_phi,
            salpha=self.sin_alpha,
            calpha=self.cos_alpha,
        )

        return x_slices, px_slices, y_slices, py_slices, zeta_slices, pzeta_slices

        # The following properties are generate by this code:
        ## for nn in 'x px y py zeta pzeta'.split():
        ##     print(f'''
        ##     @property
        ##     def slices_other_beam_{nn}_center(self):
        ##         (x_slices, px_slices, y_slices, py_slices,
        ##             zeta_slices, pzeta_slices) = self._inv_boost_slice_centers()

        ##         return self._buffer.context.linked_array_type.from_array(
        ##             {nn}_slices,
        ##             mode="readonly")

        ##     @slices_other_beam_{nn}_center.setter
        ##     def slices_other_beam_{nn}_center(self, value):
        ##         raise NotImplementedError(
        ##             "Setting slices_other_beam_{nn}_center is not implemented yet")\n''')


    @property
    def slices_other_beam_x_center(self):
        (x_slices, px_slices, y_slices, py_slices,
            zeta_slices, pzeta_slices) = self._inv_boost_slice_centers()

        return self._buffer.context.linked_array_type.from_array(
            x_slices,
            mode="readonly")

    @slices_other_beam_x_center.setter
    def slices_other_beam_x_center(self, value):
        raise NotImplementedError(
            "Setting slices_other_beam_x_center is not implemented yet")


    @property
    def slices_other_beam_px_center(self):
        (x_slices, px_slices, y_slices, py_slices,
            zeta_slices, pzeta_slices) = self._inv_boost_slice_centers()

        return self._buffer.context.linked_array_type.from_array(
            px_slices,
            mode="readonly")

    @slices_other_beam_px_center.setter
    def slices_other_beam_px_center(self, value):
        raise NotImplementedError(
            "Setting slices_other_beam_px_center is not implemented yet")


    @property
    def slices_other_beam_y_center(self):
        (x_slices, px_slices, y_slices, py_slices,
            zeta_slices, pzeta_slices) = self._inv_boost_slice_centers()

        return self._buffer.context.linked_array_type.from_array(
            y_slices,
            mode="readonly")

    @slices_other_beam_y_center.setter
    def slices_other_beam_y_center(self, value):
        raise NotImplementedError(
            "Setting slices_other_beam_y_center is not implemented yet")


    @property
    def slices_other_beam_py_center(self):
        (x_slices, px_slices, y_slices, py_slices,
            zeta_slices, pzeta_slices) = self._inv_boost_slice_centers()

        return self._buffer.context.linked_array_type.from_array(
            py_slices,
            mode="readonly")

    @slices_other_beam_py_center.setter
    def slices_other_beam_py_center(self, value):
        raise NotImplementedError(
            "Setting slices_other_beam_py_center is not implemented yet")


    @property
    def slices_other_beam_zeta_center(self):
        (x_slices, px_slices, y_slices, py_slices,
            zeta_slices, pzeta_slices) = self._inv_boost_slice_centers()

        return self._buffer.context.linked_array_type.from_array(
            zeta_slices,
            mode="readonly")

    @slices_other_beam_zeta_center.setter
    def slices_other_beam_zeta_center(self, value):
        raise NotImplementedError(
            "Setting slices_other_beam_zeta_center is not implemented yet")


    @property
    def slices_other_beam_pzeta_center(self):
        (x_slices, px_slices, y_slices, py_slices,
            zeta_slices, pzeta_slices) = self._inv_boost_slice_centers()

        return self._buffer.context.linked_array_type.from_array(
            pzeta_slices,
            mode="readonly")

    @slices_other_beam_pzeta_center.setter
    def slices_other_beam_pzeta_center(self, value):
        raise NotImplementedError(
            "Setting slices_other_beam_pzeta_center is not implemented yet")

    # The following properties are generate by this code:
    ## for nn, factor in (
    ##     ('11', '1.'),
    ##     ('12', 'self.cos_phi'),
    ##     ('13', '1.'),
    ##     ('14', 'self.cos_phi'),
    ##     ('22', '(self.cos_phi * self.cos_phi)'),
    ##     ('23', 'self.cos_phi'),
    ##     ('24', '(self.cos_phi * self.cos_phi)'),
    ##     ('33', '1.'),
    ##     ('34', 'self.cos_phi'),
    ##     ('44', '(self.cos_phi * self.cos_phi)')):

    ##     print(f"""
    ##     @property
    ##     def slices_other_beam_Sigma_{nn}(self):
    ##         return self._buffer.context.linked_array_type.from_array(
    ##               self.slices_other_beam_Sigma_{nn}_star * {factor},
    ##               mode='setitem_from_container',
    ##               container=self,
    ##               container_setitem_name='_Sigma_{nn}_setitem')

    ##     def _Sigma_{nn}_setitem(self, indx, val):
    ##         self.slices_other_beam_Sigma_{nn}_star[indx] = val / {factor}
    ##
    ##     @slices_other_beam_Sigma_{nn}.setter
    ##     def slices_other_beam_Sigma_{nn}(self, value):
    ##         self.slices_other_beam_Sigma_{nn}[:] = value\n""")


    @property
    def slices_other_beam_Sigma_11(self):
        return self._buffer.context.linked_array_type.from_array(
              self.slices_other_beam_Sigma_11_star * 1.,
              mode='setitem_from_container',
              container=self,
              container_setitem_name='_Sigma_11_setitem')

    def _Sigma_11_setitem(self, indx, val):
        self.slices_other_beam_Sigma_11_star[indx] = val / 1.

    @slices_other_beam_Sigma_11.setter
    def slices_other_beam_Sigma_11(self, value):
        self.slices_other_beam_Sigma_11[:] = value


    @property
    def slices_other_beam_Sigma_12(self):
        return self._buffer.context.linked_array_type.from_array(
              self.slices_other_beam_Sigma_12_star * self.cos_phi,
              mode='setitem_from_container',
              container=self,
              container_setitem_name='_Sigma_12_setitem')

    def _Sigma_12_setitem(self, indx, val):
        self.slices_other_beam_Sigma_12_star[indx] = val / self.cos_phi

    @slices_other_beam_Sigma_12.setter
    def slices_other_beam_Sigma_12(self, value):
        self.slices_other_beam_Sigma_12[:] = value


    @property
    def slices_other_beam_Sigma_13(self):
        return self._buffer.context.linked_array_type.from_array(
              self.slices_other_beam_Sigma_13_star * 1.,
              mode='setitem_from_container',
              container=self,
              container_setitem_name='_Sigma_13_setitem')

    def _Sigma_13_setitem(self, indx, val):
        self.slices_other_beam_Sigma_13_star[indx] = val / 1.

    @slices_other_beam_Sigma_13.setter
    def slices_other_beam_Sigma_13(self, value):
        self.slices_other_beam_Sigma_13[:] = value


    @property
    def slices_other_beam_Sigma_14(self):
        return self._buffer.context.linked_array_type.from_array(
              self.slices_other_beam_Sigma_14_star * self.cos_phi,
              mode='setitem_from_container',
              container=self,
              container_setitem_name='_Sigma_14_setitem')

    def _Sigma_14_setitem(self, indx, val):
        self.slices_other_beam_Sigma_14_star[indx] = val / self.cos_phi

    @slices_other_beam_Sigma_14.setter
    def slices_other_beam_Sigma_14(self, value):
        self.slices_other_beam_Sigma_14[:] = value


    @property
    def slices_other_beam_Sigma_22(self):
        return self._buffer.context.linked_array_type.from_array(
              self.slices_other_beam_Sigma_22_star * (self.cos_phi * self.cos_phi),
              mode='setitem_from_container',
              container=self,
              container_setitem_name='_Sigma_22_setitem')

    def _Sigma_22_setitem(self, indx, val):
        self.slices_other_beam_Sigma_22_star[indx] = val / (self.cos_phi * self.cos_phi)

    @slices_other_beam_Sigma_22.setter
    def slices_other_beam_Sigma_22(self, value):
        self.slices_other_beam_Sigma_22[:] = value


    @property
    def slices_other_beam_Sigma_23(self):
        return self._buffer.context.linked_array_type.from_array(
              self.slices_other_beam_Sigma_23_star * self.cos_phi,
              mode='setitem_from_container',
              container=self,
              container_setitem_name='_Sigma_23_setitem')

    def _Sigma_23_setitem(self, indx, val):
        self.slices_other_beam_Sigma_23_star[indx] = val / self.cos_phi

    @slices_other_beam_Sigma_23.setter
    def slices_other_beam_Sigma_23(self, value):
        self.slices_other_beam_Sigma_23[:] = value


    @property
    def slices_other_beam_Sigma_24(self):
        return self._buffer.context.linked_array_type.from_array(
              self.slices_other_beam_Sigma_24_star * (self.cos_phi * self.cos_phi),
              mode='setitem_from_container',
              container=self,
              container_setitem_name='_Sigma_24_setitem')

    def _Sigma_24_setitem(self, indx, val):
        self.slices_other_beam_Sigma_24_star[indx] = val / (self.cos_phi * self.cos_phi)

    @slices_other_beam_Sigma_24.setter
    def slices_other_beam_Sigma_24(self, value):
        self.slices_other_beam_Sigma_24[:] = value


    @property
    def slices_other_beam_Sigma_33(self):
        return self._buffer.context.linked_array_type.from_array(
              self.slices_other_beam_Sigma_33_star * 1.,
              mode='setitem_from_container',
              container=self,
              container_setitem_name='_Sigma_33_setitem')

    def _Sigma_33_setitem(self, indx, val):
        self.slices_other_beam_Sigma_33_star[indx] = val / 1.

    @slices_other_beam_Sigma_33.setter
    def slices_other_beam_Sigma_33(self, value):
        self.slices_other_beam_Sigma_33[:] = value


    @property
    def slices_other_beam_Sigma_34(self):
        return self._buffer.context.linked_array_type.from_array(
              self.slices_other_beam_Sigma_34_star * self.cos_phi,
              mode='setitem_from_container',
              container=self,
              container_setitem_name='_Sigma_34_setitem')

    def _Sigma_34_setitem(self, indx, val):
        self.slices_other_beam_Sigma_34_star[indx] = val / self.cos_phi

    @slices_other_beam_Sigma_34.setter
    def slices_other_beam_Sigma_34(self, value):
        self.slices_other_beam_Sigma_34[:] = value


    @property
    def slices_other_beam_Sigma_44(self):
        return self._buffer.context.linked_array_type.from_array(
              self.slices_other_beam_Sigma_44_star * (self.cos_phi * self.cos_phi),
              mode='setitem_from_container',
              container=self,
              container_setitem_name='_Sigma_44_setitem')

    def _Sigma_44_setitem(self, indx, val):
        self.slices_other_beam_Sigma_44_star[indx] = val / (self.cos_phi * self.cos_phi)

    @slices_other_beam_Sigma_44.setter
    def slices_other_beam_Sigma_44(self, value):
        self.slices_other_beam_Sigma_44[:] = value


# Used only in properties, not in actual tracking
def _python_boost_scalar(x, px, y, py, sigma, delta,
                  sphi, cphi, tphi, salpha, calpha):

    h = (
        delta
        + 1.0
        - np.sqrt((1.0 + delta) * (1.0 + delta) - px * px - py * py)
    )

    px_st = px / cphi - h * calpha * tphi / cphi
    py_st = py / cphi - h * salpha * tphi / cphi
    delta_st = (
        delta - px * calpha * tphi - py * salpha * tphi + h * tphi * tphi
    )

    pz_st = np.sqrt(
        (1.0 + delta_st) * (1.0 + delta_st) - px_st * px_st - py_st * py_st
    )
    hx_st = px_st / pz_st
    hy_st = py_st / pz_st
    hsigma_st = 1.0 - (delta_st + 1) / pz_st

    L11 = 1.0 + hx_st * calpha * sphi
    L12 = hx_st * salpha * sphi
    L13 = calpha * tphi

    L21 = hy_st * calpha * sphi
    L22 = 1.0 + hy_st * salpha * sphi
    L23 = salpha * tphi

    L31 = hsigma_st * calpha * sphi
    L32 = hsigma_st * salpha * sphi
    L33 = 1.0 / cphi

    x_st = L11 * x + L12 * y + L13 * sigma
    y_st = L21 * x + L22 * y + L23 * sigma
    sigma_st = L31 * x + L32 * y + L33 * sigma

    return x_st, px_st, y_st, py_st, sigma_st, delta_st

_python_boost = np.vectorize(_python_boost_scalar,
    excluded=("sphi", "cphi", "tphi", "salpha", "calpha"))

def _python_inv_boost_scalar(x_st, px_st, y_st, py_st, sigma_st, delta_st,
                  sphi, cphi, tphi, salpha, calpha):

    pz_st = np.sqrt(
        (1.0 + delta_st) * (1.0 + delta_st) - px_st * px_st - py_st * py_st
    )
    hx_st = px_st / pz_st
    hy_st = py_st / pz_st
    hsigma_st = 1.0 - (delta_st + 1) / pz_st

    Det_L = (
        1.0 / cphi
        + (hx_st * calpha + hy_st * salpha - hsigma_st * sphi) * tphi
    )

    Linv_11 = (
        1.0 / cphi + salpha * tphi * (hy_st - hsigma_st * salpha * sphi)
    ) / Det_L
    Linv_12 = (salpha * tphi * (hsigma_st * calpha * sphi - hx_st)) / Det_L
    Linv_13 = (
        -tphi
        * (
            calpha
            - hx_st * salpha * salpha * sphi
            + hy_st * calpha * salpha * sphi
        )
        / Det_L
    )

    Linv_21 = (calpha * tphi * (-hy_st + hsigma_st * salpha * sphi)) / Det_L
    Linv_22 = (
        1.0 / cphi + calpha * tphi * (hx_st - hsigma_st * calpha * sphi)
    ) / Det_L
    Linv_23 = (
        -tphi
        * (
            salpha
            - hy_st * calpha * calpha * sphi
            + hx_st * calpha * salpha * sphi
        )
        / Det_L
    )

    Linv_31 = -hsigma_st * calpha * sphi / Det_L
    Linv_32 = -hsigma_st * salpha * sphi / Det_L
    Linv_33 = (1.0 + hx_st * calpha * sphi + hy_st * salpha * sphi) / Det_L

    x_i = Linv_11 * x_st + Linv_12 * y_st + Linv_13 * sigma_st
    y_i = Linv_21 * x_st + Linv_22 * y_st + Linv_23 * sigma_st
    sigma_i = Linv_31 * x_st + Linv_32 * y_st + Linv_33 * sigma_st

    h = (delta_st + 1.0 - pz_st) * cphi * cphi

    px_i = px_st * cphi + h * calpha * tphi
    py_i = py_st * cphi + h * salpha * tphi

    delta_i = (
        delta_st
        + px_i * calpha * tphi
        + py_i * salpha * tphi
        - h * tphi * tphi
    )

    return x_i, px_i, y_i, py_i, sigma_i, delta_i

_python_inv_boost = np.vectorize(_python_inv_boost_scalar,
    excluded=("sphi", "cphi", "tphi", "salpha", "calpha"))

class TempSlicer:
    def __init__(self, bin_edges):

        bin_edges = np.sort(np.array(bin_edges))[::-1]
        self.bin_edges = bin_edges
        self.bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        self.num_slices = len(bin_edges) - 1

    def get_slice_indeces(self, particles):
        indices = np.digitize(particles.zeta, self.bin_edges, right=True)
        indices[particles.state <=0 ] = -1

        return np.array(indices, dtype=np.int64)