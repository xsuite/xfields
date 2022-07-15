# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

from re import S, U
from scipy.constants import e as qe
import xobjects as xo
import xtrack as xt

from ..general import _pkg_root

class BeamBeamBiGaussian3D(xt.BeamElement):

    _xofields = {

        'sin_phi': xo.Float64,
        'cos_phi': xo.Float64,
        'tan_phi': xo.Float64,
        'sin_alpha': xo.Float64,
        'cos_alpha': xo.Float64,

        'ref_shift_x': xo.Float64,
        'ref_shift_px': xo.Float64,
        'ref_shift_y': xo.Float64,
        'ref_shift_py': xo.Float64,
        'ref_shift_zeta': xo.Float64,
        'ref_shift_pzeta': xo.Float64,

        'post_subtract_x': xo.Float64,
        'post_subtract_px': xo.Float64,
        'post_subtract_y': xo.Float64,
        'post_subtract_py': xo.Float64,
        'post_subtract_zeta': xo.Float64,
        'post_subtract_pzeta': xo.Float64,

        'q0_other_beam': xo.Float64,

        'num_slices_other_beam': xo.Int64,

        'slices_other_beam_num_particles': xo.Float64[:],
        'slices_other_beam_zeta_star_center': xo.Float64[:],
        'slices_other_beam_x_star_center': xo.Float64[:],
        'slices_other_beam_y_star_center': xo.Float64[:],

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

    def to_dict(self):

        raise NotImplementedError('To be updated')

        dct = super().to_dict()
        for nn in self._input_param_names:
            dct[nn] = getattr(self, nn)
        # For compatibility with ducktrack:
        dct['x_bb_co'] = self.delta_x
        dct['y_bb_co'] = self.delta_y
        dct['x_co'] = self.x_CO
        dct['px_co'] = self.px_CO
        dct['y_co'] = self.y_CO
        dct['py_co'] = self.py_CO
        dct['zeta_co'] = self.sigma_CO
        dct['delta_co'] = self.delta_CO
        dct['d_x'] = self.Dx_sub
        dct['d_px'] = self.Dpx_sub
        dct['d_y'] = self.Dy_sub
        dct['d_py'] = self.Dpy_sub
        dct['d_zeta'] = self.Dsigma_sub
        dct['d_delta'] = self.Ddelta_sub
        dct['charge_slices'] = self.N_part_per_slice
        dct['sigma_11'] = self.Sig_11_0
        dct['sigma_12'] = self.Sig_12_0
        dct['sigma_13'] = self.Sig_13_0
        dct['sigma_14'] = self.Sig_14_0
        dct['sigma_22'] = self.Sig_22_0
        dct['sigma_23'] = self.Sig_23_0
        dct['sigma_24'] = self.Sig_24_0
        dct['sigma_33'] = self.Sig_33_0
        dct['sigma_34'] = self.Sig_34_0
        dct['sigma_44'] = self.Sig_44_0
        return dct


    def __init__(self, **kwargs):

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
                slices_other_beam_zeta_star_center=n_slices,
                slices_other_beam_x_star_center=n_slices,
                slices_other_beam_y_star_center=n_slices

                )
            self._from_oldinterface(params)
        else:
            super().__init__(**kwargs)
            for nn in self._input_param_names:
                if nn in kwargs.keys():
                    setattr(self, nn, kwargs[nn])

    def _from_oldinterface(self, params):
        import ducktrack

        from scipy.constants import e as qe
        bb6d_data = ducktrack.BB6Ddata.BB6D_init(
                q_part=qe, # the ducktrack input has the charge
                phi=params["phi"],
                alpha=params["alpha"],
                delta_x=params["x_bb_co"],
                delta_y=params["y_bb_co"],
                N_part_per_slice=params["charge_slices"], #
                z_slices=params["zeta_slices"],
                Sig_11_0=params["sigma_11"],
                Sig_12_0=params["sigma_12"],
                Sig_13_0=params["sigma_13"],
                Sig_14_0=params["sigma_14"],
                Sig_22_0=params["sigma_22"],
                Sig_23_0=params["sigma_23"],
                Sig_24_0=params["sigma_24"],
                Sig_33_0=params["sigma_33"],
                Sig_34_0=params["sigma_34"],
                Sig_44_0=params["sigma_44"],
                x_CO=params["x_co"],
                px_CO=params["px_co"],
                y_CO=params["y_co"],
                py_CO=params["py_co"],
                sigma_CO=params["zeta_co"],
                delta_CO=params["delta_co"],
                min_sigma_diff=1e-10,
                threshold_singular=1e-28,
                Dx_sub=params["d_x"],
                Dpx_sub=params["d_px"],
                Dy_sub=params["d_y"],
                Dpy_sub=params["d_py"],
                Dsigma_sub=params["d_zeta"],
                Ddelta_sub=params["d_delta"],
                enabled=1,
            )
        assert(
            len(bb6d_data.N_part_per_slice) ==
            len(bb6d_data.x_slices_star) ==
            len(bb6d_data.y_slices_star) ==
            len(bb6d_data.sigma_slices_star))

        self.q0_other_beam = bb6d_data.q_part/qe, # ducktrack uses coulomb
        self.sin_phi = bb6d_data.parboost.sphi,
        self.cos_phi = bb6d_data.parboost.cphi,
        self.tan_phi = bb6d_data.parboost.tphi,
        self.sin_alpha = bb6d_data.parboost.salpha,
        self.cos_alpha = bb6d_data.parboost.calpha

        self.slices_other_beam_Sigma_11_star = bb6d_data.Sigmas_0_star.Sig_11_0
        self.slices_other_beam_Sigma_12_star = bb6d_data.Sigmas_0_star.Sig_12_0
        self.slices_other_beam_Sigma_13_star = bb6d_data.Sigmas_0_star.Sig_13_0
        self.slices_other_beam_Sigma_14_star = bb6d_data.Sigmas_0_star.Sig_14_0
        self.slices_other_beam_Sigma_22_star = bb6d_data.Sigmas_0_star.Sig_22_0
        self.slices_other_beam_Sigma_23_star = bb6d_data.Sigmas_0_star.Sig_23_0
        self.slices_other_beam_Sigma_24_star = bb6d_data.Sigmas_0_star.Sig_24_0
        self.slices_other_beam_Sigma_33_star = bb6d_data.Sigmas_0_star.Sig_33_0
        self.slices_other_beam_Sigma_34_star = bb6d_data.Sigmas_0_star.Sig_34_0
        self.slices_other_beam_Sigma_44_star = bb6d_data.Sigmas_0_star.Sig_44_0

        self.slices_other_beam_num_particles = bb6d_data.N_part_per_slice
        self.slices_other_beam_zeta_star_center = bb6d_data.sigma_slices_star
        self.slices_other_beam_x_star_center = bb6d_data.x_slices_star
        self.slices_other_beam_y_star_center = bb6d_data.y_slices_star

        self.ref_shift_x = bb6d_data.x_CO + bb6d_data.delta_x
        self.ref_shift_px = bb6d_data.px_CO
        self.ref_shift_y = bb6d_data.y_CO + bb6d_data.delta_y
        self.ref_shift_py = bb6d_data.py_CO
        self.ref_shift_zeta = bb6d_data.sigma_CO
        self.ref_shift_pzeta = bb6d_data.delta_CO

        self.post_subtract_x = bb6d_data.Dx_sub
        self.post_subtract_px = bb6d_data.Dpx_sub
        self.post_subtract_y = bb6d_data.Dy_sub
        self.post_subtract_py = bb6d_data.Dpy_sub
        self.post_subtract_zeta = bb6d_data.Dsigma_sub
        self.post_subtract_pzeta = bb6d_data.Ddelta_sub

        self.min_sigma_diff = bb6d_data.min_sigma_diff
        self.threshold_singular = bb6d_data.threshold_singular

        self.num_slices_other_beam = len(bb6d_data.N_part_per_slice)

    extra_sources= [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('headers/sincos.h'),
        _pkg_root.joinpath('headers/power_n.h'),
        _pkg_root.joinpath('fieldmaps/bigaussian_src/complex_error_function.h'),
        '#define NOFIELDMAP', #TODO Remove this workaround
        _pkg_root.joinpath('fieldmaps/bigaussian_src/bigaussian.h'),
        _pkg_root.joinpath('beam_elements/beambeam_src/beambeam3d.h'),
    ]

    per_particle_kernels=[
        {'kernel_name': 'boost_particles',
        'local_particle_function_name': 'boost_local_particle'}
    ]

    # Autogenerated properties (using the following code)
    '''
    for nn, factor in (
        ('11', '1.'),
        ('12', 'self.cos_phi'),
        ('13', '1.'),
        ('14', 'self.cos_phi'),
        ('22', '(self.cos_phi * self.cos_phi)'),
        ('23', 'self.cos_phi'),
        ('24', '(self.cos_phi * self.cos_phi)'),
        ('33', '1.'),
        ('34', 'self.cos_phi'),
        ('44', '(self.cos_phi * self.cos_phi)')):

        print(f"""
        @property
        def slices_other_beam_Sigma_{nn}(self):
            return self._buffer.context.linked_array_type.from_array(
                  self.slices_other_beam_Sigma_{nn}_star * {factor},
                  mode='setitem_from_container',
                  container=self,
                  container_setitem_name='_Sigma_{nn}_setitem')

        def _Sigma_{nn}_setitem(self, indx, val):
            self.slices_other_beam_Sigma_{nn}_star[indx] = val / {factor}

        @slices_other_beam_Sigma_{nn}.setter
        def slices_other_beam_Sigma_{nn}(self, value):
            self.slices_other_beam_Sigma_{nn}[:] = value
""")

    '''

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