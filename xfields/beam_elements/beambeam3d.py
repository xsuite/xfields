# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
from scipy import special 
 
import xobjects as xo
import xtrack as xt
import xpart as xp

from ..general import _pkg_root

class BeamstrahlungTable(xo.HybridClass):
    _xofields = {
      '_index': xt.RecordIndex,
      'at_element': xo.Int64[:],
      'at_turn': xo.Int64[:],
      'particle_id': xo.Int64[:],
      'photon_id': xo.Float64[:], 
      'photon_energy': xo.Float64[:],
      'photon_critical_energy': xo.Float64[:],
      'primary_energy': xo.Float64[:],
      'rho_inv': xo.Float64[:],
        }

class BeamBeamBiGaussian3DRecord(xo.HybridClass):
    _xofields = {
        'beamstrahlungtable': BeamstrahlungTable,
       }

class BeamBeamBiGaussian3D(xt.BeamElement):

    _xofields = {

        'scale_strength': xo.Float64,

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

        'other_beam_q0': xo.Float64,

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

        # beamstrahlung 
        'flag_beamstrahlung': xo.Int64,
        'slices_other_beam_zeta_bin_width_star_beamstrahlung': xo.Float64[:],
        'other_beam_sigma_55_star_beamstrahlung': xo.Float64,

    }

    _internal_record_class = BeamBeamBiGaussian3DRecord

    _extra_c_sources= [
        _pkg_root.joinpath('headers/constants.h'),
        _pkg_root.joinpath('headers/sincos.h'),
        _pkg_root.joinpath('headers/power_n.h'),
        _pkg_root.joinpath('fieldmaps/bigaussian_src/faddeeva.h'),
        '#define NOFIELDMAP', #TODO Remove this workaround
        _pkg_root.joinpath('fieldmaps/bigaussian_src/bigaussian.h'),
        '#undef NOFIELDMAP', #TODO Remove this workaround
        _pkg_root.joinpath('beam_elements/beambeam_src/beambeam3d_transport_sigmas.h'),
        _pkg_root.joinpath('beam_elements/beambeam_src/beambeam3d_ref_frame_changes.h'),

        # beamstrahlung
        xp.general._pkg_root.joinpath('random_number_generator/rng_src/base_rng.h'),
        xp.general._pkg_root.joinpath('random_number_generator/rng_src/local_particle_rng.h'),
        _pkg_root.joinpath('headers/beamstrahlung_spectrum.h'),

        _pkg_root.joinpath('beam_elements/beambeam_src/beambeam3d.h'),
        _pkg_root.joinpath('beam_elements/beambeam_src/beambeam3d_methods_for_strongstrong.h'),

   ]

    _per_particle_kernels={
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

    def __init__(self,
                    phi=None, alpha=None, other_beam_q0=None, particles_per_macroparticle = None,
                    scale_strength = 1.,

                    slices_other_beam_num_particles=None,

                    slices_other_beam_x_center=0.,
                    slices_other_beam_px_center=0.,
                    slices_other_beam_y_center=0.,
                    slices_other_beam_py_center=0.,
                    slices_other_beam_zeta_center=None,
                    slices_other_beam_pzeta_center=0.,

                    flag_beamstrahlung=0,
 
                    slices_other_beam_zeta_bin_width_star_beamstrahlung=None,
                    other_beam_sigma_55_star_beamstrahlung=None,

                    slices_other_beam_x_center_star=None,
                    slices_other_beam_px_center_star=None,
                    slices_other_beam_y_center_star=None,
                    slices_other_beam_py_center_star=None,
                    slices_other_beam_zeta_center_star=None,
                    slices_other_beam_pzeta_center_star=None,

                    slices_other_beam_Sigma_11=None,
                    slices_other_beam_Sigma_12=None,
                    slices_other_beam_Sigma_13=None,
                    slices_other_beam_Sigma_14=None,
                    slices_other_beam_Sigma_22=None,
                    slices_other_beam_Sigma_23=None,
                    slices_other_beam_Sigma_24=None,
                    slices_other_beam_Sigma_33=None,
                    slices_other_beam_Sigma_34=None,
                    slices_other_beam_Sigma_44=None,

                    slices_other_beam_Sigma_11_star=None,
                    slices_other_beam_Sigma_12_star=None,
                    slices_other_beam_Sigma_13_star=None,
                    slices_other_beam_Sigma_14_star=None,
                    slices_other_beam_Sigma_22_star=None,
                    slices_other_beam_Sigma_23_star=None,
                    slices_other_beam_Sigma_24_star=None,
                    slices_other_beam_Sigma_33_star=None,
                    slices_other_beam_Sigma_34_star=None,
                    slices_other_beam_Sigma_44_star=None,

                    ref_shift_x=0,
                    ref_shift_px=0,
                    ref_shift_y=0,
                    ref_shift_py=0,
                    ref_shift_zeta=0,
                    ref_shift_pzeta=0,

                    other_beam_shift_x=0,
                    other_beam_shift_px=0,
                    other_beam_shift_y=0,
                    other_beam_shift_py=0,
                    other_beam_shift_zeta=0,
                    other_beam_shift_pzeta=0,

                    post_subtract_x=0, post_subtract_px=0,
                    post_subtract_y=0, post_subtract_py=0,
                    post_subtract_zeta=0, post_subtract_pzeta=0,

                    min_sigma_diff=1e-10,
                    threshold_singular = 1e-28,

                    old_interface=None,

                    config_for_update=None,

                    _sin_phi=None, _cos_phi=None, _tan_phi=None,
                    _sin_alpha=None, _cos_alpha=None,

                    **kwargs):

        if '_xobject' in kwargs.keys():
            self.xoinitialize(**kwargs)
            return

        # Collective mode (pipeline update)
        if config_for_update is not None:

            self.config_for_update = config_for_update
            self.iscollective = True
            self.track = self._track_collective # switch to specific track method

            if slices_other_beam_zeta_center is None:
                if isinstance(self.config_for_update.slicer, Slicer):
                    slices_other_beam_zeta_center = self.config_for_update.slicer.bin_centers * self.config_for_update.slicer.sigma_z
                else:
                    slices_other_beam_zeta_center = self.config_for_update.slicer.bin_centers
            # Some dummy values just to initialize the object
            if (slices_other_beam_Sigma_11 is None
                    and slices_other_beam_Sigma_11_star is None):
                slices_other_beam_Sigma_11 = 1.
            if (slices_other_beam_Sigma_12 is None
                    and slices_other_beam_Sigma_12_star is None):
                slices_other_beam_Sigma_12 = 0.
            if (slices_other_beam_Sigma_22 is None
                    and slices_other_beam_Sigma_22_star is None):
                slices_other_beam_Sigma_22 = 1.
            if (slices_other_beam_Sigma_33 is None
                    and slices_other_beam_Sigma_33_star is None):
                slices_other_beam_Sigma_33 = 1.
            if (slices_other_beam_Sigma_34 is None
                    and slices_other_beam_Sigma_34_star is None):
                slices_other_beam_Sigma_34 = 0.
            if (slices_other_beam_Sigma_44 is None
                    and slices_other_beam_Sigma_44_star is None):
                slices_other_beam_Sigma_44 = 1.

            if slices_other_beam_num_particles is None:
                slices_other_beam_num_particles = np.zeros_like(
                                            slices_other_beam_zeta_center)
            # beamstrahlung
            if slices_other_beam_zeta_bin_width_star_beamstrahlung is None and flag_beamstrahlung == 1:
                if isinstance(self.config_for_update.slicer, Slicer):
                    slices_other_beam_zeta_bin_width_star_beamstrahlung = slicer.bin_widths_beamstrahlung * slicer.sigma_z / np.cos(self.phi)
                else:
                    slices_other_beam_zeta_bin_width_star_beamstrahlung = np.abs(np.diff(self.config_for_update.slicer.bin_edges))/np.cos(self.phi)
            self.moments = None
            self.partner_moments = np.zeros(self.config_for_update.slicer.num_slices*(1+6+10),dtype=float)

            self.particles_per_macroparticle = particles_per_macroparticle


        if old_interface is not None:
            self._init_from_old_interface(old_interface=old_interface, **kwargs)
            return

        assert (slices_other_beam_zeta_center is not None
                or slices_other_beam_zeta_center_star is not None)
        assert slices_other_beam_num_particles is not None

        assert not np.isscalar(slices_other_beam_num_particles), (
                        'slices_other_beam_num_particles must be an array')

        if slices_other_beam_zeta_center is not None:
            assert not np.isscalar(slices_other_beam_zeta_center), (
                            'slices_other_beam_zeta_center must be an array')
            assert (len(slices_other_beam_zeta_center)
                        == len(slices_other_beam_num_particles))

        if slices_other_beam_zeta_center_star is not None:
            assert not np.isscalar(slices_other_beam_zeta_center_star), (
                            'slices_other_beam_zeta_center_star must be an array')
            assert (len(slices_other_beam_zeta_center_star)
                        == len(slices_other_beam_num_particles))

        # beamstrahlung
        if flag_beamstrahlung == 1:
            assert slices_other_beam_zeta_bin_width_star_beamstrahlung is not None, (
                'slices_other_beam_zeta_bin_width_star_beamstrahlung must be provided if flag_beamstrahlung = 1')
            assert not np.isscalar(slices_other_beam_zeta_bin_width_star_beamstrahlung), (
                            'slices_other_beam_zeta_bin_width_star_beamstrahlung must be an array')
            assert (len(slices_other_beam_zeta_bin_width_star_beamstrahlung)
                        == len(slices_other_beam_num_particles))
        elif flag_beamstrahlung == 2:
            assert other_beam_sigma_55_star_beamstrahlung is not None, (
                'other_beam_sigma_55_star_beamstrahlung must be provided if flag_beamstrahlung = 2')

        n_slices = len(slices_other_beam_num_particles)

        self._allocate_xobject(n_slices, **kwargs)

        #if self.iscollective:
        #    if not isinstance(self._buffer.context, xo.ContextCpu):
        #        raise NotImplementedError(
        #            'BeamBeamBiGaussian3D only works with CPU context for now')

        if phi is None:
            assert _sin_phi is not None and _cos_phi is not None and _tan_phi is not None, (
                'phi must be specified if _sin_phi, _cos_phi, _tan_phi are not')
            self._sin_phi = _sin_phi
            self._cos_phi = _cos_phi
            self._tan_phi = _tan_phi
        else:
            self._sin_phi = np.sin(phi)
            self._cos_phi = np.cos(phi)
            self._tan_phi = np.tan(phi)

        if alpha is None:
            assert _sin_alpha is not None and _cos_alpha is not None, (
                'alpha must be specified if _sin_alpha, _cos_alpha are not')
            self._sin_alpha = _sin_alpha
            self._cos_alpha = _cos_alpha
        else:
            self._sin_alpha = np.sin(alpha)
            self._cos_alpha = np.cos(alpha)

        self.num_slices_other_beam = n_slices
        self.slices_other_beam_num_particles = self._arr2ctx(np.array(
                                    slices_other_beam_num_particles))

        # Trigger properties to set corresponding starred quantities
        self._init_Sigmas(
            slices_other_beam_Sigma_11, slices_other_beam_Sigma_12,
            slices_other_beam_Sigma_13, slices_other_beam_Sigma_14,
            slices_other_beam_Sigma_22, slices_other_beam_Sigma_23,
            slices_other_beam_Sigma_24, slices_other_beam_Sigma_33,
            slices_other_beam_Sigma_34, slices_other_beam_Sigma_44,
            slices_other_beam_Sigma_11_star, slices_other_beam_Sigma_12_star,
            slices_other_beam_Sigma_13_star, slices_other_beam_Sigma_14_star,
            slices_other_beam_Sigma_22_star, slices_other_beam_Sigma_23_star,
            slices_other_beam_Sigma_24_star, slices_other_beam_Sigma_33_star,
            slices_other_beam_Sigma_34_star, slices_other_beam_Sigma_44_star,
            )

        # Initialize slice positions in the boosted frame
        self._init_starred_positions(
            slices_other_beam_num_particles,
            slices_other_beam_x_center, slices_other_beam_px_center,
            slices_other_beam_y_center, slices_other_beam_py_center,
            slices_other_beam_zeta_center, slices_other_beam_pzeta_center,
            slices_other_beam_x_center_star, slices_other_beam_px_center_star,
            slices_other_beam_y_center_star, slices_other_beam_py_center_star,
            slices_other_beam_zeta_center_star, slices_other_beam_pzeta_center_star)


        assert other_beam_q0 is not None
        self.other_beam_q0 = other_beam_q0
        self.scale_strength = scale_strength

        self.ref_shift_x = ref_shift_x
        self.ref_shift_px = ref_shift_px
        self.ref_shift_y = ref_shift_y
        self.ref_shift_py = ref_shift_py
        self.ref_shift_zeta = ref_shift_zeta
        self.ref_shift_pzeta = ref_shift_pzeta

        self.other_beam_shift_x = other_beam_shift_x
        self.other_beam_shift_px = other_beam_shift_px
        self.other_beam_shift_y = other_beam_shift_y
        self.other_beam_shift_py = other_beam_shift_py
        self.other_beam_shift_zeta = other_beam_shift_zeta
        self.other_beam_shift_pzeta = other_beam_shift_pzeta

        self.post_subtract_x = post_subtract_x
        self.post_subtract_px = post_subtract_px
        self.post_subtract_y = post_subtract_y
        self.post_subtract_py = post_subtract_py
        self.post_subtract_zeta = post_subtract_zeta
        self.post_subtract_pzeta = post_subtract_pzeta

        self.min_sigma_diff = min_sigma_diff
        self.threshold_singular = threshold_singular

        # beamstrahlung
        self.flag_beamstrahlung = flag_beamstrahlung
        self.slices_other_beam_zeta_bin_width_star_beamstrahlung = self._arr2ctx(np.array(slices_other_beam_zeta_bin_width_star_beamstrahlung))
        self.other_beam_sigma_55_star_beamstrahlung = other_beam_sigma_55_star_beamstrahlung
 
    def _allocate_xobject(self, n_slices, **kwargs):
        self.xoinitialize(
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
            slices_other_beam_zeta_bin_width_star_beamstrahlung=n_slices,  # beamstrahlung
            **kwargs
            )

    def _init_from_old_interface(self, old_interface, **kwargs):

        params=old_interface
        n_slices=len(params["charge_slices"])

        self._allocate_xobject(n_slices, **kwargs)

        self.other_beam_q0 = 1., # TODO: handle ions
        self.scale_strength = 1.

        phi = params["phi"]
        alpha = params["alpha"]
        self._sin_phi = np.sin(phi)
        self._cos_phi = np.cos(phi)
        self._tan_phi = np.tan(phi)
        self._sin_alpha = np.sin(alpha)
        self._cos_alpha = np.cos(alpha)

        self.slices_other_beam_Sigma_11 = self._arr2ctx(params['sigma_11'])
        self.slices_other_beam_Sigma_12 = self._arr2ctx(params['sigma_12'])
        self.slices_other_beam_Sigma_13 = self._arr2ctx(params['sigma_13'])
        self.slices_other_beam_Sigma_14 = self._arr2ctx(params['sigma_14'])
        self.slices_other_beam_Sigma_22 = self._arr2ctx(params['sigma_22'])
        self.slices_other_beam_Sigma_23 = self._arr2ctx(params['sigma_23'])
        self.slices_other_beam_Sigma_24 = self._arr2ctx(params['sigma_24'])
        self.slices_other_beam_Sigma_33 = self._arr2ctx(params['sigma_33'])
        self.slices_other_beam_Sigma_34 = self._arr2ctx(params['sigma_34'])
        self.slices_other_beam_Sigma_44 = self._arr2ctx(params['sigma_44'])

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
            zeta=z_slices,
            pzeta=0 * z_slices,
            sphi = self.sin_phi,
            cphi = self.cos_phi,
            tphi = self.tan_phi,
            salpha = self.sin_alpha,
            calpha = self.cos_alpha,
        )

        self.slices_other_beam_num_particles = self._arr2ctx(N_part_per_slice)

        self.slices_other_beam_x_center_star = self._arr2ctx(x_slices_star)
        self.slices_other_beam_px_center_star = self._arr2ctx(px_slices_star)
        self.slices_other_beam_y_center_star = self._arr2ctx(y_slices_star)
        self.slices_other_beam_py_center_star = self._arr2ctx(py_slices_star)
        self.slices_other_beam_zeta_center_star = self._arr2ctx(zeta_slices_star)
        self.slices_other_beam_pzeta_center_star = self._arr2ctx(pzeta_slices_star)

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

    def update_from_recieved_moments(self):
        # reference frame transformation as in https://github.com/lhcopt/lhcmask/blob/865eaf9d7b9b888c6486de00214c0c24ac93cfd3/pymask/beambeam.py#L310
        self.slices_other_beam_num_particles = self._arr2ctx(self.partner_moments[:self.num_slices_other_beam])

        self.slices_other_beam_x_center_star = self._arr2ctx(self.partner_moments[self.num_slices_other_beam:2*self.num_slices_other_beam]) * (-1.0)
        self.slices_other_beam_px_center_star = self._arr2ctx(self.partner_moments[2*self.num_slices_other_beam:3*self.num_slices_other_beam])
        self.slices_other_beam_y_center_star = self._arr2ctx(self.partner_moments[3*self.num_slices_other_beam:4*self.num_slices_other_beam])
        self.slices_other_beam_py_center_star = self._arr2ctx(self.partner_moments[4*self.num_slices_other_beam:5*self.num_slices_other_beam]) * (-1.0)
        self.slices_other_beam_zeta_center_star = self._arr2ctx(self.partner_moments[5*self.num_slices_other_beam:6*self.num_slices_other_beam])
        self.slices_other_beam_pzeta_center_star = self._arr2ctx(self.partner_moments[6*self.num_slices_other_beam:7*self.num_slices_other_beam])

        self.slices_other_beam_Sigma_11_star = self._arr2ctx(self.partner_moments[7*self.num_slices_other_beam:8*self.num_slices_other_beam])
        self.slices_other_beam_Sigma_12_star = self._arr2ctx(self.partner_moments[8*self.num_slices_other_beam:9*self.num_slices_other_beam]) * (-1.0)
        self.slices_other_beam_Sigma_13_star = self._arr2ctx(self.partner_moments[9*self.num_slices_other_beam:10*self.num_slices_other_beam]) * (-1.0)
        self.slices_other_beam_Sigma_14_star = self._arr2ctx(self.partner_moments[10*self.num_slices_other_beam:11*self.num_slices_other_beam])
        self.slices_other_beam_Sigma_22_star = self._arr2ctx(self.partner_moments[11*self.num_slices_other_beam:12*self.num_slices_other_beam])
        self.slices_other_beam_Sigma_23_star = self._arr2ctx(self.partner_moments[12*self.num_slices_other_beam:13*self.num_slices_other_beam])
        self.slices_other_beam_Sigma_24_star = self._arr2ctx(self.partner_moments[13*self.num_slices_other_beam:14*self.num_slices_other_beam]) * (-1.0)
        self.slices_other_beam_Sigma_33_star = self._arr2ctx(self.partner_moments[14*self.num_slices_other_beam:15*self.num_slices_other_beam])
        self.slices_other_beam_Sigma_34_star = self._arr2ctx(self.partner_moments[15*self.num_slices_other_beam:16*self.num_slices_other_beam]) * (-1.0)
        self.slices_other_beam_Sigma_44_star = self._arr2ctx(self.partner_moments[16*self.num_slices_other_beam:17*self.num_slices_other_beam])

    def _track_collective(self, particles, _force_suspend=False):

        if self.config_for_update._working_on_bunch is not None:
            # I am resuming a suspended calculation

            assert self.config_for_update._working_on_bunch == particles.name

            # Beam beam interaction in the boosted frame
            ret = self._apply_bb_kicks_in_boosted_frame(particles)

            if ret is not None:
                return ret # PipelineStatus
            else:
                # Back to line reference frame
                self.change_back_ref_frame_and_subtract_dipolar(particles)
                return None

        else:
            # I am working on a new bunch

            if particles._num_active_particles == 0:
                return # All particles are lost

            # Check that the element is not occupied by a bunch
            assert self.config_for_update._i_step == 0
            assert self.config_for_update._working_on_bunch is None

            self.config_for_update._working_on_bunch = particles.name

            # Slice bunch (in the lab frame)
            self.config_for_update._particles_slice_index = (
                            self.config_for_update.slicer.get_slice_indices(particles))
            self.config_for_update._other_beam_slice_index_for_particles = np.zeros_like(
                self.config_for_update._particles_slice_index)

            # Handle update frequency
            at_turn = particles._xobject.at_turn[0] # On CPU there is always an active particle in position 0
            if (self.config_for_update.update_every is not None
                    and at_turn % self.config_for_update.update_every == 0):
                self.config_for_update._do_update = True
            else:
                self.config_for_update._do_update = False

            # Change reference frame
            self.change_ref_frame(particles)

            # Can be used to test the resume without pipeline
            if _force_suspend:
                return xt.PipelineStatus(on_hold=True)

            # Beam beam interaction in the boosted frame
            ret = self._apply_bb_kicks_in_boosted_frame(particles)

            if ret is not None:
                return ret # PipelineStatus
            else:
                # Back to line reference frame
                self.change_back_ref_frame_and_subtract_dipolar(particles)
                return None

    def _apply_bb_kicks_in_boosted_frame(self, particles):

        n_slices_self_beam = self.config_for_update.slicer.num_slices

        while True:

            if self.config_for_update._do_update:

                if self.config_for_update.pipeline_manager.is_ready_to_send(self.config_for_update.element_name,
                                                     particles.name,
                                                     self.config_for_update.partner_particles_name,
                                                     particles.at_turn[0],
                                                     internal_tag=self.config_for_update._i_step):
                    # Compute moments
                    self.config_for_update.slicer.assign_slices(particles)
                    self.moments = self.config_for_update.slicer.compute_moments(particles,update_assigned_slices=False)
                    self.moments[:self.config_for_update.slicer.num_slices] *= self.particles_per_macroparticle
                    self.config_for_update.pipeline_manager.send_message(self.moments,
                                                     self.config_for_update.element_name,
                                                     particles.name,
                                                     self.config_for_update.partner_particles_name,
                                                     particles.at_turn[0],
                                                     internal_tag=self.config_for_update._i_step)

                if self.config_for_update.pipeline_manager.is_ready_to_recieve(self.config_for_update.element_name,
                                        self.config_for_update.partner_particles_name,
                                        particles.name,
                                        internal_tag=self.config_for_update._i_step):
                    self.config_for_update.pipeline_manager.recieve_message(self.partner_moments,
                                        self.config_for_update.element_name,
                                        self.config_for_update.partner_particles_name,
                                        particles.name,
                                        internal_tag=self.config_for_update._i_step)
                    self.update_from_recieved_moments()
                else:
                    return xt.PipelineStatus(on_hold=True)

            # compute interacting other beam slice ID
            self.config_for_update._other_beam_slice_index_for_particles[:] =(
                 self.config_for_update._i_step - self.config_for_update._particles_slice_index)

            self.synchro_beam_kick(particles=particles,
                        i_slice_for_particles=self.config_for_update._other_beam_slice_index_for_particles)

            self.config_for_update._i_step += 1
            if self.config_for_update._i_step == (n_slices_self_beam + self.num_slices_other_beam - 1): 
                self.config_for_update._i_step = 0
                self.config_for_update._working_on_bunch = None
                break

        return None

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
            zeta_st=zeta_star_slices,
            pzeta_st=pzeta_star_slices,
            sphi=self.sin_phi,
            cphi=self.cos_phi,
            tphi=self.tan_phi,
            salpha=self.sin_alpha,
            calpha=self.cos_alpha,
        )

        return x_slices, px_slices, y_slices, py_slices, zeta_slices, pzeta_slices

    def _init_Sigmas(self,
            slices_other_beam_Sigma_11,
            slices_other_beam_Sigma_12,
            slices_other_beam_Sigma_13,
            slices_other_beam_Sigma_14,
            slices_other_beam_Sigma_22,
            slices_other_beam_Sigma_23,
            slices_other_beam_Sigma_24,
            slices_other_beam_Sigma_33,
            slices_other_beam_Sigma_34,
            slices_other_beam_Sigma_44,

            slices_other_beam_Sigma_11_star,
            slices_other_beam_Sigma_12_star,
            slices_other_beam_Sigma_13_star,
            slices_other_beam_Sigma_14_star,
            slices_other_beam_Sigma_22_star,
            slices_other_beam_Sigma_23_star,
            slices_other_beam_Sigma_24_star,
            slices_other_beam_Sigma_33_star,
            slices_other_beam_Sigma_34_star,
            slices_other_beam_Sigma_44_star,
            ):

        # Mandatory sigmas
        assert ((slices_other_beam_Sigma_11 is None or slices_other_beam_Sigma_11_star is not None)
                or (slices_other_beam_Sigma_11 is not None or slices_other_beam_Sigma_11_star is None)), (
            "Either `slices_other_beam_Sigma_11` or `slices_other_beam_Sigma_11_star` must be provided")
        assert ((slices_other_beam_Sigma_12 is None or slices_other_beam_Sigma_12_star is not None)
                or (slices_other_beam_Sigma_12 is not None or slices_other_beam_Sigma_12_star is None)), (
            "Either `slices_other_beam_Sigma_12` or `slices_other_beam_Sigma_12_star` must be provided")
        assert ((slices_other_beam_Sigma_22 is None or slices_other_beam_Sigma_22_star is not None)
                or (slices_other_beam_Sigma_22 is not None or slices_other_beam_Sigma_22_star is None)), (
            "Either `slices_other_beam_Sigma_22` or `slices_other_beam_Sigma_22_star` must be provided")
        assert ((slices_other_beam_Sigma_33 is None or slices_other_beam_Sigma_33_star is not None)
                or (slices_other_beam_Sigma_33 is not None or slices_other_beam_Sigma_33_star is None)), (
            "Either `slices_other_beam_Sigma_33` or `slices_other_beam_Sigma_33_star` must be provided")
        assert ((slices_other_beam_Sigma_34 is None or slices_other_beam_Sigma_34_star is not None)
                or (slices_other_beam_Sigma_34 is not None or slices_other_beam_Sigma_34_star is None)), (
            "Either `slices_other_beam_Sigma_34` or `slices_other_beam_Sigma_34_star` must be provided")
        assert ((slices_other_beam_Sigma_44 is None or slices_other_beam_Sigma_44_star is not None)
                or (slices_other_beam_Sigma_44 is not None or slices_other_beam_Sigma_44_star is None)), (
            "Either `slices_other_beam_Sigma_44` or `slices_other_beam_Sigma_44_star` must be provided")

        # Coupling between transverse planes
        if slices_other_beam_Sigma_13 is None and slices_other_beam_Sigma_13_star is None:
            slices_other_beam_Sigma_13 = 0
        if slices_other_beam_Sigma_14 is None and slices_other_beam_Sigma_14_star is None:
            slices_other_beam_Sigma_14 = 0
        if slices_other_beam_Sigma_23 is None and slices_other_beam_Sigma_23_star is None:
            slices_other_beam_Sigma_23 = 0
        if slices_other_beam_Sigma_24 is None and slices_other_beam_Sigma_24_star is None:
            slices_other_beam_Sigma_24 = 0

        if slices_other_beam_Sigma_11 is not None:
            self.slices_other_beam_Sigma_11 = self._arr2ctx(slices_other_beam_Sigma_11)
        else:
            self.slices_other_beam_Sigma_11_star = self._arr2ctx(slices_other_beam_Sigma_11_star)

        if slices_other_beam_Sigma_12 is not None:
            self.slices_other_beam_Sigma_12 = self._arr2ctx(slices_other_beam_Sigma_12)
        else:
            self.slices_other_beam_Sigma_12_star = self._arr2ctx(slices_other_beam_Sigma_12_star)

        if slices_other_beam_Sigma_13 is not None:
            self.slices_other_beam_Sigma_13 = self._arr2ctx(slices_other_beam_Sigma_13)
        else:
            self.slices_other_beam_Sigma_13_star = self._arr2ctx(slices_other_beam_Sigma_13_star)

        if slices_other_beam_Sigma_14 is not None:
            self.slices_other_beam_Sigma_14 = self._arr2ctx(slices_other_beam_Sigma_14)
        else:
            self.slices_other_beam_Sigma_14_star = self._arr2ctx(slices_other_beam_Sigma_14_star)

        if slices_other_beam_Sigma_22 is not None:
            self.slices_other_beam_Sigma_22 = self._arr2ctx(slices_other_beam_Sigma_22)
        else:
            self.slices_other_beam_Sigma_22_star = self._arr2ctx(slices_other_beam_Sigma_22_star)

        if slices_other_beam_Sigma_23 is not None:
            self.slices_other_beam_Sigma_23 = self._arr2ctx(slices_other_beam_Sigma_23)
        else:
            self.slices_other_beam_Sigma_23_star = self._arr2ctx(slices_other_beam_Sigma_23_star)

        if slices_other_beam_Sigma_24 is not None:
            self.slices_other_beam_Sigma_24 = self._arr2ctx(slices_other_beam_Sigma_24)
        else:
            self.slices_other_beam_Sigma_24_star = self._arr2ctx(slices_other_beam_Sigma_24_star)

        if slices_other_beam_Sigma_33 is not None:
            self.slices_other_beam_Sigma_33 = self._arr2ctx(slices_other_beam_Sigma_33)
        else:
            self.slices_other_beam_Sigma_33_star = self._arr2ctx(slices_other_beam_Sigma_33_star)

        if slices_other_beam_Sigma_34 is not None:
            self.slices_other_beam_Sigma_34 = self._arr2ctx(slices_other_beam_Sigma_34)
        else:
            self.slices_other_beam_Sigma_34_star = self._arr2ctx(slices_other_beam_Sigma_34_star)

        if slices_other_beam_Sigma_44 is not None:
            self.slices_other_beam_Sigma_44 = self._arr2ctx(slices_other_beam_Sigma_44)
        else:
            self.slices_other_beam_Sigma_44_star = self._arr2ctx(slices_other_beam_Sigma_44_star)

    def _init_starred_positions(self,
            slices_other_beam_num_particles,
            slices_other_beam_x_center,
            slices_other_beam_px_center,
            slices_other_beam_y_center,
            slices_other_beam_py_center,
            slices_other_beam_zeta_center,
            slices_other_beam_pzeta_center,
            slices_other_beam_x_center_star,
            slices_other_beam_px_center_star,
            slices_other_beam_y_center_star,
            slices_other_beam_py_center_star,
            slices_other_beam_zeta_center_star,
            slices_other_beam_pzeta_center_star):

        if slices_other_beam_zeta_center is not None:

            # Check correct according to z, head at the first position in the arrays. Only sort non-empty slices.
            assert np.all(slices_other_beam_zeta_center[slices_other_beam_num_particles!=0][:-1]
                            >= slices_other_beam_zeta_center[slices_other_beam_num_particles!=0][1:]), (
                            'slices_other_beam_zeta_center must be sorted from to tail (descending zeta)')

            (
            x_slices_star,
            px_slices_star,
            y_slices_star,
            py_slices_star,
            zeta_slices_star,
            pzeta_slices_star,
            ) = _python_boost(
                x=slices_other_beam_x_center,
                px=slices_other_beam_px_center,
                y=slices_other_beam_y_center,
                py=slices_other_beam_py_center,
                zeta=slices_other_beam_zeta_center,
                pzeta=slices_other_beam_pzeta_center,
                sphi=self.sin_phi,
                cphi=self.cos_phi,
                tphi=self.tan_phi,
                salpha=self.sin_alpha,
                calpha=self.cos_alpha,
            )

        # User-provided value has priority
        if slices_other_beam_x_center_star is not None:
            self.slices_other_beam_x_center_star = slices_other_beam_x_center_star
        else:
            self.slices_other_beam_x_center_star = self._arr2ctx(x_slices_star)

        if slices_other_beam_px_center_star is not None:
            self.slices_other_beam_px_center_star = slices_other_beam_px_center_star
        else:
            self.slices_other_beam_px_center_star = self._arr2ctx(px_slices_star)

        if slices_other_beam_y_center_star is not None:
            self.slices_other_beam_y_center_star = slices_other_beam_y_center_star
        else:
            self.slices_other_beam_y_center_star = self._arr2ctx(y_slices_star)

        if slices_other_beam_py_center_star is not None:
            self.slices_other_beam_py_center_star = slices_other_beam_py_center_star
        else:
            self.slices_other_beam_py_center_star = self._arr2ctx(py_slices_star)

        if slices_other_beam_zeta_center_star is not None:
            self.slices_other_beam_zeta_center_star = slices_other_beam_zeta_center_star
        else:
            self.slices_other_beam_zeta_center_star = self._arr2ctx(zeta_slices_star)

        if slices_other_beam_pzeta_center_star is not None:
            self.slices_other_beam_pzeta_center_star = slices_other_beam_pzeta_center_star
        else:
            self.slices_other_beam_pzeta_center_star = self._arr2ctx(pzeta_slices_star)

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
def _python_boost_scalar(x, px, y, py, zeta, pzeta,
                  sphi, cphi, tphi, salpha, calpha):

    h = (
        pzeta
        + 1.0
        - np.sqrt((1.0 + pzeta) * (1.0 + pzeta) - px * px - py * py)
    )

    px_st = px / cphi - h * calpha * tphi / cphi
    py_st = py / cphi - h * salpha * tphi / cphi
    pzeta_st = (
        pzeta - px * calpha * tphi - py * salpha * tphi + h * tphi * tphi
    )

    pz_st = np.sqrt(
        (1.0 + pzeta_st) * (1.0 + pzeta_st) - px_st * px_st - py_st * py_st
    )
    hx_st = px_st / pz_st
    hy_st = py_st / pz_st
    hzeta_st = 1.0 - (pzeta_st + 1) / pz_st

    L11 = 1.0 + hx_st * calpha * sphi
    L12 = hx_st * salpha * sphi
    L13 = calpha * tphi

    L21 = hy_st * calpha * sphi
    L22 = 1.0 + hy_st * salpha * sphi
    L23 = salpha * tphi

    L31 = hzeta_st * calpha * sphi
    L32 = hzeta_st * salpha * sphi
    L33 = 1.0 / cphi

    x_st = L11 * x + L12 * y + L13 * zeta
    y_st = L21 * x + L22 * y + L23 * zeta
    zeta_st = L31 * x + L32 * y + L33 * zeta

    return x_st, px_st, y_st, py_st, zeta_st, pzeta_st

_python_boost = np.vectorize(_python_boost_scalar,
    excluded=("sphi", "cphi", "tphi", "salpha", "calpha"))

def _python_inv_boost_scalar(x_st, px_st, y_st, py_st, zeta_st, pzeta_st,
                  sphi, cphi, tphi, salpha, calpha):

    pz_st = np.sqrt(
        (1.0 + pzeta_st) * (1.0 + pzeta_st) - px_st * px_st - py_st * py_st
    )
    hx_st = px_st / pz_st
    hy_st = py_st / pz_st
    hzeta_st = 1.0 - (pzeta_st + 1) / pz_st

    Det_L = (
        1.0 / cphi
        + (hx_st * calpha + hy_st * salpha - hzeta_st * sphi) * tphi
    )

    Linv_11 = (
        1.0 / cphi + salpha * tphi * (hy_st - hzeta_st * salpha * sphi)
    ) / Det_L
    Linv_12 = (salpha * tphi * (hzeta_st * calpha * sphi - hx_st)) / Det_L
    Linv_13 = (
        -tphi
        * (
            calpha
            - hx_st * salpha * salpha * sphi
            + hy_st * calpha * salpha * sphi
        )
        / Det_L
    )

    Linv_21 = (calpha * tphi * (-hy_st + hzeta_st * salpha * sphi)) / Det_L
    Linv_22 = (
        1.0 / cphi + calpha * tphi * (hx_st - hzeta_st * calpha * sphi)
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

    Linv_31 = -hzeta_st * calpha * sphi / Det_L
    Linv_32 = -hzeta_st * salpha * sphi / Det_L
    Linv_33 = (1.0 + hx_st * calpha * sphi + hy_st * salpha * sphi) / Det_L

    x_i = Linv_11 * x_st + Linv_12 * y_st + Linv_13 * zeta_st
    y_i = Linv_21 * x_st + Linv_22 * y_st + Linv_23 * zeta_st
    zeta_i = Linv_31 * x_st + Linv_32 * y_st + Linv_33 * zeta_st

    h = (pzeta_st + 1.0 - pz_st) * cphi * cphi

    px_i = px_st * cphi + h * calpha * tphi
    py_i = py_st * cphi + h * salpha * tphi

    pzeta_i = (
        pzeta_st
        + px_i * calpha * tphi
        + py_i * salpha * tphi
        - h * tphi * tphi
    )

    return x_i, px_i, y_i, py_i, zeta_i, pzeta_i

_python_inv_boost = np.vectorize(_python_inv_boost_scalar,
    excluded=("sphi", "cphi", "tphi", "salpha", "calpha"))

class TempSlicer:
    def __init__(self, bin_edges):

        bin_edges = np.sort(np.array(bin_edges))[::-1]
        self.bin_edges = bin_edges
        self.bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        self.num_slices = len(bin_edges) - 1

    def get_slice_indices(self, particles):
        context = particles._context
        if isinstance(context, xo.ContextPyopencl):
            raise NotImplementedError
 
        bin_edges = context.nparray_to_context_array(self.bin_edges)

        digitize = particles._context.nplike_lib.digitize  # only works with cpu and cupy
        indices = digitize(particles.zeta, bin_edges, right=True)
        indices -= 1 # In digitize, 0 means before the first edge
        indices[particles.state <=0 ] = -1

        indices_out = context.zeros(shape=indices.shape, dtype=np.int64)
        indices_out[:] = indices
        return indices_out

    def assign_slices(self, particles):
        particles.slice = self.get_slice_indices(particles)

    def compute_moments(self, particles, update_assigned_slices=True, threshold_num_macroparticles=20):
        if update_assigned_slices:
            self.assign_slices(particles)

        slice_moments = np.zeros(self.num_slices*(1+6+10),dtype=float)
        for i_slice in range(self.num_slices):
            mask = (particles.slice == i_slice) & (particles.state >0)  # skip lost particles (1: alive, 0 lost)
            slice_moments[i_slice]                   = 0 if len(particles.x[mask]) < threshold_num_macroparticles else len(particles.x[mask])                                    # nb part
            slice_moments[self.num_slices+i_slice]   = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float(particles.x[mask].sum())/slice_moments[i_slice]     # <x>
            slice_moments[2*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float(particles.px[mask].sum())/slice_moments[i_slice]    # <px>
            slice_moments[3*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float(particles.y[mask].sum())/slice_moments[i_slice]     # <y>
            slice_moments[4*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float(particles.py[mask].sum())/slice_moments[i_slice]    # <py>
            slice_moments[5*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float(particles.zeta[mask].sum())/slice_moments[i_slice]  # <z>
            slice_moments[6*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float(particles.delta[mask].sum())/slice_moments[i_slice] # <pz> # TODO mhy pzeta doesn't work?

            x_diff  = 0 if len(particles.x[mask]) < threshold_num_macroparticles else particles.x[mask]-slice_moments[self.num_slices+i_slice]
            px_diff = 0 if len(particles.x[mask]) < threshold_num_macroparticles else particles.px[mask]-slice_moments[2*self.num_slices+i_slice]
            y_diff  = 0 if len(particles.x[mask]) < threshold_num_macroparticles else particles.y[mask]-slice_moments[3*self.num_slices+i_slice]
            py_diff = 0 if len(particles.x[mask]) < threshold_num_macroparticles else particles.py[mask]-slice_moments[4*self.num_slices+i_slice]
            slice_moments[7*self.num_slices+i_slice]  = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((x_diff**2).sum())/slice_moments[i_slice]             # Sigma_11
            slice_moments[8*self.num_slices+i_slice]  = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((x_diff*px_diff).sum())/slice_moments[i_slice]      # Sigma_12
            slice_moments[9*self.num_slices+i_slice]  = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((x_diff*y_diff).sum())/slice_moments[i_slice]       # Sigma_13
            slice_moments[10*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((x_diff*py_diff).sum())/slice_moments[i_slice]     # Sigma_14
            slice_moments[11*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((px_diff**2).sum())/slice_moments[i_slice]           # Sigma_22
            slice_moments[12*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((px_diff*y_diff).sum())/slice_moments[i_slice]     # Sigma_23
            slice_moments[13*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((px_diff*py_diff).sum())/slice_moments[i_slice]    # Sigma_24
            slice_moments[14*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((y_diff**2).sum())/slice_moments[i_slice]            # Sigma_33
            slice_moments[15*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((y_diff*py_diff).sum())/slice_moments[i_slice]     # Sigma_34
            slice_moments[16*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((py_diff**2).sum())/slice_moments[i_slice]           # Sigma_44

        return slice_moments

# this might replace TempSlicer
class Slicer:
    def __init__(self, n_slices, sigma_z, mode="unibin"):

        assert isinstance(n_slices, int) and n_slices>0, ("'n_slices' must be a positive integer!")
        assert mode in ["unicharge", "unibin", "improved"], ("Accepted values for 'mode': 'unicharge', 'unibin', 'improved'")

        # bin params are in units of RMS bunch length
        if mode=="unicharge":
            z_k_arr, l_k_arr, w_k_arr, dz_k_arr = self.unicharge(n_slices)
        elif mode=="unibin":
            z_k_arr, l_k_arr, w_k_arr, dz_k_arr = self.unibin(n_slices)
        elif mode=="improved":
            z_k_arr, l_k_arr, w_k_arr, dz_k_arr = self.improved(n_slices)

        self.num_slices  = n_slices
        self.sigma_z     = sigma_z 
        self.bin_centers = z_k_arr
        self.bin_edges   = l_k_arr
        self.bin_weights = w_k_arr
        self.bin_widths_beamstrahlung = dz_k_arr

    def rho(self, z):
        """
        Gaussian charge density.
        Rho has a unit of [1/m].
        z = z_k/sigma_z [1], normalized by bunch length in the frame where the slicing takes place
        """
        return np.exp(-z**2/(2)) / (np.sqrt(2*np.pi))

    def unicharge(self, n_slices):
        """
        Uniform charge slicing.
        """
    
        # these are units of sigma_z
        z_k_arr_unicharge = np.zeros(n_slices)  # should be n_slices long, ordered from + to -
        l_k_arr_unicharge = np.zeros(n_slices+1)  # bin edges, n_slices+1 long
        w_k_arr_unicharge = np.zeros(n_slices)  # bin weights, used for bunch intensity normalization
        half = int((n_slices + 1) / 2)
        n_odd = n_slices % 2
        w_k_arr_unicharge[:half] = 1 / n_slices  # fill up initial values, e.g. n_slices=300-> fill up elements [0,149]; 301: [0,150]
        l_k_arr_unicharge[0] = -5  # leftmost bin edge
        w_k_sum = 0 # sum of weights: integral of gaussian up to l_k
        rho_upper = 0 # start from top of distribution (positive end, l_upper=inf)
        
        # go from bottom end toward 0 (=middle of Gaussian)
        for j in range(half):
        
            w_k_sum += 2*w_k_arr_unicharge[j] # integrate rho up to and including bin j
    
            # get bin center
            if n_odd and j == half-1:  # center bin (z_c=0)
                z_k_arr_unicharge[j] = 0
            else:  # all other bins
                rho_lower = rho_upper
    
                # get upper bin boundary
                arg = w_k_sum - 1
                l_upper = np.sqrt(2)*special.erfinv(arg)
                l_k_arr_unicharge[j+1] = l_upper
                rho_upper = self.rho(l_upper) 
                
                # get z_k: center of momentum
                z_k_arr_unicharge[j] = (rho_upper - rho_lower) / w_k_arr_unicharge[j]
        
        # mirror for positive half
        z_k_arr_unicharge[half:] = -z_k_arr_unicharge[n_slices-half-1::-1]  # bin centers
        w_k_arr_unicharge[half:] =  w_k_arr_unicharge[n_slices-half-1::-1]  # bin weights, used for bunch intensity normalization
        l_k_arr_unicharge[half:] = -l_k_arr_unicharge[n_slices-half::-1]  # bin edges
        dz_k_arr_unicharge       = np.diff(l_k_arr_unicharge)  # for beamstrahlung
        l_k_arr_unicharge        = l_k_arr_unicharge[::-1]

        return z_k_arr_unicharge, l_k_arr_unicharge, w_k_arr_unicharge, dz_k_arr_unicharge

    def unibin(self, n_slices):
        """
        Uniform bin slicing.
        """

        # these are units of sigma_z
        z_k_list_unibin = []  # should be n_slices long, ordered from + to -
    
        m = 1 if not n_slices%2 else 0
    
        # dmitry goes from +n_slices/2 to -n_slices/2-1 (50-(-51) for 101 slices); hirata goes from n_slices to 0
        for k in range(int(n_slices/2), -int(n_slices/2)-(1-m), -1):
        
            # slices extend from -N*sigma to +N*sigma
            N = 5
            z_k = (2*k - m) / (n_slices - 1) * N * special.erf(np.sqrt(n_slices / 6))
            z_k_list_unibin.append(z_k)
    
        z_k_arr_unibin = np.array(z_k_list_unibin)  # bin centers
        w_k_arr_unibin = np.exp(-z_k_arr_unibin**2/2) # proportional, but these are not yet not normalized
        w_k_arr_unibin = w_k_arr_unibin / np.sum(w_k_arr_unibin) # bin weights, used for bunch intensity normalization
        dz_i = -np.diff(z_k_arr_unibin)[0]
        l_k_arr_unibin = np.hstack([z_k_arr_unibin+dz_i/2, z_k_arr_unibin[-1]-dz_i/2])  # bin edges
        dz_k_array_unibin = np.ones(n_slices)*dz_i  # for beamstrahlung

        return z_k_arr_unibin, l_k_arr_unibin, w_k_arr_unibin, dz_k_array_unibin

    def improved(self, n_slices):
        """
        This method is a mix between uniform bin and charge. It finds the slice centers by iteration.
        """

        # these are units of sigma_z
        z_k_arr_improved = np.zeros(n_slices)  # should be n_slices long, ordered from + to -
        l_k_arr_improved = np.zeros(n_slices+1)  # bin edges, n_slices+1 long
        w_k_arr_improved = np.zeros(n_slices)  # bin weights, used for bunch intensity normalization
        half = int((n_slices + 1) / 2)
        n_odd = n_slices % 2
        w_k_arr_improved[:half] = 1 / n_slices  # fill up initial values, e.g. n_slices=300-> fill up elements [0,149]; 301: [0,150]
        l_k_arr_improved[0] = -5  # leftmost bin edge

        k_max = min(1000, 20*n_slices)  # max iterations for l_k
        
        for i in range(k_max+1):
            w_k_sum = 0 # sum of weights: integral of gaussian up to l_k
            rho_upper = 0 # start from top of distribution (positive end, l_upper=inf)
            
            # go from bottom toward 0 (=middle of Gaussian)
            for j in range(half):
            
                w_k_sum += 2*w_k_arr_improved[j] # integrate rho up to including current bin
        
                # get z_k
                if n_odd and j == half-1:  # center bin (z_c=0)
                    z_k_arr_improved[j] = 0
                else:  # all other bins
                    rho_lower = rho_upper
        
                    arg = w_k_sum - 1
                    l_upper = np.sqrt(2)*special.erfinv(arg)
        
                    l_k_arr_improved[j+1] = l_upper
                    
                    rho_upper = self.rho(l_upper)  # to cancel 1/sigma_z in rho
                    
                    # get z_k: center of momentum
                    z_k_arr_improved[j] = (rho_upper - rho_lower) / w_k_arr_improved[j]
                    
                # get w_k
                if i < k_max:
                    w_k_arr_improved[j] = np.exp( -z_k_arr_improved[j]**2 / 4 )
            
            # renormalize w_k
            if i < k_max:
                w_int = 2*np.sum(w_k_arr_improved[:half]) - n_odd * w_k_arr_improved[half-1]
                w_k_arr_improved[:half] = w_k_arr_improved[:half] / w_int
        
        # mirror for negative half
        z_k_arr_improved[half:] = -z_k_arr_improved[n_slices-half-1::-1]  # bin centers
        w_k_arr_improved[half:] =  w_k_arr_improved[n_slices-half-1::-1]  # bin weights, used for bunch intensity normalization
        l_k_arr_improved[half:] = -l_k_arr_improved[n_slices-half::-1]  # bin edges
        dz_k_arr_improved       = np.diff(l_k_arr_improved)  # for beamstrahlung
        l_k_arr_improved        = l_k_arr_improved[::-1]

        return z_k_arr_improved, l_k_arr_improved, w_k_arr_improved, dz_k_arr_improved

    def get_slice_indices(self, particles):
        context = particles._context
        if isinstance(context, xo.ContextPyopencl):
            raise NotImplementedError
 
        bin_edges = context.nparray_to_context_array(self.bin_edges) * self.sigma_z  # bin params are in units of RMS bunch length

        digitize = particles._context.nplike_lib.digitize  # only works with cpu and cupy
        indices = digitize(particles.zeta, bin_edges, right=True)
        indices -= 1 # In digitize, 0 means before the first edge
        indices[particles.state <=0 ] = -1

        indices_out = context.zeros(shape=indices.shape, dtype=np.int64)
        indices_out[:] = indices
        return indices_out

    def assign_slices(self, particles):
        particles.slice = self.get_slice_indices(particles)

    def compute_moments(self, particles, update_assigned_slices=True, threshold_num_macroparticles=20):
        if update_assigned_slices:
            self.assign_slices(particles)

        slice_moments = np.zeros(self.num_slices*(1+6+10),dtype=float)
        for i_slice in range(self.num_slices):
            mask = (particles.slice == i_slice) & (particles.state >0)  # skip lost particles (1: alive, 0 lost)
            slice_moments[i_slice]                   = 0 if len(particles.x[mask]) < threshold_num_macroparticles else len(particles.x[mask])                                    # nb part
            slice_moments[self.num_slices+i_slice]   = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float(particles.x[mask].sum())/slice_moments[i_slice]     # <x>
            slice_moments[2*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float(particles.px[mask].sum())/slice_moments[i_slice]    # <px>
            slice_moments[3*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float(particles.y[mask].sum())/slice_moments[i_slice]     # <y>
            slice_moments[4*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float(particles.py[mask].sum())/slice_moments[i_slice]    # <py>
            slice_moments[5*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float(particles.zeta[mask].sum())/slice_moments[i_slice]  # <z>
            slice_moments[6*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float(particles.delta[mask].sum())/slice_moments[i_slice] # <pz> # TODO mhy pzeta doesn't work?

            x_diff  = 0 if len(particles.x[mask]) < threshold_num_macroparticles else particles.x[mask]-slice_moments[self.num_slices+i_slice]
            px_diff = 0 if len(particles.x[mask]) < threshold_num_macroparticles else particles.px[mask]-slice_moments[2*self.num_slices+i_slice]
            y_diff  = 0 if len(particles.x[mask]) < threshold_num_macroparticles else particles.y[mask]-slice_moments[3*self.num_slices+i_slice]
            py_diff = 0 if len(particles.x[mask]) < threshold_num_macroparticles else particles.py[mask]-slice_moments[4*self.num_slices+i_slice]
            slice_moments[7*self.num_slices+i_slice]  = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((x_diff**2).sum())/slice_moments[i_slice]             # Sigma_11
            slice_moments[8*self.num_slices+i_slice]  = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((x_diff*px_diff).sum())/slice_moments[i_slice]      # Sigma_12
            slice_moments[9*self.num_slices+i_slice]  = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((x_diff*y_diff).sum())/slice_moments[i_slice]       # Sigma_13
            slice_moments[10*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((x_diff*py_diff).sum())/slice_moments[i_slice]     # Sigma_14
            slice_moments[11*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((px_diff**2).sum())/slice_moments[i_slice]           # Sigma_22
            slice_moments[12*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((px_diff*y_diff).sum())/slice_moments[i_slice]     # Sigma_23
            slice_moments[13*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((px_diff*py_diff).sum())/slice_moments[i_slice]    # Sigma_24
            slice_moments[14*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((y_diff**2).sum())/slice_moments[i_slice]            # Sigma_33
            slice_moments[15*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((y_diff*py_diff).sum())/slice_moments[i_slice]     # Sigma_34
            slice_moments[16*self.num_slices+i_slice] = 0 if len(particles.x[mask]) < threshold_num_macroparticles else float((py_diff**2).sum())/slice_moments[i_slice]           # Sigma_44

        return slice_moments

class ConfigForUpdateBeamBeamBiGaussian3D:

    def __init__(self,
        pipeline_manager=None,
        element_name=None,
        slicer=None,
        partner_particles_name=None,
        update_every=None):

        self.pipeline_manager = pipeline_manager
        self.element_name = element_name
        self.slicer = slicer
        self.partner_particles_name = partner_particles_name
        self.update_every = update_every

        self._i_step = 0
        self._working_on_bunch = None
        self._particles_slice_index = None

