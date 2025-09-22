# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np

import xobjects as xo
import xtrack as xt

from ..general import _pkg_root

class BeamstrahlungTable(xo.HybridClass):
    """
    Buffer size should be larger than the number of expected BS photons emitted. Test on single collision for estimate.
    If more photons are emitted than the buffer size, the surplus data will be dropped.
    Photons from multiple turns and beambeam elements are stored in the same buffer.
    Photons are 'macro' i.e. represent the dynamics of bunch_intensity/n_macroparticles real BS photons.
    This implies that the number of emitted BS photons scales linearly with n_macroparticles.

    Fields:
     _index: custom C struct for metadata
     at_element: [1] element index in the xtrack.Line object, starts with 0
     at_turn: [1] turn index, starts with 0
     particle_id: [1] array index in the xpart.Particles object of the primary macroparticle emitting the photon
     primary_energy: [eV] total energy of primary macroparticle before emission of this photon
     photon_id: [1] counter for photons emitted from the same primary in the same collision with a single slice
        photon_energy: [eV] total energy of a beamstrahlung photon
     photon_critical_energy (with quantum BS only): [eV] critical energy of a beamstrahlung photon
     rho_inv (with quantum BS only): [m^-1] (Fr/dz) inverse bending radius of the primary macroparticle
     n_avg (with mean BS only): [1] avg. number of emitted beamstrahlung photons from 1 macroparticle in collision with a single slice
     delta_avg (with mean BS only): [1] avg. rel. E loss for 1 macroparticle in collision with a single slice
    """
    _xofields = {
      '_index': xt.RecordIndex,
      'at_element': xo.Int64[:],
      'at_turn': xo.Int64[:],
      'particle_id': xo.Int64[:],
      'primary_energy': xo.Float64[:],
      'photon_id': xo.Float64[:],
      'photon_energy': xo.Float64[:],
      'photon_critical_energy': xo.Float64[:],
      'rho_inv': xo.Float64[:],
      'n_avg': xo.Float64[:],
      'delta_avg': xo.Float64[:],
        }


class BhabhaTable(xo.HybridClass):
    """
    Buffer size should be larger than the number of expected BH photons emitted. Test on single collision for estimate.
    If more photons are emitted than the buffer size, the surplus data will be dropped.
    Photons from multiple turns and beambeam elements are stored in the same buffer.
    Photons are 'real' i.e. represent the dynamics of a single real BH photon.
    This implies that the number of emitted BH photons does not depend on n_macroparticles.

    Fields:
     _index: custom C struct for metadata
     at_element: [1] element index in the xtrack.Line object, starts with 0
     at_turn: [1] turn index, starts with 0
     particle_id: [1] array index in the xpart.Particles object of the primary macroparticle emitting the photon
     primary_energy: [eV] total energy of primary macroparticle before emission of this photon
     photon_id: [1] counter for photons emitted from the same primary in the same collision with a single slice
     photon_x, photon_y, photon_z: [m] (boosted) coordinates of the scattered Bhabha photon
     photon_px, photon_py: [m] (boosted) momenta of the scattered Bhabha photon
     photon_energy: [eV] total energy of the scattered Bhabha photon
     primary_scattering_angle: [rad] scattering angle of scattered primary macroparticle
     photon_scattering_angle: [rad] scattering angle of scattered Bhabha photon
    """
    _xofields = {
      '_index': xt.RecordIndex,
      'at_element': xo.Int64[:],
      'at_turn': xo.Int64[:],
      'particle_id': xo.Int64[:],
      'primary_energy': xo.Float64[:],
      'photon_id': xo.Int64[:],
      'photon_x': xo.Float64[:],
      'photon_y': xo.Float64[:],
      'photon_z': xo.Float64[:],
      'photon_px': xo.Float64[:],
      'photon_py': xo.Float64[:],
      'photon_energy': xo.Float64[:],
      'primary_scattering_angle': xo.Float64[:],
      'photon_scattering_angle': xo.Float64[:],
        }


class LumiTable(xo.HybridClass):
    """
    Buffer size should be equal to the number of tracking turns.
    If more turns are tracked than the buffer size, the surplus data will be dropped.
    Luminosity contributions from multiple beambeam elements in the same tracking turn are aggregated in the same buffer element.

    Fields:
     _index: custom C struct for metadata
     at_element: [1] element index in the xtrack.Line object, starts with 0
     luminosity: [m^-2] integrated luminosity per bunch crossing for one turn
    """
    _xofields = {
      '_index': xt.RecordIndex,
      'at_element': xo.Int64[:],
      'luminosity': xo.Float64[:],
        }


class BeamBeamBiGaussian3DRecord(xo.HybridClass):
    _xofields = {
        'beamstrahlungtable': BeamstrahlungTable,
        'bhabhatable': BhabhaTable,
        'lumitable': LumiTable,
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
        'slices_other_beam_zeta_bin_width_beamstrahlung': xo.Float64[:],
        'slices_other_beam_zeta_bin_width_star_beamstrahlung': xo.Float64[:],
        'slices_other_beam_sqrtSigma_11_beamstrahlung': xo.Float64[:],
        'slices_other_beam_sqrtSigma_33_beamstrahlung': xo.Float64[:],
        'slices_other_beam_sqrtSigma_55_beamstrahlung': xo.Float64[:],

         #bhabha
         'flag_bhabha': xo.Int64,
         'compt_x_min': xo.Float64,
         'compt_scale': xo.Float64,
         'flag_beamsize_effect': xo.Int64,

         #lumi
         'flag_luminosity': xo.Int64,

    }

    _internal_record_class = BeamBeamBiGaussian3DRecord

    _rename = {'flag_beamstrahlung': '_flag_beamstrahlung',
               'flag_bhabha': '_flag_bhabha'}

    _depends_on = [xt.RandomUniform]

    _extra_c_sources= [
        _pkg_root.joinpath('headers/constants.h'),
        xt.general._pkg_root.joinpath('headers/atomicadd.h'),
        _pkg_root.joinpath('headers/sincos.h'),
        _pkg_root.joinpath('headers/power_n.h'),
        _pkg_root.joinpath('headers','particle_states.h'),
        _pkg_root.joinpath('fieldmaps/bigaussian_src/faddeeva.h'),
        _pkg_root.joinpath('fieldmaps/bigaussian_src/bigaussian.h'),
        _pkg_root.joinpath('beam_elements/beambeam_src/beambeam3d_transport_sigmas.h'),
        _pkg_root.joinpath('beam_elements/beambeam_src/beambeam3d_ref_frame_changes.h'),

        # beamstrahlung
        _pkg_root.joinpath('headers/beamstrahlung_spectrum.h'),
        _pkg_root.joinpath('headers/bhabha_spectrum.h'),
        _pkg_root.joinpath('beam_elements/beambeam_src/beambeam3d.h'),
        _pkg_root.joinpath(
            'beam_elements/beambeam_src/beambeam3d_methods_for_strongstrong.h'),

   ]

    _per_particle_kernels={
        'synchro_beam_kick': xo.Kernel(
            c_name='BeamBeam3D_selective_apply_synchrobeam_kick_local_particle',
            args=[
                xo.Arg(xo.Int64, pointer=True, name='i_slice_for_particles'),
            ]),
        'change_ref_frame': xo.Kernel(
            c_name='BeamBeamBiGaussian3D_change_ref_frame_local_particle',
            args=[]),
        'change_back_ref_frame_and_subtract_dipolar': xo.Kernel(
            c_name='BeamBeamBiGaussian3D_change_back_ref_frame_and_subtract_dipolar_local_particle',
            args=[]),
    }

    def __init__(self,
                    phi=None, alpha=None, other_beam_q0=None,
                    scale_strength = 1.,

                    slices_other_beam_num_particles=None,

                    slices_other_beam_x_center=0.,
                    slices_other_beam_px_center=0.,
                    slices_other_beam_y_center=0.,
                    slices_other_beam_py_center=0.,
                    slices_other_beam_zeta_center=None,
                    slices_other_beam_pzeta_center=0.,

                    flag_beamstrahlung=0,
                    slices_other_beam_zeta_bin_width_beamstrahlung=None,
                    slices_other_beam_zeta_bin_width_star_beamstrahlung=None,
                    slices_other_beam_sqrtSigma_11_beamstrahlung=None,
                    slices_other_beam_sqrtSigma_33_beamstrahlung=None,
                    slices_other_beam_sqrtSigma_55_beamstrahlung=None,

                    flag_bhabha=0,
                    compt_x_min=1e-4,
                    compt_scale=1.,
                    flag_beamsize_effect=1,

                    flag_luminosity=0,

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

        """
        3D beam-beam element in the soft-Gaussian approximation.

        Args:
             phi (float): Half crossing angle in [rad]. 
             alpha (float): Crossing plane. E.g. CMS (y-z crossing) has `alpha=pi/2` and ATLAS (x-z crossing) has `alpha=0`.
             other_beam_q0 (float): Charge sign of opposing beam. -1 for electrons, +1 for protons or positrons.
             scale_strength (float): Used to scale beam-beam force strength. Scales `other_beam_q0`.
             slices_other_beam_num_particles (float array): Number of real charges per slice in the opposing bunch. Length of the array is the number of longitudinal slices.
             slices_other_beam_{x,px,y,py,zeta,pzeta}_center (float array): Array storing the per-slice centroid variables of the opposing bunch, in the boosted accelerator frame. Length of the array is the number of longitudinal slices. 
             flag_beamstrahlung (int): Flag to simulate beamstrahlung. 0: OFF, 1: ON (mean, only for testing), 2: ON (quantum, use this for simulations)
             slices_other_beam_zeta_bin_width_beamstrahlung (float array): Array storing the longitudinal bin width in the unboosted accelerator frame. Length of the array is the number of longitudinal slices. Used for beamstrahlung only.
             slices_other_beam_zeta_bin_width_star_beamstrahlung (float array): Array storing the longitudinal bin width in the boosted accelerator frame. Length of the array is the number of longitudinal slices. Used for beamstrahlung only. Obtained as slices_other_beam_zeta_bin_width_beamstrahlung/cos(phi). 
             slices_other_beam_sqrtSigma_{135}{135}_beamstrahlung (float array): Array storing the per-slice standard deviations of x (=1), y (=3) and zeta (=5) of the opposing bunch, in the unboosted accelerator frame. Length of the array is the number of longitudinal slices. Used for beamstrahlung only. 
             flag_bhabha (int): Flag to simulate small angle radiative Bhabha scattering. 1: ON (quantum), 0: OFF
             compt_x_min (float): Low energy cut on virtual photon spectrum, used for Bhabha, in units of [gamma^-2] where gamma is the rel. Lorentz factor.
             compt_scale (float): Scaling factor to scale up photon generation, used for Bhabha.
             flag_beamsize_effect (int): Flag to simulate beamsize effect, used for Bhabha. 1: ON, 0: OFF. Results in ~factor 2 reduction in cross section.
             flag_luminosity (int): Flag to record soft-Gaussian luminosity per bunch crossing in a buffer. Luminosity will be in units of [m^-2].
             slices_other_beam_{x,px,y,py,zeta,pzeta}_center_star (float array): Array storing the per-slice centroid variables of the opposing bunch, in the unboosted accelerator frame. Length of the array is the number of longitudinal slices. 
             slices_other_beam_Sigma_{1234}{1234} (float array): Array storing the per-slice statistical moments (variances and covariances of x (=1), px (=2), y (=3), py (=4)) of the opposing bunch, in the unboosted accelerator frame. Length of the array is the number of longitudinal slices. 
             slices_other_beam_Sigma_{1234}{1234}_star (float array): Array storing the per-slice statistical moments (variances and covariances of x (=1), px (=2), y (=3), py (=4)) of the opposing bunch, in the boosted accelerator frame. Length of the array is the number of longitudinal slices. 
             ref_shift_{x,px,y,py,zeta,pzeta} (float): Closed orbit shift, subtracted from each macroparticle before collision and added back after.
             other_beam_shift_{x,px,y,py,zeta,pzeta} (float): Closed orbit shift of the opposing beam (used in weak-strong mode). Subtracted and from the opposing slice centroids before collision and added back after.
             post_subtract_{x,px,y,py,zeta,pzeta} (float): Additional quantity that is subtracted after the beam-beam collision. Used e.g. for dipole kick.
             min_sigma_diff (float): Round beam kick (~2x faster) is used instead of elliptical kick, if `fabs(sigma_x-sigma_y) < min_sigma_diff`.
             threshold_singular (float): Small number used to handle singularities when transporting slice moments from interaction point (IP) to collision point (CP).
             old_interface (dict): Dictionary containing parameters with the old interface. (obsolete)
             config_for_update (xfields.ConfigForUpdateBeamBeamBiGaussian3D): Used for (quasi-)strong-strong beam-beam and `None` for weak-strong. See documentation of xfields.ConfigForUpdateBeamBeamBiGaussian3D.
             _sin_phi (float): Sine of half crossing angle.
             _cos_phi (float): Cosine of half crossing angle.
             _tan_phi (float): Tangent of half crossing angle.
             _sin_alpha (float): Sine of crossing plane.
             _cos_alpha (float): Cosine of crossing plane.
        """

        if '_xobject' in kwargs.keys():
            self.xoinitialize(**kwargs)
            return

        # Collective mode (pipeline update)
        if config_for_update is not None:

            self.config_for_update = config_for_update
            self.iscollective = True
            self.track = self._track_collective # switch to specific track method

            if slices_other_beam_zeta_center is None:
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
            self.moments = None


        else:
            self.config_for_update = None

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

        n_slices = len(slices_other_beam_num_particles)

        self._allocate_xobject(n_slices, **kwargs)

        if config_for_update is not None:
            self.partner_moments = self._buffer.context.nplike_lib.zeros(
                self.config_for_update.slicer.num_slices*(1+6+10), dtype=float)

        _init_alpha_phi(self, phi=phi, alpha=alpha,
                    _sin_phi=_sin_phi, _cos_phi=_cos_phi, _tan_phi=_tan_phi,
                    _sin_alpha=_sin_alpha, _cos_alpha=_cos_alpha)

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

        # Initialize beamstrahlung related quantities
        self._init_beamstrahlung(
            flag_beamstrahlung,
            slices_other_beam_zeta_bin_width_beamstrahlung,
            slices_other_beam_zeta_bin_width_star_beamstrahlung,
            slices_other_beam_sqrtSigma_11_beamstrahlung,
            slices_other_beam_sqrtSigma_33_beamstrahlung,
            slices_other_beam_sqrtSigma_55_beamstrahlung,
        )

        # initialize bhabha
        self._init_bhabha(flag_bhabha, compt_x_min, compt_scale, flag_beamsize_effect)

        self._init_luminosity(flag_luminosity)

        if other_beam_q0 is None:
            other_beam_q0 = 0

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
            slices_other_beam_zeta_bin_width_beamstrahlung=n_slices,  # beamstrahlung
            slices_other_beam_zeta_bin_width_star_beamstrahlung=n_slices,
            slices_other_beam_sqrtSigma_11_beamstrahlung=n_slices,
            slices_other_beam_sqrtSigma_33_beamstrahlung=n_slices,
            slices_other_beam_sqrtSigma_55_beamstrahlung=n_slices,
            **kwargs
            )

    def _init_beamstrahlung(self,
            flag_beamstrahlung,
            slices_other_beam_zeta_bin_width_beamstrahlung,
            slices_other_beam_zeta_bin_width_star_beamstrahlung,
            slices_other_beam_sqrtSigma_11_beamstrahlung,
            slices_other_beam_sqrtSigma_33_beamstrahlung,
            slices_other_beam_sqrtSigma_55_beamstrahlung,
            ):

        if self.config_for_update is not None:
            if slices_other_beam_zeta_bin_width_star_beamstrahlung is None:
                    if slices_other_beam_zeta_bin_width_beamstrahlung is not None:
                        slices_other_beam_zeta_bin_width_star_beamstrahlung = (
                            slices_other_beam_zeta_bin_width_beamstrahlung
                                / np.cos(self.phi))
                    else:
                        slices_other_beam_zeta_bin_width_star_beamstrahlung = (
                            self.config_for_update.slicer.bin_widths_beamstrahlung
                                / np.cos(self.phi))

        if slices_other_beam_zeta_bin_width_beamstrahlung is not None:
            assert not np.isscalar(slices_other_beam_zeta_bin_width_beamstrahlung), (
                'slices_other_beam_zeta_bin_width_beamstrahlung must be an array')
            assert (len(slices_other_beam_zeta_bin_width_beamstrahlung)
                == len(self.slices_other_beam_num_particles))
            slices_other_beam_zeta_bin_width_star_beamstrahlung = (
                slices_other_beam_zeta_bin_width_beamstrahlung / np.cos(self.phi))
            self.slices_other_beam_zeta_bin_width_beamstrahlung = self._arr2ctx(
                np.array(slices_other_beam_zeta_bin_width_beamstrahlung))
            self.slices_other_beam_zeta_bin_width_star_beamstrahlung = self._arr2ctx(
                np.array(slices_other_beam_zeta_bin_width_star_beamstrahlung))
        elif slices_other_beam_zeta_bin_width_star_beamstrahlung is not None:
            assert not np.isscalar(slices_other_beam_zeta_bin_width_star_beamstrahlung), (
                'slices_other_beam_zeta_bin_width_star_beamstrahlung must be an array')
            assert (len(slices_other_beam_zeta_bin_width_star_beamstrahlung)
                == len(self.slices_other_beam_num_particles))
            slices_other_beam_zeta_bin_width_beamstrahlung = (
                np.array(slices_other_beam_zeta_bin_width_star_beamstrahlung) * np.cos(self.phi))
            self.slices_other_beam_zeta_bin_width_beamstrahlung = self._arr2ctx(
                slices_other_beam_zeta_bin_width_beamstrahlung)
            self.slices_other_beam_zeta_bin_width_star_beamstrahlung = self._arr2ctx(
                slices_other_beam_zeta_bin_width_star_beamstrahlung)
        else:
            self.slices_other_beam_zeta_bin_width_star_beamstrahlung[:] = 0
            self.slices_other_beam_zeta_bin_width_beamstrahlung[:] = 0

        if slices_other_beam_sqrtSigma_11_beamstrahlung is not None:
            assert not np.isscalar(slices_other_beam_sqrtSigma_11_beamstrahlung), (
                'slices_other_beam_sqrtSigma_11_beamstrahlung must be an array')
            assert (len(slices_other_beam_sqrtSigma_11_beamstrahlung)
                == len(self.slices_other_beam_num_particles))
            self.slices_other_beam_sqrtSigma_11_beamstrahlung = self._arr2ctx(
                np.array(slices_other_beam_sqrtSigma_11_beamstrahlung))
        else:
            self.slices_other_beam_sqrtSigma_11_beamstrahlung[:] = 0

        if slices_other_beam_sqrtSigma_33_beamstrahlung is not None:
            assert not np.isscalar(slices_other_beam_sqrtSigma_33_beamstrahlung), (
                'slices_other_beam_sqrtSigma_33_beamstrahlung must be an array')
            assert (len(slices_other_beam_sqrtSigma_33_beamstrahlung)
                == len(self.slices_other_beam_num_particles))
            self.slices_other_beam_sqrtSigma_33_beamstrahlung = self._arr2ctx(
                np.array(slices_other_beam_sqrtSigma_33_beamstrahlung))
        else:
            self.slices_other_beam_sqrtSigma_33_beamstrahlung[:] = 0

        if slices_other_beam_sqrtSigma_55_beamstrahlung is not None:
            assert not np.isscalar(slices_other_beam_sqrtSigma_55_beamstrahlung), (
                'slices_other_beam_sqrtSigma_55_beamstrahlung must be an array')
            assert (len(slices_other_beam_sqrtSigma_55_beamstrahlung)
                == len(self.slices_other_beam_num_particles))
            self.slices_other_beam_sqrtSigma_55_beamstrahlung = self._arr2ctx(
                np.array(slices_other_beam_sqrtSigma_55_beamstrahlung))
        else:
            self.slices_other_beam_sqrtSigma_55_beamstrahlung[:] = 0

        self.flag_beamstrahlung = flag_beamstrahlung # Trigger property setter

    @property
    def flag_beamstrahlung(self):
        return self._flag_beamstrahlung

    @flag_beamstrahlung.setter
    def flag_beamstrahlung(self, flag_beamstrahlung):
        ctx2np  = self._context.nparray_from_context_array
        if flag_beamstrahlung == 1:  # avg photon emission, requires unboosted strong bunch slice RMS
            if np.any(ctx2np(
                self.slices_other_beam_sqrtSigma_11_beamstrahlung == 0)):
                raise ValueError(
                    'slices_other_beam_sqrtSigma_11_beamstrahlung'
                    'needs to be correctly set')
            if np.any(ctx2np(
                self.slices_other_beam_sqrtSigma_33_beamstrahlung == 0)):
                raise ValueError(
                    'slices_other_beam_sqrtSigma_33_beamstrahlung '
                    'needs to be correctly set')
            if np.any(ctx2np(
                self.slices_other_beam_sqrtSigma_55_beamstrahlung == 0)):
                raise ValueError(
                    'slices_other_beam_sqrtSigma_55_beamstrahlung '
                    'needs to be correctly set')
        elif flag_beamstrahlung == 2:  # quantum BS, requires boosted strong z bin widths
            if np.any(ctx2np(
                self.slices_other_beam_zeta_bin_width_star_beamstrahlung == 0)):
                raise ValueError(
                    'slices_other_beam_zeta_bin_width_star_beamstrahlung '
                    'needs to be correctly set')
        self._flag_beamstrahlung = flag_beamstrahlung

    def _init_bhabha(self, flag_bhabha, compt_x_min, compt_scale, flag_beamsize_effect):
        self.flag_beamsize_effect = flag_beamsize_effect
        self.compt_x_min = compt_x_min
        self.compt_scale = compt_scale
        self.flag_bhabha = flag_bhabha # Trigger property setter

    @property
    def flag_bhabha(self):
        return self._flag_bhabha

    @flag_bhabha.setter
    def flag_bhabha(self, flag_bhabha):
        if flag_bhabha == 1:
            if self.compt_x_min <= 0:
                raise ValueError('compt_x_min must be larger than 0')
            if self.compt_scale <= 0:
                raise ValueError('compt_scale must be larger than 0')
        self._flag_bhabha = flag_bhabha

    def _init_luminosity(self, flag_luminosity):
        self.flag_luminosity = flag_luminosity

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

    def update_from_received_moments(self):
        # reference frame transformation as in https://github.com/lhcopt/lhcmask/blob/865eaf9d7b9b888c6486de00214c0c24ac93cfd3/pymask/beambeam.py#L310
        self.slices_other_beam_num_particles = self._arr2ctx(self.partner_moments[:self.num_slices_other_beam]) # this contains num real particles, the total elementary charge per slice
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

    def _track_non_collective(self, particles, i_slices=None):

            # Change reference frame
            self.change_ref_frame(particles)

            self._debug_part_after_boost = particles.copy()

            i_slices = i_slices or range(len(self.slices_other_beam_zeta_center_star))

            # Beam beam interaction in the boosted frame
            for ii in i_slices:
                self.synchro_beam_kick(particles=particles,
                    i_slice_for_particles=ii + 0 * particles.particle_id)

            # Back to lab frame
            self.change_back_ref_frame_and_subtract_dipolar(particles)


    def _track_collective(self, particles, _force_suspend=False):
        if self.config_for_update._working_on_bunch is not None:
            # I am resuming a suspended calculation

            assert self.config_for_update._working_on_bunch == particles.name

            # Beam beam interaction in the boosted frame
            ret = self._sync_with_partner_and_apply_bb_kicks_in_boosted_frame(particles)

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
            ret = self._sync_with_partner_and_apply_bb_kicks_in_boosted_frame(particles)

            if ret is not None:
                return ret # PipelineStatus
            else:
                # Back to line reference frame
                self.change_back_ref_frame_and_subtract_dipolar(particles)
                return None

    def _sync_with_partner_and_apply_bb_kicks_in_boosted_frame(self, particles):

        n_slices_self_beam = self.config_for_update.slicer.num_slices

        while True: # loop over slices

            # recompute and communicate slice moments; if QSS only update before first step
            if (self.config_for_update._do_update and (not self.config_for_update.quasistrongstrong
                or self.config_for_update._i_step == 0)):

                status = self._compute_moments_and_sync_with_partner(particles)

                if status is not None:
                    return status # pipeline hold

            # compute interacting other beam slice ID
            self.config_for_update._other_beam_slice_index_for_particles[:] =(
                 self.config_for_update._i_step - self.config_for_update._particles_slice_index)

            self.synchro_beam_kick(particles=particles,
                        i_slice_for_particles=self.config_for_update._other_beam_slice_index_for_particles)

            self.config_for_update._i_step += 1
            if self.config_for_update._i_step == (
                            n_slices_self_beam + self.num_slices_other_beam - 1):
                self.config_for_update._i_step = 0
                self.config_for_update._working_on_bunch = None
                break

        return None

    def _compute_moments_and_sync_with_partner(self, particles):

        # Find at_turn from the first active particle
        ii = 0
        while particles.state[ii] != 1:
            ii += 1
        at_turn = int(particles.at_turn[ii])

        pipeline_manager = self.config_for_update.pipeline_manager

        if pipeline_manager.is_ready_to_send(self.config_for_update.element_name,
                                                particles.name,
                                                self.config_for_update.partner_particles_name,
                                                at_turn,
                                                internal_tag=self.config_for_update._i_step):
            # Compute moments
            self.moments = self.config_for_update.slicer.compute_moments(particles)  # assign to slices happens inside here

            pipeline_manager.send_message(self.moments,
                                                self.config_for_update.element_name,
                                                particles.name,
                                                self.config_for_update.partner_particles_name,
                                                at_turn,
                                                internal_tag=self.config_for_update._i_step)
        if pipeline_manager.is_ready_to_receive(self.config_for_update.element_name,
                                self.config_for_update.partner_particles_name,
                                particles.name,
                                internal_tag=self.config_for_update._i_step):
            pipeline_manager.receive_message(self.partner_moments, self.config_for_update.element_name,
                                             self.config_for_update.partner_particles_name, particles.name,
                                             internal_tag=self.config_for_update._i_step)
            self.update_from_received_moments()
        else:
            return xt.PipelineStatus(on_hold=True)

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
        _init_alpha_phi(self, phi=value)

        # Trigger properties to set corresponding starred quantities
        self._init_Sigmas(
            self.slices_other_beam_Sigma_11,
            self.slices_other_beam_Sigma_12,
            self.slices_other_beam_Sigma_13,
            self.slices_other_beam_Sigma_14,
            self.slices_other_beam_Sigma_22,
            self.slices_other_beam_Sigma_23,
            self.slices_other_beam_Sigma_24,
            self.slices_other_beam_Sigma_33,
            self.slices_other_beam_Sigma_34,
            self.slices_other_beam_Sigma_44,

            self.slices_other_beam_Sigma_11_star,
            self.slices_other_beam_Sigma_12_star,
            self.slices_other_beam_Sigma_13_star,
            self.slices_other_beam_Sigma_14_star,
            self.slices_other_beam_Sigma_22_star,
            self.slices_other_beam_Sigma_23_star,
            self.slices_other_beam_Sigma_24_star,
            self.slices_other_beam_Sigma_33_star,
            self.slices_other_beam_Sigma_34_star,
            self.slices_other_beam_Sigma_44_star,
        )

        # Initialize slice positions in the boosted frame. Depends on phi, alpha
        self._init_starred_positions(
            self.slices_other_beam_num_particles,
            self.slices_other_beam_x_center, self.slices_other_beam_px_center,
            self.slices_other_beam_y_center, self.slices_other_beam_py_center,
            self.slices_other_beam_zeta_center, self.slices_other_beam_pzeta_center,
            self.slices_other_beam_x_center_star, self.slices_other_beam_px_center_star,
            self.slices_other_beam_y_center_star, self.slices_other_beam_py_center_star,
            self.slices_other_beam_zeta_center_star, self.slices_other_beam_pzeta_center_star)

        # Initialize beamstrahlung related quantities. Depends on phi, alpha
        self._init_beamstrahlung(
            self.flag_beamstrahlung,
            self.slices_other_beam_zeta_bin_width_beamstrahlung,
            self.slices_other_beam_zeta_bin_width_star_beamstrahlung,
            self.slices_other_beam_sqrtSigma_11_beamstrahlung,
            self.slices_other_beam_sqrtSigma_33_beamstrahlung,
            self.slices_other_beam_sqrtSigma_55_beamstrahlung,
        )

    @property
    def alpha(self):
        return np.arctan2(self.sin_alpha, self.cos_alpha)

    @alpha.setter
    def alpha(self, value):
        _init_alpha_phi(self, alpha=value)

        # Trigger properties to set corresponding starred quantities
        self._init_Sigmas(
            self.slices_other_beam_Sigma_11,
            self.slices_other_beam_Sigma_12,
            self.slices_other_beam_Sigma_13,
            self.slices_other_beam_Sigma_14,
            self.slices_other_beam_Sigma_22,
            self.slices_other_beam_Sigma_23,
            self.slices_other_beam_Sigma_24,
            self.slices_other_beam_Sigma_33,
            self.slices_other_beam_Sigma_34,
            self.slices_other_beam_Sigma_44,

            self.slices_other_beam_Sigma_11_star,
            self.slices_other_beam_Sigma_12_star,
            self.slices_other_beam_Sigma_13_star,
            self.slices_other_beam_Sigma_14_star,
            self.slices_other_beam_Sigma_22_star,
            self.slices_other_beam_Sigma_23_star,
            self.slices_other_beam_Sigma_24_star,
            self.slices_other_beam_Sigma_33_star,
            self.slices_other_beam_Sigma_34_star,
            self.slices_other_beam_Sigma_44_star,
        )

        # Initialize slice positions in the boosted frame
        self._init_starred_positions(
            self.slices_other_beam_num_particles,
            self.slices_other_beam_x_center, self.slices_other_beam_px_center,
            self.slices_other_beam_y_center, self.slices_other_beam_py_center,
            self.slices_other_beam_zeta_center, self.slices_other_beam_pzeta_center,
            self.slices_other_beam_x_center_star, self.slices_other_beam_px_center_star,
            self.slices_other_beam_y_center_star, self.slices_other_beam_py_center_star,
            self.slices_other_beam_zeta_center_star, self.slices_other_beam_pzeta_center_star)

        # Initialize beamstrahlung related quantities
        self._init_beamstrahlung(
            self.flag_beamstrahlung,
            self.slices_other_beam_zeta_bin_width_beamstrahlung,
            self.slices_other_beam_zeta_bin_width_star_beamstrahlung,
            self.slices_other_beam_sqrtSigma_11_beamstrahlung,
            self.slices_other_beam_sqrtSigma_33_beamstrahlung,
            self.slices_other_beam_sqrtSigma_55_beamstrahlung,
        )

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

        # unboosted sigmas preferred
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

        if isinstance(slices_other_beam_num_particles, (list, tuple)):
            slices_other_beam_num_particles = np.array(slices_other_beam_num_particles)
        if isinstance(slices_other_beam_x_center, (list, tuple)):
            slices_other_beam_x_center = np.array(slices_other_beam_x_center)
        if isinstance(slices_other_beam_px_center, (list, tuple)):
            slices_other_beam_px_center = np.array(slices_other_beam_px_center)
        if isinstance(slices_other_beam_y_center, (list, tuple)):
            slices_other_beam_y_center = np.array(slices_other_beam_y_center)
        if isinstance(slices_other_beam_py_center, (list, tuple)):
            slices_other_beam_py_center = np.array(slices_other_beam_py_center)
        if isinstance(slices_other_beam_zeta_center, (list, tuple)):
            slices_other_beam_zeta_center = np.array(slices_other_beam_zeta_center)
        if isinstance(slices_other_beam_pzeta_center, (list, tuple)):
            slices_other_beam_pzeta_center = np.array(slices_other_beam_pzeta_center)
        if isinstance(slices_other_beam_x_center_star, (list, tuple)):
            slices_other_beam_x_center_star = np.array(slices_other_beam_x_center_star)
        if isinstance(slices_other_beam_px_center_star, (list, tuple)):
            slices_other_beam_px_center_star = np.array(slices_other_beam_px_center_star)
        if isinstance(slices_other_beam_y_center_star, (list, tuple)):
            slices_other_beam_y_center_star = np.array(slices_other_beam_y_center_star)
        if isinstance(slices_other_beam_py_center_star, (list, tuple)):
            slices_other_beam_py_center_star = np.array(slices_other_beam_py_center_star)
        if isinstance(slices_other_beam_zeta_center_star, (list, tuple)):
            slices_other_beam_zeta_center_star = np.array(slices_other_beam_zeta_center_star)
        if isinstance(slices_other_beam_pzeta_center_star, (list, tuple)):
            slices_other_beam_pzeta_center_star = np.array(slices_other_beam_pzeta_center_star)

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


class ConfigForUpdateBeamBeamBiGaussian3D:

    def __init__(self,
        pipeline_manager=None,
        element_name=None,
        slicer=None,
        partner_particles_name=None,
        update_every=None,
        quasistrongstrong=None):

        self.pipeline_manager = pipeline_manager
        self.element_name = element_name
        self.slicer = slicer
        self.partner_particles_name = partner_particles_name
        self.update_every = update_every
        self.quasistrongstrong = quasistrongstrong

        self._i_step = 0
        self._working_on_bunch = None
        self._particles_slice_index = None

def _init_alpha_phi(bb, phi=None, alpha=None,
                    _sin_phi=None, _cos_phi=None, _tan_phi=None,
                    _sin_alpha=None, _cos_alpha=None):

    if phi is None and _sin_phi is not None:
        bb._sin_phi = _sin_phi
        bb._cos_phi = _cos_phi
        bb._tan_phi = _tan_phi
    else:
        if phi is None:
            phi = 0
        bb._sin_phi = np.sin(phi)
        bb._cos_phi = np.cos(phi)
        bb._tan_phi = np.tan(phi)

    if alpha is None and _sin_alpha is not None:
        bb._sin_alpha = _sin_alpha
        bb._cos_alpha = _cos_alpha
    else:
        if alpha is None:
            alpha = 0
        bb._sin_alpha = np.sin(alpha)
        bb._cos_alpha = np.cos(alpha)
