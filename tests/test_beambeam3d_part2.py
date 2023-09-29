import numpy as np
import pytest

import xobjects as xo
import xtrack as xt
import xfields as xf
import xpart as xp

from xobjects.test_helpers import for_all_test_contexts

import ducktrack as dtk


def sigma_configurations():
    print('decoupled round beam')
    (Sig_11_0, Sig_12_0, Sig_13_0,
    Sig_14_0, Sig_22_0, Sig_23_0,
    Sig_24_0, Sig_33_0, Sig_34_0, Sig_44_0) = (
    20e-06,  0.,  0.,
    0., 0., 0.,
    0., 20e-6, 0., 0.)
    yield (Sig_11_0, Sig_12_0, Sig_13_0, Sig_14_0, Sig_22_0, Sig_23_0, Sig_24_0,
            Sig_33_0, Sig_34_0, Sig_44_0)

    print('decoupled tall beam')
    (Sig_11_0, Sig_12_0, Sig_13_0,
    Sig_14_0, Sig_22_0, Sig_23_0,
    Sig_24_0, Sig_33_0, Sig_34_0, Sig_44_0) = (
    20e-06,  0.,  0.,
    0., 0., 0.,
    0., 40e-6, 0., 0.)
    yield (Sig_11_0, Sig_12_0, Sig_13_0, Sig_14_0, Sig_22_0, Sig_23_0, Sig_24_0,
            Sig_33_0, Sig_34_0, Sig_44_0)

    print('decoupled fat beam')
    (Sig_11_0, Sig_12_0, Sig_13_0,
    Sig_14_0, Sig_22_0, Sig_23_0,
    Sig_24_0, Sig_33_0, Sig_34_0, Sig_44_0) = (
    40e-06,  0.,  0.,
    0., 0., 0.,
    0., 20e-6, 0., 0.)
    yield (Sig_11_0, Sig_12_0, Sig_13_0, Sig_14_0, Sig_22_0, Sig_23_0, Sig_24_0,
            Sig_33_0, Sig_34_0, Sig_44_0)

    print('coupled beam')
    (Sig_11_0, Sig_12_0, Sig_13_0,
    Sig_14_0, Sig_22_0, Sig_23_0,
    Sig_24_0, Sig_33_0, Sig_34_0, Sig_44_0) = (
    8.4282060230000004e-06,  1.8590458800000001e-07,  -3.5512334410000001e-06,
    -3.8254462239999997e-08, 4.101510281e-09, -7.5517657920000006e-08,
    -8.1134615060000002e-10, 1.031446898e-05, 1.177863077e-07, 1.3458251810000001e-09)
    yield (Sig_11_0, Sig_12_0, Sig_13_0, Sig_14_0, Sig_22_0, Sig_23_0, Sig_24_0,
            Sig_33_0, Sig_34_0, Sig_44_0)


for_all_sigma_configurations = pytest.mark.parametrize(
    'Sig_11_0, Sig_12_0, Sig_13_0, Sig_14_0, Sig_22_0, Sig_23_0, Sig_24_0,'
    'Sig_33_0, Sig_34_0, Sig_44_0',
    list(sigma_configurations()),
    ids=['decoupled round beam', 'decoupled tall beam', 'decoupled fat beam',
         'coupled beam'],
)


@for_all_test_contexts
@for_all_sigma_configurations
def test_beambeam3d_gx_gy_singularity(test_context, Sig_11_0, Sig_12_0, Sig_13_0,
                                      Sig_14_0, Sig_22_0, Sig_23_0, Sig_24_0,
                                      Sig_33_0, Sig_34_0, Sig_44_0):
    # crossing plane
    alpha = 0

    # crossing angle
    phi = 0

    # separations
    x_bb_co=0
    y_bb_co=0
    charge_slices=np.array([1e16, 2e16, 5e16])
    z_slices=np.array([-6., 0.2, 5.5])

    x_co = 0
    px_co = 0
    y_co = 0
    py_co = 0
    zeta_co = 0
    delta_co = 0

    d_x=0
    d_px=0
    d_y=-0
    d_py=0
    d_zeta=0
    d_delta=0

    Sig_11_0 = Sig_11_0 + np.zeros_like(charge_slices)
    Sig_12_0 = Sig_12_0 + np.zeros_like(charge_slices)
    Sig_13_0 = Sig_13_0 + np.zeros_like(charge_slices)
    Sig_14_0 = Sig_14_0 + np.zeros_like(charge_slices)
    Sig_22_0 = Sig_22_0 + np.zeros_like(charge_slices)
    Sig_23_0 = Sig_23_0 + np.zeros_like(charge_slices)
    Sig_24_0 = Sig_24_0 + np.zeros_like(charge_slices)
    Sig_33_0 = Sig_33_0 + np.zeros_like(charge_slices)
    Sig_34_0 = Sig_34_0 + np.zeros_like(charge_slices)
    Sig_44_0 = Sig_44_0 + np.zeros_like(charge_slices)

    bb = xf.BeamBeamBiGaussian3D(

        _context=test_context,

        phi=phi, alpha=alpha, other_beam_q0=1,

        slices_other_beam_num_particles=charge_slices[::-1],
        slices_other_beam_zeta_center=z_slices[::-1],

        slices_other_beam_Sigma_11=Sig_11_0,
        slices_other_beam_Sigma_12=Sig_12_0,
        slices_other_beam_Sigma_13=Sig_13_0,
        slices_other_beam_Sigma_14=Sig_14_0,
        slices_other_beam_Sigma_22=Sig_22_0,
        slices_other_beam_Sigma_23=Sig_23_0,
        slices_other_beam_Sigma_24=Sig_24_0,
        slices_other_beam_Sigma_33=Sig_33_0,
        slices_other_beam_Sigma_34=Sig_34_0,
        slices_other_beam_Sigma_44=Sig_44_0,

        ref_shift_x=x_co,
        ref_shift_px=px_co,
        ref_shift_y=y_co,
        ref_shift_py=py_co,
        ref_shift_zeta=zeta_co,
        ref_shift_pzeta=delta_co,

        other_beam_shift_x=x_bb_co,
        other_beam_shift_y=y_bb_co,

        post_subtract_x=d_x,
        post_subtract_px=d_px,
        post_subtract_y=d_y,
        post_subtract_py=d_py,
        post_subtract_zeta=d_zeta,
        post_subtract_pzeta=d_delta,
    )

    ctx2np = bb._context.nparray_from_context_array # Patch for Pyopencl
    bb.slices_other_beam_Sigma_11[1] = ctx2np(bb.slices_other_beam_Sigma_11)[0]
    bb.slices_other_beam_Sigma_12[1] = ctx2np(bb.slices_other_beam_Sigma_12)[0]
    bb.slices_other_beam_Sigma_13[1] = ctx2np(bb.slices_other_beam_Sigma_13)[0]
    bb.slices_other_beam_Sigma_14[1] = ctx2np(bb.slices_other_beam_Sigma_14)[0]
    bb.slices_other_beam_Sigma_22[1] = ctx2np(bb.slices_other_beam_Sigma_22)[0]
    bb.slices_other_beam_Sigma_23[1] = ctx2np(bb.slices_other_beam_Sigma_23)[0]
    bb.slices_other_beam_Sigma_24[1] = ctx2np(bb.slices_other_beam_Sigma_24)[0]
    bb.slices_other_beam_Sigma_33[1] = ctx2np(bb.slices_other_beam_Sigma_33)[0]
    bb.slices_other_beam_Sigma_34[1] = ctx2np(bb.slices_other_beam_Sigma_34)[0]
    bb.slices_other_beam_Sigma_44[1] = ctx2np(bb.slices_other_beam_Sigma_44)[0]

    part= xp.Particles(_context=test_context, p0c=6500e9)

    part.name = 'beam1_bunch1'

    bb.track(part)

    part.move(xo.context_default)
    assert not np.isnan(part.px[0])
    assert not np.isnan(part.py[0])

@for_all_test_contexts
@for_all_sigma_configurations
def test_beambeam3d_collective(test_context, Sig_11_0, Sig_12_0, Sig_13_0,
                               Sig_14_0, Sig_22_0, Sig_23_0, Sig_24_0, Sig_33_0,
                               Sig_34_0, Sig_44_0):

    if isinstance(test_context, xo.ContextPyopencl):
        pytest.skip("Not implemented for OpenCL")
        return

    if isinstance(test_context, xo.ContextCupy):
        pytest.skip("Not implemented for cupy")
        return

    # crossing plane
    alpha = 0.7

    # crossing angle
    phi = 0.8

    # separations
    x_bb_co=5e-3
    y_bb_co=-4e-3
    charge_slices=np.array([1e16, 2e16, 5e16])
    z_slices=np.array([-6., 0.2, 5.5])

    x_co = 2e-3
    px_co= 1e-6
    y_co=-3e-3
    py_co=-2e-6
    zeta_co=0.01
    delta_co=1.2e-3

    d_x=1.5e-3
    d_px=1.6e-6
    d_y=-1.7e-3
    d_py=-1.8e-6
    d_zeta=0.019
    d_delta=3e-4

    Sig_11_0 = Sig_11_0 + np.zeros_like(charge_slices)
    Sig_12_0 = Sig_12_0 + np.zeros_like(charge_slices)
    Sig_13_0 = Sig_13_0 + np.zeros_like(charge_slices)
    Sig_14_0 = Sig_14_0 + np.zeros_like(charge_slices)
    Sig_22_0 = Sig_22_0 + np.zeros_like(charge_slices)
    Sig_23_0 = Sig_23_0 + np.zeros_like(charge_slices)
    Sig_24_0 = Sig_24_0 + np.zeros_like(charge_slices)
    Sig_33_0 = Sig_33_0 + np.zeros_like(charge_slices)
    Sig_34_0 = Sig_34_0 + np.zeros_like(charge_slices)
    Sig_44_0 = Sig_44_0 + np.zeros_like(charge_slices)

    print('------------------------')
    print(locals())

    bb_dtk = dtk.elements.BeamBeam6D(
        phi=phi, alpha=alpha,
        x_bb_co=x_bb_co,
        y_bb_co=y_bb_co,
        charge_slices=charge_slices,
        zeta_slices=z_slices,
        sigma_11=Sig_11_0[0],
        sigma_12=Sig_12_0[0],
        sigma_13=Sig_13_0[0],
        sigma_14=Sig_14_0[0],
        sigma_22=Sig_22_0[0],
        sigma_23=Sig_23_0[0],
        sigma_24=Sig_24_0[0],
        sigma_33=Sig_33_0[0],
        sigma_34=Sig_34_0[0],
        sigma_44=Sig_44_0[0],
        x_co=x_co,
        px_co=px_co,
        y_co=y_co,
        py_co=py_co,
        zeta_co=zeta_co,
        delta_co=delta_co,
        d_x=d_x,
        d_px=d_px,
        d_y=d_y,
        d_py=d_py,
        d_zeta=d_zeta,
        d_delta=d_delta
    )

    slicer = xf.TempSlicer(n_slices=3, sigma_z=1, mode="unibin")
    # slicer = xf.TempSlicer(bin_edges = [-10, -5, 0, 5, 10])
    config_for_update=xf.ConfigForUpdateBeamBeamBiGaussian3D(
        pipeline_manager=None,
        element_name=None,
        slicer=slicer,
        update_every=None, # Never updates (test in weakstrong mode)
    )


    bb = xf.BeamBeamBiGaussian3D(
        _context=test_context,

        config_for_update=config_for_update,

        phi=phi, alpha=alpha, other_beam_q0=1,

        slices_other_beam_num_particles=charge_slices[::-1],
        slices_other_beam_zeta_center=z_slices[::-1],

        slices_other_beam_Sigma_11=Sig_11_0,
        slices_other_beam_Sigma_12=Sig_12_0,
        slices_other_beam_Sigma_13=Sig_13_0,
        slices_other_beam_Sigma_14=Sig_14_0,
        slices_other_beam_Sigma_22=Sig_22_0,
        slices_other_beam_Sigma_23=Sig_23_0,
        slices_other_beam_Sigma_24=Sig_24_0,
        slices_other_beam_Sigma_33=Sig_33_0,
        slices_other_beam_Sigma_34=Sig_34_0,
        slices_other_beam_Sigma_44=Sig_44_0,

        ref_shift_x=x_co,
        ref_shift_px=px_co,
        ref_shift_y=y_co,
        ref_shift_py=py_co,
        ref_shift_zeta=zeta_co,
        ref_shift_pzeta=delta_co,

        other_beam_shift_x=x_bb_co,
        other_beam_shift_y=y_bb_co,

        post_subtract_x=d_x,
        post_subtract_px=d_px,
        post_subtract_y=d_y,
        post_subtract_py=d_py,
        post_subtract_zeta=d_zeta,
        post_subtract_pzeta=d_delta,
    )

    dtk_part = dtk.TestParticles(
        p0c=6500e9,
        x=-1.23e-3,
        px = 50e-3,
        y = 2e-3,
        py = 27e-3,
        sigma = 3.,
        delta = 2e-4)

    part= xp.Particles(_context=test_context, **dtk_part.to_dict())

    part.name = 'beam1_bunch1'

    ret = bb.track(part, _force_suspend=True)
    assert ret.on_hold
    ret = bb.track(part)
    assert ret is None

    bb_dtk.track(dtk_part)

    for cc in 'x px y py zeta delta'.split():
        val_test = getattr(part, cc)[0]
        val_ref = getattr(dtk_part, cc)
        print('')
        print(f'ducktrack: {cc} = {val_ref:.12e}')
        print(f'xsuite:    {cc} = {val_test:.12e}')
        assert np.isclose(val_test, val_ref, rtol=0, atol=5e-12)


@for_all_test_contexts
@for_all_sigma_configurations
def test_beambeam3d_old_interface(test_context, Sig_11_0, Sig_12_0, Sig_13_0,
                                  Sig_14_0, Sig_22_0, Sig_23_0, Sig_24_0,
                                  Sig_33_0, Sig_34_0, Sig_44_0):
    # crossing plane
    alpha = 0.7

    # crossing angle
    phi = 0.8

    # separations
    x_bb_co=5e-3
    y_bb_co=-4e-3
    charge_slices=np.array([1e16, 2e16, 5e16])
    z_slices=np.array([-6., 0.2, 5.5])

    x_co = 2e-3
    px_co= 1e-6
    y_co=-3e-3
    py_co=-2e-6
    zeta_co=0.01
    delta_co=1.2e-3

    d_x=1.5e-3
    d_px=1.6e-6
    d_y=-1.7e-3
    d_py=-1.8e-6
    d_zeta=0.019
    d_delta=3e-4

    Sig_11_0 = Sig_11_0 + np.zeros_like(charge_slices)
    Sig_12_0 = Sig_12_0 + np.zeros_like(charge_slices)
    Sig_13_0 = Sig_13_0 + np.zeros_like(charge_slices)
    Sig_14_0 = Sig_14_0 + np.zeros_like(charge_slices)
    Sig_22_0 = Sig_22_0 + np.zeros_like(charge_slices)
    Sig_23_0 = Sig_23_0 + np.zeros_like(charge_slices)
    Sig_24_0 = Sig_24_0 + np.zeros_like(charge_slices)
    Sig_33_0 = Sig_33_0 + np.zeros_like(charge_slices)
    Sig_34_0 = Sig_34_0 + np.zeros_like(charge_slices)
    Sig_44_0 = Sig_44_0 + np.zeros_like(charge_slices)

    print('------------------------')
    print(locals())

    bb_dtk = dtk.elements.BeamBeam6D(
        phi=phi, alpha=alpha,
        x_bb_co=x_bb_co,
        y_bb_co=y_bb_co,
        charge_slices=charge_slices,
        zeta_slices=z_slices,
        sigma_11=Sig_11_0[0],
        sigma_12=Sig_12_0[0],
        sigma_13=Sig_13_0[0],
        sigma_14=Sig_14_0[0],
        sigma_22=Sig_22_0[0],
        sigma_23=Sig_23_0[0],
        sigma_24=Sig_24_0[0],
        sigma_33=Sig_33_0[0],
        sigma_34=Sig_34_0[0],
        sigma_44=Sig_44_0[0],
        x_co=x_co,
        px_co=px_co,
        y_co=y_co,
        py_co=py_co,
        zeta_co=zeta_co,
        delta_co=delta_co,
        d_x=d_x,
        d_px=d_px,
        d_y=d_y,
        d_py=d_py,
        d_zeta=d_zeta,
        d_delta=d_delta
    )

    bb = xf.BeamBeamBiGaussian3D(old_interface=bb_dtk.to_dict(), _context=test_context)

    dtk_part = dtk.TestParticles(
        p0c=6500e9,
        x=-1.23e-3,
        px = 50e-3,
        y = 2e-3,
        py = 27e-3,
        sigma = 3.,
        delta = 2e-4)

    part=xp.Particles(_context=test_context, **dtk_part.to_dict())

    bb.track(part)

    bb_dtk.track(dtk_part)

    part.move(_context=xo.ContextCpu())
    for cc in 'x px y py zeta delta'.split():
        val_test = getattr(part, cc)[0]
        val_ref = getattr(dtk_part, cc)
        print('')
        print(f'ducktrack: {cc} = {val_ref:.12e}')
        print(f'xsuite:    {cc} = {val_test:.12e}')
        assert np.isclose(val_test, val_ref, rtol=0, atol=5e-12)

     # Scaling down bb:
    bb.scale_strength = 0
    part_before_tracking = part.copy()
    part.move(_context=test_context)
    bb.track(part)
    part.move(_context=xo.ContextCpu())

    for cc in 'x px y py zeta delta'.split():
        val_test = getattr(part, cc)[0]
        val_ref = getattr(part_before_tracking, cc)[0]
        print('')
        print(f'before: {cc} = {val_ref:.12e}')
        print(f'after bb off:    {cc} = {val_test:.12e}')
        assert np.allclose(val_test, val_ref, rtol=0, atol=1e-14)
