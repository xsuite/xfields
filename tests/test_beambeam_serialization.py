import json

import numpy as np
import pytest

import xfields as xf
import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts


def _make_beambeam(element_class, context, scale_strength=1.):
    kwargs = dict(_context=context, other_beam_q0=1,
                  scale_strength=scale_strength)
    if element_class is xf.BeamBeamBiGaussian2D:
        kwargs.update(other_beam_beta0=1., other_beam_num_particles=1e11,
                      other_beam_Sigma_11=1e-6, other_beam_Sigma_33=1e-6)
    else:
        kwargs.update(
            phi=0., alpha=0.,
            slices_other_beam_num_particles=[1e11],
            slices_other_beam_zeta_center=[0.],
            slices_other_beam_Sigma_11=[1e-6],
            slices_other_beam_Sigma_12=[0.],
            slices_other_beam_Sigma_22=[1e-12],
            slices_other_beam_Sigma_33=[1e-6],
            slices_other_beam_Sigma_34=[0.],
            slices_other_beam_Sigma_44=[1e-12],
        )
    return element_class(**kwargs)


@pytest.mark.parametrize('element_class',
                         [xf.BeamBeamBiGaussian2D, xf.BeamBeamBiGaussian3D])
@pytest.mark.parametrize('scale_strength', [0., 1., 0.3, -0.5])
@pytest.mark.parametrize('copy_to_cpu', [True, False])
@for_all_test_contexts
def test_beambeam_scale_strength_roundtrip(
        test_context, element_class, scale_strength, copy_to_cpu):
    element = _make_beambeam(element_class, test_context, scale_strength)
    data = element.to_dict(copy_to_cpu=copy_to_cpu)
    restored = element_class.from_dict(data, _context=test_context)
    assert restored.scale_strength == scale_strength


@pytest.mark.parametrize('element_class',
                         [xf.BeamBeamBiGaussian2D, xf.BeamBeamBiGaussian3D])
@pytest.mark.parametrize('build_on_cpu', [True, False])
@for_all_test_contexts
def test_beambeam_filling_pattern_json_roundtrip(
        test_context, element_class, build_on_cpu, tmp_path):
    build_context = xo.context_default if build_on_cpu else test_context
    line = xt.Line(elements={
        'bb_off': _make_beambeam(element_class, build_context),
        'bb_on': _make_beambeam(element_class, build_context),
    })
    line.vars['beambeam_scale'] = 1.
    for name in line.element_names:
        line.vars[f'{name}_scale_strength'] = line.vars['beambeam_scale']
        line.element_refs[name].scale_strength = (
            line.vars[f'{name}_scale_strength'])
    # A missing opposing bunch disables this lens independently of the global knob.
    line.vars['bb_off_scale_strength'] = 0.

    path = tmp_path / 'line.json'
    line.to_json(path)
    restored = xt.Line.from_json(path, _context=build_context)
    assert restored['bb_off'].scale_strength == 0.
    assert restored['bb_on'].scale_strength == 1.
    assert restored.vars.val['bb_off_scale_strength'] == 0.

    restored.vars['beambeam_scale'] = 0.
    assert restored['bb_off'].scale_strength == 0.
    assert restored['bb_on'].scale_strength == 0.

    restored.build_tracker(_context=test_context)
    for name in restored.element_names:
        assert restored[name]._context is test_context
    particles = xt.Particles(_context=test_context, p0c=7e12, x=1e-3, y=2e-3)
    before = particles.copy()
    restored.track(particles)
    for coord in ('x', 'px', 'y', 'py', 'zeta', 'delta'):
        np.testing.assert_array_equal(
            test_context.nparray_from_context_array(getattr(particles, coord)),
            test_context.nparray_from_context_array(getattr(before, coord)))

    restored.vars['beambeam_scale'] = 0.5
    assert restored['bb_off'].scale_strength == 0.
    assert restored['bb_on'].scale_strength == 0.5
    reference = restored.copy(_context=xo.context_default)
    reference.build_tracker()
    reference_particles = particles.copy(_context=xo.context_default)
    reference.track(reference_particles)
    restored.track(particles)
    assert np.any(test_context.nparray_from_context_array(particles.px) != 0.)
    for coord in ('x', 'px', 'y', 'py', 'zeta', 'delta'):
        np.testing.assert_allclose(
            test_context.nparray_from_context_array(getattr(particles, coord)),
            getattr(reference_particles, coord), rtol=1e-11, atol=1e-18)


def _beambeam3d_slice_kwargs(num_slices, starred, starred_width):
    factors = np.linspace(1., 1.5, num_slices)
    kwargs = dict(phi=0.15, alpha=0.3, other_beam_q0=1., scale_strength=0.3,
                  flag_beamstrahlung=1,
                  slices_other_beam_num_particles=1e11 * factors)
    suffix = '_star' if starred else ''
    for coord, value in dict(x=1e-4, px=2e-6, y=-3e-5, py=-4e-6,
                             zeta=1e-3, pzeta=5e-4).items():
        kwargs[f'slices_other_beam_{coord}_center{suffix}'] = value * factors[::-1]
    for ij, value in {'11': 1e-6, '12': 1e-10, '13': 2e-10, '14': 3e-12,
                      '22': 2e-12, '23': 4e-12, '24': 5e-14,
                      '33': 2e-6, '34': 6e-10, '44': 3e-12}.items():
        kwargs[f'slices_other_beam_Sigma_{ij}{suffix}'] = value * factors
    width_suffix = '_star' if starred_width else ''
    kwargs[f'slices_other_beam_zeta_bin_width{width_suffix}_beamstrahlung'] = 1e-3 * factors
    for ij, value in {'11': 1e-3, '33': 2e-3, '55': 3e-3}.items():
        kwargs[f'slices_other_beam_sqrtSigma_{ij}_beamstrahlung'] = value * factors
    return kwargs


@pytest.mark.parametrize('num_slices', [1, 3])
@pytest.mark.parametrize('starred', [True, False])
@pytest.mark.parametrize('starred_width', [True, False])
@pytest.mark.parametrize('array_type', ['numpy', 'list', 'context'])
@for_all_test_contexts
def test_beambeam3d_slice_arrays_roundtrip(
        test_context, num_slices, starred, starred_width, array_type):
    kwargs = _beambeam3d_slice_kwargs(num_slices, starred, starred_width)
    reference = xf.BeamBeamBiGaussian3D(_context=xo.context_default, **kwargs)
    array_names = [name for name in reference._xofields
                   if name.startswith('slices_other_beam_')]

    for name, value in kwargs.items():
        if isinstance(value, np.ndarray):
            if array_type == 'list':
                kwargs[name] = value.tolist()
            elif array_type == 'context':
                kwargs[name] = test_context.nparray_to_context_array(value)

    element = xf.BeamBeamBiGaussian3D(_context=test_context, **kwargs)
    # Exercise construction plus host-array, device-array and JSON reloads.
    elements = [element]
    for copy_to_cpu in (True, False):
        elements.append(xf.BeamBeamBiGaussian3D.from_dict(
            element.to_dict(copy_to_cpu=copy_to_cpu), _context=test_context))
    elements.append(xf.BeamBeamBiGaussian3D.from_dict(
        json.loads(json.dumps(element.to_dict(), cls=xo.JEncoder)),
        _context=test_context))

    for restored in elements:
        assert restored._context is test_context
        assert restored.scale_strength == reference.scale_strength
        assert restored.flag_beamstrahlung == 1
        assert restored.num_slices_other_beam == num_slices
        for name in array_names:
            np.testing.assert_allclose(
                test_context.nparray_from_context_array(getattr(restored, name)),
                getattr(reference, name), rtol=2e-14, atol=0,
                err_msg=name)


@for_all_test_contexts
def test_beambeam3d_beamstrahlung_tuple_arrays(test_context):
    kwargs = _beambeam3d_slice_kwargs(3, starred=True, starred_width=True)
    reference = xf.BeamBeamBiGaussian3D(**kwargs)
    tuple_names = [name for name, value in kwargs.items()
                   if name.endswith('_beamstrahlung')
                   and isinstance(value, np.ndarray)]
    tuple_names.append('slices_other_beam_num_particles')
    for name in tuple_names:
        kwargs[name] = tuple(kwargs[name])
    element = xf.BeamBeamBiGaussian3D(_context=test_context, **kwargs)
    for name in tuple_names:
        np.testing.assert_array_equal(
            test_context.nparray_from_context_array(getattr(element, name)),
            getattr(reference, name))
