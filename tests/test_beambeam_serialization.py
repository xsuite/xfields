import numpy as np
import pytest

import xfields as xf
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
@for_all_test_contexts
def test_beambeam_filling_pattern_json_roundtrip(
        test_context, element_class, tmp_path):
    line = xt.Line(elements={
        'bb_off': _make_beambeam(element_class, test_context),
        'bb_on': _make_beambeam(element_class, test_context),
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
    restored = xt.Line.from_json(path, _context=test_context)
    assert restored['bb_off'].scale_strength == 0.
    assert restored['bb_on'].scale_strength == 1.
    assert restored.vars.val['bb_off_scale_strength'] == 0.

    restored.vars['beambeam_scale'] = 0.
    assert restored['bb_off'].scale_strength == 0.
    assert restored['bb_on'].scale_strength == 0.

    restored.build_tracker(_context=test_context)
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
    restored.track(particles)
    assert np.any(test_context.nparray_from_context_array(particles.px) != 0.)
