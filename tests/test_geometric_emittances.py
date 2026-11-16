import numpy as np
import pytest
import xtrack as xt
import xfields as xf

from test_rigid_bunch_beambeam import _make_toy_environment
from xfields.config_tools.beambeam_config_tools.config_tools import compute_twiss_and_madpoints_at_bb


def test_frozen_spacecharge_geometric_emittances():
    def make_line():
        line = xt.Line(elements=[xt.LineSegmentMap(length=100, qx=0.28, qy=0.31,
                                                  betx=3., bety=5.)],
                       particle_ref=xt.Particles(p0c=2e9))
        line.build_tracker()
        return line
    ln, lg = make_line(), make_line()
    bg = ln.particle_ref.beta0[0] * ln.particle_ref.gamma0[0]
    kwargs = dict(longitudinal_profile=xf.LongitudinalProfileQGaussian(
                      number_of_particles=1e10, sigma_z=0.1),
                  sigma_z=0.1, s_spacecharge=np.array([0.]), delta_rms=1e-3)
    xf.install_spacecharge_frozen(line=ln, nemitt_x=2e-6, nemitt_y=3e-6, **kwargs)
    xf.install_spacecharge_frozen(line=lg, gemitt_x=2e-6/bg, gemitt_y=3e-6/bg, **kwargs)
    for key in ('sigma_x', 'sigma_y', 'length'):
        assert getattr(ln['spacecharge_0'], key) == pytest.approx(getattr(lg['spacecharge_0'], key))
    before = list(lg.element_names)
    with pytest.raises(ValueError, match='Cannot provide both'):
        xf.install_spacecharge_frozen(line=lg, nemitt_x=2e-6, gemitt_x=2e-6/bg,
                                      nemitt_y=3e-6, **kwargs)
    assert list(lg.element_names) == before


def test_beambeam_geometric_covariance_different_energies():
    env = _make_toy_environment()
    env.acw.particle_ref = xt.Particles(p0c=3e12)
    result = compute_twiss_and_madpoints_at_bb(
        env.cw, env.acw, {'cw': {'ip1': ['ip1']}, 'acw': {'ip1': ['ip1']}},
        gemitt_x=2e-9, gemitt_y=3e-9, survey_separation=False)
    for beam in ('cw', 'acw'):
        tw = result['twiss'][beam]
        cov = result['covariance'][beam]
        np.testing.assert_allclose(cov.Sigma11, tw.betx * 2e-9, rtol=1e-10)
        np.testing.assert_allclose(cov.Sigma33, tw.bety * 3e-9, rtol=1e-10)


@pytest.mark.parametrize('mode', ['particles', 'rigid_bunch'])
@pytest.mark.parametrize('plane', ['x', 'y'])
def test_beambeam_conflicting_emittances(mode, plane):
    env = xt.Environment()
    env.extra_config['xfields_beambeam'] = {'mode': mode}
    kwargs = dict(nemitt_x=2e-6, nemitt_y=3e-6, num_particles=1e10)
    kwargs[f'gemitt_{plane}'] = 1e-6
    with pytest.raises(ValueError, match='Cannot provide both'):
        env.xfields.configure_beambeam_interactions(**kwargs)
