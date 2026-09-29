# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2025.                   #
# ########################################### #
import numpy as np
import pytest

import xobjects as xo
from xobjects.test_helpers import (
    allow_kernel_compilation, for_all_test_contexts)
import xtrack as xt
import xfields as xf


#############################################################
# Shared beam parameters
#############################################################
NEMITT_X         = 1e-5
NEMITT_Y         = 1e-7
SIGMA_Z          = 4e-3
SIGMA_DELTA      = 1e-3
BUNCH_INTENSITY = 4e9


#############################################################
# Module-level fixture: toy ring, twiss, and LMA
#############################################################
@pytest.fixture(scope='module')
def toy_ring():
    """
    Build a FODO-like toy ring with TouschekScattering elements and apertures,
    run Twiss and the local momentum acceptance exactly once for the whole
    test session.

    Yields a dict with keys:
      line  - the xtrack.Line
      twiss - the 4d Twiss table
      lma   - the xt.Table of local momentum acceptance
    """
    lbend = 3.0
    angle = np.pi / 2
    lquad = 0.3

    env = xt.Environment()
    line = env.new_line(components=[
        env.new('mqf.1', xt.Quadrupole, length=lquad, k1= 0.1),
        env.new('d1.1',  xt.Drift,      length=1.0),
        env.new('mb1.1', xt.Bend,       length=lbend, angle=angle),
        env.new('d2.1',  xt.Drift,      length=1.0),

        env.new('mqd.1', xt.Quadrupole, length=lquad, k1=-0.7),
        env.new('d3.1',  xt.Drift,      length=1.0),
        env.new('mb2.1', xt.Bend,       length=lbend, angle=angle),
        env.new('d4.1',  xt.Drift,      length=1.0),

        env.new('mqf.2', xt.Quadrupole, length=lquad, k1= 0.1),
        env.new('d1.2',  xt.Drift,      length=1.0),
        env.new('mb1.2', xt.Bend,       length=lbend, angle=angle),
        env.new('d2.2',  xt.Drift,      length=1.0),

        env.new('mqd.2', xt.Quadrupole, length=lquad, k1=-0.7),
        env.new('d3.2',  xt.Drift,      length=1.0),
        env.new('mb2.2', xt.Bend,       length=lbend, angle=angle),
        env.new('d4.2',  xt.Drift,      length=1.0),
    ])

    line.set_particle_ref('electron', p0c=1e9)
    line.configure_bend_model(core='full', edge=None)

    # Insert one TouschekScattering at the entrance of every magnet,
    # plus one at the very end of the line — all in a single batch insert.
    tab = line.get_table()
    tab_bends_quads = tab.rows[
        (tab.element_type == 'Bend') | (tab.element_type == 'Quadrupole')
    ]
    placements = []
    for ii, nn in enumerate(tab_bends_quads.name):
        tscatter_name = f'TScatter.{ii}'
        env.elements[tscatter_name] = xf.TouschekScattering()
        placements.append(env.place(tscatter_name, at=0.0, from_=nn))

    # Last TouschekScattering at the end of the line
    tscatter_name = f'TScatter.{ii+1}'
    env.elements[tscatter_name] = xf.TouschekScattering()
    placements.append(env.place(tscatter_name, at=tab.s[-1]))

    line.insert(placements)

    # Rectangular apertures around every non-drift/non-marker element
    aper_size = 0.040  # m
    tab = line.get_table()
    needs_aperture = tab.rows.match_not(element_type='Drift.*|Marker|').name
    placements = []
    for nn in needs_aperture:
        env.new(f'{nn}_aper_entry', xt.LimitRect,
                min_x=-aper_size, max_x=aper_size,
                min_y=-aper_size, max_y=aper_size)
        placements.append(env.place(f'{nn}_aper_entry', at=f'{nn}@start'))
        env.new(f'{nn}_aper_exit', xt.LimitRect,
                min_x=-aper_size, max_x=aper_size,
                min_y=-aper_size, max_y=aper_size)
        placements.append(env.place(f'{nn}_aper_exit', at=f'{nn}@end'))
    line.insert(placements)

    # Twiss
    tw = line.twiss(method='4d')
    tw.particle_on_co.move(_context=line._context)

    # LMA
    tab = line.get_table()
    elements = tab.rows.match(element_type='TouschekScattering').name
    lma = line.get_local_momentum_acceptance(
        elements=elements,
        twiss=tw,
        nemitt_x=NEMITT_X,
        nemitt_y=NEMITT_Y,
        y_offset=1e-12,
        delta_negative_limit=-0.012,
        delta_positive_limit=+0.012,
        delta_step_size=0.001,
        n_turns=64,
        method='4d',
        with_progress=False,
        verbose=False,
    )

    yield dict(line=line, twiss=tw, lma=lma)


def _build_tracker_or_skip(line, test_context):
    line.discard_tracker()
    try:
        line.build_tracker(_context=test_context)
    except Exception as err:
        if (
            isinstance(test_context, xo.ContextCpu)
            and test_context.openmp_enabled
        ):
            pytest.skip(
                f"OpenMP tracker compilation failed for {test_context}: {err}"
            )
        raise


#############################################################
# Helper: fresh (unscaled) copy of LMA and new TouschekStudy
#############################################################
def _fresh_lma(toy_ring_data):
    """Return an independent copy of the module-level LMA table."""
    lma = toy_ring_data['lma']
    return xt.Table({
        'name':   lma.name.copy(),
        's':      lma.s.copy(),
        'delta_neg': lma.delta_neg.copy(),
        'delta_pos': lma.delta_pos.copy(),
    })

def _toy_ring_variant(toy_ring_data, rename=None, drop=()):
    """
    Copy of the toy ring, with the TouschekScattering elements renamed
    (``rename(name, index) -> new name``, index among the
    TouschekScattering elements) and/or the elements in ``drop`` removed.
    The LMA of the toy ring still applies (it is interpolated in s).
    Returns the new line and its 4d Twiss.
    """
    line     = toy_ring_data['line']
    env      = line.env
    tab      = line.get_table()
    ts_names = list(tab.rows.match(element_type='TouschekScattering').name)
    names = []
    for nn in line.element_names:
        if nn in drop:
            continue
        if rename is not None and nn in ts_names:
            new_name = rename(nn, ts_names.index(nn))
            env.elements[new_name] = xf.TouschekScattering()
            nn = new_name
        names.append(nn)
    # Same magnets and apertures (shared, not modified), new line
    new_line = env.new_line(components=names)
    new_line.particle_ref = line.particle_ref.copy()
    return new_line, new_line.twiss(method='4d')

def _build_study(line, lma, twiss=None, *, n_scattering_events=int(1e6), **kwargs):
    """
    Convenience factory for TouschekStudy with sensible test defaults.
    """
    defaults = dict(
        local_momentum_acceptance=lma,
        local_momentum_acceptance_scale=0.85,
        nemitt_x=NEMITT_X,
        nemitt_y=NEMITT_Y,
        sigma_z=SIGMA_Z,
        sigma_delta=SIGMA_DELTA,
        bunch_intensity=BUNCH_INTENSITY,
        n_scattering_events=n_scattering_events,
        nx=3, ny=3, nz=3,
        weight_retention_fraction=0.99,
        seed=1997,
        method='4d',
    )
    defaults.update(kwargs)
    return xf.TouschekStudy(line, twiss=twiss, **defaults)


#############################################################
# Tests
#############################################################
class TestTouschekStudyInit:
    """Unit-tests for TouschekStudy constructor validation."""

    def test_construction_with_normalised_emittances(self, toy_ring):
        """Study constructed with nemitt_x/y converts to geometric emittances."""
        line = toy_ring['line']
        lma  = _fresh_lma(toy_ring)
        tw   = toy_ring['twiss']
        study = _build_study(line, lma, tw)
        beta0  = line.particle_ref.beta0[0]
        gamma0 = line.particle_ref.gamma0[0]
        assert study.gemitt_x == pytest.approx(NEMITT_X / (beta0 * gamma0))
        assert study.gemitt_y == pytest.approx(NEMITT_Y / (beta0 * gamma0))

    def test_construction_with_geometric_emittances(self, toy_ring):
        """Study constructed with gemitt_x/y stores them directly."""
        line   = toy_ring['line']
        lma    = _fresh_lma(toy_ring)
        tw     = toy_ring['twiss']
        beta0  = line.particle_ref.beta0[0]
        gamma0 = line.particle_ref.gamma0[0]
        gx = NEMITT_X / (beta0 * gamma0)
        gy = NEMITT_Y / (beta0 * gamma0)
        study = _build_study(line, lma, tw, gemitt_x=gx, gemitt_y=gy,
                             nemitt_x=None, nemitt_y=None)
        assert study.gemitt_x == pytest.approx(gx)
        assert study.gemitt_y == pytest.approx(gy)

    def test_lma_is_scaled_in_place(self, toy_ring):
        """LMA columns must be multiplied by local_momentum_acceptance_scale."""
        line          = toy_ring['line']
        tw            = toy_ring['twiss']
        lma_fresh     = _fresh_lma(toy_ring)
        delta_neg_before = lma_fresh.delta_neg.copy()
        delta_pos_before = lma_fresh.delta_pos.copy()
        scale = 0.85
        _build_study(line, lma_fresh, tw,
                       local_momentum_acceptance_scale=scale)
        np.testing.assert_allclose(lma_fresh.delta_neg, delta_neg_before * scale)
        np.testing.assert_allclose(lma_fresh.delta_pos, delta_pos_before * scale)

    def test_raises_on_missing_line(self, toy_ring):
        with pytest.raises(ValueError, match=r'`line` is required'):
            xf.TouschekStudy(
                line=None,
                local_momentum_acceptance=_fresh_lma(toy_ring),
                sigma_z=SIGMA_Z, sigma_delta=SIGMA_DELTA,
                bunch_intensity=BUNCH_INTENSITY, n_scattering_events=10,
                nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y,
            )

    def test_raises_on_missing_particle_ref(self, toy_ring):
        line = xt.Line(
            elements=[xf.TouschekScattering()],
            element_names=['ts'],
        )
        with pytest.raises(ValueError, match=r'particle_ref'):
            xf.TouschekStudy(
                line,
                local_momentum_acceptance=_fresh_lma(toy_ring),
                sigma_z=SIGMA_Z, sigma_delta=SIGMA_DELTA,
                bunch_intensity=BUNCH_INTENSITY, n_scattering_events=10,
                nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y,
            )

    @pytest.mark.parametrize('missing_key', [
        'sigma_z', 'sigma_delta', 'bunch_intensity', 'n_scattering_events',
    ])
    def test_raises_on_missing_required_kwarg(self, toy_ring, missing_key):
        """Each required kwarg should raise ValueError when absent."""
        line = toy_ring['line']
        required = dict(
            local_momentum_acceptance=_fresh_lma(toy_ring),
            sigma_z=SIGMA_Z, sigma_delta=SIGMA_DELTA,
            bunch_intensity=BUNCH_INTENSITY, n_scattering_events=10,
            nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y,
        )
        kw = {k: v for k, v in required.items() if k != missing_key}
        with pytest.raises(ValueError):
            xf.TouschekStudy(line, **kw)

    def test_accepts_deprecated_n_simulated_alias(self, toy_ring):
        line = toy_ring['line']
        with pytest.warns(FutureWarning, match='n_simulated'):
            study = xf.TouschekStudy(
                line,
                local_momentum_acceptance=_fresh_lma(toy_ring),
                sigma_z=SIGMA_Z, sigma_delta=SIGMA_DELTA,
                bunch_intensity=BUNCH_INTENSITY, n_simulated=10,
                nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y,
            )
        assert study.n_scattering_events == 10

    def test_raises_on_invalid_rng(self, toy_ring):
        line = toy_ring['line']
        with pytest.raises(ValueError, match='rng'):
            xf.TouschekStudy(
                line,
                local_momentum_acceptance=_fresh_lma(toy_ring),
                sigma_z=SIGMA_Z, sigma_delta=SIGMA_DELTA,
                bunch_intensity=BUNCH_INTENSITY, n_scattering_events=10,
                nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y,
                rng='invalid',
            )

    def test_raises_on_both_nemitt_and_gemitt(self, toy_ring):
        line = toy_ring['line']
        lma  = _fresh_lma(toy_ring)
        tw   = toy_ring['twiss']
        with pytest.raises(ValueError, match=r'not both'):
            _build_study(line, lma, tw, gemitt_x=1e-9, gemitt_y=1e-11)

    def test_raises_on_neither_nemitt_nor_gemitt(self, toy_ring):
        line = toy_ring['line']
        with pytest.raises(ValueError, match=r'must provide'):
            xf.TouschekStudy(
                line,
                local_momentum_acceptance=_fresh_lma(toy_ring),
                sigma_z=SIGMA_Z, sigma_delta=SIGMA_DELTA,
                bunch_intensity=BUNCH_INTENSITY, n_scattering_events=10,
            )

    def test_raises_on_wrong_lma_type(self, toy_ring):
        """Non-xt.Table local_momentum_acceptance must raise TypeError."""
        import pandas as pd
        line = toy_ring['line']
        bad  = pd.DataFrame({'name': ['a'], 's': [0.0],
                             'delta_neg': [-0.01], 'delta_pos': [0.01]})
        with pytest.raises(TypeError, match=r'xt\.Table'):
            xf.TouschekStudy(
                line,
                local_momentum_acceptance=bad,
                sigma_z=SIGMA_Z, sigma_delta=SIGMA_DELTA,
                bunch_intensity=BUNCH_INTENSITY, n_scattering_events=10,
                nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y,
            )

    def test_raises_on_missing_lma_columns(self, toy_ring):
        """LMA table missing required columns must raise ValueError."""
        line = toy_ring['line']
        lma  = toy_ring['lma']
        # Build an xt.Table without 'delta_pos'
        bad  = xt.Table({'name': lma.name, 's': lma.s, 'delta_neg': lma.delta_neg})
        with pytest.raises(ValueError, match=r'missing columns'):
            xf.TouschekStudy(
                line,
                local_momentum_acceptance=bad,
                sigma_z=SIGMA_Z, sigma_delta=SIGMA_DELTA,
                bunch_intensity=BUNCH_INTENSITY, n_scattering_events=10,
                nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y,
            )

    def test_raises_on_lma_nan_values(self, toy_ring):
        """LMA table with NaN values must raise ValueError."""
        line       = toy_ring['line']
        lma        = toy_ring['lma']
        delta_neg_bad = lma.delta_neg.copy().astype(float)
        delta_neg_bad[0] = np.nan
        bad = xt.Table({'name': lma.name, 's': lma.s,
                        'delta_neg': delta_neg_bad, 'delta_pos': lma.delta_pos})
        with pytest.raises(ValueError, match=r'NaN'):
            xf.TouschekStudy(
                line,
                local_momentum_acceptance=bad,
                sigma_z=SIGMA_Z, sigma_delta=SIGMA_DELTA,
                bunch_intensity=BUNCH_INTENSITY, n_scattering_events=10,
                nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y,
            )

    def test_raises_when_no_touschek_elements(self, toy_ring):
        """A line with no TouschekScattering must raise ValueError."""
        env2 = xt.Environment()
        bare = env2.new_line(components=[env2.new('d1', xt.Drift, length=1.0)])
        bare.set_particle_ref('electron', p0c=1e9)
        with pytest.raises(ValueError, match=r'TouschekScattering'):
            xf.TouschekStudy(
                bare,
                local_momentum_acceptance=toy_ring['lma'],
                sigma_z=SIGMA_Z, sigma_delta=SIGMA_DELTA,
                bunch_intensity=BUNCH_INTENSITY, n_scattering_events=10,
                nemitt_x=NEMITT_X, nemitt_y=NEMITT_Y,
            )


class TestTouschekStudyInitialise:
    """Tests for TouschekStudy.initialise_touschek()."""

    @pytest.fixture(autouse=True)
    def _setup(self, toy_ring):
        """Build a fresh study and initialise it."""
        self.line   = toy_ring['line']
        self.tw     = toy_ring['twiss']
        tab         = self.line.get_table()
        self.tnames = tab.rows.match(element_type='TouschekScattering').name
        lma         = _fresh_lma(toy_ring)
        self.study  = _build_study(self.line, lma, self.tw)
        self.study.initialise_touschek()

    def test_all_elements_are_configured(self):
        assert len(self.tnames) > 0
        for nn in self.tnames:
            assert isinstance(self.line[nn], xf.TouschekScattering)

    def test_p0c_positive_and_finite(self):
        for nn in self.tnames:
            el = self.line[nn]
            assert np.isfinite(el.p0c) and el.p0c > 0

    def test_integrated_piwinski_rate_nonnegative(self):
        for nn in self.tnames:
            el = self.line[nn]
            assert np.isfinite(el.integrated_piwinski_rate)
            assert el.integrated_piwinski_rate >= 0

    def test_local_piwinski_rate_nonnegative(self):
        for nn in self.tnames:
            el = self.line[nn]
            assert np.isfinite(el.piwinski_rate)
            assert el.piwinski_rate >= 0

    def test_lma_sign_convention(self):
        """delta_neg ≤ 0 and delta_pos ≥ 0 must hold at every element."""
        for nn in self.tnames:
            el = self.line[nn]
            assert el.delta_neg <= 0, f'{nn}: delta_neg={el.delta_neg} must be ≤ 0'
            assert el.delta_pos >= 0, f'{nn}: delta_pos={el.delta_pos} must be ≥ 0'

    def test_beam_params_stored_correctly(self):
        for nn in self.tnames:
            el = self.line[nn]
            assert el.bunch_intensity == pytest.approx(BUNCH_INTENSITY)
            assert el.sigma_z          == pytest.approx(SIGMA_Z)
            assert el.sigma_delta      == pytest.approx(SIGMA_DELTA)

    def test_nz_not_increased_by_nz_eff_logic(self):
        """The nz-clamp must never raise nz above the requested value of 3."""
        for nn in self.tnames:
            el = self.line[nn]
            assert el.nz <= 3.0 + 1e-12, \
                f'{nn}: nz={el.nz} exceeds requested nz=3.0'

    def test_twiss_populated_after_initialise(self):
        assert self.study.twiss is not None

    def test_partial_initialise_single_element(self):
        """
        Calling initialise_touschek(element=nn) should (re-)configure only
        that element.  Doubling bunch_intensity must increase the rate
        because the Piwinski rate scales as N_b^2.
        """
        nn       = self.tnames[0]
        el       = self.line[nn]
        old_rate = el.integrated_piwinski_rate
        self.study.bunch_intensity *= 2
        self.study.initialise_touschek(element=nn)
        assert el.integrated_piwinski_rate > old_rate

    def test_partial_initialise_raises_on_wrong_type(self):
        with pytest.raises(TypeError, match=r'string'):
            self.study.initialise_touschek(element=42)

    def test_partial_initialise_raises_on_missing_element(self):
        with pytest.raises(ValueError, match=r'not present'):
            self.study.initialise_touschek(element='not_in_the_line')

    def test_partial_initialise_raises_on_non_touschek_element(self):
        tab   = self.line.get_table()
        other = tab.rows.match(element_type='Drift').name[0]
        with pytest.raises(TypeError, match=r'not a TouschekScattering'):
            self.study.initialise_touschek(element=other)

    def test_run_without_particles_returns_scalar_result(self):
        result = self.study.run(track=False)

        assert result.element_names == self.study.elements
        assert result.rate_scattering > 0
        assert result.lifetime_scattering > 0
        assert result.rate_tracking is None
        assert result.lifetime_tracking is None
        assert result.particles_by_element is None
        assert result.particles is None
        assert result.lost_particles is None

        cols = set(result.local_rates._col_names)
        assert "integrated_piwinski_rate" in cols
        assert "total_mc_rate" not in cols
        assert "num_lost_particles" not in cols

    def test_xtrack_rng_generates_particles(self, toy_ring):
        line = toy_ring['line']
        tw = toy_ring['twiss']
        lma = _fresh_lma(toy_ring)
        study = _build_study(
            line, lma, tw, n_scattering_events=int(1e5), rng='xtrack')
        study.initialise_touschek()

        result = study.run(track=False, keep_particles=True)

        assert study.rng == 'xtrack'
        assert result.particles is not None
        assert len(result.particles.x) > 0
        for nn in study.elements:
            assert line[nn].rng == 'xtrack'


class TestTouschekScattering:
    """Tests for the TouschekScattering.scatter() method."""

    @pytest.fixture(autouse=True)
    def _setup(self, toy_ring):
        self.line = toy_ring['line']
        tw        = toy_ring['twiss']
        lma       = _fresh_lma(toy_ring)
        study     = _build_study(self.line, lma, tw)
        study.initialise_touschek()

        # Scatter from the first TouschekScattering element only (fast)
        tab        = self.line.get_table()
        tnames     = tab.rows.match(element_type='TouschekScattering').name
        self.nn    = tnames[0]
        self.el    = self.line[self.nn]
        self.parts = self.el.scatter()
        self.alive = self.parts.filter(self.parts.state == 1)

    def test_scatter_returns_particles(self):
        assert isinstance(self.parts, xt.Particles)

    def test_some_particles_selected(self):
        assert len(self.alive.x) > 0

    def test_coordinates_finite(self):
        for attr in ('x', 'px', 'y', 'py', 'zeta', 'delta'):
            vals = getattr(self.alive, attr)
            assert np.all(np.isfinite(vals)), f'Non-finite values in {attr}'

    def test_weights_finite_and_nonnegative(self):
        assert np.all(np.isfinite(self.alive.weight))
        assert np.all(self.alive.weight >= 0)

    def test_at_element_matches_line_index(self):
        expected = self.line.element_names.index(self.nn)
        assert int(self.alive.at_element[0]) == expected

    def test_scattered_delta_outside_lma(self):
        """
        The C kernel selects only particles whose delta lies outside the LMA;
        every returned particle must satisfy delta < delta_neg or delta > delta_pos.
        """
        d       = self.alive.delta
        outside = (d < self.el.delta_neg) | (d > self.el.delta_pos)
        assert np.all(outside), 'Some scattered particles have delta inside the LMA'

    def test_total_mc_rate_recorded(self):
        assert np.isfinite(self.el.total_mc_rate)
        assert self.el.total_mc_rate >= 0

    def test_ignored_rate_fraction(self):
        """ignored_rate stores the rate not represented by retained particles."""
        assert self.el.ignored_rate == pytest.approx(
            (1 - self.el.weight_retention_fraction) * self.el.total_mc_rate,
            rel=1e-9)

    def test_theta_log_populated(self):
        """theta_log must map particle IDs → scattering angles in (0, π)."""
        assert isinstance(self.el.theta_log, dict)
        assert len(self.el.theta_log) > 0
        for pid, theta in self.el.theta_log.items():
            assert np.isfinite(theta)
            assert 0 < theta < np.pi, f'theta={theta} out of (0, π)'

    def test_weight_sum_is_finite_and_positive(self):
        assert np.isfinite(self.alive.weight.sum())
        assert self.alive.weight.sum() > 0


N_EVENTS_RETENTION = int(2e5)


def _generate_by_element(toy_ring, weight_retention_fraction):
    """
    Generate (no tracking) at every TouschekScattering element of the toy
    ring with a fixed seed; return, per element, the coordinates and weights
    of the returned particles and the integrated Piwinski rate.
    """
    line  = toy_ring['line']
    study = _build_study(line, _fresh_lma(toy_ring), toy_ring['twiss'],
                         n_scattering_events=N_EVENTS_RETENTION,
                         weight_retention_fraction=weight_retention_fraction)
    study.initialise_touschek()
    result = study.run(track=False, generate_particles=True,
                       keep_particles=True)
    out = {}
    for nn in study.elements:
        part  = result.particles_by_element[nn]
        alive = part.state > 0
        coords = np.stack([part.x[alive], part.px[alive], part.y[alive],
                           part.py[alive], part.zeta[alive],
                           part.delta[alive]], axis=1)
        out[nn] = dict(
            keys=[tuple(cc) for cc in coords],
            weight=part.weight[alive].copy(),
            rate=line[nn].integrated_piwinski_rate)
    return out


@pytest.fixture(scope='module')
def full_sample(toy_ring):
    """All the scattered particles: pickPart is skipped."""
    return _generate_by_element(toy_ring, 1.0)


class TestTouschekWeightRetention:
    """
    With weight_retention_fraction < 1, pickPart keeps the highest-weight
    particles. Every kept particle must carry its own weight, i.e. the weight
    it has in the full sample (weight_retention_fraction = 1, where pickPart
    is skipped), generated with the same seed.
    """

    @pytest.mark.parametrize('weight_retention_fraction', [0.99, 0.999])
    def test_kept_particles_carry_their_own_weight(
            self, toy_ring, full_sample, weight_retention_fraction):
        kept_sample = _generate_by_element(toy_ring, weight_retention_fraction)
        for nn, kept in kept_sample.items():
            full = full_sample[nn]
            assert len(full['weight']) == N_EVENTS_RETENTION
            assert 0 < len(kept['weight']) < N_EVENTS_RETENTION

            # Same seed: the kept particles are a subset of the full sample
            weight_of = dict(zip(full['keys'], full['weight']))
            missing = [kk for kk in kept['keys'] if kk not in weight_of]
            assert not missing, f'{nn}: kept particles not in the full sample'
            expected = np.array([weight_of[kk] for kk in kept['keys']])

            np.testing.assert_allclose(
                kept['weight'], expected, rtol=1e-12, atol=0,
                err_msg=f'{nn}: kept particles carry the weight of others')

            # The full sample is normalised to the section rate; the kept
            # particles carry the retained fraction of it
            assert full['weight'].sum() == pytest.approx(full['rate'], rel=1e-9)
            assert kept['weight'].sum() == pytest.approx(
                weight_retention_fraction * kept['rate'], rel=2e-3)


class TestTouschekLocalRates:
    """Per-element diagnostics of TouschekStudy.local_rates."""

    def test_counts_only_generated_particles(self, toy_ring):
        """
        The particle buffer returned by scatter() has twice the capacity
        needed: its unused slots must not enter num_particles and sum_weight.
        """
        line  = toy_ring['line']
        study = _build_study(line, _fresh_lma(toy_ring), toy_ring['twiss'],
                             n_scattering_events=int(1e5))
        study.initialise_touschek()
        result = study.run(track=False, keep_particles=True)
        lr     = result.local_rates
        for ii, nn in enumerate(study.elements):
            part  = result.particles_by_element[nn]
            alive = part.state > 0
            assert lr['num_particles'][ii] == np.sum(alive)
            assert lr['sum_weight'][ii] == pytest.approx(
                np.sum(part.weight[alive]), rel=1e-12)
            assert lr['sum_weight'][ii] == pytest.approx(
                study.weight_retention_fraction
                * lr['integrated_piwinski_rate'][ii], rel=2e-3)


class TestTouschekLossStates:
    """
    Every particle with state <= 0 is lost: on an aperture (0) or in other
    ways, e.g. absorbed in an xcoll collimator (negative states).
    """

    def test_all_loss_states_enter_the_loss_rate(self, toy_ring, monkeypatch):
        line  = toy_ring['line']
        study = _build_study(line, _fresh_lma(toy_ring), toy_ring['twiss'],
                             n_scattering_events=int(2e4))
        study.initialise_touschek()

        def fake_track(particles, **kwargs):
            # Stand-in for tracking: a third lost on an aperture, a third
            # absorbed in a collimator (xcoll LOST_ON_MATERIAL = -330)
            alive = np.flatnonzero(particles.state > 0)
            particles.state[alive[0::3]] = 0
            particles.state[alive[1::3]] = -330

        monkeypatch.setattr(line, 'track', fake_track)
        result = study.run(track=True, n_turns=1, keep_particles=True)

        lost_weight, n_lost = 0.0, 0
        for nn in study.elements:
            part = result.particles_by_element[nn]
            lost = (part.state <= 0) & (part.state > xt.particles.LAST_INVALID_STATE)
            assert np.any(part.state[lost] == 0)
            assert np.any(part.state[lost] == -330)
            lost_weight += np.sum(part.weight[lost])
            n_lost += np.sum(lost)

        assert result.rate_tracking == pytest.approx(lost_weight, rel=1e-12)
        assert len(result.lost_particles.x) == n_lost
        lr = result.local_rates
        assert np.sum(lr['num_lost_particles']) == n_lost
        assert np.sum(lr['sum_lost_weight']) == pytest.approx(lost_weight, rel=1e-12)


class TestTouschekSections:
    """Lattice sections represented by the TouschekScattering elements."""

    @pytest.mark.parametrize('rename', [
        lambda nn, ii: f'ts.{ii + 1}',              # numbered from 1
        lambda nn, ii: f'ts_{chr(ord("a") + ii)}',  # no digits
    ], ids=['numbered_from_1', 'no_digits'])
    def test_single_element_matches_whole_ring(self, toy_ring, rename):
        """
        initialise_touschek(element) must integrate the section preceding
        `element` whatever its name.
        """
        line, tw = _toy_ring_variant(toy_ring, rename=rename)
        study = _build_study(line, _fresh_lma(toy_ring), tw)
        study.initialise_touschek()
        whole_ring = {nn: line[nn].integrated_piwinski_rate
                      for nn in study.elements}

        for nn in study.elements:
            line[nn].integrated_piwinski_rate = 0.0
            single = _build_study(line, _fresh_lma(toy_ring), tw)
            single.initialise_touschek(element=nn)
            assert line[nn].integrated_piwinski_rate > 0
            assert line[nn].integrated_piwinski_rate == pytest.approx(
                whole_ring[nn], rel=1e-12), nn

    def test_last_element_must_be_at_the_end_of_the_line(self, toy_ring):
        """
        Without a TouschekScattering at the end of the line, the part of the
        line after the last one would not be accounted for in any section:
        this is refused, whole ring or single element.
        """
        line     = toy_ring['line']
        tab      = line.get_table()
        ts_names = list(tab.rows.match(element_type='TouschekScattering').name)
        assert tab['s', ts_names[-1]] == pytest.approx(tab.s[-1])

        line2, tw2 = _toy_ring_variant(toy_ring, drop=(ts_names[-1],))
        study2 = _build_study(line2, _fresh_lma(toy_ring), tw2)
        with pytest.raises(ValueError, match=r'must sit at the end of the line'):
            study2.initialise_touschek()
        with pytest.raises(ValueError, match=r'must sit at the end of the line'):
            study2.initialise_touschek(element=ts_names[0])

    def test_first_section_starts_at_s_zero(self, toy_ring):
        """
        The section of the first element goes from s = 0 to it (an empty one
        if it sits at s = 0, as the first TSCATTER of ELEGANT: see
        test_touschek_elegant_reference.py). Compare with an independent
        trapezoid of the local rate over the rows of the line up to it.
        """
        line     = toy_ring['line']
        tab      = line.get_table()
        ts_names = list(tab.rows.match(element_type='TouschekScattering').name)
        first    = ts_names[0]
        assert 0 < tab['s', first] < tab.s[-1]

        study = _build_study(line, _fresh_lma(toy_ring), toy_ring['twiss'])
        study.initialise_touschek()

        names = list(tab.name[:list(tab.name).index(first) + 1])
        s     = np.array([tab['s', nn] for nn in names])
        assert s[0] == 0.0
        rates = np.array([study._compute_piwinski_scattering_rate(nn)
                          for nn in names])
        expected = np.trapezoid(rates, s) / toy_ring['twiss'].line_length
        assert expected > 0
        assert line[first].integrated_piwinski_rate == pytest.approx(
            expected, rel=1e-12)


class TestTouschekNoScatteringEvents:
    """n_scattering_events = 0 is a valid setting, but cannot generate."""

    def test_scatter_with_zero_events_raises(self, toy_ring):
        line  = toy_ring['line']
        study = _build_study(line, _fresh_lma(toy_ring), toy_ring['twiss'],
                             n_scattering_events=0)
        study.initialise_touschek()
        # Rates only: nothing to generate, no error
        assert study.run(track=False).rate_scattering > 0
        # Generation: a clear error (it used to kill the process with a
        # division by zero in the C kernel)
        with pytest.raises(ValueError, match='n_scattering_events'):
            study.run(track=False, generate_particles=True)


class TestPiwinskiIntegral:
    """
    Unit tests for the Piwinski integral helper.
    """

    def test_integral_positive(self):
        val = xf.TouschekStudy._compute_piwinski_integral(
            0.01, B1=5.0, B2=3.0)
        assert val > 0

    def test_integral_decreases_with_larger_tm(self):
        """A larger momentum cut-off (larger tm) must give a smaller integral."""
        B1, B2 = 5.0, 3.0
        assert (
            xf.TouschekStudy._compute_piwinski_integral(0.001, B1, B2)
            > xf.TouschekStudy._compute_piwinski_integral(0.10, B1, B2)
        )

    def test_integral_decreases_with_larger_B1(self):
        """Larger B1 (tighter beam) must suppress the integral."""
        tm, B2 = 0.01, 2.0
        assert (
            xf.TouschekStudy._compute_piwinski_integral(tm, B1=3.0, B2=B2)
            > xf.TouschekStudy._compute_piwinski_integral(
                tm, B1=10.0, B2=B2)
        )

    def test_b2_is_the_square_root(self):
        assert xf.TouschekStudy._compute_piwinski_b2(5.0, 9.0) == pytest.approx(3.0)
        assert xf.TouschekStudy._compute_piwinski_b2(5.0, 0.0) == 0.0

    def test_b2_rounding_below_zero_gives_zero(self):
        """B2^2 slightly negative from rounding must give 0, not NaN."""
        B1 = 5.0
        assert xf.TouschekStudy._compute_piwinski_b2(B1, -1e-12 * B1**2) == 0.0

    def test_b2_negative_raises(self):
        with pytest.raises(ValueError, match='B2'):
            xf.TouschekStudy._compute_piwinski_b2(5.0, -1.0)

    def test_integral_finite_for_large_B2_t(self):
        """The asymptotic I0 branch (B2*t > 500) must return a finite positive value."""
        val = xf.TouschekStudy._compute_piwinski_integral(
            0.001, B1=600.0, B2=599.0)
        assert np.isfinite(val) and val > 0


class TestEndToEndLifetime:
    """
    Minimal end-to-end smoke test: scatter, track, merge, lifetime.
    Verifies that the result is a physically plausible positive finite number.
    """

    @for_all_test_contexts(excluding=('ContextCupy', 'ContextPyopencl'))
    @allow_kernel_compilation
    def test_lifetime_positive_finite(self, toy_ring, test_context):
        line = toy_ring['line']
        tw   = toy_ring['twiss']
        lma  = _fresh_lma(toy_ring)

        study = _build_study(line, lma, tw, n_scattering_events=int(2e5))
        study.initialise_touschek()
        _build_tracker_or_skip(line, test_context)
        result = study.run(track=True, n_turns=128)

        rate_tracking = result.rate_tracking

        assert rate_tracking > 0, 'No particles were lost — something is wrong'

        assert np.isfinite(result.lifetime_tracking)
        assert result.lifetime_tracking > 0
        assert np.isfinite(result.lifetime_scattering)
        assert result.lifetime_scattering > 0
        assert result.element_names == study.elements
        assert result.particles_by_element is None
        assert result.particles is None
        assert result.lost_particles is None
