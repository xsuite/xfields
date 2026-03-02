# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2025.                   #
# ########################################### #
import numpy as np
import pytest

import xobjects as xo
import xtrack as xt
import xfields as xf

# --- Helpers -----------------------------------------------------------------
def _build_toy_ring_with_touschek_and_apertures(
    *,
    aper_size=0.040,
    install_apertures=True,
):
    lbend = 3.0
    angle = np.pi / 2

    lquad = 0.3
    k1qf = 0.1
    k1qd = 0.7

    env = xt.Environment()
    line = env.new_line(components=[
        env.new('mqf.1', xt.Quadrupole, length=lquad, k1=k1qf),
        env.new('d1.1',  xt.Drift, length=1.0),
        env.new('mb1.1', xt.Bend, length=lbend, angle=angle),
        env.new('d2.1',  xt.Drift, length=1.0),

        env.new('mqd.1', xt.Quadrupole, length=lquad, k1=-k1qd),
        env.new('d3.1',  xt.Drift, length=1.0),
        env.new('mb2.1', xt.Bend, length=lbend, angle=angle),
        env.new('d4.1',  xt.Drift, length=1.0),

        env.new('mqf.2', xt.Quadrupole, length=lquad, k1=k1qf),
        env.new('d1.2',  xt.Drift, length=1.0),
        env.new('mb1.2', xt.Bend, length=lbend, angle=angle),
        env.new('d2.2',  xt.Drift, length=1.0),

        env.new('mqd.2', xt.Quadrupole, length=lquad, k1=-k1qd),
        env.new('d3.2',  xt.Drift, length=1.0),
        env.new('mb2.2', xt.Bend, length=lbend, angle=angle),
        env.new('d4.2',  xt.Drift, length=1.0),
    ])

    line.set_particle_ref('electron', p0c=1e9)
    line.configure_bend_model(core='full', edge=None)

    # Insert Touschek scattering centers (one per magnet, plus one at end)
    tab0 = line.get_table()
    tab_magnets = tab0.rows[(tab0.element_type == 'Bend') | (tab0.element_type == 'Quadrupole')]

    for ii, nn in enumerate(tab_magnets.name):
        name = f'TScatter_{ii}'
        env.elements[name] = xf.TouschekScattering()
        line.insert(name, at=0.0, from_=nn)

    # Last TouschekScattering at end of the line
    name_last = f'TScatter_{ii + 1}'
    env.elements[name_last] = xf.TouschekScattering()
    line.insert(name_last, at=float(line.get_length()))

    # Install rectangular apertures
    if install_apertures:
        tab = line.get_table()
        needs_aperture = np.unique(tab.element_type)[
            ~np.isin(np.unique(tab.element_type), ["", "Drift", "Marker"])
        ]

        placements = []
        for nn, ee in zip(tab.name, tab.element_type):
            if ee not in needs_aperture:
                continue

            env.new(
                f'{nn}_aper_entry', xt.LimitRect,
                min_x=-aper_size, max_x=aper_size,
                min_y=-aper_size, max_y=aper_size
            )
            placements.append(env.place(f'{nn}_aper_entry', at=f'{nn}@start'))

            env.new(
                f'{nn}_aper_exit', xt.LimitRect,
                min_x=-aper_size, max_x=aper_size,
                min_y=-aper_size, max_y=aper_size
            )
            placements.append(env.place(f'{nn}_aper_exit', at=f'{nn}@end'))

        line.insert(placements)

    return env, line


def _compute_fast_lma_at_touschek_centers(line, *, nemitt_x, nemitt_y):
    # Fast-ish settings for tests
    return line.momentum_aperture(
        include_type_pattern="TouschekScattering",
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        y_offset=1e-12,
        delta_negative_limit=-0.012,
        delta_positive_limit=+0.012,
        delta_step_size=0.001,
        n_turns=64,
        method="4d",
        with_progress=False,
        verbose=False,
    ).to_pandas()


# --- Tests -------------------------------------------------------------------
def test_touschek_manager_initialise_configures_elements():
    """
    Check that TouschekManager.initialise_touschek() configures all TouschekScattering
    elements with consistent optics, momentum aperture, and positive rates.
    """
    nemitt_x = 1e-5
    nemitt_y = 1e-7
    sigma_z = 4e-3
    sigma_delta = 1e-3
    bunch_population = 4e9

    _, line = _build_toy_ring_with_touschek_and_apertures()

    # LMA at Touschek centers (DataFrame-like as expected by TouschekManager)
    df_ma = _compute_fast_lma_at_touschek_centers(line, nemitt_x=nemitt_x, nemitt_y=nemitt_y)

    # Keep runtime reasonable
    tm = xf.TouschekManager(
        line,
        momentum_aperture=df_ma,
        momentum_aperture_scale=0.85,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        sigma_delta=sigma_delta,
        bunch_population=bunch_population,
        n_simulated=1e6,
        nx=3, ny=3, nz=3,
        ignored_portion=0.01,
        seed=1997,
        method="4d",
    )

    tm.initialise_touschek()

    tab = line.get_table()
    tnames = tab.rows[tab.element_type == "TouschekScattering"].name
    assert len(tnames) > 0

    for nn in tnames:
        el = line[nn]
        assert isinstance(el, xf.TouschekScattering)

        # Config fields should be set
        assert np.isfinite(el._p0c) and el._p0c > 0
        assert np.isfinite(el._integrated_piwinski_rate) and el._integrated_piwinski_rate >= 0
        assert np.isfinite(el.piwinski_rate) and el.piwinski_rate >= 0

        # Acceptance should have correct sign convention
        assert el._deltaN <= 0
        assert el._deltaP >= 0

        # Consistency
        assert el._bunch_population == pytest.approx(bunch_population)
        assert el._sigma_z == pytest.approx(sigma_z)
        assert el._sigma_delta == pytest.approx(sigma_delta)

        # nz may be reduced by nz_eff logic, but never increased
        assert el._nz <= 3.0 + 1e-15


def test_touschek_scattering_scatter_returns_particles():
    """
    Run scatter() on one TouschekScattering element and check that:
      - returned object is xt.Particles,
      - some particles are selected,
      - weights and deltas are finite,
      - at_element points to the TouschekScattering location.
    """
    nemitt_x = 1e-5
    nemitt_y = 1e-7
    sigma_z = 4e-3
    sigma_delta = 1e-3
    bunch_population = 4e9

    _, line = _build_toy_ring_with_touschek_and_apertures()

    df_ma = _compute_fast_lma_at_touschek_centers(line, nemitt_x=nemitt_x, nemitt_y=nemitt_y)

    tm = xf.TouschekManager(
        line,
        momentum_aperture=df_ma,
        momentum_aperture_scale=0.85,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        sigma_delta=sigma_delta,
        bunch_population=bunch_population,
        n_simulated=1e6,
        nx=3, ny=3, nz=3,
        ignored_portion=0.01,
        seed=1997,
        method="4d",
    )
    tm.initialise_touschek()

    tab = line.get_table()
    tnames = tab.rows[tab.element_type == "TouschekScattering"].name
    nn = tnames[0]
    el = line[nn]

    parts = el.scatter()
    assert isinstance(parts, xt.Particles)

    # "Selected" particles are those with state==1 immediately after scatter
    alive = parts.filter(parts.state == 1)
    assert alive._capacity > 0
    assert len(alive.x) > 0

    # Sanity of generated coordinates and weights
    assert np.all(np.isfinite(alive.x))
    assert np.all(np.isfinite(alive.px))
    assert np.all(np.isfinite(alive.y))
    assert np.all(np.isfinite(alive.py))
    assert np.all(np.isfinite(alive.zeta))
    assert np.all(np.isfinite(alive.delta))
    assert np.all(np.isfinite(alive.weight))
    assert np.all(alive.weight >= 0)

    # Particles should be located at the element index
    expected_idx = line.element_names.index(nn)
    assert int(alive.at_element[0]) == expected_idx

    # Element should have recorded MC totals
    assert np.isfinite(el.total_mc_rate)
    assert el.total_mc_rate >= 0


def test_touschek_manager_validates_momentum_aperture_columns():
    """
    TouschekManager should reject malformed momentum_aperture objects.
    """
    _, line = _build_toy_ring_with_touschek_and_apertures()

    # Minimal fake DataFrame-like with missing columns
    import pandas as pd
    bad = pd.DataFrame({"s": [0.0], "deltan": [-0.01]})  # missing 'deltap'

    with pytest.raises(ValueError, match=r"missing columns"):
        xf.TouschekManager(
            line,
            momentum_aperture=bad,
            sigma_z=4e-3,
            sigma_delta=1e-3,
            bunch_population=1.0,
            n_simulated=10,
            nemitt_x=1e-6,
            nemitt_y=1e-6,
            method="4d",
        )
