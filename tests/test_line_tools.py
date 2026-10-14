# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import xtrack as xt


def _line():
    env = xt.Environment()
    line = env.new_line(components=[env.new('d1', xt.Drift, length=1.0)])
    line.set_particle_ref('electron', p0c=1e9)
    return line


def test_spacecharge_install_frozen_facade(monkeypatch):
    line = _line()
    calls = {}

    def fake_install_spacecharge_frozen(**kwargs):
        calls.update(kwargs)
        return 'installed'

    monkeypatch.setattr(
        'xfields.config_tools.install_spacecharge_frozen',
        fake_install_spacecharge_frozen)

    result = line.xfields.spacecharge_install_frozen(
        num_spacecharge_interactions=3)

    assert result == 'installed'
    assert calls['line'] is line
    assert calls['num_spacecharge_interactions'] == 3


def test_spacecharge_replace_with_quasi_frozen_facade(monkeypatch):
    line = _line()
    calls = {}

    def fake_replace_spacecharge_with_quasi_frozen(**kwargs):
        calls.update(kwargs)
        return ['sc0']

    monkeypatch.setattr(
        'xfields.config_tools.replace_spacecharge_with_quasi_frozen',
        fake_replace_spacecharge_with_quasi_frozen)

    result = line.xfields.spacecharge_replace_with_quasi_frozen(
        update_mean_x_on_track=False)

    assert result == ['sc0']
    assert calls['line'] is line
    assert calls['update_mean_x_on_track'] is False


def test_spacecharge_replace_with_pic_facade(monkeypatch):
    line = _line()
    calls = {}

    def fake_replace_spacecharge_with_PIC(**kwargs):
        calls.update(kwargs)
        return 'pic_collection', ['pic0']

    monkeypatch.setattr(
        'xfields.config_tools.replace_spacecharge_with_PIC',
        fake_replace_spacecharge_with_PIC)

    result = line.xfields.spacecharge_replace_with_pic(nx_grid=8)

    assert result == ('pic_collection', ['pic0'])
    assert calls['line'] is line
    assert calls['nx_grid'] == 8


def test_ibs_configure_facade(monkeypatch):
    line = _line()
    calls = {}

    def fake_configure_intrabeam_scattering(line_arg, **kwargs):
        calls['line'] = line_arg
        calls.update(kwargs)
        return 'configured'

    monkeypatch.setattr(
        'xfields.ibs.configure_intrabeam_scattering',
        fake_configure_intrabeam_scattering)

    result = line.xfields.ibs_configure(
        element='ibs_kick', name='ibskick', at=0, update_every=10)

    assert result == 'configured'
    assert calls['line'] is line
    assert calls['element'] == 'ibs_kick'
    assert calls['name'] == 'ibskick'
    assert calls['at'] == 0
    assert calls['update_every'] == 10
