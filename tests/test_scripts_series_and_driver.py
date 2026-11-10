"""Tests for two driver scripts: capture-time series split and run_watched outcome handling."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


series = load('series_split_from_poses')
rw = load('run_watched')


# ------------------------------------------------------------- capture-time split

def gaps_for(sizes, between=100.0, within=15.0):
    """Per-photo gaps for consecutive series of the given sizes."""
    out = []
    for j, n in enumerate(sizes):
        out += [within] * (n - 1)
        if j < len(sizes) - 1:
            out.append(between)
    return out


def test_split_finds_series_boundaries():
    cuts, acc = series.capture_time_cuts(gaps_for([57, 50, 38], between=110), 3)
    assert cuts == [57, 107]
    assert sorted(c for _, c in acc) == [57, 107]


def test_split_ignores_gap_after_first_photo():
    # The PAHMA 5-613 case: a 110 s pause after the first (test) photo is larger than
    # the real 98 s boundary; it must not create a one-photo series.
    g = gaps_for([57, 50, 38])
    g[0] = 110.0          # after photo 0
    g[56] = 98.0          # real boundary after photo 56
    g[106] = 127.0        # real boundary after photo 106
    cuts, acc = series.capture_time_cuts(g, 3)
    assert cuts == [57, 107]
    assert [s for s, _ in acc] == [127.0, 98.0]


def test_split_ignores_gap_before_last_photo():
    g = gaps_for([20, 20])
    g[-1] = 500.0         # pause before the final photo
    cuts, _ = series.capture_time_cuts(g, 2)
    assert cuts == [20]


def test_split_min_series_size_is_configurable():
    g = gaps_for([2, 18])
    assert series.capture_time_cuts(g, 2, min_series=2)[0] == [2]
    with pytest.raises(ValueError):
        series.capture_time_cuts([10.0, 10.0, 10.0], 3, min_series=3)


# ------------------------------------------------------------- run_watched outcomes

def res(status='complete', stop_reason=None):
    return {'status': status, 'stop_reason': stop_reason, 'elapsed_seconds': 1.0}


def run(status, message=None):
    return {'status': status, 'failures': [{'message': message}] if message else []}


@pytest.mark.parametrize('watchdog, metashape, expected', [
    (res(), run('complete'), 'ok'),
    (None, run('complete'), 'ok_unrecorded'),
    (None, run('failed', 'boom'), 'watchdog_vanished'),
    (res('stopped_or_failed', 'swap growth 7000 MiB'), run('failed', 'x'), 'watchdog_stop'),
    (res('watchdog_interrupted'), run('failed', 'x'), 'watchdog_stop'),
    (res('stopped_or_failed'), run('failed', 'Metashape license unavailable or checkout denied'), 'license'),
    (res('stopped_or_failed'), run('failed', 'alignment failed configured camera/tie-point criteria'),
     'alignment_failed'),
    (res('stopped_or_failed'), run('failed', 'RuntimeError: disk full'), 'failed'),
])
def test_classify(watchdog, metashape, expected):
    assert rw.classify(watchdog, metashape) == expected


def test_watchdog_stop_wins_over_alignment_message():
    r = run('failed', 'alignment failed configured camera/tie-point criteria')
    assert rw.classify(res('stopped_or_failed', 'timeout'), r) == 'watchdog_stop'


def driver(tmp_path, monkeypatch, outcomes):
    """Run run_watched.main with attempt() and the license check replaced by fakes."""
    calls = []

    def fake_attempt(manifest, out, a):
        calls.append(out.name)
        return outcomes[out.name]

    monkeypatch.setattr(rw, 'attempt', fake_attempt)
    monkeypatch.setattr(rw, 'license_preflight', lambda path: True)
    monkeypatch.setattr(rw.time, 'sleep', lambda s: None)
    manifests = []
    for name in outcomes:
        if '-retry' not in name:
            m = tmp_path / f'{name}.json'
            m.write_text(json.dumps({}))
            manifests.append(str(m))
    code = 0
    try:
        rw.main(['--manifests', *manifests, '--root', str(tmp_path), '--output-dir', str(tmp_path / 'out')])
    except SystemExit as e:
        code = e.code
    return calls, code


def test_alignment_failure_is_recorded_and_driver_continues(tmp_path, monkeypatch):
    calls, code = driver(tmp_path, monkeypatch, {'a': 'ok', 'b': 'alignment_failed', 'c': 'ok'})
    assert calls == ['a', 'b', 'c']
    assert code == 0


def test_watchdog_stop_still_stops_the_driver(tmp_path, monkeypatch):
    calls, code = driver(tmp_path, monkeypatch, {'a': 'watchdog_stop', 'b': 'ok'})
    assert calls == ['a']
    assert code == 2


def test_other_failure_still_stops_the_driver(tmp_path, monkeypatch):
    calls, code = driver(tmp_path, monkeypatch, {'a': 'failed', 'b': 'ok'})
    assert calls == ['a']
    assert code == 2
