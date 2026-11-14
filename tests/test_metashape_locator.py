import json
import os
import sys
from pathlib import Path

from bd3d_quality import metashape_locator as ml

FAKE = Path(__file__).parent / 'fake_metashape'


def make_python(path, metashape=True, sleep=0):
    """An executable 'python': this interpreter, with or without the fake Metashape module."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = f'export PYTHONPATH="{FAKE}"\n' if metashape else 'export PYTHONPATH=""\n'
    pause = f'sleep {sleep}\n' if sleep else ''
    path.write_text(f'#!/bin/sh\n{extra}{pause}exec {sys.executable} "$@"\n')
    path.chmod(0o755)
    return str(path)


def write_log(path, python):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"command: [{python!r}, '-B', '-m', 'bd3d_quality.resource_monitor']\ncomplete\n")
    return path


def layout(tmp_path):
    repo, home, apps = tmp_path / 'repo', tmp_path / 'home', tmp_path / 'Applications'
    for d in (repo, home, apps):
        d.mkdir()
    return repo, home, apps


def test_candidate_order_and_dedupe(tmp_path):
    repo, home, apps = layout(tmp_path)
    (repo / '.env').write_text('# comment\nexport BD3D_METASHAPE_PYTHON="/env/file/python"\nOTHER=1\n')
    ml.save_choice('/saved/python', repo / '.coverage-app.json')
    old = write_log(repo / 'results/a/run1-watchdog.log', '/old/log/python')
    os.utime(old, (1, 1))
    write_log(repo / 'runs/x/new-watchdog.log', '/new/log/python')
    write_log(repo / 'results/gpu-pilot-d8/ref50-watchdog.log', '/ref50/python')
    write_log(repo / 'results/b/dup-watchdog.log', '/saved/python')          # duplicate of the saved one
    (repo / 'results/b/broken-watchdog.log').write_text('not a command line\n')
    venv = make_python(home / '.venvs/ms/bin/python')
    repo_venv = make_python(repo / '.venv/bin/python')
    bundled = apps / 'MetashapePro.app/Contents/Frameworks/Python.framework/Versions/3.12/bin'
    make_python(bundled / 'python3.12')
    make_python(bundled / 'python3.12-config')                               # not an interpreter
    got = ml.candidates(env={'BD3D_METASHAPE_PYTHON': '/env/var/python'}, repo=repo, home=home,
                        applications=apps, settings=repo / '.coverage-app.json')
    assert [p for _, p in got] == ['/env/var/python', '/env/file/python', '/saved/python', '/ref50/python',
                                   '/new/log/python', '/old/log/python', venv, repo_venv,
                                   str(bundled / 'python3.12')]
    sources = dict((p, s) for s, p in got)
    assert sources['/env/file/python'] == 'BD3D_METASHAPE_PYTHON (.env)'
    assert sources['/ref50/python'] == 'watchdog log results/gpu-pilot-d8/ref50-watchdog.log'


def test_first_working_candidate_wins(tmp_path):
    good = make_python(tmp_path / 'good/python')
    also_good = make_python(tmp_path / 'later/python')
    plain = make_python(tmp_path / 'plain/python', metashape=False)
    tried = []
    search = ml.find_metashape_python(
        [('env', str(tmp_path / 'missing/python')), ('saved', plain), ('log', good), ('venv', also_good)],
        on_try=lambda *a: tried.append(a))
    assert search.found == ml.Found(good, '2.2.1', 'log')
    assert [t[2] for t in tried] == [False, False, True]                      # stops at the first success
    reasons = {p: r for _, p, r in search.tried}
    assert reasons[str(tmp_path / 'missing/python')] == 'not found'
    assert 'Metashape' in reasons[plain]                                     # ModuleNotFoundError line


def test_none_found_reports_every_reason(tmp_path):
    plain = make_python(tmp_path / 'plain/python', metashape=False)
    noexec = tmp_path / 'noexec/python'
    noexec.parent.mkdir()
    noexec.write_text('')
    search = ml.find_metashape_python([('a', plain), ('b', str(noexec))])
    assert search.found is None and [r for _, _, r in search.tried][1] == 'not executable'


def test_probe_times_out(tmp_path):
    slow = make_python(tmp_path / 'slow/python', sleep=5)
    assert ml.probe(slow, timeout=0.5) == (False, 'timed out after 0.5 s')


def test_settings_file_round_trip_keeps_other_keys(tmp_path):
    f = tmp_path / 's.json'
    assert ml.load_saved(f) is None
    f.write_text(json.dumps({'other': 1}))
    ml.save_choice('/x/python', f)
    assert json.loads(f.read_text()) == {'other': 1, 'metashape_python': '/x/python'}
    assert ml.load_saved(f) == '/x/python'
    f.write_text('not json')
    assert ml.load_saved(f) is None


def test_settings_file_is_gitignored():
    import subprocess
    r = subprocess.run(['git', 'check-ignore', '-q', str(ml.SETTINGS_FILE)], cwd=ml.REPO)
    assert r.returncode == 0
