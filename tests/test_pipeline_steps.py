import json
import os
import stat
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from bd3d_quality import pipeline_steps as ps
from bd3d_quality.metashape_runner import run_reconstruction

FAKE = Path(__file__).parent / 'fake_metashape'


def fake_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location('fake_metashape_module', FAKE / 'Metashape.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_python(tmp_path, **env):
    """An 'interpreter that can import Metashape': this Python plus the fake module."""
    exports = ''.join(f'export {k}={v}\n' for k, v in env.items())
    p = tmp_path / 'metashape-python'
    p.write_text(f'#!/bin/sh\nexport PYTHONPATH="$PYTHONPATH:{FAKE}"\n{exports}exec {sys.executable} "$@"\n')
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


# ---------------------------------------------------------------- settings and manifests

def test_pilot_settings_match_reference_runs():
    ref = json.loads((ps.REPO / 'results/gpu-pilot-d8/ref50/manifest.json').read_text())['configuration']
    assert ps.pilot_config('mesh').to_dict() == ref
    assert 'Metal GPU' in ps.settings_text() and 'depth_downscale 8' in ps.settings_text()


def test_alignment_and_mesh_manifests_are_seed_compatible(dataset, tmp_path):
    root, _ = dataset
    api = fake_module()
    align = ps.alignment_manifest(root, 'dog')
    mesh = ps.mesh_manifest(align)
    assert align['configuration']['profile'] == 'alignment' and mesh['configuration']['profile'] == 'mesh'
    assert mesh['images'] == align['images'] and mesh['experiment_id'] != align['experiment_id']
    # the existing seed path accepts the pair: alignment-only run, then mesh seeded from it
    run_reconstruction(align, root, tmp_path / 'a', api=api)
    assert ps.is_alignment_checkpoint(tmp_path / 'a')
    state = run_reconstruction(mesh, root, tmp_path / 'm', api=api, seed_checkpoint=tmp_path / 'a')
    assert state['status'] == 'complete' and (tmp_path / 'm/raw-mesh.obj').exists()
    assert not ps.is_alignment_checkpoint(tmp_path / 'm')


def test_write_manifest_never_overwrites(tmp_path):
    ps.write_manifest(tmp_path / 'm/x.json', {'a': 1})
    with pytest.raises(FileExistsError):
        ps.write_manifest(tmp_path / 'm/x.json', {'a': 2})


def test_capture_names_are_unique(tmp_path):
    assert ps.capture_name('/data/ARF dog', tmp_path) == 'ARF-dog'
    (tmp_path / 'ARF-dog-align').mkdir()
    assert ps.capture_name('/data/ARF dog', tmp_path) == 'ARF-dog-2'


def test_watched_command_and_limits():
    cmd = ps.watched_command('/py', Path('/m.json'), Path('/photos'), Path('/out/r-mesh'), ps.MESH_LIMITS,
                             seed=Path('/out/r-align'))
    i = cmd.index('--')
    monitor, child = cmd[:i], cmd[i + 1:]
    assert monitor[:4] == ['/py', '-B', '-m', 'bd3d_quality.resource_monitor']
    for flag, value in [('--seconds', '7200'), ('--rss-gib', '6'), ('--swap-growth-mib', '6144')]:
        assert monitor[monitor.index(flag) + 1] == value
    assert {'--seconds-note', '--rss-note', '--limit-note'} <= set(monitor)
    assert child[-2:] == ['--seed-checkpoint', '/out/r-align'] and 'reconstruct' in child
    align = ps.ALIGN_LIMITS.args()
    assert align[:2] == ['--seconds', '1800'] and '--seconds-note' in align and '--rss-gib' not in align


# ---------------------------------------------------------------- paths

def test_inspect_folder_reasons(tmp_path):
    ok, p, why = ps.inspect_folder(tmp_path / 'nope' / 'run')
    assert not ok and 'does not exist' in why and str(tmp_path) in why
    f = tmp_path / 'file.txt'
    f.write_text('x')
    assert ps.inspect_folder(f)[2] == 'is a file, not a folder'
    assert ps.inspect_folder(tmp_path, need=['cameras.csv'])[2] == 'has no cameras.csv'
    (tmp_path / 'cameras.csv').write_text('label\n')
    assert ps.inspect_folder(tmp_path, need=['cameras.csv']) == (True, tmp_path.resolve(), 'ok')


@pytest.mark.skipif(os.geteuid() == 0, reason='root ignores permissions')
def test_inspect_folder_permission_denied_is_not_missing(tmp_path):
    locked = tmp_path / 'locked'
    (locked / 'run').mkdir(parents=True)
    locked.chmod(0)
    try:
        assert not (locked / 'run').exists()                 # what the old check saw
        ok, _, why = ps.inspect_folder(locked / 'run')
        assert not ok and 'permission denied' in why
    finally:
        locked.chmod(0o755)


# ---------------------------------------------------------------- status

def write_res(run, **res):
    run.with_name(run.name + '-resources.json').write_text(json.dumps(res))


def test_run_outcome(tmp_path):
    run = tmp_path / 'r'
    assert ps.run_outcome(run) == ('not started', '')
    run.mkdir()
    assert ps.run_outcome(run)[0] == 'running'
    (run / 'run.json').write_text(json.dumps({'status': 'complete'}))
    write_res(run, status='complete', elapsed_seconds=12.0)
    assert ps.run_outcome(run) == ('done', '12 s')
    write_res(run, status='watchdog_interrupted')
    assert ps.run_outcome(run)[0] == 'cancelled'
    (run / 'run.json').write_text(json.dumps({'status': 'failed', 'failures': [
        {'message': 'alignment failed configured camera/tie-point criteria'}]}))
    write_res(run, status='stopped_or_failed')
    assert ps.run_outcome(run) == ('failed', 'alignment failed the camera/tie-point criteria')
    write_res(run, status='stopped_or_failed', stop_reason='process_RSS_limit')
    assert ps.run_outcome(run) == ('failed', 'watchdog stop: process_RSS_limit')


# ---------------------------------------------------------------- scoring

def write_run(d, mesh, shift=0.0):
    d.mkdir()
    lines = [f'v {float(a) + shift} {b} {c}' if l.startswith('v ') else l
             for l in Path(mesh).read_text().splitlines() for _, a, b, c in [(l.split() + ['0'] * 4)[:4]]]
    (d / 'raw-mesh.obj').write_text('\n'.join(lines) + '\n')
    rows = ['label,aligned,x,y,z'] + [f'c{i},True,{x + shift},{y},{z}' for i, (x, y, z) in
                                      enumerate([(3, 0, 0), (0, 3, 0), (-3, 0, 1), (0, -3, 2)])]
    (d / 'cameras.csv').write_text('\n'.join(rows) + '\n')


def test_score_run_identical_meshes(tmp_path, tetra):
    write_run(tmp_path / 'ref', tetra)
    write_run(tmp_path / 'run', tetra, shift=5.0)            # same shape and cameras, moved frame
    row = ps.score_run(tmp_path / 'run', tmp_path / 'ref')
    assert row['label_v3'] < 1e-6 and row['camera_shared_images'] == 4
    assert 'missing RMSE 0.0000' in ps.score_text(row) and 'extra RMSE 0.0000' in ps.score_text(row)


def test_score_run_needs_both_meshes(tmp_path, tetra):
    write_run(tmp_path / 'ref', tetra)
    (tmp_path / 'run').mkdir()
    with pytest.raises(FileNotFoundError, match='raw-mesh.obj'):
        ps.score_run(tmp_path / 'run', tmp_path / 'ref')


# ---------------------------------------------------------------- watched process

def test_watched_process_streams_and_logs(tmp_path):
    run = tmp_path / 'r'
    lines = []
    script = ("import pathlib,sys,time\n"
              f"pathlib.Path({str(run)!r}+'-console.log').write_text('metashape says hi\\n')\n"
              "print('watchdog line', flush=True)\n")
    code = ps.WatchedProcess([sys.executable, '-c', script], run, lines.append, env=dict(os.environ)).run(poll=0.05)
    assert code == 0 and 'watchdog line' in lines and '  | metashape says hi' in lines
    assert 'watchdog line' in (tmp_path / 'r-watchdog.log').read_text()


def test_watched_process_cancel_stops_process_group(tmp_path):
    marker = tmp_path / 'grandchild.pid'
    script = ("import subprocess,sys,time\n"
              "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
              f"open({str(marker)!r}, 'w').write(str(c.pid))\n"
              "print('started', flush=True); time.sleep(60)\n")
    wp = ps.WatchedProcess([sys.executable, '-c', script], tmp_path / 'r', lambda s: None, env=dict(os.environ))
    t = threading.Thread(target=wp.run, kwargs={'poll': 0.05})
    t.start()
    for _ in range(100):
        if marker.exists() and marker.read_text():
            break
        time.sleep(0.05)
    wp.cancel()
    t.join(timeout=10)
    assert not t.is_alive() and wp.cancelled and wp.proc.returncode != 0
    pid = int(marker.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail('grandchild survived cancel')


# ---------------------------------------------------------------- end to end through the real watchdog

@pytest.mark.skipif(sys.platform != 'darwin', reason='resource_monitor samples macOS sysctl')
def test_align_then_mesh_through_watchdog(dataset, tmp_path):
    root, _ = dataset
    py = fake_python(tmp_path)
    assert ps.license_ok(py) == (True, 'activated')
    paths = ps.CapturePaths(tmp_path / 'out', 'dog')
    align = ps.alignment_manifest(root, 'dog')
    ps.write_manifest(paths.align_manifest, align)
    lines = []
    code = ps.WatchedProcess(ps.watched_command(py, paths.align_manifest, root, paths.align_run, ps.ALIGN_LIMITS),
                             paths.align_run, lines.append).run(poll=0.1)
    assert code == 0 and ps.run_outcome(paths.align_run)[0] == 'done', '\n'.join(lines)
    assert ps.is_alignment_checkpoint(paths.align_run) and (paths.align_run / 'cameras.csv').exists()
    res = json.loads((tmp_path / 'out/dog-align-resources.json').read_text())
    assert res['limits']['seconds'] == 1800 and 'coverage app' in res['limit_overrides']['seconds']['note']

    ps.write_manifest(paths.mesh_manifest, ps.mesh_manifest(align))
    cmd = ps.watched_command(py, paths.mesh_manifest, root, paths.mesh_run, ps.MESH_LIMITS, seed=paths.align_run)
    assert ps.WatchedProcess(cmd, paths.mesh_run, lines.append).run(poll=0.1) == 0, '\n'.join(lines)
    assert ps.run_outcome(paths.mesh_run)[0] == 'done' and (paths.mesh_run / 'raw-mesh.obj').exists()
    res = json.loads((tmp_path / 'out/dog-mesh-resources.json').read_text())
    assert res['limits']['RSS_bytes'] == 6 * 1024 ** 3 and res['limits']['swap_growth_MiB'] == 6144
    assert ps.score_run(paths.mesh_run, paths.mesh_run)['label_v3'] < 1e-9


def test_license_check_reports_missing_license(tmp_path):
    ok, why = ps.license_ok(fake_python(tmp_path, FAKE_METASHAPE_ACTIVATED=0))
    assert not ok and 'license unavailable' in why
    assert ps.license_ok('')[1].startswith('BD3D_METASHAPE_PYTHON is not set')


# ---------------------------------------------------------------- cancel output, run names, image-quality cache

@pytest.mark.skipif(sys.platform != 'darwin', reason='resource_monitor samples macOS sysctl')
def test_watchdog_cancel_prints_one_line_not_a_traceback(tmp_path):
    import signal
    import subprocess
    run = tmp_path / 'r'
    proc = subprocess.Popen([sys.executable, '-m', 'bd3d_quality.resource_monitor', '--output', str(run),
                             '--seconds', '60', '--', sys.executable, '-c', 'import time; time.sleep(30)'],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
                            env={**os.environ, 'PYTHONPATH': str(ps.REPO / 'src')})
    console = tmp_path / 'r-console.log'
    for _ in range(200):
        if console.exists():
            break
        time.sleep(0.05)
    time.sleep(0.5)
    os.killpg(proc.pid, signal.SIGTERM)
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 2 and 'Traceback' not in out
    assert out.strip().splitlines()[-1] == f'cancelled by user; outputs kept in {run.resolve()} (record: r-resources.json)'
    res = json.loads((tmp_path / 'r-resources.json').read_text())
    assert res['status'] == 'watchdog_interrupted'
    assert ps.run_outcome(run)[0] == 'cancelled'


def test_run_names_skip_leftovers_of_cancelled_runs(tmp_path):
    assert ps.unused_run_name(tmp_path, 'dog-mesh') == 'dog-mesh'
    (tmp_path / 'dog-mesh-watchdog.log').write_text('')                 # cancelled before the folder existed
    assert ps.unused_run_name(tmp_path, 'dog-mesh') == 'dog-mesh-2'
    (tmp_path / 'manifests').mkdir()
    (tmp_path / 'manifests' / 'dog-mesh-2.json').write_text('{}')     # manifest written, run never started
    assert ps.unused_run_name(tmp_path, 'dog-mesh') == 'dog-mesh-3'
    (tmp_path / 'manifests' / 'dog-align.json').write_text('{}')
    assert ps.capture_name('/x/dog', tmp_path) == 'dog-2'
    assert ps.run_name_taken(tmp_path, 'dog-align') and not ps.run_name_taken(tmp_path, 'dog-2-align')


def write_cache(path, manifest, skip=0, sources=1):
    rows = ['path,sha256,image_quality,source_run_id,source_experiment_id,metashape_version']
    for i, e in enumerate(manifest['images'][skip:]):
        rows.append(f'{e["path"]},{e["sha256"]},0.8,run{i % sources},exp,2.2.1')
    path.write_text('\n'.join(rows) + '\n')
    return path


def test_find_quality_cache(tmp_path):
    ref = tmp_path / 'set' / 'ref'
    ref.mkdir(parents=True)
    assert ps.find_quality_cache(ref) is None and ps.find_quality_cache(None) is None
    (tmp_path / 'set' / ps.CACHE_NAME).write_text('x')
    assert ps.find_quality_cache(ref) == tmp_path / 'set' / ps.CACHE_NAME            # beside the run
    (ref / ps.CACHE_NAME).write_text('x')
    assert ps.find_quality_cache(ref) == ref / ps.CACHE_NAME                         # in the run wins
    real = ps.find_quality_cache(ps.REPO / 'results/pahma-5-613-v2/ref145b')
    assert real == ps.REPO / 'results/pahma-5-613-v2' / ps.CACHE_NAME
    assert ps.find_quality_cache(ps.REPO / 'results/gpu-pilot-d8/ref50') is None


def test_cache_coverage(dataset, tmp_path):
    _, m = dataset
    assert ps.cache_coverage(write_cache(tmp_path / 'full.csv', m), m) == (True, 'cache covers all 3 photos')
    assert ps.cache_coverage(write_cache(tmp_path / 'part.csv', m, skip=1), m) == (
        False, 'cache covers 2 of 3 photos (matched by SHA-256)')
    assert not ps.cache_coverage(write_cache(tmp_path / 'mixed.csv', m, sources=2), m)[0]
    assert ps.cache_coverage(tmp_path / 'missing.csv', m)[0] is False
    cmd = ps.watched_command('/py', 'm.json', 'p', 'r', ps.ALIGN_LIMITS, image_quality_cache=tmp_path / 'full.csv')
    assert cmd[-2:] == ['--image-quality-cache', str((tmp_path / 'full.csv').resolve())]


@pytest.mark.skipif(sys.platform != 'darwin', reason='resource_monitor samples macOS sysctl')
def test_align_with_image_quality_cache_through_watchdog(dataset, tmp_path):
    root, _ = dataset
    paths = ps.CapturePaths(tmp_path / 'out', 'dog')
    align = ps.alignment_manifest(root, 'dog')
    ps.write_manifest(paths.align_manifest, align)
    cache = write_cache(tmp_path / ps.CACHE_NAME, align)
    cmd = ps.watched_command(fake_python(tmp_path), paths.align_manifest, root, paths.align_run, ps.ALIGN_LIMITS,
                             image_quality_cache=cache)
    lines = []
    assert ps.WatchedProcess(cmd, paths.align_run, lines.append).run(poll=0.1) == 0, '\n'.join(lines)
    assert ps.image_quality_source(paths.align_run) == f'cache {cache.resolve()} (3 images)'
