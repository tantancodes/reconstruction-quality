"""Regression tests for geometry evaluator v3 (src/bd3d_quality/evaluation.py, scripts/score_runs_v3.py).

The golden rows in tests/data/v3_golden.json were written by score_runs_v3.py with the
original evaluator loaded from the Aphoma fork, before it was moved into the package.
"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import v3_fixture
from bd3d_quality import evaluation as ev

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
GOLDEN = json.loads((HERE / 'data' / 'v3_golden.json').read_text())


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / 'scripts' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


score_runs_v3 = load_script('score_runs_v3')


def same(new, old):
    """Compare a fresh value with the CSV text written by the original evaluator."""
    if isinstance(new, (float, int)) and not isinstance(new, bool):
        return new == pytest.approx(float(old), rel=1e-9, abs=1e-15)
    return str(new) == old


@pytest.mark.parametrize('case,kwargs', [('ok', {}), ('badscale', {'camera_scale': 1.3 * 1.05})])
def test_matches_original_evaluator(tmp_path, case, kwargs):
    ref, cand = v3_fixture.make_runs(tmp_path, **kwargs)
    _, rows = score_runs_v3.score_runs(ref, [cand])
    row, golden = rows[0], GOLDEN[case]
    assert set(golden) <= set(row)
    bad = {k: (row[k], golden[k]) for k in golden if not same(row[k], golden[k])}
    assert not bad


def test_settings_signature_unchanged():
    assert ev.v3_settings()['signature'] == GOLDEN['settings_signature'] == 'ce52f0ae78eec23f'


def test_camera_start_recovers_the_frame(tmp_path):
    ref_dir, cand_dir = v3_fixture.make_runs(tmp_path)
    ref = ev.Reference.from_files(ref_dir / 'raw-mesh.obj', ref_dir / 'cameras.csv')
    row, T = score_candidate(ref, cand_dir)
    true = np.eye(4)
    true[:3, :3] = v3_fixture.SCALE * v3_fixture.rotation(v3_fixture.ROTATION_DEG)
    true[:3, 3] = v3_fixture.OFFSET
    composed = T['camera_init_transform'] @ true  # candidate frame -> reference frame, applied to the true map
    assert np.allclose(composed, np.eye(4), atol=1e-9)
    assert row['camera_shared_images'] == 18
    assert row['fitness_at_camera_init'] > 0.95
    assert row['qc_v3'] == 'pass' and row['scale_check'] == 'agree'


def test_missing_patch_shows_as_missing_surface(tmp_path):
    ref_dir, cand_dir = v3_fixture.make_runs(tmp_path)
    row, _ = score_candidate(ev.Reference.from_files(ref_dir / 'raw-mesh.obj', ref_dir / 'cameras.csv'), cand_dir)
    assert row['r2c_rmse_norm'] > 2 * row['c2r_rmse_norm']  # reference -> candidate = missing surface
    assert row['label_v3'] == pytest.approx(np.sqrt((row['r2c_rmse_norm'] ** 2 + row['c2r_rmse_norm'] ** 2) / 2))


def test_same_surface_in_another_frame_scores_zero(tmp_path):
    ref_dir, cand_dir = v3_fixture.make_runs(tmp_path, drop_patch=False, bump=0.0)
    row, _ = score_candidate(ev.Reference.from_files(ref_dir / 'raw-mesh.obj', ref_dir / 'cameras.csv'), cand_dir)
    assert row['label_v3'] < 1e-9
    assert row['icp_scale_v3'] == pytest.approx(1 / v3_fixture.SCALE, rel=1e-9)


def test_scale_check_flags_camera_icp_disagreement(tmp_path):
    ref_dir, cand_dir = v3_fixture.make_runs(tmp_path, camera_scale=v3_fixture.SCALE * 1.05)
    row, _ = score_candidate(ev.Reference.from_files(ref_dir / 'raw-mesh.obj', ref_dir / 'cameras.csv'), cand_dir)
    assert row['scale_check'] == 'disagree'
    assert abs(row['scale_ratio'] - 1) > ev.SCALE_TOLERANCE


def test_qc_thresholds():
    assert [ev.qc(f) for f in (1.0, 0.95, 0.9499, 0.80, 0.79)] == ['pass', 'pass', 'caution', 'caution', 'manual_review']


def test_centroid_start_icp_is_the_v2_loop():
    """Small offset, rotation and scale from a centroid start: ICP recovers them (v1/v2 setting)."""
    g = np.arange(6.0)
    target = np.stack(np.meshgrid(g, g * 1.3, g * 0.7, indexing='ij'), -1).reshape(-1, 3)
    source = (1.02 * target @ v3_fixture.rotation(2).T + [0.1, -0.05, 0.2])[::-1]
    out = ev.icp(source, target, ev.centroid_start(source, target), 0.3, 60, 1e-12, 0.0, with_scale=True)
    assert out['fitness'] == 1.0
    assert out['scale'] == pytest.approx(1 / 1.02, rel=1e-9)
    assert np.allclose(source @ out['transform'][:3, :3].T + out['transform'][:3, 3], target[::-1], atol=1e-9)


def score_candidate(ref, cand_dir):
    return ev.score_candidate(ref, ev.load_obj_mesh(cand_dir / 'raw-mesh.obj'), ev.load_cameras(cand_dir / 'cameras.csv'))


# --------------------------------------------------------------- real data (opt-in: pytest -m local)

LOCAL = [  # (stored labels_v3.csv, reference run, candidate runs' parent folders)
    ('results/gpu-pilot-d8-ext/geometry-v3/labels_v3.csv', 'results/gpu-pilot-d8/ref50',
     ['results/gpu-pilot-d8', 'results/gpu-pilot-d8-ext']),
    ('results/pahma-5-613-v2/geometry-v3-all/labels_v3.csv', 'results/pahma-5-613-v2/ref145b',
     ['results/pahma-5-613-v2']),
]


@pytest.mark.local
@pytest.mark.parametrize('stored,reference,parents', LOCAL)
def test_real_runs_reproduce_stored_labels(stored, reference, parents):
    """Re-score real runs (meshes are local-only) and compare every column at full precision."""
    ref = REPO / reference
    if not (ref / 'raw-mesh.obj').exists():
        pytest.skip('local run folders not present')
    old = list(csv.DictReader(open(REPO / stored, newline='')))
    cands = [next(REPO / p / r['run'] for p in parents if (REPO / p / r['run'] / 'raw-mesh.obj').exists()) for r in old]
    _, rows = score_runs_v3.score_runs(ref, cands)
    for new, prev in zip(rows, old):
        assert {k: str(v) for k, v in new.items()} == prev
