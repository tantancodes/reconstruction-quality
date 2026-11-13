"""Regression test for scripts/combine_labels.py (one labels.csv per artifact)."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'combine_labels.py'
spec = importlib.util.spec_from_file_location('combine_labels', SCRIPT)
combine_labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(combine_labels)

SCORE = {'images': '30', 'label_v3': '0.0085', 'qc_v3': 'pass', 'scale_check': 'agree', 'fitness_v3': '0.99',
         'r2c_rmse_norm': '0.0101', 'c2r_rmse_norm': '0.0065', 'r2c_p95_norm': '0.02', 'c2r_p95_norm': '0.01',
         'r2c_frac_beyond_cutoff': '0.003', 'c2r_frac_beyond_cutoff': '0.001',
         'completeness_at_0.01': '0.91', 'accuracy_at_0.01': '0.96', 'candidate_mesh_sha256': 'ab' * 32}


def write_scores(folder, rows):
    folder.mkdir(parents=True)
    with open(folder / 'labels_v3.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'run_id', 'experiment_id', *SCORE])
        w.writeheader()
        w.writerows(rows)


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, 'argv', ['combine_labels.py', *map(str, args)])
    combine_labels.main()


def test_combines_scores_with_run_context(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    (runs / 'side40').mkdir(parents=True)
    (runs / 'side40' / 'run.json').write_text(json.dumps({'image_quality_source': {'method': 'cache'}}))
    (runs / 'side40' / 'metrics-initial_alignment.json').write_text(json.dumps(
        {'metrics': {'reprojection_rmse_px': 0.43, 'aligned_camera_count': 30}}))
    (runs / 'side40-resources.json').write_text(json.dumps({'elapsed_seconds': 412.5}))
    write_scores(tmp_path / 'scores', [
        {'run': 'side40', 'run_id': 'r1', 'experiment_id': 'bd3d-1', **SCORE},
        {'run': 'elsewhere', 'run_id': 'r2', 'experiment_id': 'bd3d-2', **SCORE,  # no run folder: context left empty
         'r2c_rmse_norm': '0.002', 'c2r_rmse_norm': '0.004'},
    ])
    out = tmp_path / 'labels.csv'
    run(monkeypatch, '--artifact', 'arf-dog', '--scores', tmp_path / 'scores', '--runs-dir', runs, '--output', out)

    rows = list(csv.DictReader(open(out, newline='')))
    assert list(rows[0]) == combine_labels.COLS
    a, b = rows
    m, e = 0.0101, 0.0065
    assert a == {
        'artifact': 'arf-dog', 'run': 'side40', 'images': '30', 'label_v3': '0.0085', 'qc_v3': 'pass',
        'scale_check': 'agree', 'fitness_v3': '0.99',
        'missing_surface_rmse_norm': '0.0101', 'missing_surface_p95_norm': '0.02',
        'missing_surface_frac_beyond_cutoff': '0.003',
        'extra_surface_rmse_norm': '0.0065', 'extra_surface_p95_norm': '0.01',
        'extra_surface_frac_beyond_cutoff': '0.001',
        'completeness_at_0.01': '0.91', 'accuracy_at_0.01': '0.96', 'dominant_error': 'missing',
        'missing_share_of_sq_error': str(m * m / (m * m + e * e)),
        'run_reprojection_rmse_px': '0.43', 'aligned_cameras': '30', 'elapsed_seconds': '412.5',
        'image_quality_source': 'cache', 'run_id': 'r1', 'experiment_id': 'bd3d-1',
        'candidate_mesh_sha256': 'ab' * 32}
    assert b['dominant_error'] == 'extra'
    assert b['run_reprojection_rmse_px'] == b['aligned_cameras'] == b['elapsed_seconds'] == b['image_quality_source'] == ''


def test_refuses_to_overwrite(tmp_path, monkeypatch):
    write_scores(tmp_path / 'scores', [{'run': 'x', 'run_id': 'r', 'experiment_id': 'e', **SCORE}])
    out = tmp_path / 'labels.csv'
    out.write_text('keep me')
    with pytest.raises(SystemExit):
        run(monkeypatch, '--artifact', 'a', '--scores', tmp_path / 'scores', '--runs-dir', tmp_path, '--output', out)
    assert out.read_text() == 'keep me'
