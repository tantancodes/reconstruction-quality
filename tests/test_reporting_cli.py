import copy
import csv
import subprocess
import sys
import pytest
from bd3d_quality.cli import main
from bd3d_quality.image_features import features_for_manifest
from bd3d_quality.manifests import create_manifest
from bd3d_quality.provenance import write_json, read_json
from bd3d_quality.reporting import summarize, build_report


def test_summary_csv(dataset, tmp_path):
    root, m = dataset
    f = features_for_manifest(m, root)
    s = summarize(m, features=f)
    p = tmp_path / 'summary.json'; write_json(p, s)
    rows = build_report([p], tmp_path / 'report')
    with (tmp_path / 'report/experiments.csv').open() as h: actual = list(csv.DictReader(h))
    assert len(actual) == 1 and actual[0]['run_id'] == ''
    assert rows[0]['status'] == 'not_reconstructed'
    assert (tmp_path / 'report/feature_dictionary.json').exists()
    with pytest.raises(ValueError, match='duplicate'): build_report([p,p], tmp_path / 'duplicates')
    assert not (tmp_path / 'duplicates').exists()


def test_feature_provenance_mismatch(dataset):
    root, m = dataset
    f = features_for_manifest(m, root); f['experiment_id'] = 'other'
    with pytest.raises(ValueError): summarize(m, features=f)


def test_incompatible_feature_configs(dataset, tmp_path):
    root, m = dataset; m2 = create_manifest(root, 'other-object')
    ps = []
    for i, (manifest, dim) in enumerate([(m, 64), (m2, 96)]):
        p = tmp_path / f'{i}.json'
        write_json(p, summarize(manifest, features=features_for_manifest(manifest, root, dim))); ps.append(p)
    with pytest.raises(ValueError, match='incompatible'): build_report(ps, tmp_path / 'report')


def test_inconsistent_summary_rejected(dataset, tmp_path):
    root, m = dataset; s = summarize(m); s['row']['image_count'] = 999
    p = tmp_path / 'summary.json'; write_json(p, s)
    with pytest.raises(ValueError, match='inconsistent'): build_report([p], tmp_path / 'report')


def test_cli_validation_and_errors(dataset, tmp_path, capsys):
    root, m = dataset; p = tmp_path / 'manifest.json'; write_json(p, m)
    assert main(['manifest', 'validate', str(p), '--root', str(root)]) == 0
    (root / '0.png').unlink()
    assert main(['manifest', 'validate', str(p), '--root', str(root)]) == 2
    assert main(['features', 'extract', str(p), '--root', str(root), '--output', str(tmp_path / 'f.json')]) == 2
    assert not (tmp_path / 'f.json').exists()


def test_end_to_end_unlicensed(tmp_path):
    out = tmp_path / 'demo'
    subprocess.run([sys.executable, 'examples/synthetic_workflow.py', str(out)], check=True, capture_output=True)
    assert read_json(out / 'validation.json')['valid']
    assert read_json(out / 'geometry.json')['status'] == 'complete'
    assert read_json(out / 'report/validation.json')['row_count'] == 1


def test_geometry_requires_recorded_mesh(dataset, tetra):
    from bd3d_quality.geometry_quality import compare_geometry
    root, m = dataset
    g = compare_geometry(tetra, tetra, reference_id='synthetic', units='mm', sample_count=100)
    with pytest.raises(ValueError, match='raw mesh'): summarize(m, geometry=g)


def test_incomplete_pair_set(dataset):
    from bd3d_quality.local_match_features import match_manifest
    root, m = dataset
    matches = match_manifest(m, root); matches['pairs'].pop()
    with pytest.raises(ValueError, match='incomplete'): summarize(m, matches=matches)
