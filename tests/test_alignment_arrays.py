"""Synthetic metadata/arrays only. No PyTorch, real images or model fitting."""
import copy

import numpy as np
import pytest

from bd3d_quality.alignment_arrays import load_alignment_arrays, content_set_id
from bd3d_quality.historical_alignment import build_dataset
from bd3d_quality.provenance import digest, read_json, sha256_file, write_json
from bd3d_quality.tabular_dataset import PRE_FEATURES


def synthetic_record(i, photos, label):
    images = [{'path': name+'.jpg', 'sha256': digest('synthetic-photo:'+name)} for name in photos]
    sid = 'sha256:'+digest(images)
    return {'schema_version': 1, 'kind': 'historical_alignment_run', 'record_id': f'fixture-{i}',
        'source_category': 'synthetic_test_only', 'artifact_id': 'arf-dog', 'artifact_group_id': 'arf-dog',
        'acquisition_session_id': None, 'session_status': 'unknown', 'reconstruction_run_id': f'fixture-run-{i}',
        'source_image_set_id': sid, 'image_count': len(images), 'source_images': images,
        'image_provenance_status': 'verified_hash_manifest', 'image_match_evidence': 'SYNTHETIC ONLY',
        'image_features': dict(zip(PRE_FEATURES, [len(images), 100+i, 20+i, 300+i, 5+i/10, 1000+i, 50+i])),
        'feature_provenance': {'stage': 'pre_alignment_images_only', 'source_image_set_id': sid},
        'metashape_version': 'synthetic-version', 'configuration_status': 'partially_reported',
        'alignment_configuration': {'sparse_downscale': 1},
        'label': {'rmse_pixels': label, 'units': 'pixels',
            'status': 'computed_historical_verified' if label is not None else 'missing',
            'definition_id': 'synthetic-pixel-rms', 'stage': 'synthetic-stage',
            'source': {'file_path': 'SYNTHETIC-NOT-REAL', 'section': 'fixture', 'evidence_sha256': digest(f'fixture-report-{i}')}},
        'diagnostics': {'aligned_camera_count': len(images), 'tie_point_count': 100}}


@pytest.fixture
def bundle(tmp_path):
    r = [synthetic_record(0, ['a','b','c'], .3), synthetic_record(1, ['c','b','a'], .301),
         synthetic_record(2, ['b','c','d'], .4), synthetic_record(3, ['c','d','e'], .5),
         synthetic_record(4, ['e','f','g'], None), synthetic_record(5, ['f','g','h'], .6)]
    r[1]['image_features'] = copy.deepcopy(r[0]['image_features'])
    r[5]['image_features'] = None; r[5]['feature_provenance'] = None
    path = tmp_path/'bundle'; build_dataset(r, path)
    return path


def update_fixture_hash(bundle, name):
    h = read_json(bundle/'file_hashes.json'); h[name] = sha256_file(bundle/name)
    write_json(bundle/'file_hashes.json', h, replace=True)


def test_default_strict_gate_stays_closed_exploratory_is_explicit(bundle):
    before = sha256_file(bundle/'arrays.npz')
    strict = load_alignment_arrays(bundle)
    assert strict.X.shape == (0,7) and strict.y.shape == (0,1)
    data = load_alignment_arrays(bundle, mode='exploratory')
    assert data.X.shape == (4,7) and data.y.shape == (4,1)
    assert data.X.dtype == data.y.dtype == np.float32
    assert data.X.flags.c_contiguous and data.X_mask.dtype == bool
    assert not data.strict_ready_mask.any() and data.feature_names == tuple(PRE_FEATURES)
    assert len(set(data.content_groups)) == 3 and len(set(data.artifact_groups)) == 1
    assert data.content_groups[0] == data.content_groups[1]
    assert sha256_file(bundle/'arrays.npz') == before
    assert len(load_alignment_arrays(bundle)) == 0


def test_all_rows_preserve_missing_values_and_masks(bundle):
    data = load_alignment_arrays(bundle, mode='exploratory', paired_only=False)
    assert len(data) == 6
    assert np.isnan(data.y[4,0]) and not data.y_mask[4,0]
    assert np.isnan(data.X[5]).all() and not data.X_mask[5].any()
    assert data.y_mask[5,0]
    sample = data[0]
    assert set(sample) == {'x','y','x_mask','y_mask','strict_ready'}
    sample['x'][0] = -999
    assert data.X[0,0] == 3  # samples are writable copies, never mutable views of evidence


def test_float64_analysis_and_filters(bundle):
    data = load_alignment_arrays(bundle, mode='exploratory', dtype=np.float64)
    assert data.y[1,0] == .301 and data.X.dtype == np.float64
    assert len(load_alignment_arrays(bundle, mode='exploratory', artifact_group='not-present')) == 0
    with pytest.raises(ValueError, match='mode'): load_alignment_arrays(bundle, mode='automatic')
    with pytest.raises(ValueError, match='float32/float64'): load_alignment_arrays(bundle, dtype=np.float16)


def test_explicit_subset_preserves_row_metadata_and_rejects_implicit_replication(bundle):
    data = load_alignment_arrays(bundle, mode='exploratory')
    subset = data.take([3,0])
    assert subset.record_ids == (data.record_ids[3], data.record_ids[0])
    np.testing.assert_array_equal(subset.X, data.X[[3,0]])
    np.testing.assert_array_equal(subset.source_indices, data.source_indices[[3,0]])
    assert not subset.strict_ready_mask.any() and len(data.take([])) == 0
    for indices in [[0,0], [-1], [100], [1.5], [True]]:
        with pytest.raises(ValueError): data.take(indices)


def test_array_tampering_rejected_even_if_file_hash_is_updated(bundle):
    with np.load(bundle/'arrays.npz', allow_pickle=False) as a:
        values = {k: a[k] for k in a.files}
    values['model_input_row_mask'][0] = True
    np.savez_compressed(bundle/'arrays.npz', **values)
    update_fixture_hash(bundle, 'arrays.npz')
    with pytest.raises(ValueError, match='array/record mismatch'):
        load_alignment_arrays(bundle, mode='exploratory')


def test_source_hash_and_feature_order_protect_against_leakage(bundle):
    schema = read_json(bundle/'schema.json')
    schema['feature_columns'][0] = 'diagnostic_tie_point_count'
    write_json(bundle/'schema.json', schema, replace=True)
    with pytest.raises(ValueError, match='hash mismatch'): load_alignment_arrays(bundle)
    update_fixture_hash(bundle, 'schema.json')
    with pytest.raises(ValueError, match='feature/target contract'): load_alignment_arrays(bundle)


def test_source_record_reordering_rejected(bundle):
    r = read_json(bundle/'records.json'); r['records'].reverse()
    write_json(bundle/'records.json', r, replace=True); update_fixture_hash(bundle, 'records.json')
    with pytest.raises(ValueError, match='row-order'): load_alignment_arrays(bundle)
