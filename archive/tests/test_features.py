import numpy as np
from PIL import Image
import pytest
from bd3d_legacy.image_features import extract_image_features, features_for_manifest
from bd3d_legacy.local_match_features import match_manifest


def test_constant_image(tmp_path):
    p = tmp_path / 'constant.png'
    Image.new('RGB', (40, 20), (100, 100, 100)).save(p)
    f = extract_image_features(p)
    assert f['width_px'] == 40 and f['aspect_ratio'] == 2
    assert f['brightness_mean'] == 100
    assert f['contrast_std'] == f['entropy_bits'] == f['laplacian_variance'] == 0


def test_features_bound_to_manifest(dataset):
    root, m = dataset
    record = features_for_manifest(m, root, 64)
    assert len(record['images']) == 3
    assert record['images'][0]['analysis_width_px'] == 64
    assert record['experiment_id'] == m['experiment_id']
    (root / '0.png').unlink()
    with pytest.raises(ValueError): features_for_manifest(m, root)


def test_tiny_image(tmp_path):
    p = tmp_path / 'tiny.png'; Image.new('L', (1, 1)).save(p)
    f = extract_image_features(p)
    assert f['laplacian_variance'] is None and f['missing_reasons']


def test_pair_matches(dataset):
    root, m = dataset
    r = match_manifest(m, root)
    assert len(r['pairs']) == 3
    assert all(p['ratio_match_count_a_to_b'] >= 0 for p in r['pairs'])
    assert r == match_manifest(m, root)


def test_constant_no_descriptors(tmp_path):
    from bd3d_quality.manifests import create_manifest
    for i in range(2): Image.new('L', (64,64)).save(tmp_path / f'{i}.png')
    result = match_manifest(create_manifest(tmp_path, 'constant'), tmp_path)
    assert result['pairs'][0]['ratio_match_count_a_to_b'] == 0


def test_invalid_dimension(dataset):
    root, m = dataset
    with pytest.raises(ValueError): features_for_manifest(m, root, 0)
