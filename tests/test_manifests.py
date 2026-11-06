import copy
import hashlib
import shutil
import pytest
from bd3d_quality.config import ReconstructionConfig
from bd3d_quality.manifests import create_manifest, experiment_identity, validate_manifest, safe_path
from bd3d_quality.provenance import sha256_file, write_json


def test_identity_portable(dataset, tmp_path):
    root, m = dataset
    other = tmp_path / 'moved'
    shutil.copytree(root, other)
    m2 = create_manifest(other, 'test-object', description='different description')
    assert m['experiment_id'] == m2['experiment_id']
    assert validate_manifest(m2, other)['files_verified']
    m2['configuration']['keypoint_limit'] += 1
    assert experiment_identity(m2) != m['experiment_id']
    assert not validate_manifest(m2, other)['valid']


def test_hash(dataset):
    root, m = dataset
    p = root / m['images'][0]['path']
    assert sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.parametrize('mutation', ['count', 'duplicate', 'hash', 'size', 'empty', 'configuration'])
def test_bad_manifest(dataset, mutation):
    root, original = dataset
    m = copy.deepcopy(original)
    if mutation == 'count': m['image_count'] = 9
    elif mutation == 'duplicate': m['images'][1] = m['images'][0]
    elif mutation == 'hash': m['images'][0]['sha256'] = 'x' * 64
    elif mutation == 'size': m['images'][0]['size_bytes'] = 0
    elif mutation == 'empty': m['images'] = []
    elif mutation == 'configuration': m['configuration'] = {}
    assert not validate_manifest(m, root)['valid']


@pytest.mark.parametrize('name', ['/tmp/a.jpg', '../escape.jpg', 'a/../x.jpg', 'a\\b.jpg', './a.jpg'])
def test_path_escape(tmp_path, name):
    with pytest.raises(ValueError): safe_path(tmp_path, name)


def test_symlink_escape(tmp_path):
    root = tmp_path / 'root'; root.mkdir()
    outside = tmp_path / 'outside.png'; outside.write_bytes(b'x')
    (root / 'link.png').symlink_to(outside)
    with pytest.raises(ValueError): safe_path(root, 'link.png')


@pytest.mark.parametrize('action', ['missing', 'changed', 'unreadable'])
def test_bad_inputs(dataset, action):
    root, m = dataset
    p = root / m['images'][0]['path']
    if action == 'missing': p.unlink()
    else: p.write_bytes(b'not an image')
    assert not validate_manifest(m, root)['valid']
    if action == 'unreadable':
        with pytest.raises(OSError): create_manifest(root, 'test')


def test_exclusive_write(tmp_path):
    p = tmp_path / 'evidence.json'
    write_json(p, {'a': 1})
    with pytest.raises(FileExistsError): write_json(p, {'a': 2})
    assert '1' in p.read_text()


@pytest.mark.parametrize('kwargs', [{'profile': 'unknown'}, {'match_downscale': 3}, {'keypoint_limit': -1},
                                  {'min_alignment_fraction': float('nan')}, {'cpu_only': 'false'}])
def test_config_validation(kwargs):
    with pytest.raises(ValueError): ReconstructionConfig(**kwargs)
