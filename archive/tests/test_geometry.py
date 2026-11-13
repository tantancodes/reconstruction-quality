import numpy as np
import pytest
from bd3d_legacy.geometry_quality import compare_geometry, load_obj_mesh, sample_surface


def compare(a, b, **kwargs):
    return compare_geometry(a, b, reference_id='synthetic', units='mm', sample_count=1000, **kwargs)


def test_self(tetra):
    r = compare(tetra, tetra)
    assert r['status'] == 'complete'
    assert r['metrics']['chamfer_mean_squared'] < 1e-20
    assert r['registration']['fitness'] == 1


def test_translation(tetra, tmp_path):
    p = tmp_path / 'translated.obj'
    lines = []
    for line in tetra.read_text().splitlines():
        if line.startswith('v '):
            xyz = np.array([float(v) for v in line.split()[1:]]) + [7, -3, 2]
            line = 'v ' + ' '.join(map(str, xyz))
        lines.append(line)
    p.write_text('\n'.join(lines))
    r = compare(p, tetra)
    assert r['status'] == 'complete' and r['metrics']['chamfer_mean_squared'] < 1e-20
    assert np.allclose(np.array(r['registration']['transform_candidate_to_reference'])[:3,3], [-7,3,-2])


def test_scale_rejection(tetra, tmp_path):
    p = tmp_path / 'scaled.obj'
    p.write_text('\n'.join('v ' + ' '.join(str(float(v) * 10) for v in line.split()[1:])
        if line.startswith('v ') else line for line in tetra.read_text().splitlines()))
    r = compare(p, tetra)
    assert r['status'] == 'invalid' and r['metrics'] is None
    assert 'major_scale_or_coverage_inconsistency' in r['flags']


@pytest.mark.parametrize('text', ['v nan 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3',
                                    'v 0 0 0\nf 1 2 3', 'v 0 0 0\nv 1 0 0\nv 2 0 0\nf 1 2 3'])
def test_invalid_mesh(tmp_path, text):
    p = tmp_path / 'bad.obj'; p.write_text(text)
    with pytest.raises(ValueError): compare(p, p)


def test_ground_truth_requires_evidence(tetra):
    with pytest.raises(ValueError, match='evidence'): compare(tetra, tetra, reference_kind='validated_ground_truth')


def test_invalid_settings(tetra):
    with pytest.raises(ValueError): compare(tetra, tetra, scale_tolerance=float('nan'))


def test_registration_failure_suppresses_metrics(tetra, tmp_path):
    p = tmp_path / 'other.obj'
    p.write_text('v 0 0 0\nv 2 0 0\nv 0 2 0\nf 1 2 3\n')
    result = compare(p, tetra, correspondence_fraction=1e-8, scale_tolerance=0.99)
    assert result['status'] == 'invalid'
    assert 'registration_failed' in result['flags'] and result['metrics'] is None


def test_negative_obj_indices(tmp_path):
    p = tmp_path / 'negative.obj'
    p.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n')
    assert load_obj_mesh(p).triangles.tolist() == [[0,1,2]]
