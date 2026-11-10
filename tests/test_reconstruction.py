from types import SimpleNamespace as NS
import pytest
from bd3d_quality.manifests import experiment_identity
from bd3d_quality.metashape_runner import run_reconstruction
from bd3d_quality.provenance import read_json, write_json
from bd3d_quality.reconstruction_metrics import collect_metrics


def test_alignment_and_complete_resume(dataset, tmp_path, fake_api):
    root, m = dataset
    out = tmp_path / 'run'
    r = run_reconstruction(m, root, out, api=fake_api)
    assert r['status'] == 'complete'
    metrics = r['stages'][0]['metrics']
    assert metrics['aligned_camera_count'] == 3 and metrics['valid_tie_point_count'] == 1
    assert metrics['reprojection_rmse_px'] is None
    assert (out / 'raw-tie-points.ply').exists()
    assert run_reconstruction(m, root, out, resume=True, api=fake_api)['run_id'] == r['run_id']
    assert fake_api.app.gpu_mask == 3
    with pytest.raises(FileExistsError): run_reconstruction(m, root, out, api=fake_api)


def test_dense_mesh_stages(dataset, tmp_path, fake_api):
    root, m = dataset
    m['configuration']['profile'] = 'mesh'; m['experiment_id'] = experiment_identity(m)
    r = run_reconstruction(m, root, tmp_path / 'run', api=fake_api)
    assert [s['stage'] for s in r['stages']] == ['initial_alignment', 'dense_reconstruction', 'mesh_generation']
    assert r['stages'][0]['metrics']['dense_point_count'] is None
    assert r['stages'][1]['metrics']['dense_point_count'] == 27
    assert r['stages'][2]['metrics']['mesh_face_count'] == 1


@pytest.mark.parametrize('tamper', ['project', 'artifact', 'identity'])
def test_resume_rejects_tampering(dataset, tmp_path, fake_api, tamper):
    root, m = dataset; out = tmp_path / 'run'
    run_reconstruction(m, root, out, api=fake_api)
    before = (out / 'run.json').read_bytes()
    if tamper == 'identity':
        m['configuration']['keypoint_limit'] += 1; m['experiment_id'] = experiment_identity(m)
    else:
        (out / ('project.psx' if tamper == 'project' else 'raw-tie-points.ply')).write_text('changed')
    with pytest.raises(ValueError): run_reconstruction(m, root, out, resume=True, api=fake_api)
    assert (out / 'run.json').read_bytes() == before


def test_clean_checkpoint_resume(dataset, tmp_path, fake_api):
    root, m = dataset; out = tmp_path / 'run'
    r = run_reconstruction(m, root, out, api=fake_api)
    r['status'] = 'checkpointed'; r['finished_at'] = None
    write_json(out / 'run.json', r, replace=True)
    assert run_reconstruction(m, root, out, resume=True, api=fake_api)['status'] == 'complete'


def test_license_failure(dataset, tmp_path, fake_api):
    root, m = dataset; out = tmp_path / 'run'; fake_api.app.activated = False
    with pytest.raises(RuntimeError, match='license'): run_reconstruction(m, root, out, api=fake_api)
    r = read_json(out / 'run.json')
    assert r['status'] == 'failed' and r['failures']
    with pytest.raises(ValueError, match='clean checkpoints'):
        run_reconstruction(m, root, out, resume=True, api=fake_api)


def test_alignment_failure(dataset, tmp_path, fake_api, monkeypatch):
    root, m = dataset; out = tmp_path / 'run'
    monkeypatch.setattr(fake_api.Chunk, 'alignCameras', lambda self: None)
    with pytest.raises(RuntimeError, match='alignment failed'): run_reconstruction(m, root, out, api=fake_api)
    assert (out / 'metrics-initial_alignment.json').exists()
    assert read_json(out / 'run.json')['status'] == 'failed'


def test_no_camera_alignment():
    c = NS(cameras=[NS(transform=None)], tie_points=None, point_cloud=None, model=None)
    r = collect_metrics(c, 1)
    assert r['metrics']['aligned_camera_count'] == 0
    assert r['metrics']['tie_point_count'] is None
    assert r['missing_reasons']['tie_point_count']


def test_reprojection_method():
    class Camera:
        transform = True
        def error(self, point, projection): return NS(norm=lambda: 5.0)
    camera = Camera()
    chunk = NS(cameras=[camera], tie_points=NS(points=[NS(valid=True, track_id=7, coord='xyz')],
        projections={camera: [NS(track_id=7, coord='uv')]}))
    r = collect_metrics(chunk, 1)
    assert r['metrics']['reprojection_rmse_px'] == 5
    assert r['metrics']['reprojection_observation_count'] == 1


def test_lock_blocks_concurrent_runner(dataset, tmp_path, fake_api):
    root, m = dataset; out = tmp_path / 'run'
    out.mkdir(); (out / '.run.lock').touch()
    with pytest.raises(FileExistsError): run_reconstruction(m, root, out, resume=True, api=fake_api)


def test_api_version_failure(dataset, tmp_path, fake_api):
    root, m = dataset; fake_api.app.version = '2.3.0'
    with pytest.raises(RuntimeError, match='targets Metashape 2.2'):
        run_reconstruction(m, root, tmp_path / 'run', api=fake_api)
    assert read_json(tmp_path / 'run/run.json')['status'] == 'failed'


def test_dense_empty_failure_preserves_alignment(dataset, tmp_path, fake_api, monkeypatch):
    root, m = dataset
    m['configuration']['profile'] = 'dense'; m['experiment_id'] = experiment_identity(m)
    monkeypatch.setattr(fake_api.Chunk, 'buildPointCloud', lambda self: setattr(self, 'point_cloud', NS(point_count=0)))
    with pytest.raises(RuntimeError, match='no points'):
        run_reconstruction(m, root, tmp_path / 'run', api=fake_api)
    r = read_json(tmp_path / 'run/run.json')
    assert r['stages'][0]['stage'] == 'initial_alignment' and r['status'] == 'failed'
    assert fake_api.app.cpu_enable is False


def test_metric_artifact_tampering(dataset, tmp_path, fake_api):
    root, m = dataset; out = tmp_path / 'run'
    run_reconstruction(m, root, out, api=fake_api)
    (out / 'metrics-initial_alignment.json').write_text('{}')
    with pytest.raises(ValueError, match='artifact mismatch'):
        run_reconstruction(m, root, out, resume=True, api=fake_api)


def test_project_fingerprint_excludes_transient_lock(tmp_path):
    from bd3d_quality.metashape_runner import project_fingerprints
    (tmp_path / 'project.psx').write_text('project')
    folder = tmp_path / 'project.files'; folder.mkdir()
    (folder / 'lock').write_text('transient process ID')
    (folder / 'doc.xml').write_text('scientific state')
    r = project_fingerprints(tmp_path)
    assert 'project.files/lock' not in r and 'project.files/doc.xml' in r


def test_vendor_project_lock_blocks_resume(dataset, tmp_path, fake_api):
    root, m = dataset; out = tmp_path / 'run'
    run_reconstruction(m, root, out, api=fake_api)
    folder = out / 'project.files'; folder.mkdir()
    (folder / 'lock').touch()
    with pytest.raises(ValueError, match='project lock'):
        run_reconstruction(m, root, out, resume=True, api=fake_api)


# --- per-camera table -------------------------------------------------------

def _camera_table_chunk(chunk_matrix):
    import numpy as np

    class Camera:
        def __init__(self, label, transform, meta):
            self.label, self.transform, self.meta = label, transform, meta
        def error(self, point, projection):
            return NS(norm=lambda: projection)  # residual magnitude encoded in the fake coord

    pose = np.array([[0, 0, 1, 1], [0, 1, 0, 2], [-1, 0, 0, 3], [0, 0, 0, 1]], dtype=float)  # +Z -> +X
    aligned = Camera('a.jpg', pose, {'Image/Quality': '0.75'})
    unaligned = Camera('b.jpg', None, {})
    points = [NS(valid=True, track_id=1, coord='p1'), NS(valid=True, track_id=2, coord='p2'),
              NS(valid=False, track_id=3, coord='p3')]
    projections = {aligned: [NS(track_id=1, coord=3.0), NS(track_id=2, coord=4.0), NS(track_id=3, coord=99.0)],
                   unaligned: [NS(track_id=1, coord=1.0)]}
    return NS(cameras=[aligned, unaligned], transform=NS(matrix=chunk_matrix),
              tie_points=NS(points=points, projections=projections), point_cloud=None, model=None)


class _RowMatrix:
    """Mimics Metashape.Matrix, which exposes rows rather than numpy semantics."""
    def __init__(self, a): self.a = a
    def row(self, i): return iter(self.a[i])


def _chunk_matrix():
    import numpy as np
    m = 2.0 * np.eye(4); m[3, 3] = 1; m[0, 3] = 10  # scale 2, shift x by 10
    return m


@pytest.mark.parametrize('wrap', [lambda m: m, _RowMatrix])
def test_camera_table_pose_and_residuals(wrap):
    from bd3d_quality.reconstruction_metrics import collect_camera_table
    a, b = collect_camera_table(_camera_table_chunk(wrap(_chunk_matrix())))
    assert a['label'] == 'a.jpg' and a['aligned'] is True
    assert (a['x'], a['y'], a['z']) == (12.0, 4.0, 6.0)
    assert (a['view_x'], a['view_y'], a['view_z']) == pytest.approx((1.0, 0.0, 0.0))  # unit length despite scale
    assert a['n_projections'] == 2  # projection onto invalid tie point excluded
    assert a['reprojection_rmse_px'] == pytest.approx((12.5) ** 0.5)
    assert a['image_quality'] == 0.75 and a['missing_reasons'] is None


def test_camera_table_unaligned_and_missing_quality():
    from bd3d_quality.reconstruction_metrics import collect_camera_table
    _, b = collect_camera_table(_camera_table_chunk(_chunk_matrix()))
    assert b['label'] == 'b.jpg' and b['aligned'] is False
    assert all(b[k] is None for k in ['x', 'y', 'z', 'view_x', 'view_y', 'view_z',
                                       'reprojection_rmse_px', 'image_quality'])
    assert b['n_projections'] == 1  # matches exist even though the pose was not solved
    for field in ['pose', 'reprojection_rmse_px', 'image_quality']:
        assert field in b['missing_reasons']


def test_camera_table_absence_is_not_zero():
    from bd3d_quality.reconstruction_metrics import collect_camera_table
    import numpy as np
    no_ties = NS(cameras=[NS(label='c', transform=np.eye(4), meta={})], transform=NS(matrix=np.eye(4)),
                 tie_points=None)
    (row,) = collect_camera_table(no_ties)
    assert row['n_projections'] is None and row['reprojection_rmse_px'] is None
    assert 'tie points not built' in row['missing_reasons']
    chunk = _camera_table_chunk(_chunk_matrix())
    chunk.tie_points.projections[chunk.cameras[0]] = []
    a, _ = collect_camera_table(chunk)
    assert a['n_projections'] == 0 and a['reprojection_rmse_px'] is None
    assert 'no valid observations' in a['missing_reasons']


def test_camera_table_bad_chunk_transform_keeps_rows():
    from bd3d_quality.reconstruction_metrics import collect_camera_table
    a, b = collect_camera_table(_camera_table_chunk(None))
    assert a['x'] is None and 'chunk transform unavailable' in a['missing_reasons']
    assert a['reprojection_rmse_px'] is not None  # residuals do not depend on the chunk transform


def test_camera_table_matches_pooled_rmse():
    from bd3d_quality.reconstruction_metrics import collect_camera_table
    chunk = _camera_table_chunk(_chunk_matrix())
    rows = [r for r in collect_camera_table(chunk) if r['reprojection_rmse_px'] is not None]
    pooled = (sum(r['n_projections'] * r['reprojection_rmse_px'] ** 2 for r in rows)
              / sum(r['n_projections'] for r in rows)) ** 0.5
    m = collect_metrics(chunk, 2)['metrics']
    assert pooled == pytest.approx(m['reprojection_rmse_px'])
    assert sum(r['n_projections'] for r in rows) == m['reprojection_observation_count']


def test_runner_writes_cameras_csv(dataset, tmp_path, fake_api, monkeypatch):
    import csv
    from bd3d_quality.provenance import sha256_file
    def analyze(self, cameras):
        for i, c in enumerate(cameras):
            c.label, c.meta = f'{i}.png', {'Image/Quality': '0.9'}
    monkeypatch.setattr(fake_api.Chunk, 'analyzeImages', analyze, raising=False)
    root, m = dataset; out = tmp_path / 'run'
    r = run_reconstruction(m, root, out, api=fake_api)
    with (out / 'cameras.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert [row['label'] for row in rows] == ['0.png', '1.png', '2.png']
    assert all(row['image_quality'] == '0.9' for row in rows)
    assert rows[0]['x'] == ''  # fake pose is unusable: written blank, never 0
    recorded = next(a for a in r['artifacts'] if a['path'] == 'cameras.csv')
    assert recorded['sha256'] == sha256_file(out / 'cameras.csv')
    assert not r['warnings']
    (out / 'cameras.csv').write_text('tampered')
    with pytest.raises(ValueError, match='artifact mismatch'):
        run_reconstruction(m, root, out, resume=True, api=fake_api)


def test_runner_analyze_images_failure_is_recorded(dataset, tmp_path, fake_api, monkeypatch):
    def broken(self, cameras): raise RuntimeError('license feature missing')
    monkeypatch.setattr(fake_api.Chunk, 'analyzeImages', broken, raising=False)
    root, m = dataset; out = tmp_path / 'run'
    r = run_reconstruction(m, root, out, api=fake_api)
    assert r['status'] == 'complete'
    (w,) = r['warnings']
    assert w['field'] == 'image_quality' and 'license feature missing' in w['missing_reason']
    assert (out / 'cameras.csv').exists()


def test_failed_alignment_still_writes_cameras_csv(dataset, tmp_path, fake_api, monkeypatch):
    root, m = dataset; out = tmp_path / 'run'
    monkeypatch.setattr(fake_api.Chunk, 'alignCameras', lambda self: None)
    with pytest.raises(RuntimeError, match='alignment failed'):
        run_reconstruction(m, root, out, api=fake_api)
    assert (out / 'cameras.csv').read_text().count('False') == 3
    assert any(a['path'] == 'cameras.csv' for a in read_json(out / 'run.json')['artifacts'])
