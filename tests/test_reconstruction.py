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
