"""Isolated Metashape 2.2 integration, raw artifacts and verified checkpoints."""
from __future__ import annotations
import csv
import importlib
import os
import shutil
import tempfile
from pathlib import Path
import time
from uuid import uuid4
from .config import ReconstructionConfig, ResourceProtocol
from .manifests import require_valid, safe_path
from .provenance import now, write_json, read_json, sha256_file, software_versions, digest
from .reconstruction_metrics import CAMERA_TABLE_COLUMNS, collect_camera_table, collect_metrics


def write_camera_table(path, rows):
    """Atomic CSV publish; None is written as an empty cell (missing, not zero)."""
    path = Path(path)
    fd, name = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CAMERA_TABLE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def project_fingerprints(output):
    paths = [output / 'project.psx']
    folder = output / 'project.files'
    if folder.exists():
        # Metashape removes this transient process lock when Document closes.
        paths.extend(p for p in sorted(folder.rglob('*'))
                     if p.is_file() and p != folder / 'lock')
    return {p.relative_to(output).as_posix(): sha256_file(p) for p in paths}


IMAGE_QUALITY_CACHE_COLUMNS = ['path', 'sha256', 'image_quality', 'source_run_id', 'source_experiment_id',
                               'metashape_version']


def apply_image_quality_cache(chunk, manifest, cache_path):
    """Set camera.meta['Image/Quality'] from a cache CSV instead of running analyzeImages.

    Every manifest image must have a cache row with the same SHA-256 and a finite value;
    otherwise this raises (no silent partial cache). Returns the provenance record.
    """
    cache_path = Path(cache_path).resolve()
    with cache_path.open(newline='') as f:
        rows = {r['sha256']: r for r in csv.DictReader(f)}
    cameras = list(chunk.cameras)
    if len(cameras) != len(manifest['images']):
        raise ValueError('camera count differs from manifest; cannot map image-quality cache')
    sources = set()
    for camera, entry in zip(cameras, manifest['images']):
        row = rows.get(entry['sha256'])
        if row is None or row['image_quality'] in ('', None):
            raise ValueError(f'image-quality cache has no value for {entry["path"]}')
        value = float(row['image_quality'])
        if value != value or value in (float('inf'), float('-inf')):
            raise ValueError(f'nonfinite cached image quality for {entry["path"]}')
        camera.meta['Image/Quality'] = row['image_quality']
        sources.add((row['source_run_id'], row['source_experiment_id'], row['metashape_version']))
    if len(sources) != 1:
        raise ValueError('image-quality cache rows come from more than one source run')
    (run_id, experiment_id, version), = sources
    return {'method': 'cache', 'cache_path': str(cache_path), 'cache_sha256': sha256_file(cache_path),
            'source_run_id': run_id, 'source_experiment_id': experiment_id,
            'source_metashape_version': version, 'images': len(cameras)}


def run_reconstruction(manifest, root, output, *, resume=False, api=None, seed_checkpoint=None,
                       image_quality_cache=None):
    require_valid(manifest, root)
    output = Path(output).resolve()
    config = ReconstructionConfig(**manifest['configuration'])
    protocol = ResourceProtocol(**manifest['execution_protocol']) if 'execution_protocol' in manifest else None
    if seed_checkpoint is not None and resume:
        raise ValueError('seed checkpoint and resume are mutually exclusive')
    seed_state = verify_seed(seed_checkpoint, manifest, root) if seed_checkpoint is not None else None
    if resume:
        if not output.is_dir():
            raise ValueError('resume directory does not exist')
    else:
        output.mkdir(parents=True, exist_ok=False)
    lock = output / '.run.lock'
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    state = None
    started = time.perf_counter()
    previous_gpu = None
    previous_cpu = None
    try:
        if resume:
            if (output / 'project.files' / 'lock').exists():
                raise ValueError('Metashape project lock exists; close the other document before resuming')
            state = read_json(output / 'run.json')
            if state['experiment_id'] != manifest['experiment_id'] or state['configuration'] != config.to_dict():
                state = None
                raise ValueError('resume provenance/settings mismatch')
            if state['status'] not in {'checkpointed', 'complete'}:
                state = None
                raise ValueError('only clean checkpoints can resume; failed/interrupted runs require a new directory')
            if state['project_files'] != project_fingerprints(output):
                state = None
                raise ValueError('project checkpoint content mismatch')
            for artifact in state['artifacts']:
                p = output / artifact['path']
                if sha256_file(p) != artifact['sha256']:
                    state = None
                    raise ValueError('recorded artifact mismatch')
            if state['status'] == 'complete':
                return state
        else:
            state = {'schema_version': 1, 'kind': 'reconstruction',
                     'run_id': str(uuid4()), 'experiment_id': manifest['experiment_id'],
                     'dataset_id': manifest['dataset_id'], 'configuration': config.to_dict(),
                     'started_at': now(), 'finished_at': None, 'status': 'running',
                     'software': software_versions(), 'stages': [], 'warnings': [],
                     'failures': [], 'artifacts': [], 'project_files': {},
                     'planned_stages': ['initial_alignment'] +
                         (['dense_reconstruction'] if config.profile in {'dense', 'mesh'} else []) +
                         (['mesh_generation'] if config.profile == 'mesh' else []),
                     'skipped_stages': ['refinement', 'cleanup', 'texture'],
                     'limitations': ['No refinement or mesh postprocessing performed; raw outputs preserved.']}
            if seed_state:
                state['parent_checkpoint'] = {'run_id': seed_state['run_id'],
                    'experiment_id': seed_state['experiment_id'], 'project_files': seed_state['project_files']}
            if protocol:
                state['execution_protocol'] = manifest['execution_protocol']
                state['planned_stages'].insert(1, 'depth_maps')
            write_json(output / 'manifest.json', manifest)
            write_json(output / 'run.json', state)
        api = api or importlib.import_module('Metashape')
        state['software']['Metashape'] = str(api.app.version)
        if not str(api.app.version).startswith('2.2.'):
            raise RuntimeError('runner targets Metashape 2.2.x; validate adapter before using another version')
        if not api.app.activated:
            raise RuntimeError('Metashape license unavailable or checkout denied')
        if config.cpu_only:
            previous_gpu = api.app.gpu_mask
            previous_cpu = api.app.cpu_enable
            api.app.gpu_mask = 0
            api.app.cpu_enable = True
        doc = api.Document()
        if seed_state:
            source = Path(seed_checkpoint)
            shutil.copy2(source / 'project.psx', output / 'project.psx')
            shutil.copytree(source / 'project.files', output / 'project.files')
        if resume or seed_state:
            doc.open(str(output / 'project.psx'))
            if len(doc.chunks) != 1:
                raise ValueError('checkpoint must contain exactly one chunk')
            chunk = doc.chunks[0]
            expected = [str(safe_path(root, e['path'])) for e in manifest['images']]
            actual = [str(Path(c.photo.path).resolve()) for c in chunk.cameras]
            if actual != expected:
                raise ValueError('checkpoint camera manifest mismatch (use original root or start a fresh run)')
        else:
            chunk = doc.addChunk()
            chunk.addPhotos([str(safe_path(root, e['path'])) for e in manifest['images']])
            if image_quality_cache is not None:
                # Reuse per-image Image/Quality computed once on a parent run; verified by image SHA-256.
                state['image_quality_source'] = apply_image_quality_cache(chunk, manifest, image_quality_cache)
                write_json(output / 'run.json', state, replace=True)
            else:
                state['image_quality_source'] = {'method': 'analyzeImages'}
                try:
                    # Writes camera.meta['Image/Quality'] for cameras.csv; never fatal.
                    chunk.analyzeImages(chunk.cameras)
                except Exception as exc:
                    state['warnings'].append({'stage': 'image_analysis', 'field': 'image_quality',
                                              'missing_reason': f'analyzeImages failed: {type(exc).__name__}: {exc}'})
        completed = {s['stage'] for s in state['stages']}

        def checkpoint(stage, duration, operation_seconds=None):
            snapshot = {'stage': stage, 'recorded_at': now(), 'runtime_seconds': duration,
                        **collect_metrics(chunk, manifest['image_count'])}
            if operation_seconds is not None:
                snapshot['operation_seconds'] = operation_seconds
            metric_path = output / f'metrics-{stage}.json'
            write_json(metric_path, snapshot)
            state['artifacts'].append({'path': metric_path.name, 'sha256': sha256_file(metric_path),
                                       'size_bytes': metric_path.stat().st_size})
            state['stages'].append(snapshot)
            doc.save(str(output / 'project.psx'))
            state['project_files'] = project_fingerprints(output)
            state['status'] = 'checkpointed'
            write_json(output / 'run.json', state, replace=True)
            return snapshot

        def export(path, action):
            action()
            p = output / path
            if not p.is_file() or p.stat().st_size == 0:
                raise RuntimeError(f'empty/missing raw export: {path}')
            state['artifacts'].append({'path': path, 'sha256': sha256_file(p), 'size_bytes': p.stat().st_size})

        if seed_state:
            snapshot = checkpoint('initial_alignment', 0.0, {'inherited_alignment_runtime': seed_state['stages'][0]['runtime_seconds']})
            completed.add('initial_alignment')
        if 'initial_alignment' not in completed:
            t = time.perf_counter()
            chunk.matchPhotos(downscale=config.match_downscale, keypoint_limit=config.keypoint_limit,
                              tiepoint_limit=config.tiepoint_limit, generic_preselection=config.generic_preselection,
                              reference_preselection=config.reference_preselection)
            chunk.alignCameras()
            snapshot = checkpoint('initial_alignment', time.perf_counter() - t)
        else:
            snapshot = next(s for s in state['stages'] if s['stage'] == 'initial_alignment')
        if not any(a['path'] == 'cameras.csv' for a in state['artifacts']):
            # Before the alignment criteria check, so failed alignments keep their table too.
            write_camera_table(output / 'cameras.csv', collect_camera_table(chunk))
            p = output / 'cameras.csv'
            state['artifacts'].append({'path': p.name, 'sha256': sha256_file(p), 'size_bytes': p.stat().st_size})
            write_json(output / 'run.json', state, replace=True)
        metrics = snapshot['metrics']
        if (metrics['camera_count'] != manifest['image_count'] or
            metrics['aligned_camera_count'] < config.min_aligned_cameras or
            metrics['alignment_fraction'] < config.min_alignment_fraction or
            (metrics['valid_tie_point_count'] or 0) < config.min_valid_tie_points):
            raise RuntimeError('alignment failed configured camera/tie-point criteria')
        if not any(a['path'] == 'raw-tie-points.ply' for a in state['artifacts']):
            export('raw-tie-points.ply', lambda: chunk.exportPointCloud(
                str(output / 'raw-tie-points.ply'), source_data=api.TiePointsData,
                format=api.PointCloudFormatPLY))
        if config.profile in {'dense', 'mesh'} and 'dense_reconstruction' not in completed:
            t = time.perf_counter()
            depth_options = {} if protocol is None else {
                'max_neighbors': protocol.max_neighbors, 'workitem_size_cameras': protocol.workitem_size_cameras,
                'max_workgroup_size': protocol.max_workgroup_size, 'subdivide_task': True}
            if protocol and 'depth_maps' in completed:
                depth_seconds = 0.0  # Existing depth checkpoint reused, no rebuilding.
            else:
                chunk.buildDepthMaps(downscale=config.depth_downscale, filter_mode=getattr(api, config.depth_filter), **depth_options)
                depth_seconds = time.perf_counter() - t
                if protocol:
                    checkpoint("depth_maps", depth_seconds)
            cloud_started = time.perf_counter()
            chunk.buildPointCloud()
            cloud_seconds = time.perf_counter() - cloud_started
            if (getattr(chunk.point_cloud, "point_count", 0) or 0) <= 0:
                raise RuntimeError("dense reconstruction returned no points")
            export_started = time.perf_counter()
            export('raw-dense.ply', lambda: chunk.exportPointCloud(
                str(output / 'raw-dense.ply'), source_data=api.PointCloudData, format=api.PointCloudFormatPLY))
            checkpoint('dense_reconstruction', time.perf_counter() - t,
                       {'depth_maps': depth_seconds, 'point_cloud': cloud_seconds,
                        'dense_export': time.perf_counter() - export_started})
        if config.profile == 'mesh' and 'mesh_generation' not in completed:
            t = time.perf_counter()
            model_options = {} if protocol is None else {
                'face_count_custom': 0, 'trimming_radius': 0, 'build_texture': False,
                'workitem_size_cameras': protocol.workitem_size_cameras,
                'max_workgroup_size': protocol.max_workgroup_size}
            chunk.buildModel(source_data=api.DepthMapsData, surface_type=api.Arbitrary,
                             interpolation=api.EnabledInterpolation, face_count=api.CustomFaceCount if protocol else api.HighFaceCount, **model_options)
            if chunk.model is None or not len(chunk.model.faces):
                raise RuntimeError('mesh generation returned no faces')
            build_seconds = time.perf_counter() - t
            export_started = time.perf_counter()
            export('raw-mesh.obj', lambda: chunk.exportModel(str(output / 'raw-mesh.obj'),
                    format=api.ModelFormatOBJ, save_texture=False, save_normals=True, save_colors=False))
            checkpoint('mesh_generation', time.perf_counter() - t,
                       {'mesh_build': build_seconds, 'mesh_export': time.perf_counter() - export_started})
        state['status'] = 'complete'
        state['finished_at'] = now()
        state['runtime_seconds_this_invocation'] = time.perf_counter() - started
        write_json(output / 'run.json', state, replace=True)
        return state
    except Exception as exc:
        if state is not None:
            state['status'] = 'failed'
            state['finished_at'] = now()
            state['failures'].append({'type': type(exc).__name__, 'message': str(exc)})
            write_json(output / 'run.json', state, replace=True)
        raise
    finally:
        if previous_gpu is not None:
            api.app.gpu_mask = previous_gpu
            api.app.cpu_enable = previous_cpu
        lock.unlink(missing_ok=True)


def verify_seed(source, manifest, root):
    """Verify closed, complete alignment-only parent; never modify its files."""
    source = Path(source).resolve()
    if (source / '.run.lock').exists() or (source / 'project.files/lock').exists():
        raise ValueError('seed checkpoint is locked')
    state = read_json(source / 'run.json')
    parent = read_json(source / 'manifest.json')
    require_valid(parent, root)
    if state['status'] != 'complete' or [s['stage'] for s in state['stages']] != ['initial_alignment']:
        raise ValueError('seed must be a complete alignment-only checkpoint')
    if parent['experiment_id'] != state['experiment_id'] or parent['configuration'] != state['configuration']:
        raise ValueError('seed ledger identity/settings mismatch')
    if parent['images'] != manifest['images'] or parent['dataset_id'] != manifest['dataset_id']:
        raise ValueError('seed input provenance mismatch')
    for k in ['match_downscale', 'keypoint_limit', 'tiepoint_limit', 'generic_preselection', 'reference_preselection']:
        if parent['configuration'][k] != manifest['configuration'][k]:
            raise ValueError('seed alignment settings mismatch')
    if state['project_files'] != project_fingerprints(source):
        raise ValueError('seed project content mismatch')
    for a in state['artifacts']:
        if sha256_file(source / a['path']) != a['sha256']:
            raise ValueError('seed artifact mismatch')
    return state
