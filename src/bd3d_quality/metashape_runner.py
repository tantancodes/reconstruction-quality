"""Isolated Metashape 2.2 integration, raw artifacts and verified checkpoints."""
from __future__ import annotations
import importlib
import os
from pathlib import Path
import time
from uuid import uuid4
from .config import ReconstructionConfig
from .manifests import require_valid, safe_path
from .provenance import now, write_json, read_json, sha256_file, software_versions, digest
from .reconstruction_metrics import collect_metrics


def project_fingerprints(output):
    paths = [output / 'project.psx']
    folder = output / 'project.files'
    if folder.exists():
        # Metashape removes this transient process lock when Document closes.
        paths.extend(p for p in sorted(folder.rglob('*'))
                     if p.is_file() and p != folder / 'lock')
    return {p.relative_to(output).as_posix(): sha256_file(p) for p in paths}


def run_reconstruction(manifest, root, output, *, resume=False, api=None):
    require_valid(manifest, root)
    output = Path(output).resolve()
    config = ReconstructionConfig(**manifest['configuration'])
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
        if resume:
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
        completed = {s['stage'] for s in state['stages']}

        def checkpoint(stage, duration):
            snapshot = {'stage': stage, 'recorded_at': now(), 'runtime_seconds': duration,
                        **collect_metrics(chunk, manifest['image_count'])}
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

        if 'initial_alignment' not in completed:
            t = time.perf_counter()
            chunk.matchPhotos(downscale=config.match_downscale, keypoint_limit=config.keypoint_limit,
                              tiepoint_limit=config.tiepoint_limit, generic_preselection=config.generic_preselection,
                              reference_preselection=config.reference_preselection)
            chunk.alignCameras()
            snapshot = checkpoint('initial_alignment', time.perf_counter() - t)
        else:
            snapshot = next(s for s in state['stages'] if s['stage'] == 'initial_alignment')
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
            chunk.buildDepthMaps(downscale=config.depth_downscale, filter_mode=getattr(api, config.depth_filter))
            chunk.buildPointCloud()
            if (getattr(chunk.point_cloud, "point_count", 0) or 0) <= 0:
                raise RuntimeError("dense reconstruction returned no points")
            export('raw-dense.ply', lambda: chunk.exportPointCloud(
                str(output / 'raw-dense.ply'), source_data=api.PointCloudData, format=api.PointCloudFormatPLY))
            checkpoint('dense_reconstruction', time.perf_counter() - t)
        if config.profile == 'mesh' and 'mesh_generation' not in completed:
            t = time.perf_counter()
            chunk.buildModel(source_data=api.DepthMapsData, surface_type=api.Arbitrary,
                             interpolation=api.EnabledInterpolation, face_count=api.HighFaceCount)
            if chunk.model is None or not len(chunk.model.faces):
                raise RuntimeError('mesh generation returned no faces')
            export('raw-mesh.obj', lambda: chunk.exportModel(str(output / 'raw-mesh.obj'),
                    format=api.ModelFormatOBJ, save_texture=False, save_normals=True, save_colors=False))
            checkpoint('mesh_generation', time.perf_counter() - t)
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
