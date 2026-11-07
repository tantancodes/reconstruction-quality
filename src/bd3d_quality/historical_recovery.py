"""Recover historical JSON labels and join small caches; never reads image pixels."""
from __future__ import annotations

import itertools
import math
from pathlib import Path

from .historical_alignment import require, validate_run
from .provenance import digest, read_json, sha256_file
from .tabular_dataset import PRE_FEATURES


def bounded_json(path, limit=2_000_000):
    path = Path(path)
    require(path.stat().st_size <= limit, f'metadata exceeds byte budget: {path}')
    return read_json(path)


def cache_subset_features(images, features, matches):
    """Hash join and aggregate an existing complete, directional pair cache."""
    for cache in (features, matches):
        require(cache['settings_signature'] == digest(cache['settings']), 'cache settings mismatch')
    inputs = {(i['path'], i['sha256']): i for i in features['images']}
    orb = {i['path']: i for i in matches['images']}
    require(len(inputs) == len(features['images']), 'duplicate cached image')
    require(len(orb) == len(matches['images']), 'duplicate cached ORB image')
    selected = [(i['path'], i['sha256']) for i in images]
    require(len(set(selected)) == len(selected) >= 2, 'invalid selected image set')
    require(set(selected).issubset(inputs), 'manifest not hash-matched to image cache')
    for name, sha in selected:
        require(name in orb and orb[name]['sha256'] == sha, 'ORB image hash mismatch')
    pairs = {(p['path_a'], p['path_b']): p['ratio_match_count_a_to_b'] for p in matches['pairs']}
    require(len(pairs) == len(matches['pairs']), 'duplicate cached pair')
    require(set(pairs) == set(itertools.combinations(orb, 2)), 'incomplete/different-direction pair cache')
    selected_names = {name for name, _ in selected}
    ordered = [name for name in orb if name in selected_names]
    def mean(values):
        values = list(values)
        require(bool(values) and all(type(v) in (int, float) and math.isfinite(v) for v in values),
                'missing/nonfinite cached measurement')
        return sum(values) / len(values)
    result = dict(zip(PRE_FEATURES, [len(selected),
        *(mean(inputs[i][k] for i in selected) for k in
          ['brightness_mean', 'contrast_std', 'laplacian_variance', 'entropy_bits']),
        mean(orb[name]['orb_keypoint_count'] for name in ordered),
        mean(pairs[p] for p in itertools.combinations(ordered, 2))]))
    return result


def recover_legacy_report(path, features, matches, cache_provenance):
    """A source-declared manifest is checked, not invented for unlinked reports."""
    path = Path(path)
    report = bounded_json(path, 262_144)
    run, alignment = report['run'], report['alignment']
    experiment = report.get('experiment') or {}
    count = run['input_image_count']
    manifest_path = experiment.get('manifest_path')
    images, manifest_evidence, values, feature_source = [], None, None, None
    image_set = None
    if manifest_path:
        manifest = bounded_json(manifest_path, 262_144)
        manifest_hash = sha256_file(Path(manifest_path))
        require(manifest_hash == experiment.get('manifest_sha256'), 'report manifest hash mismatch')
        declared_counts = [manifest[k] for k in ('images_used_count', 'image_count') if k in manifest]
        require(bool(declared_counts) and all(n == count for n in declared_counts), 'manifest declared count mismatch')
        require(count == experiment['images_used_count'] == len(manifest['images']),
                'report/manifest count mismatch')
        require(experiment['artifact_id'] == manifest['artifact_id'], 'manifest artifact mismatch')
        require(experiment['experiment_id'] == manifest['experiment_id'], 'manifest experiment mismatch')
        images = [{'path':i['filename'], 'sha256':i['sha256']} for i in manifest['images']]
        image_set = 'sha256:' + digest(images)
        values = cache_subset_features(images, features, matches)
        manifest_evidence = {'manifest_path':manifest_path, 'manifest_sha256':manifest_hash,
            'method':'Report-declared manifest SHA-256 and filename/content-hash join to prior image and pair caches',
            'limit':'Original images were not read or rehashed this pass. Manifest declarations cannot independently prove bytes present at historical execution.'}
        feature_source = {'stage':'pre_alignment_images_only', 'source_image_set_id':image_set,
            **cache_provenance, 'source':'Aggregated prior scalar caches only; no image decoding or feature extraction'}
    value = (alignment.get('reprojection_error_px') or {}).get('rmse')
    stage = alignment.get('snapshot_stage')
    require(value is None or (type(value) in (int, float) and math.isfinite(value) and value >= 0),
            'invalid historical pixel RMSE')
    require(value is None or stage == 'post_error_reduction_pre_dense',
            'unreviewed legacy numeric measurement stage')
    artifact = experiment.get('artifact_id') or 'arf-dog'
    require(run['project_name'].startswith('ARF_Dog_'), 'unreviewed artifact attribution')
    settings = run.get('effective_settings')
    record = {'schema_version':1, 'kind':'historical_alignment_run',
        'record_id':'archival-report:'+report['run_id'], 'observation_unit':'alignment_run_report',
        'source_category':'local_archival_report', 'artifact_id':artifact, 'artifact_group_id':artifact,
        'acquisition_session_id':None, 'session_status':'unknown_shared_source_pool',
        'reconstruction_run_id':report['run_id'], 'run_id_status':'historical_report_uuid',
        'source_image_set_id':image_set, 'image_count':count, 'source_images':images,
        'image_provenance_status':'verified_hash_manifest' if manifest_path else 'missing',
        'image_match_evidence':manifest_evidence, 'image_features':values, 'feature_provenance':feature_source,
        'metashape_version':run['metashape_api_version'],
        'alignment_configuration':{'sparse_downscale':(settings or {}).get('sparse_downscale'), 'keypoint_limit':None,
            'tiepoint_limit':None, 'preselection':None, 'historical_effective_profile':settings},
        'configuration_status':'partially_reported',
        'experiment_metadata':experiment,
        'run_metadata':{k:run.get(k) for k in ('project_name','started_at_utc','finished_at_utc','sparse_quality')},
        'label':{'rmse_pixels':value, 'units':'pixels',
            'status':'computed_historical_verified' if value is not None else 'missing',
            'definition_id':'aphoma_legacy_valid_tie_point_observation_rmse_pixels',
            'stage':stage, 'stage_status':'explicit_historical_snapshot',
            'source':{'provider':'local_archival_report', 'file_path':str(path),
                'section':'alignment.reprojection_error_px.rmse; snapshot_stage='+str(stage),
                'page':None, 'evidence_sha256':sha256_file(path)}},
        'diagnostics':{'aligned_camera_count':alignment.get('aligned_cameras'),
            'tie_point_count':alignment.get('valid_tie_point_count')},
        'availability':{'report':'verified local JSON',
            'photographs':'manifest/cache hash-linked' if manifest_path else 'run-specific manifest unavailable',
            'feature_extraction':'cached scalars reused' if values else 'blocked by missing run/image association'},
        'field_evidence':{'artifact':'experiment manifest or explicitly named ARF Dog project; one related group',
            'settings':'Partial source effective profile; unreported settings are not inferred from present code'},
        'warnings':['Repeated runs and overlapping subsets of ARF Dog are not independent artifacts.',
            'A post-error-reduction label is not an initial-alignment label or physical-accuracy measurement.',
            'Missing RMSE on failed snapshots is retained as null, never zero.']}
    validate_run(record)
    audit = {'run_id':report['run_id'], 'source_path':str(path), 'source_sha256':sha256_file(path),
        'source_bytes':path.stat().st_size, 'image_count':count, 'label_stage':stage, 'rmse_pixels':value,
        'manifest_verified':bool(manifest_path), 'cached_features_joined':values is not None,
        'source_run_success':report.get('output_status', {}).get('success')}
    return record, audit


def initial_alignment_inventory(results):
    """Keep parent-inherited snapshots visible but never count them as new labels."""
    found = []
    for path in sorted(Path(results).glob('*/run.json')):
        report = bounded_json(path)
        for stage in report.get('stages', []):
            if stage['stage'] != 'initial_alignment':
                continue
            metrics_path = path.parent / 'metrics-initial_alignment.json'
            metric_doc = bounded_json(metrics_path)
            require(metric_doc == stage, 'embedded and standalone initial metrics differ')
            expected = next(a['sha256'] for a in report['artifacts'] if a['path'] == metrics_path.name)
            require(sha256_file(metrics_path) == expected, 'initial metric evidence hash mismatch')
            parent = report.get('continuation_of') or report.get('parent_checkpoint')
            found.append({'run_id':report['run_id'], 'report_path':str(path),
                'report_sha256':sha256_file(path), 'metric_path':str(metrics_path), 'metric_sha256':expected,
                'stage':'initial_alignment', 'rmse_pixels':stage['metrics']['reprojection_rmse_px'],
                'input_image_count':stage['metrics']['input_image_count'], 'dataset_id':report['dataset_id'],
                'alignment_configuration':report['configuration'], 'software':report['software']['Metashape'],
                'parent_reference':parent, 'new_alignment_event':not bool(parent),
                'run_status_ledger':report['status'],
                'limit':'Separate initial-stage inventory, not pooled with legacy post-refinement labels; derivative rasters form a separate stratum.'})
    return {'snapshots':found, 'distinct_initial_alignment_events':sum(r['new_alignment_event'] for r in found),
        'inherited_snapshots_not_new_labels':sum(not r['new_alignment_event'] for r in found),
        'artifact_group':'arf-dog', 'additional_independent_artifact_groups':0,
        'process_status':'Ledger status is historical metadata, not evidence of a live process.'}
