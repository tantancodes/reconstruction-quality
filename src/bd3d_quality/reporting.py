"""One row per condition identity; explicit joins and compatibility checks."""
import csv
from pathlib import Path
import numpy as np
from .manifests import require_valid
from .provenance import digest, now, read_json, sha256_file, write_json


def summarize(manifest, *, features=None, matches=None, run=None, geometry=None, sources=None):
    require_valid(manifest)
    eid = manifest['experiment_id']
    for record, kind in [(features, 'image_features'), (matches, 'local_matches'), (run, 'reconstruction')]:
        if record and (record.get('experiment_id') != eid or record.get('kind') != kind):
            raise ValueError(f'{kind} provenance mismatch')
    for record in [features, matches, geometry]:
        if record and record.get('settings_signature') != digest(record['settings']):
            raise ValueError('analysis settings signature mismatch')
    if run and run.get('execution_protocol') != manifest.get('execution_protocol'):
        raise ValueError('run execution protocol mismatch')
    if run and run['configuration'] != manifest['configuration']:
        raise ValueError('run configuration mismatch')
    if geometry:
        if not run or geometry['candidate']['sha256'] not in [a['sha256'] for a in run['artifacts'] if a['path'].endswith('.obj')]:
            raise ValueError('geometry candidate is not a recorded raw mesh artifact')
    row = {'experiment_id': eid, 'dataset_id': manifest['dataset_id'],
           'configuration_signature': digest(manifest['configuration']) if 'execution_protocol' not in manifest else
               digest({'configuration': manifest['configuration'], 'execution_protocol': manifest['execution_protocol']}),
           'profile': manifest['configuration']['profile'], 'image_count': manifest['image_count'],
           'status': run['status'] if run else 'not_reconstructed', 'run_id': run['run_id'] if run else None}
    if features:
        expected = {e['path']: e['sha256'] for e in manifest['images']}
        actual = {e['path']: e['sha256'] for e in features['images']}
        if actual != expected or len(features['images']) != manifest['image_count']:
            raise ValueError('feature image provenance mismatch')
        row['image_settings_signature'] = features['settings_signature']
        row['image_software_signature'] = digest(features['software'])
        for key in ['brightness_mean', 'contrast_std', 'laplacian_variance', 'entropy_bits', 'width_px', 'height_px', 'size_bytes', 'aspect_ratio']:
            values = [e[key] for e in features['images'] if e.get(key) is not None]
            row[f'image_{key}_mean'] = float(np.mean(values)) if values else None
            row[f'image_{key}_measured_count'] = len(values)
    if matches:
        expected = {e['path']: e['sha256'] for e in manifest['images']}
        if {e['path']: e['sha256'] for e in matches['images']} != expected or len(matches['images']) != len(expected):
            raise ValueError('match image provenance mismatch')
        pairs = [(p['path_a'], p['path_b']) for p in matches['pairs']]
        from itertools import combinations
        if set(pairs) != set(combinations(expected, 2)) or len(set(pairs)) != len(pairs):
            raise ValueError('match pairs incomplete or duplicated')
        row['match_software_signature'] = digest(matches['software'])
        row['match_settings_signature'] = matches['settings_signature']
        counts = [p['ratio_match_count_a_to_b'] for p in matches['pairs']]
        row['pair_count'] = len(counts)
        row['pair_ratio_match_count_mean'] = float(np.mean(counts)) if counts else None
    if run:
        row['reconstruction_software_signature'] = digest(run['software'])
        seen = set()
        for stage in run['stages']:
            label = stage['stage']
            if label in seen:
                raise ValueError('duplicate stage measurement')
            seen.add(label)
            for key, value in stage['metrics'].items():
                row[f'{label}__{key}'] = value
            row[f'{label}__runtime_seconds'] = stage['runtime_seconds']
    if geometry:
        row.update({'geometry_status': geometry['status'], 'reference_id': geometry['reference_id'],
                    'reference_sha256': geometry['reference']['sha256'],
                    'geometry_settings_signature': geometry['settings_signature'], 'geometry_units': geometry['units'],
                    'geometry_software_signature': digest(geometry['software']),
                    'registration_fitness': geometry['registration']['fitness'],
                    'geometry_chamfer_mean_squared': geometry['metrics']['chamfer_mean_squared'] if geometry['metrics'] else None})
    return {'schema_version': 1, 'kind': 'experiment_summary', 'created_at': now(),
            'experimental_unit': 'one input-and-configuration condition; one selected reconstruction run',
            'row': row, 'manifest': manifest, 'features': features, 'matches': matches,
            'run': run, 'geometry': geometry, 'source_files': sources or [],
            'missing_value_semantics': 'JSON null / CSV empty = unavailable or unmeasured; zero is measured zero'}


def flatten_summary(summary):
    if summary.get('schema_version') != 1 or summary.get('kind') != 'experiment_summary':
        raise ValueError('unsupported summary schema')
    recomputed = summarize(summary['manifest'], features=summary.get('features'),
        matches=summary.get('matches'), run=summary.get('run'), geometry=summary.get('geometry'))
    if summary['row'] != recomputed['row']:
        raise ValueError('summary row inconsistent with source records')
    return dict(summary['row'])


def build_report(summary_paths, output):
    rows, seen, compatibility = [], set(), {}
    sources = []
    for path in summary_paths:
        row = flatten_summary(read_json(path))
        eid = row['experiment_id']
        if eid in seen:
            raise ValueError(f'duplicate experimental condition: {eid}; select one run explicitly')
        seen.add(eid)
        for key in ['configuration_signature', 'image_settings_signature', 'match_settings_signature',
                    'geometry_settings_signature', 'reference_sha256', 'geometry_units',
                    'image_software_signature', 'match_software_signature', 'reconstruction_software_signature',
                    'geometry_software_signature']:
            value = row.get(key)
            if value is not None:
                if key in compatibility and compatibility[key] != value:
                    raise ValueError(f'incompatible {key}; build separate comparison tables')
                compatibility[key] = value
        sources.append({'filename': Path(path).name, 'sha256': sha256_file(Path(path)), 'experiment_id': eid})
        rows.append(row)
    if not rows:
        raise ValueError('at least one summary required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (output / 'experiments.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'experiments.json', {'schema_version': 1, 'rows': rows, 'sources': sources})
    write_json(output / 'validation.json', {'valid': True, 'row_count': len(rows), 'unique_conditions': len(seen),
            'compatibility': compatibility, 'status_counts': {s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows})}})
    write_json(output / 'feature_dictionary.json', {'missing': 'null/empty means unavailable',
        'stage_separator': '__', 'fields': {f: field_description(f) for f in fields}})
    return rows


def field_description(name):
    if '__' in name:
        stage, metric = name.split('__', 1)
        return f'{metric} measured at {stage}; see docs/FEATURE_DICTIONARY.md for units and definitions'
    return {'experiment_id': 'SHA-256 of ordered inputs, dataset ID, effective configuration and schema',
            'geometry_chamfer_mean_squared': 'Half the sum of directional mean squared nearest-sample distances; units squared',
            'status': 'Selected run status or not_reconstructed', 'image_count': 'Declared, validated manifest image count'}.get(name,
            f'{name}; see docs/FEATURE_DICTIONARY.md')
