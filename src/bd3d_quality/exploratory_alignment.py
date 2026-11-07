"""EXPLORATORY / NOT VALIDATED descriptive analysis; no fitting or inference."""
from __future__ import annotations

from collections import Counter, defaultdict
import base64
import csv
from html import escape
from pathlib import Path

import numpy as np

from .alignment_arrays import STATUS, TARGET, content_set_id, load_alignment_arrays
from .historical_alignment import compatibility_key, require
from .provenance import sha256_file, software_versions, write_json
from .tabular_dataset import PRE_FEATURES

SHORT_NAMES = ['Photo count', 'Brightness', 'Contrast', 'Laplacian variance',
               'Entropy', 'ORB keypoints', 'Pair ratio matches', 'Pixel RMSE']
AXIS_NAMES = ['Number of photographs', 'Mean grayscale brightness (0–255)',
              'Mean grayscale contrast (SD)', 'Mean Laplacian variance (intensity²)',
              'Mean image entropy (bits)', 'Mean ORB keypoint count',
              'Mean one-way ratio-match count', 'Reprojection RMSE (pixels)']


def summarize(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    result = {'count': len(finite), 'missing': len(values)-len(finite)}
    for name, value in [('min', np.min), ('q25', lambda a: np.quantile(a, .25)),
                        ('median', np.median), ('mean', np.mean),
                        ('q75', lambda a: np.quantile(a, .75)), ('max', np.max),
                        ('population_sd', np.std)]:
        result[name] = float(value(finite)) if len(finite) else None
    return result


def correlations(values, weights=None, *, rank=False):
    """Pairwise finite weighted Pearson, optionally weighted midrank Spearman.

    No p-values/intervals: observations are dependent. Constant columns yield
    null, including their diagonal, rather than an invented correlation.
    """
    values = np.asarray(values, dtype=float)
    weights = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    require(weights.shape == (len(values),) and np.isfinite(weights).all() and (weights > 0).all(),
            'positive finite row weights required')
    result = []; counts = []
    for i in range(values.shape[1]):
        row = []; ns = []
        for j in range(values.shape[1]):
            use = np.isfinite(values[:, i]) & np.isfinite(values[:, j])
            x, y, w = values[use, i].copy(), values[use, j].copy(), weights[use]
            ns.append(int(use.sum()))
            if len(x) < 2:
                row.append(None); continue
            if rank:
                def midrank(a):
                    _, inverse = np.unique(a, return_inverse=True)
                    mass = np.bincount(inverse, weights=w)
                    return (np.cumsum(mass)-mass/2)[inverse]
                x, y = midrank(x), midrank(y)
            x -= np.average(x, weights=w); y -= np.average(y, weights=w)
            denom = np.sqrt(np.dot(w, x*x)*np.dot(w, y*y))
            row.append(float(np.clip(np.dot(w, x*y)/denom, -1, 1)) if denom > 0 else None)
        result.append(row); counts.append(ns)
    return {'coefficients': result, 'pairwise_finite_counts': counts}


def dependence(records):
    groups = defaultdict(list)
    for r in records:
        groups[content_set_id(r)].append(r)
    require(None not in groups, 'exploratory rows require complete content-hash sets')
    entries = []
    for number, (group, rows) in enumerate(groups.items(), 1):
        labels = [r['label']['rmse_pixels'] for r in rows]
        vectors = [[r['image_features'][k] for k in PRE_FEATURES] for r in rows]
        entries.append({'group_label': f'G{number:02}', 'content_set_id': group,
            'image_count': rows[0]['image_count'], 'run_count': len(rows),
            'record_ids': [r['record_id'] for r in rows],
            'strata': sorted({compatibility_key(r) for r in rows}),
            'identical_cached_features_within_tolerance': bool(np.allclose(vectors, vectors[0], rtol=1e-12, atol=1e-10)),
            'rmse_summary': summarize(labels)})
    hashes = [{i['sha256'] for i in r['source_images']} for r in records]
    overlaps = []; adjacency = [set() for _ in records]
    for i in range(len(records)):
        for j in range(i+1, len(records)):
            shared = len(hashes[i] & hashes[j]); union = len(hashes[i] | hashes[j])
            if shared:
                adjacency[i].add(j); adjacency[j].add(i)
            overlaps.append({'a': records[i]['record_id'], 'b': records[j]['record_id'],
                'shared_images': shared, 'union_images': union, 'jaccard': shared/union,
                'same_content_set': content_set_id(records[i]) == content_set_id(records[j])})
    remaining = set(range(len(records))); components = []
    while remaining:
        pending = [min(remaining)]; component = set()
        while pending:
            n = pending.pop()
            if n in component: continue
            component.add(n); pending.extend(adjacency[n]-component)
        remaining -= component
        components.append([records[n]['record_id'] for n in sorted(component)])
    return {'status': STATUS, 'artifact_groups': sorted({r['artifact_group_id'] for r in records}),
        'unique_content_sets': len(groups), 'unique_source_photo_hashes': len(set().union(*hashes)),
        'repeated_content_set_groups': sum(len(r) > 1 for r in groups.values()),
        'rows_beyond_one_per_content_set': len(records)-len(groups), 'content_sets': entries,
        'overlap_pairs': overlaps, 'photo_overlap_connected_components': components,
        'limit': 'Distinct subsets, repeats and strata are not independent artifacts. No photo-disjoint split is implied.'}


def csv_file(path, rows):
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['analysis_status', *rows[0]])
        writer.writeheader()
        writer.writerows({'analysis_status': STATUS, **row} for row in rows)


def analyze(bundle, output):
    """Analyze metadata only into a new output directory. Never calls a baseline."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    data = load_alignment_arrays(bundle, mode='exploratory', artifact_group='arf-dog', dtype=np.float64)
    require(len(data) > 1, 'at least two complete exploratory ARF Dog pairs required')
    records = data.records
    output = Path(output)
    require(not output.resolve().is_relative_to(Path(bundle).resolve()), 'output cannot be inside historical bundle')
    output.mkdir(parents=True, exist_ok=False)
    values = np.column_stack([data.X, data.y])
    columns = [*PRE_FEATURES, TARGET]
    dep = dependence(records)
    counts = Counter(data.content_groups)
    weights = np.asarray([1/counts[g] for g in data.content_groups])
    corr = {'status': STATUS, 'columns': columns,
        'row_weighted_pearson': correlations(values), 'row_weighted_spearman': correlations(values, rank=True),
        'set_balanced_pearson': correlations(values, weights),
        'set_balanced_spearman': correlations(values, weights, rank=True),
        'weight_definition': 'Each image-content set has total weight 1; repeated runs share that weight.',
        'limit': 'Descriptive, pooled across partial profile strata; weighting does not create independence or validate causal effects.'}
    summary = {name: summarize(values[:, k]) for k, name in enumerate(columns)}
    missing = []
    all_records = data.source_records
    parent = np.array([[*((r.get('image_features') or {}).get(k) for k in PRE_FEATURES),
                        r['label']['rmse_pixels'] if r['label']['status'] in
                        {'document_verified', 'computed_historical_verified'} else None] for r in all_records], dtype=float)
    for scope, array in [('all_source_records', parent), ('selected_exploratory_pairs', values)]:
        for k, name in enumerate(columns):
            n = int((~np.isfinite(array[:, k])).sum())
            missing.append({'scope': scope, 'column': name, 'rows': len(array),
                            'missing_count': n, 'missing_fraction': n/len(array)})
    strata = sorted(set(data.strata)); stratum_labels = {s: f'S{i+1}' for i,s in enumerate(strata)}
    selected_rows = [{'record_id': r['record_id'], 'run_id': r['reconstruction_run_id'],
        'artifact_group': r['artifact_group_id'], 'content_set_id': content_set_id(r),
        'source_image_set_id': r['source_image_set_id'], 'stratum': compatibility_key(r),
        'stratum_label': stratum_labels[compatibility_key(r)], 'strict_ready': bool(data.strict_ready_mask[i]),
        **dict(zip(columns, values[i].tolist()))} for i,r in enumerate(records)]
    excluded = [{'record_id': r['record_id'], 'artifact_group': r['artifact_group_id'],
        'missing_predictors': [k for k in PRE_FEATURES if (r.get('image_features') or {}).get(k) is None],
        'verified_label_available': r['label']['status'] in {'document_verified', 'computed_historical_verified'},
        'reason': 'Outside artifact scope and/or incomplete verified label/image-only predictor pair'}
        for i,r in enumerate(all_records) if i not in set(data.source_indices)]
    provenance = {'status': STATUS, 'selected_rows': len(data), 'source_records': len(all_records),
        'strict_ready_selected': int(data.strict_ready_mask.sum()),
        'artifact_groups': sorted(set(data.artifact_groups)),
        'selected_missing_acquisition_session_ids': sum(r.get('acquisition_session_id') is None for r in records),
        'selected_missing_alignment_fields': {k: sum(r['alignment_configuration'].get(k) is None for r in records)
                                              for k in ['keypoint_limit', 'tiepoint_limit', 'preselection']},
        'configuration_status_counts': dict(Counter(r['configuration_status'] for r in records)),
        'label_stage_counts': dict(Counter(r['label']['stage'] for r in records)),
        'software_version_counts': dict(Counter(r['metashape_version'] for r in records)),
        'strata': [{'label': stratum_labels[s], 'key': s, 'rows': data.strata.count(s),
            'content_sets': len({g for g,t in zip(data.content_groups, data.strata) if t == s}),
            'configuration': next(r['alignment_configuration'] for r in records if compatibility_key(r) == s)} for s in strata],
        'excluded_rows': excluded,
        'limits': ['All selected records describe one ARF Dog artifact and overlapping photo subsets.',
            'Report-declared manifest and cached hashes are linked; historical execution bytes were not independently proven or rehashed here.',
            'Acquisition sessions are unknown; alignment/optimization configuration is partial.',
            'Labels are post-error-reduction pixel residuals, not initial-alignment error or physical mesh accuracy.',
            'Profile keys include downstream settings; pooled EDA does not establish settings equivalence.',
            'No preprocessing, model, validation split, confidence interval or significance test was fitted.']}
    write_json(output/'summary.json', {'status': STATUS, 'columns': summary})
    write_json(output/'correlations.json', corr)
    write_json(output/'dependence.json', dep)
    write_json(output/'provenance-and-selection.json', provenance)
    write_json(output/'missingness.json', {'status': STATUS, 'fields': missing,
        'selected_metadata': {'acquisition_session_id_missing': provenance['selected_missing_acquisition_session_ids'],
                              'missing_alignment_fields': provenance['selected_missing_alignment_fields'],
                              'configuration_status': provenance['configuration_status_counts']},
        'imputation_performed': False})
    csv_file(output/'selected-pairs.csv', selected_rows)
    csv_file(output/'missingness.csv', missing)
    csv_file(output/'summary.csv', [{'column': k, **v} for k,v in summary.items()])
    csv_file(output/'correlations.csv', [{'method': method, 'a': a, 'b': b,
        'coefficient': corr[method]['coefficients'][i][j], 'finite_rows': corr[method]['pairwise_finite_counts'][i][j]}
        for method in ['row_weighted_pearson', 'row_weighted_spearman', 'set_balanced_pearson', 'set_balanced_spearman']
        for i,a in enumerate(columns) for j,b in enumerate(columns)])
    np.savez_compressed(output/'exploratory-arrays.npz', X_pre=data.X.astype(np.float32),
        y_rmse_pixels=data.y.astype(np.float32), X_pre_mask=data.X_mask, y_rmse_pixels_mask=data.y_mask,
        strict_ready_mask=data.strict_ready_mask, analysis_status=np.asarray(STATUS),
        feature_columns=np.asarray(PRE_FEATURES), target_column=np.asarray(TARGET))

    figures = []
    def save(fig, name, title):
        fig.suptitle(STATUS+'\n'+title, y=.975, va='top', fontsize=14, fontweight='bold')
        fig.text(.5, .015, f'{len(data)} dependent runs • one artifact • descriptive only • no model fitted', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .90))
        fig.savefig(output/(name+'.png'), dpi=140)
        fig.savefig(output/(name+'.pdf'), metadata={'Title': STATUS+' — '+title, 'CreationDate': None, 'ModDate': None})
        plt.close(fig); figures.append((name, title))

    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for k, ax in enumerate(axes.flat):
        ax.hist(values[:, k], bins='auto', color='#397c9e', edgecolor='white')
        ax.set(xlabel=AXIS_NAMES[k], ylabel='Run count', title=SHORT_NAMES[k])
        ax.ticklabel_format(axis='x', style='plain', useOffset=False)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    save(fig, 'feature-distributions', 'Seven image-only predictors and pixel-RMSE distribution')

    colors = ['#0072b2', '#d55e00', '#009e73', '#cc79a7', '#e69f00']
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for k, ax in enumerate(axes.flat):
        if k == 7:
            ax.axis('off'); ax.text(.05, .5, 'Colors: partial historical profile strata\nPoints may overlap; no jitter or fitted line.\nNo causal or generalization claim.', va='center')
            continue
        for j,s in enumerate(strata):
            use = np.asarray(data.strata) == s
            ax.scatter(data.X[use, k], data.y[use, 0], color=colors[j % len(colors)],
                       marker=['o','s','^'][j % 3], label=f'{stratum_labels[s]} (n={int(use.sum())})',
                       s=40-5*j, alpha=.8)
        ax.set(xlabel=AXIS_NAMES[k], ylabel='Reprojection RMSE (pixels)', title=SHORT_NAMES[k])
        ax.ticklabel_format(axis='both', style='plain', useOffset=False)
    axes.flat[0].legend(fontsize=8)
    save(fig, 'features-versus-rmse', 'Feature versus reported RMSE; partial strata shown separately')

    for method in ['pearson', 'spearman']:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        for ax, weighting in zip(axes, ['row_weighted', 'set_balanced']):
            matrix = np.array(corr[weighting+'_'+method]['coefficients'], dtype=float)
            im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdBu_r')
            ax.set(xticks=range(8), yticks=range(8), xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
                   title=weighting.replace('_', ' ').title())
            plt.setp(ax.get_xticklabels(), rotation=55, ha='right', fontsize=9)
            ax.tick_params(axis='y', labelsize=9)
            for i in range(8):
                for j in range(8):
                    v = matrix[i, j]
                    ax.text(j, i, f'{v:.2f}' if np.isfinite(v) else 'NA', ha='center', va='center',
                            fontsize=9, color='white' if abs(v) > .65 else 'black')
            fig.colorbar(im, ax=ax, fraction=.045, pad=.04)
        save(fig, 'correlation-'+method, method.title()+' correlations; no p-values or independence assumption')

    fig, ax = plt.subplots(figsize=(12, 7))
    positions = np.arange(8)
    for shift, scope, color in [(-.18, 'all_source_records', '#8c8c8c'), (.18, 'selected_exploratory_pairs', '#0072b2')]:
        rows = [r for r in missing if r['scope'] == scope]
        counts_missing = [r['missing_count'] for r in rows]
        ax.barh(positions+shift, counts_missing, height=.34, color=color, label=f'{scope} (n={rows[0]["rows"]})')
        for y,n in zip(positions+shift, counts_missing): ax.text(n+.08, y, str(n), va='center', fontsize=9)
    ax.set(yticks=positions, yticklabels=SHORT_NAMES, xlabel='Missing values (not imputed)', xlim=(0, max(1, max(r['missing_count'] for r in missing))+2))
    ax.legend(loc='lower right')
    save(fig, 'missingness', 'Complete-pair selection excludes missing values; it does not repair provenance')

    group_rows = [next(r for r in records if content_set_id(r) == entry['content_set_id']) for entry in dep['content_sets']]
    sets = [{i['sha256'] for i in r['source_images']} for r in group_rows]
    overlap = np.array([[len(a & b)/len(a | b) for b in sets] for a in sets])
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(overlap, vmin=0, vmax=1, cmap='Blues')
    labels = [f'{g["group_label"]} n={g["image_count"]} r={g["run_count"]}' for g in dep['content_sets']]
    ax.set(xticks=range(len(sets)), yticks=range(len(sets)), xticklabels=labels, yticklabels=labels)
    plt.setp(ax.get_xticklabels(), rotation=70, ha='right', fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    fig.colorbar(im, ax=ax, label='Photo-content Jaccard overlap')
    save(fig, 'photo-overlap', f'{len(sets)} distinct content sets; n=photos, r=runs; all from one artifact')

    html = ['<!doctype html><meta charset="utf-8"><title>'+STATUS+'</title>',
        '<style>body{font:16px system-ui;max-width:1300px;margin:32px auto;padding:0 24px;color:#192b36}img{width:100%}h1{color:#9b350b}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}</style>',
        '<h1>'+STATUS+'</h1><h2>ARF Dog descriptive analysis</h2>',
        f'<p>{len(data)} runs; {dep["unique_content_sets"]} distinct image sets; one artifact; strict-ready selected rows: {int(data.strict_ready_mask.sum())}.</p>',
        '<p>No baseline, Ridge or PyTorch model was fitted. These are dependent, selected complete cases; correlations do not establish causation, physical accuracy or held-out-artifact generalization.</p>',
        '<ul>'+''.join('<li>'+escape(s)+'</li>' for s in provenance['limits'])+'</ul>',
        '<p>Numeric outputs: <a href="summary.csv">summary</a> · <a href="correlations.csv">correlations</a> · <a href="missingness.csv">missingness</a> · <a href="selected-pairs.csv">selected rows</a> · <a href="dependence.json">repeat/overlap audit</a>.</p>']
    for name, title in figures:
        encoded = base64.b64encode((output/(name+'.png')).read_bytes()).decode('ascii')
        html.append('<h2>'+escape(title)+'</h2><img alt="'+escape(STATUS+' — '+title)+'" src="data:image/png;base64,'+encoded+'"><p><a href="'+name+'.pdf">PDF export</a></p>')
    (output/'report.html').write_text('\n'.join(html), encoding='utf-8')
    write_json(output/'analysis-manifest.json', {'status': STATUS, 'mode': data.mode,
        'source_bundle': str(Path(bundle).resolve()), 'source_hashes': data.source_hashes,
        'source_indices': data.source_indices.tolist(), 'feature_columns': PRE_FEATURES, 'target': TARGET,
        'dtype_for_analysis': 'float64 source JSON values', 'dtype_for_export': 'float32',
        'models_fitted': False, 'strict_gate_modified': False, 'original_photos_read': False,
        'software': {**software_versions(), 'matplotlib': matplotlib.__version__},
        'analysis_source_sha256': sha256_file(Path(__file__)),
        'loader_source_sha256': sha256_file(Path(__file__).with_name('alignment_arrays.py'))})
    write_json(output/'file-hashes.json', {'status': STATUS, 'files': {p.name: sha256_file(p)
        for p in sorted(output.iterdir()) if p.is_file()}})
    return {'status': STATUS, 'rows': len(data), 'unique_content_sets': dep['unique_content_sets'],
            'strict_ready': int(data.strict_ready_mask.sum()), 'output': str(output)}
