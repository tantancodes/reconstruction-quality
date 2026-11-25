"""Public data for the web viewer: camera tables and measured labels of the 28 labelled builds.

Writes viewer/public/ from files already in git, so the viewer can be hosted without any restricted
asset (no photos, meshes, dense clouds or Metashape projects):

  public/cameras/<capture>/<run>.csv   copies of the tracked cameras.csv files (builds and references)
  public/datasets.json                 one entry per build: object, capture group, reference, and the
                                       v3 label with its QC and 1%-threshold accuracy/completeness
  public/predictions.json              the saved out-of-fold predictions of the set models (seed means per
                                       build), with per-split MAE checked against the saved summary.csv and
                                       Spearman rank correlation inside each held-out fold

The builds and labels are read with scripts/pilot_set_models.load_runs, the same loader the models
were trained and evaluated with, so the viewer shows exactly the experiments' labels.

  python scripts/export_viewer_public.py
"""
import csv
import json
from collections import defaultdict
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import numpy as np  # noqa: E402
import pilot_set_models as pilot  # noqa: E402

OUT = REPO / 'viewer' / 'public'
COFFIN_LABELS = REPO / 'results/pahma-5-613-v2/labels-all.csv'
# Reference build of each capture group (the build every label in the group is measured against).
REFERENCES = {
    'arf-dog/hist16-cpu': REPO / 'results/hist16-old-alignment/_reference/cameras.csv',
    'arf-dog/gpu-d8': REPO / 'results/gpu-pilot-d8/ref50/cameras.csv',
    'pahma-5-613/gpu-d8': REPO / 'results/pahma-5-613-v2/ref145b/cameras.csv',
}
OBJECTS = {'arf-dog': 'ARF dog', 'pahma-5-613': 'PAHMA 5-613 coffin'}
LABEL_FILES = [  # (file, run-id column) holding qc_v3 and the 1% accuracy/completeness of each build
    (REPO / 'results/geometry-labels-v3/labels_v3.csv', 'experiment_id'),
    (REPO / 'results/gpu-pilot-d8-ext/labels.csv', 'run'),
    (REPO / 'results/gpu-pilot-d8/geometry-v3/labels_v3.csv', 'run'),
    (COFFIN_LABELS, 'run'),
]


def rel(p):
    return str(Path(p).resolve().relative_to(REPO))


def label_rows():
    rows = {}
    for path, key in LABEL_FILES:
        if path.is_file():
            with open(path, newline='') as f:
                for r in csv.DictReader(f):
                    rows.setdefault(r[key], {**r, '_source': rel(path)})
    return rows


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def commit():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# Saved out-of-fold predictions: (id, predictions, summary, producer, description of the splits).
PREDICTIONS = [
    ('pilot', REPO / 'results/pilot-real-v3-all/predictions.csv', REPO / 'results/pilot-real-v3-all/summary.csv',
     'scripts/pilot_set_models.py'),
    ('geometry-aware', REPO / 'experiments/geometry-aware/results/real/predictions.csv',
     REPO / 'experiments/geometry-aware/results/real/summary.csv', 'experiments/geometry-aware/run_experiments.py'),
]
SPLITS = {
    'LORO': ('leave one build out', 'Builds from one capture share photos, so a near-duplicate of the held-out build is in '
             'training: optimistic, not a test of generalisation.'),
    'LOGO': ('leave one capture group out', '3 folds: dog CPU pipeline (16), dog GPU pipeline (7), coffin (5).'),
    'LOAO': ('leave one object out', '2 folds: dog (23 builds) and coffin (5). The only test on an unseen object.'),
}


def ranks(x):
    """Average ranks (ties share their mean rank), for Spearman's rho without scipy."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind='mergesort')
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def predictions(dataset_ids):
    out = {'splits': {k: {'name': n, 'note': t} for k, (n, t) in SPLITS.items()}, 'sources': []}
    for sid, path, summary_path, producer in PREDICTIONS:
        if not path.is_file():
            print(f'skip {sid}: {rel(path) if path.is_relative_to(REPO) else path} not found')
            continue
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r['capture'] = r.get('group') or f"{r['artifact']}/{r['source']}"
            r['artifact'] = r['capture'].split('/')[0]
        per_seed, per_run = defaultdict(list), defaultdict(list)
        for r in rows:
            per_seed[(r['scheme'], r['model'], r['seed'])].append(abs(float(r['predicted']) - float(r['actual'])))
            per_run[(r['scheme'], r['model'], r['capture'], r['run'], r['held_out'])].append((float(r['predicted']), float(r['actual'])))
        builds = []
        for (scheme, model, capture, run, held), v in per_run.items():
            p = np.array([x for x, _ in v])
            builds.append({'scheme': scheme, 'model': model, 'dataset': f'{capture}/{run}', 'capture': capture, 'run': run,
                           'held_out': held, 'measured': v[0][1], 'predicted': float(p.mean()),
                           'predicted_sd': float(p.std(ddof=1)) if len(p) > 1 else 0.0, 'seeds': len(p)})
            assert f'{capture}/{run}' in dataset_ids, f'{capture}/{run} has a prediction but no dataset entry'
        with open(summary_path, newline='') as f:
            saved = {(r['scheme'], r['model']): float(r['mae_mean']) for r in csv.DictReader(f)}
        summary = []
        for scheme, model in sorted({(k[0], k[1]) for k in per_seed}):
            maes = np.array([np.mean(v) for (s_, m_, _), v in per_seed.items() if s_ == scheme and m_ == model])
            if (scheme, model) in saved:   # the viewer must show exactly the saved results
                assert abs(maes.mean() - saved[(scheme, model)]) < 1e-9, (sid, scheme, model, maes.mean(), saved[(scheme, model)])
            # rank correlation inside each held-out fold (LORO: inside each object), on seed-mean predictions
            within = {}
            B = [b for b in builds if b['scheme'] == scheme and b['model'] == model]
            fold = (lambda b: b['capture'].split('/')[0]) if scheme == 'LORO' else (lambda b: b['held_out'])
            for g in sorted({fold(b) for b in B}):
                sel = [b for b in B if fold(b) == g]
                within[g] = {'n': len(sel), 'rho': spearman([b['predicted'] for b in sel], [b['measured'] for b in sel])}
            summary.append({'scheme': scheme, 'model': model, 'seeds': len(maes), 'mae_mean': float(maes.mean()),
                            'mae_sd': float(maes.std(ddof=1)) if len(maes) > 1 else 0.0, 'within': within})
        out['sources'].append({'id': sid, 'predictions': rel(path), 'summary': rel(summary_path), 'produced_by': producer,
                               'builds': builds, 'metrics': summary})
    return out


def main():
    if OUT.exists():
        shutil.rmtree(OUT / 'cameras', ignore_errors=True)
    (OUT / 'cameras').mkdir(parents=True, exist_ok=True)
    extra = label_rows()
    datasets = []

    def copy(src, capture, run):
        dest = OUT / 'cameras' / capture.replace('/', '_') / f'{run}.csv'
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        return str(dest.relative_to(OUT.parent))

    for capture, ref in REFERENCES.items():
        artifact = capture.split('/')[0]
        datasets.append({'id': f'{capture}/{ref.parent.name}', 'object': OBJECTS[artifact], 'artifact': artifact,
                         'capture': capture, 'run': ref.parent.name, 'role': 'reference',
                         'cameras': copy(ref, capture, ref.parent.name), 'cameras_source': rel(ref)})
    for r in pilot.load_runs(COFFIN_LABELS):
        capture = f"{r['artifact']}/{r['source']}"
        e = extra.get(r['run'], {})
        datasets.append({
            'id': f"{capture}/{r['run']}", 'object': OBJECTS[r['artifact']], 'artifact': r['artifact'],
            'capture': capture, 'run': r['run'], 'role': 'build',
            'cameras': copy(r['cameras'], capture, r['run']), 'cameras_source': rel(r['cameras']),
            'reference': f'{capture}/{REFERENCES[capture].parent.name}',
            'label': r['label'], 'qc': e.get('qc_v3') or None,
            'completeness_at_1pct': num(e.get('completeness_at_0.01')), 'accuracy_at_1pct': num(e.get('accuracy_at_0.01')),
            'reprojection_px': r['reproj'], 'label_source': e.get('_source'),
        })
    manifest = {
        'generated_by': 'scripts/export_viewer_public.py', 'commit': commit(),
        'label': 'v3 label: symmetric surface RMSE between the build mesh and its reference mesh after camera-initialised '
                 'similarity ICP, divided by the reference bounding-box diagonal (docs/METHOD.md). Lower is better.',
        'restricted': 'Photos, meshes, dense clouds and Metashape projects are not public. Open them locally in the '
                      'viewer (File > Open folder); they are read in the browser tab and never uploaded.',
        'datasets': datasets,
    }
    (OUT / 'datasets.json').write_text(json.dumps(manifest, indent=1))
    preds = predictions({d['id'] for d in datasets})
    preds['generated_by'], preds['commit'] = 'scripts/export_viewer_public.py', manifest['commit']
    (OUT / 'predictions.json').write_text(json.dumps(preds, separators=(',', ':')))
    for src in preds['sources']:
        print(f"predictions {src['id']}: {len(src['builds'])} build rows, MAE matches {src['summary']}")
    n = sum(d['role'] == 'build' for d in datasets)
    print(f'{n} builds and {len(REFERENCES)} references -> {OUT.relative_to(REPO)}')


if __name__ == '__main__':
    main()
