"""Rebuild the 16 historical FAST_DENSE ARF-dog subsets as current-format manifests.

Each historical manifest (Aphoma fork, experiments/manifests/**) lists filenames
and SHA-256s. We map every image to its index in the current verified 50-image
parent manifest, check the hash matches, and write a subset manifest with the
current configuration (default: config/alignment.json). The historical
experiment_id, manifest SHA-256 and run_id are kept in `selection` so cameras.csv
can be joined back to the geometric labels without guessing.

No Metashape. Usage:
  PYTHONPATH=src python3 scripts/historical_subset_manifests.py \
      --labels ~/Documents/ChatGPT/bd3d/analysis_outputs/fast_dense_geometry_dataset_v1/fast_dense_geometry_modeling_dataset_v1.csv \
      --parent results/arf-dog-50-manifest.json --root ~/Downloads/Jpg-processed \
      --output experiments/historical-fast-dense-16
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

from bd3d_quality.manifests import experiment_identity, require_valid
from bd3d_quality.provenance import digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', type=Path, required=True)
    ap.add_argument('--parent', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--config', type=Path, default=Path('config/alignment.json'))
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()

    parent = json.loads(a.parent.read_text())
    config = json.loads(a.config.read_text())
    index = {e['path']: i for i, e in enumerate(parent['images'])}
    rows = list(csv.DictReader(a.labels.open(newline='')))
    hist_root = a.labels.resolve().parents[2]  # .../bd3d (manifest_path is relative to it)
    a.output.mkdir(parents=True, exist_ok=False)
    plan = []
    for r in rows:
        mpath = hist_root / r['manifest_path']
        raw = mpath.read_bytes()
        if hashlib.sha256(raw).hexdigest() != r['manifest_sha256']:
            raise SystemExit(f'historical manifest hash mismatch: {mpath}')
        old = json.loads(raw)
        idx = []
        for im in old['images']:
            i = index[im['filename']]
            if parent['images'][i]['sha256'] != im['sha256']:
                raise SystemExit(f'image hash mismatch: {im["filename"]}')
            idx.append(i)
        if idx != sorted(idx):
            raise SystemExit(f'{old["experiment_id"]}: historical order differs from parent order')
        m = copy.deepcopy(parent)
        m['images'] = [parent['images'][i] for i in idx]
        m['image_count'] = len(idx)
        m['configuration'] = config
        m['selection'] = {
            'version': 1, 'strategy': 'historical', 'seed': old.get('random_seed'),
            'parent_experiment_id': parent['experiment_id'],
            'parent_image_list_signature': digest(parent['images']),
            'indices_zero_based': idx, 'order': 'parent manifest order (matches historical canonical order)',
            'historical_experiment_id': old['experiment_id'],
            'historical_manifest_sha256': r['manifest_sha256'],
            'historical_run_id': r['run_id'],
        }
        m['description'] = f'Historical FAST_DENSE subset {old["experiment_id"]} re-expressed for alignment-only rerun'
        m['experiment_id'] = experiment_identity(m)
        require_valid(m, a.root)
        name = old['experiment_id']
        (a.output / f'{name}.json').write_text(json.dumps(m, indent=2, sort_keys=True) + '\n')
        plan.append({'name': name, 'manifest': f'{name}.json', 'experiment_id': m['experiment_id'],
                     'historical_run_id': r['run_id'], 'image_count': len(idx)})
    (a.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(f'wrote {len(plan)} manifests to {a.output}')


if __name__ == '__main__':
    main()
