"""Candidate manifests for PAHMA 5-613 from the reference's confirmed series split.

Series are numbered by increasing elevation: 1 = circuit facing walls, 2 = 45 deg
circuit, 3 = near top-down. All candidates keep the parent's configuration
(mesh, Metal GPU, depth_downscale 8), so they differ only in image selection.

  drop_s3        all images except series 3
  drop_s2        all images except series 2
  random30_s<N>  30 random images, random.Random(N).sample over parent order

  PYTHONPATH=src python3 scripts/make_pahma_manifests.py \
      --parent experiments/pahma-5-613/parent-145.json --root ~/Documents/bd3d-data/PAHMA-5-613 \
      --series results/pahma-5-613-v1/series_split/series.csv --column series_pose \
      --seeds 20261110 20261111 --output experiments/pahma-5-613/candidates
"""
import argparse
import copy
import csv
import json
from pathlib import Path

from bd3d_quality.experiment_design import subset_manifest
from bd3d_quality.manifests import experiment_identity, require_valid
from bd3d_quality.provenance import digest


def drop_series(parent, series_of, drop, series_file, column):
    keep = [i for i, e in enumerate(parent['images']) if series_of[Path(e['path']).stem] != drop]
    m = copy.deepcopy(parent)
    m['images'] = [parent['images'][i] for i in keep]
    m['image_count'] = len(keep)
    m['selection'] = {'version': 1, 'strategy': f'drop_series_{drop}', 'seed': None,
                      'parent_experiment_id': parent['experiment_id'],
                      'parent_image_list_signature': digest(parent['images']),
                      'indices_zero_based': keep, 'series_file': str(series_file), 'series_column': column,
                      'order': 'parent manifest order (capture order)'}
    m['description'] = f'PAHMA 5-613: series {drop} removed ({len(keep)} images), mesh, Metal GPU, depth_downscale 8'
    m['experiment_id'] = experiment_identity(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parent', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--series', type=Path, required=True)
    ap.add_argument('--column', default='series_pose')
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    parent = json.loads(a.parent.read_text())
    series_of = {r['label']: int(r[a.column]) for r in csv.DictReader(open(a.series, newline=''))}
    if set(series_of) != {Path(e['path']).stem for e in parent['images']}:
        raise SystemExit('series file does not cover exactly the parent images (unaligned reference camera?)')
    out = {'drop_s3': drop_series(parent, series_of, 3, a.series, a.column),
           'drop_s2': drop_series(parent, series_of, 2, a.series, a.column)}
    for s in a.seeds:
        m = subset_manifest(parent, 30, 'random', seed=s)
        m['description'] = f'PAHMA 5-613: 30 random images (seed {s}), mesh, Metal GPU, depth_downscale 8'
        out[f'random30_s{s}'] = m
    a.output.mkdir(parents=True, exist_ok=False)
    for name, m in out.items():
        require_valid(m, a.root.expanduser())
        (a.output / f'{name}.json').write_text(json.dumps(m, indent=2, sort_keys=True) + '\n')
        print(name, m['image_count'], m['experiment_id'])


if __name__ == '__main__':
    main()
