"""Candidate manifests for PAHMA 5-613 from the reference's confirmed series split.

Series are numbered by increasing elevation: 1 = circuit facing walls, 2 = 45 deg
circuit, 3 = near top-down. All candidates keep the parent's configuration
(mesh, Metal GPU, depth_downscale 8), so they differ only in image selection.

  drop_s3        all images except series 3
  drop_s2        all images except series 2
  random30_s<N>  30 random images, random.Random(N).sample over parent order
  side40_s1      40% of series 1 removed as one contiguous azimuth sector: series-1 cameras
                 sorted by azimuth_deg (series.csv), round(0.4*n1) consecutive ones removed
                 starting at sorted position round(0.3*n1). Series 2 and 3 kept.
  random60_s<N>  60 random images (--random60-seed)

  PYTHONPATH=src python3 scripts/make_pahma_manifests.py \
      --parent experiments/pahma-5-613/parent-145.json --root <bd3d-data>/PAHMA-5-613 \
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


def side_removed_series1(parent, rows, column, fraction, series_file):
    s1 = sorted((r for r in rows if int(r[column]) == 1), key=lambda r: float(r['azimuth_deg']))
    n1 = len(s1)
    k, start = round(fraction * n1), round(0.3 * n1)
    removed_labels = {r['label'] for r in s1[start:start + k]}
    keep = [i for i, e in enumerate(parent['images']) if Path(e['path']).stem not in removed_labels]
    removed = [i for i, e in enumerate(parent['images']) if Path(e['path']).stem in removed_labels]
    az = [float(r['azimuth_deg']) for r in s1[start:start + k]]
    m = copy.deepcopy(parent)
    m['images'] = [parent['images'][i] for i in keep]
    m['image_count'] = len(keep)
    m['selection'] = {'version': 1, 'strategy': 'series1_contiguous_side_removed', 'seed': None,
                      'parent_experiment_id': parent['experiment_id'],
                      'parent_image_list_signature': digest(parent['images']),
                      'indices_zero_based': keep, 'removed_indices_zero_based': removed,
                      'rule': f'series-1 cameras sorted by azimuth; {k} of {n1} consecutive removed from sorted '
                              f'position {start} (azimuth {min(az):.1f}..{max(az):.1f} deg in reference pose frame)',
                      'series_file': str(series_file), 'series_column': column,
                      'order': 'parent manifest order (capture order)'}
    m['description'] = (f'PAHMA 5-613: {k}/{n1} series-1 images removed as one azimuth sector '
                        f'({len(keep)} images), mesh, Metal GPU, depth_downscale 8')
    m['experiment_id'] = experiment_identity(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parent', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--series', type=Path, required=True)
    ap.add_argument('--column', default='series_pose')
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--side-fraction', type=float, default=0.4)
    ap.add_argument('--random60-seed', type=int, default=20261112)
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
    rows = list(csv.DictReader(open(a.series, newline='')))
    out['side40_s1'] = side_removed_series1(parent, rows, a.column, a.side_fraction, a.series)
    m = subset_manifest(parent, 60, 'random', seed=a.random60_seed)
    m['description'] = f'PAHMA 5-613: 60 random images (seed {a.random60_seed}), mesh, Metal GPU, depth_downscale 8'
    out[f'random60_s{a.random60_seed}'] = m
    a.output.mkdir(parents=True, exist_ok=False)
    for name, m in out.items():
        require_valid(m, a.root.expanduser())
        (a.output / f'{name}.json').write_text(json.dumps(m, indent=2, sort_keys=True) + '\n')
        print(name, m['image_count'], m['experiment_id'])


if __name__ == '__main__':
    main()
