"""Manifests for the GPU dense pilot: two ARF-dog candidates + the 50-image reference.

All three share one configuration: config/mesh.json with depth_downscale=8 and
cpu_only=false (Metal GPU), so they differ only in image selection.

  random10   10 images, random.Random(20261109).sample over the parent order
  side40     40% contiguous side removed: the 50 images are two 25-image rings
             (upper: parent positions 0-24, lower: 25-49) in acquisition order.
             Ring positions 7-16 are removed from both rings (20 images), i.e. one
             ~150 deg azimuth sector (about 137-268 deg in the historical reference
             frame). 30 images kept.
  ref50      all 50 images

  PYTHONPATH=src python3 scripts/make_gpu_pilot_manifests.py \
      --parent results/arf-dog-50-manifest.json --root <dog-photos> \
      --output experiments/gpu-pilot-d8
"""
import argparse
import copy
import json
from pathlib import Path

from bd3d_quality.experiment_design import subset_manifest
from bd3d_quality.manifests import experiment_identity, require_valid
from bd3d_quality.provenance import digest

SEED = 20261109
RING = 25
REMOVED_RING_POSITIONS = range(7, 17)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parent', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--config', type=Path, default=Path('config/mesh.json'))
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()

    parent = json.loads(a.parent.read_text())
    if parent['image_count'] != 2 * RING:
        raise SystemExit('expected the 50-image two-ring parent')
    config = json.loads(a.config.read_text()) | {'depth_downscale': 8, 'cpu_only': False}
    base = copy.deepcopy(parent)
    base['configuration'] = config
    base['experiment_id'] = experiment_identity(base)

    ref = copy.deepcopy(base)
    ref['description'] = 'GPU pilot reference: all 50 images, depth_downscale 8, Metal GPU'

    rnd = subset_manifest(base, 10, 'random', seed=SEED)
    rnd['description'] = f'GPU pilot candidate: 10 random images (seed {SEED}), depth_downscale 8, Metal GPU'

    removed = {ring * RING + i for ring in (0, 1) for i in REMOVED_RING_POSITIONS}
    keep = [i for i in range(2 * RING) if i not in removed]
    side = copy.deepcopy(base)
    side['images'] = [base['images'][i] for i in keep]
    side['image_count'] = len(keep)
    side['selection'] = {
        'version': 1, 'strategy': 'contiguous_side_removed', 'seed': None,
        'parent_experiment_id': base['experiment_id'], 'parent_image_list_signature': digest(base['images']),
        'indices_zero_based': keep, 'removed_indices_zero_based': sorted(removed),
        'rule': 'two 25-image rings in acquisition order; ring positions 7-16 removed from both (40%)',
        'order': 'parent manifest order (acquisition order)'}
    side['description'] = 'GPU pilot candidate: 40% contiguous side removed (30 images), depth_downscale 8, Metal GPU'
    side['experiment_id'] = experiment_identity(side)

    a.output.mkdir(parents=True, exist_ok=False)
    for name, m in (('random10', rnd), ('side40', side), ('ref50', ref)):
        require_valid(m, a.root)
        (a.output / f'{name}.json').write_text(json.dumps(m, indent=2, sort_keys=True) + '\n')
        print(name, m['image_count'], m['experiment_id'], [e['path'] for e in m['images']][:3], '...')


if __name__ == '__main__':
    main()
