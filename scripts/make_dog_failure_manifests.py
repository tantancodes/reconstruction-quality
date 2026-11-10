"""More ARF-dog failure-case manifests at the GPU pilot settings (mesh, Metal GPU, depth_downscale 8).

Built from experiments/gpu-pilot-d8/ref50.json (the existing GPU dog reference), so the
configuration is identical and only image selection differs. Scored against results/gpu-pilot-d8/ref50.

  random15_s<N>, random20_s<N>   random.Random(N).sample over parent order, seeds 20261110, 20261111
  side25                         25% contiguous side removed: ring positions 9-14 (6 of 25 = 24%)
                                 removed from both 25-image rings. Same sector centre as side40
                                 (positions 7-16), so side25's gap lies inside side40's.

  PYTHONPATH=src python3 scripts/make_dog_failure_manifests.py --root ~/Downloads/Jpg-processed \
      --output experiments/gpu-pilot-d8-ext
"""
import argparse
import copy
import json
from pathlib import Path

from bd3d_quality.experiment_design import subset_manifest
from bd3d_quality.manifests import experiment_identity, require_valid
from bd3d_quality.provenance import digest

SEEDS = (20261110, 20261111)
RING = 25
REMOVED_RING_POSITIONS = range(9, 15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', type=Path, default=Path('experiments/gpu-pilot-d8/ref50.json'))
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    base = json.loads(a.reference.read_text())
    if base['image_count'] != 2 * RING:
        raise SystemExit('expected the 50-image two-ring reference')
    out = {}
    for count in (15, 20):
        for seed in SEEDS:
            m = subset_manifest(base, count, 'random', seed=seed)
            m['description'] = f'GPU dog failure case: {count} random images (seed {seed}), depth_downscale 8, Metal GPU'
            out[f'random{count}_s{seed}'] = m
    removed = {ring * RING + i for ring in (0, 1) for i in REMOVED_RING_POSITIONS}
    keep = [i for i in range(2 * RING) if i not in removed]
    side = copy.deepcopy(base)
    side['images'] = [base['images'][i] for i in keep]
    side['image_count'] = len(keep)
    side['selection'] = {
        'version': 1, 'strategy': 'contiguous_side_removed', 'seed': None,
        'parent_experiment_id': base['experiment_id'], 'parent_image_list_signature': digest(base['images']),
        'indices_zero_based': keep, 'removed_indices_zero_based': sorted(removed),
        'rule': 'two 25-image rings in acquisition order; ring positions 9-14 removed from both (24%)',
        'order': 'parent manifest order (acquisition order)'}
    side['description'] = 'GPU dog failure case: 25% contiguous side removed (38 images), depth_downscale 8, Metal GPU'
    side['experiment_id'] = experiment_identity(side)
    out['side25'] = side
    a.output.mkdir(parents=True, exist_ok=False)
    for name, m in out.items():
        require_valid(m, a.root.expanduser())
        (a.output / f'{name}.json').write_text(json.dumps(m, indent=2, sort_keys=True) + '\n')
        print(name, m['image_count'], m['experiment_id'])


if __name__ == '__main__':
    main()
