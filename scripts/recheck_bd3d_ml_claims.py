"""Recompute three earlier claims from the Aphoma bd3d-ml branch (docs/RESULTS.md) with v3 labels.

The bd3d-ml analysis used v1 geometric labels, which the evaluation audit
(results/geometry-labels-v3/NOTES.md) showed were registration failures for 7 of 16 runs.
This script first reproduces each claim from the original modeling table with the v1 label
(to confirm the method matches), then repeats it with label_v3 joined by run_id.

  claim 1  structured 30-image contrast: central gap / remove-strongest-overlap (5.7x)
  claim 2  Spearman rho between sparse reprojection RMSE and geometric error (-0.17)
  claim 3  two-feature early-SfM Ridge (tie points/image, valid observations/image; alpha 1,
           StandardScaler in-fold), leave-one-condition-out RMSE improvement over the
           dummy, image-count and reprojection baselines (10.7%, 11.2%, 13.7%)

  python3 scripts/recheck_bd3d_ml_claims.py \
      --dataset <bd3d>/analysis_outputs/fast_dense_geometry_dataset_v1/fast_dense_geometry_modeling_dataset_v1.csv \
      --output results/bd3d-ml-claims-recheck
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
FEATURES = {'dummy': [], 'image_count': ['images_used_count'], 'reprojection': ['reprojection_rmse_px'],
            'early_sfm': ['tie_points_per_image', 'valid_observations_per_image']}
PUBLISHED = {'central_gap_over_remove_strongest': 5.7, 'spearman_reprojection_vs_label': -0.17,
             'loco_improvement_vs_dummy_pct': 10.7, 'loco_improvement_vs_image_count_pct': 11.2,
             'loco_improvement_vs_reprojection_pct': 13.7}


def loco_rmse(d, y, feats):
    pred = np.zeros(len(d))
    for i in range(len(d)):
        tr = np.arange(len(d)) != i
        if not feats:
            pred[i] = y[tr].mean()
        else:
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(d.loc[tr, feats], y[tr])
            pred[i] = m.predict(d.loc[[i], feats])[0]
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def claims(d, label):
    y = d[label].to_numpy()
    s = d.set_index('experiment_id')[label]
    cg = s['arf-dog-structured-central-gap-30-v1']
    rs = s[[i for i in s.index if 'remove-strongest' in i][0]]
    r = {k: loco_rmse(d, y, f) for k, f in FEATURES.items()}
    return {'central_gap_label': cg, 'remove_strongest_label': rs, 'central_gap_over_remove_strongest': cg / rs,
            'spearman_reprojection_vs_label': float(spearmanr(d.reprojection_rmse_px, y).statistic),
            **{f'loco_rmse_{k}': v for k, v in r.items()},
            'loco_improvement_vs_dummy_pct': 100 * (1 - r['early_sfm'] / r['dummy']),
            'loco_improvement_vs_image_count_pct': 100 * (1 - r['early_sfm'] / r['image_count']),
            'loco_improvement_vs_reprojection_pct': 100 * (1 - r['early_sfm'] / r['reprojection'])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--labels', type=Path, default=REPO / 'results/geometry-labels-v3/labels_v3.csv')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.dataset, low_memory=False).copy()
    v3 = pd.read_csv(a.labels)[['run_id', 'label_v1', 'label_v3']]
    d = d.merge(v3, on='run_id', validate='one_to_one').reset_index(drop=True)
    if len(d) != 16 or not np.allclose(d.geometry_y_normalized_symmetric_rmse, d.label_v1):
        raise SystemExit('modeling table does not match the 16 v1-labelled runs')
    v1, v3r = claims(d, 'label_v1'), claims(d, 'label_v3')
    rows = [{'quantity': k, 'published_bd3d_ml': PUBLISHED.get(k, ''), 'reproduced_v1': v1[k], 'recomputed_v3': v3r[k]}
            for k in v1]
    a.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(a.output / 'claims.csv', index=False)
    (a.output / 'settings.json').write_text(json.dumps({
        'dataset_rows': len(d), 'dataset_file': a.dataset.name, 'labels': str(a.labels.relative_to(REPO)),
        'validation': 'leave-one-condition-out = leave one run out (16 conditions, one run each)',
        'model': 'Ridge(alpha=1.0) with StandardScaler fitted in each training fold; dummy = training mean'}, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
