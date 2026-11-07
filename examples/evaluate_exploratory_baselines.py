"""EXPLORATORY / NOT VALIDATED optional future mean/Ridge evaluation.

PREPARED, NOT RUN in the EDA phase. No PyTorch implementation or dependency.
Execution requires an explicit flag and one full compatibility-stratum key.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from bd3d_quality.alignment_arrays import STATUS, load_alignment_arrays
from bd3d_quality.exploratory_splits import plan_within_artifact_folds
from bd3d_quality.historical_alignment import require
from bd3d_quality.provenance import sha256_file, write_json


def evaluate(bundle, output, stratum, alpha):
    """Called only by explicit future execution; fits a fixed-alpha weighted Ridge."""
    require(np.isfinite(alpha) and alpha > 0, 'positive finite fixed Ridge alpha required')
    output = Path(output)
    require(not output.exists(), 'baseline output must be a new directory')
    require(not output.resolve().is_relative_to(Path(bundle).resolve()), 'output cannot be inside historical bundle')
    data = load_alignment_arrays(bundle, mode='exploratory', artifact_group='arf-dog', dtype=np.float64)
    plan = plan_within_artifact_folds(data, stratum)
    predictions = []; parameters = []
    for fold in plan['folds']:
        train, test = fold['train_indices'], fold['test_indices']
        X, y = data.X[train], data.y[train, 0]
        repeats = Counter(data.content_groups[i] for i in train)
        w = np.asarray([1/repeats[data.content_groups[i]] for i in train])
        center = np.average(X, axis=0, weights=w)
        scale = np.sqrt(np.average((X-center)**2, axis=0, weights=w))
        scale = np.where(scale > 1e-12, scale, 1.)
        Z = (X-center)/scale
        y_center = float(np.average(y, weights=w))
        coef = np.linalg.solve(Z.T @ (w[:, None]*Z)+alpha*np.eye(X.shape[1]),
                               Z.T @ (w*(y-y_center)))
        pred = y_center+(data.X[test]-center)/scale @ coef
        for i, value in zip(test, pred):
            predictions.append({'record_id': data.record_ids[i], 'content_set_id': data.content_groups[i],
                'observed_rmse_pixels': float(data.y[i, 0]), 'mean_prediction_pixels': y_center,
                'ridge_prediction_pixels': float(value),
                'shared_training_photo_hashes': fold['shared_photo_hashes_train_test']})
        parameters.append({'held_out_content_set': fold['held_out_content_set'],
            'training_center': center.tolist(), 'training_scale': scale.tolist(),
            'training_target_mean': y_center, 'ridge_coefficients_standardized': coef.tolist()})
    counts = Counter(p['content_set_id'] for p in predictions)
    balanced = np.asarray([1/counts[p['content_set_id']] for p in predictions])
    metrics = {}
    for name in ['mean', 'ridge']:
        errors = np.asarray([p[name+'_prediction_pixels']-p['observed_rmse_pixels'] for p in predictions])
        metrics[name] = {}
        for weighting, w in [('run_weighted', np.ones(len(errors))), ('content_set_balanced', balanced)]:
            metrics[name][weighting] = {'mae_pixels': float(np.average(abs(errors), weights=w)),
                'rmse_pixels': float(np.sqrt(np.average(errors**2, weights=w)))}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'EXPLORATORY-baselines.json', {'status': STATUS, 'plan': plan,
        'ridge_alpha_fixed_not_tuned': alpha, 'features': list(data.feature_names), 'metrics': metrics,
        'predictions': predictions, 'fold_parameters': parameters, 'source_hashes': data.source_hashes,
        'training_weights': 'Each training content set has total weight 1; preprocessing uses training rows only.',
        'evaluation_weights': 'Run-level residuals, also weighted 1/repeat_count per content set; labels are not averaged across strata.',
        'strict_ready_rows': int(data.strict_ready_mask.sum()), 'strict_gate_modified': False,
        'scope': 'Within-Dog interpolation/extrapolation only; overlapping source photos, unknown sessions and partial settings remain. No held-out-artifact generalization.',
        'script_sha256': sha256_file(Path(__file__))})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle'); p.add_argument('output')
    p.add_argument('--stratum', required=True, help='Full compatibility key; strata are never silently pooled')
    p.add_argument('--ridge-alpha', type=float, default=1.)
    p.add_argument('--execute-exploratory-baselines', action='store_true')
    args = p.parse_args()
    if not args.execute_exploratory_baselines:
        p.error('No model fitted. Future execution requires --execute-exploratory-baselines.')
    evaluate(args.bundle, args.output, args.stratum, args.ridge_alpha)
