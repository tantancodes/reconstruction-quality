"""One labels.csv per artifact: v3 label plus missing- vs extra-surface breakdown and run context.

missing surface = reference -> candidate (r2c): reference surface the candidate does not reconstruct.
extra surface   = candidate -> reference (c2r): candidate surface with no reference counterpart
                  (spurious geometry, noise, misplaced parts).
dominant_error  = whichever RMSE (normalized by reference bbox diagonal) is larger.
The v3 label itself is the symmetric RMSE: sqrt((c2r_rmse^2 + r2c_rmse^2) / 2).

  python3 scripts/combine_labels.py --artifact pahma-5-613 --scores results/pahma-5-613-v2/geometry-v3 \
      --runs-dir results/pahma-5-613-v2 --output results/pahma-5-613-v2/labels.csv
"""
import argparse
import csv
import json
from pathlib import Path

COLS = ['artifact', 'run', 'images', 'label_v3', 'qc_v3', 'scale_check', 'fitness_v3',
        'missing_surface_rmse_norm', 'missing_surface_p95_norm', 'missing_surface_frac_beyond_cutoff',
        'extra_surface_rmse_norm', 'extra_surface_p95_norm', 'extra_surface_frac_beyond_cutoff',
        'completeness_at_0.01', 'accuracy_at_0.01', 'dominant_error', 'missing_share_of_sq_error',
        'run_reprojection_rmse_px', 'aligned_cameras', 'elapsed_seconds', 'image_quality_source',
        'run_id', 'experiment_id', 'candidate_mesh_sha256']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', required=True)
    ap.add_argument('--scores', type=Path, nargs='+', required=True, help='geometry-v3 folders (labels_v3.csv)')
    ap.add_argument('--runs-dir', type=Path, nargs='+', required=True, help='folders holding the run directories')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    out = []
    for folder in a.scores:
        for r in csv.DictReader(open(folder / 'labels_v3.csv', newline='')):
            run_dir = next((d / r['run'] for d in a.runs_dir if (d / r['run'] / 'run.json').exists()), None)
            run = json.loads((run_dir / 'run.json').read_text()) if run_dir else {}
            align = json.loads((run_dir / 'metrics-initial_alignment.json').read_text())['metrics'] if run_dir else {}
            res_path = run_dir.with_name(run_dir.name + '-resources.json') if run_dir else None
            res = json.loads(res_path.read_text()) if res_path and res_path.exists() else {}
            m, e = float(r['r2c_rmse_norm']), float(r['c2r_rmse_norm'])
            src = run.get('image_quality_source') or {}
            out.append({
                'artifact': a.artifact, 'run': r['run'], 'images': r['images'], 'label_v3': r['label_v3'],
                'qc_v3': r['qc_v3'], 'scale_check': r['scale_check'], 'fitness_v3': r['fitness_v3'],
                'missing_surface_rmse_norm': m, 'missing_surface_p95_norm': r['r2c_p95_norm'],
                'missing_surface_frac_beyond_cutoff': r['r2c_frac_beyond_cutoff'],
                'extra_surface_rmse_norm': e, 'extra_surface_p95_norm': r['c2r_p95_norm'],
                'extra_surface_frac_beyond_cutoff': r['c2r_frac_beyond_cutoff'],
                'completeness_at_0.01': r['completeness_at_0.01'], 'accuracy_at_0.01': r['accuracy_at_0.01'],
                'dominant_error': 'missing' if m > e else 'extra',
                'missing_share_of_sq_error': m * m / (m * m + e * e) if m or e else '',
                'run_reprojection_rmse_px': align.get('reprojection_rmse_px', ''),
                'aligned_cameras': align.get('aligned_camera_count', ''),
                'elapsed_seconds': res.get('elapsed_seconds', ''),
                'image_quality_source': src.get('method', ''),
                'run_id': r['run_id'], 'experiment_id': r['experiment_id'],
                'candidate_mesh_sha256': r['candidate_mesh_sha256']})
    if a.output.exists():
        raise SystemExit(f'{a.output} exists; refusing to overwrite')
    with open(a.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(out)
    print(f'wrote {len(out)} rows to {a.output}')


if __name__ == '__main__':
    main()
