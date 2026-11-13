"""Score pipeline runs against a reference run with geometry evaluator v3.

Each run directory is one written by `bd3d-quality reconstruct` (cameras.csv, raw-mesh.obj,
run.json). The method is in src/bd3d_quality/evaluation.py: camera-initialised similarity
ICP, then the symmetric surface RMSE normalised by the reference bounding-box diagonal.
No Metashape needed.

  python scripts/score_runs_v3.py --reference results/gpu-pilot-d8/ref50 \
      --candidates results/gpu-pilot-d8/random10 results/gpu-pilot-d8/side40 \
      --output results/gpu-pilot-d8/geometry-v3

Writes <output>/labels_v3.csv (one row per candidate) and <output>/settings.json.
"""
import argparse
import csv
import json
from pathlib import Path

from bd3d_quality.evaluation import LONG_ITERATIONS, Reference, load_cameras, load_obj_mesh, score_candidate, v3_settings
from bd3d_quality.provenance import sha256_file

ROW = ['label_v3', 'qc_v3', 'fitness_v3', 'fitness_at_camera_init', 'icp_scale_v3', 'camera_scale', 'scale_ratio',
       'scale_check', 'iterations_v3', 'converged_v3', 'camera_shared_images', 'camera_shape_residual']
CHECKS = ['label_v3_rigid_camera_scale', 'fitness_rigid_camera_scale', f'label_v3_{LONG_ITERATIONS}it',
          f'scale_v3_{LONG_ITERATIONS}it', f'converged_v3_{LONG_ITERATIONS}it']
SUMMARY = ('run', 'images', 'label_v3', 'qc_v3', 'fitness_v3', 'fitness_at_camera_init', 'icp_scale_v3',
           'camera_scale', 'scale_check', 'converged_v3', 'c2r_rmse_norm', 'r2c_rmse_norm',
           'r2c_frac_beyond_cutoff', 'label_v3_rigid_camera_scale', f'label_v3_{LONG_ITERATIONS}it')


def score_runs(reference, candidates):
    """Return the prepared reference and one labels_v3.csv row per candidate run directory."""
    ref = Reference.from_files(reference / 'raw-mesh.obj', reference / 'cameras.csv')
    rows = []
    for cand in candidates:
        cams = load_cameras(cand / 'cameras.csv')
        r, _ = score_candidate(ref, load_obj_mesh(cand / 'raw-mesh.obj'), cams)
        run = json.loads((cand / 'run.json').read_text())
        distances = [k for k in r if k not in ROW and k not in CHECKS]
        rows.append({'run': cand.name, 'run_id': run['run_id'], 'experiment_id': run['experiment_id'],
                     'images': len(cams), **{k: r[k] for k in ROW + distances},
                     'label_v3_rigid_camera_scale': r['label_v3_rigid_camera_scale'],
                     f'label_v3_{LONG_ITERATIONS}it': r[f'label_v3_{LONG_ITERATIONS}it'],
                     'candidate_mesh_sha256': sha256_file(cand / 'raw-mesh.obj')})
        print(json.dumps({k: rows[-1][k] for k in SUMMARY}), flush=True)
    return ref, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', type=Path, required=True, help='reference run directory')
    ap.add_argument('--candidates', type=Path, nargs='+', required=True, help='candidate run directories')
    ap.add_argument('--output', type=Path, required=True, help='new folder for labels_v3.csv and settings.json')
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    ref, rows = score_runs(a.reference, a.candidates)
    with open(a.output / 'labels_v3.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (a.output / 'settings.json').write_text(json.dumps(
        {'evaluator_v3': v3_settings(), 'reference_run': str(a.reference),
         'reference_mesh_sha256': sha256_file(a.reference / 'raw-mesh.obj'),
         'reference_bounding_box_diagonal': ref.diag}, indent=2))


if __name__ == '__main__':
    main()
