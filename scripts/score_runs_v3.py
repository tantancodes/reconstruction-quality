"""Score new pipeline runs against a reference run with geometry evaluator v3.

Same method as rescore_camera_init_icp.py (camera-initialised similarity ICP, v1
sampling and settings, 2.5% camera/ICP scale check), applied to run directories
written by `bd3d_quality.cli reconstruct` (cameras.csv + raw-mesh.obj).

  python3 scripts/score_runs_v3.py --fork ~/Documents/ChatGPT/bd3d \
      --reference results/gpu-pilot-d8/ref50 \
      --candidates results/gpu-pilot-d8/random10 results/gpu-pilot-d8/side40 \
      --output results/gpu-pilot-d8/geometry-v3
"""
import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from rescore_camera_init_icp import (LONG_ITERATIONS, SCALE_TOLERANCE, camera_transform,  # noqa: E402
                                     load_cameras, refine, scale_of, score)
from rescore_similarity_icp import load_v1, qc, sha256  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fork', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--candidates', type=Path, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    v1 = load_v1(a.fork.expanduser().resolve())
    s = v1.GeometryEvaluationSettings()
    settings = {**asdict(s), 'evaluator_version': 3,
                'registration_method': 'camera_similarity_initialization_then_similarity_point_to_point_icp',
                'scale_check_tolerance': SCALE_TOLERANCE}
    settings['signature'] = hashlib.sha256(repr(sorted(settings.items())).encode()).hexdigest()[:16]

    ref_obj = a.reference / 'raw-mesh.obj'
    ref_mesh = v1.load_obj_mesh(ref_obj)
    ref_cams = load_cameras(a.reference / 'cameras.csv')
    diag = float(np.linalg.norm(np.ptp(ref_mesh.vertices, axis=0)))
    cutoff = s.icp_max_correspondence_distance_fraction * diag
    reg_ref = v1.sample_surface(ref_mesh, s.registration_sample_count, s.random_seed)
    a.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for cand in a.candidates:
        cand_mesh = v1.load_obj_mesh(cand / 'raw-mesh.obj')
        reg_cand = v1.sample_surface(cand_mesh, s.registration_sample_count, s.random_seed)
        T0, n_shared, resid = camera_transform(load_cameras(cand / 'cameras.csv'), ref_cams)
        init_d, _ = cKDTree(reg_ref).query(v1.apply_transform(reg_cand, T0), workers=1)
        args = (cutoff, s.icp_max_iterations, s.icp_relative_tolerance, s.icp_absolute_tolerance_fraction * diag)
        sim = refine(reg_cand, reg_ref, T0, *args, with_scale=True)
        rig = refine(reg_cand, reg_ref, T0, *args, with_scale=False)
        long = refine(reg_cand, reg_ref, T0, cutoff, LONG_ITERATIONS, *args[2:], with_scale=True)
        m = score(v1, s, ref_mesh, cand_mesh, sim['transform'], diag, cutoff)
        ratio = sim['scale'] / scale_of(T0)
        run = json.loads((cand / 'run.json').read_text())
        row = {'run': cand.name, 'run_id': run['run_id'], 'experiment_id': run['experiment_id'],
               'images': len(load_cameras(cand / 'cameras.csv')),
               'label_v3': m['y'], 'qc_v3': qc(sim['fitness']), 'fitness_v3': sim['fitness'],
               'fitness_at_camera_init': float((init_d <= cutoff).mean()),
               'icp_scale_v3': sim['scale'], 'camera_scale': scale_of(T0), 'scale_ratio': ratio,
               'scale_check': 'agree' if abs(ratio - 1) <= SCALE_TOLERANCE else 'disagree',
               'iterations_v3': sim['iterations'], 'converged_v3': sim['converged'],
               'camera_shared_images': n_shared, 'camera_shape_residual': resid,
               **{k: v for k, v in m.items() if k != 'y'},
               'label_v3_rigid_camera_scale': score(v1, s, ref_mesh, cand_mesh, rig['transform'], diag, cutoff)['y'],
               f'label_v3_{LONG_ITERATIONS}it': score(v1, s, ref_mesh, cand_mesh, long['transform'], diag, cutoff)['y'],
               'candidate_mesh_sha256': sha256(cand / 'raw-mesh.obj')}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ('run', 'images', 'label_v3', 'qc_v3', 'fitness_v3', 'fitness_at_camera_init',
                                               'icp_scale_v3', 'camera_scale', 'scale_check', 'converged_v3',
                                               'c2r_rmse_norm', 'r2c_rmse_norm', 'r2c_frac_beyond_cutoff',
                                               'label_v3_rigid_camera_scale', f'label_v3_{LONG_ITERATIONS}it')}), flush=True)
    with open(a.output / 'labels_v3.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (a.output / 'settings.json').write_text(json.dumps(
        {'evaluator_v3': settings, 'reference_run': str(a.reference), 'reference_mesh_sha256': sha256(ref_obj),
         'reference_bounding_box_diagonal': diag}, indent=2))


if __name__ == '__main__':
    main()
