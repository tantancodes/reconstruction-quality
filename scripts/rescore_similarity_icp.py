"""Rescore the 16 historical FAST_DENSE meshes with similarity ICP (labels v2).

Why: the historical alignments are unscaled local frames (no markers/scale bars),
yet evaluator v1 registered candidate meshes to the reference with rigid ICP
only, so arbitrary frame-scale differences were counted as surface error.

What changes vs v1: the per-iteration ICP fit also solves one uniform scale
(Umeyama). Everything else is v1: mesh loading and area-weighted sampling are
imported unchanged from the historical evaluator, same seeds, sample counts,
60 iterations, tolerances, centroid initialisation and 5% correspondence cutoff
(fraction of the reference bounding-box diagonal).

Checks and diagnostics:
  step 0  v1 evaluator re-run (imported, unmodified) must reproduce stored labels,
          and this script's rigid path must reproduce v1 exactly;
  step 1  scale from cameras (similarity Procrustes of each run's cameras onto the
          reference's, shared images) as an independent check on the ICP scale;
  step 3  fraction of points beyond the cutoff and their share of squared error;
  sens.   rigid and similarity with 300 iterations (reported, not the label).

No Metashape. The historical label CSV and reports are read only.

  python3 scripts/rescore_similarity_icp.py \
      --fork ~/Documents/ChatGPT/bd3d \
      --cameras results/hist16-old-alignment --output results/geometry-labels-v2
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

QC = [(0.95, 'pass'), (0.80, 'caution'), (-1, 'manual_review')]  # historical METHODS.md rule
SCALE_FLAG = 0.01  # |icp_scale / camera_scale - 1| above this -> scale_check = 'disagree'


def qc(fitness):
    return next(name for lo, name in QC if fitness >= lo)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_v1(fork):
    spec = importlib.util.spec_from_file_location('geometry_quality_v1', fork / 'analysis' / 'geometry_quality.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look the module up here
    spec.loader.exec_module(mod)
    return mod


def fit(source, target, with_scale):
    """Least-squares rotation+translation (+ uniform scale) mapping source -> target."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    A, B = source - mu_s, target - mu_t
    U, S, Vt = np.linalg.svd(np.einsum('ni,nj->ij', A, B, optimize=False))
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    s = float((S * np.diag(D)).sum() / (A ** 2).sum()) if with_scale else 1.0
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = mu_t - s * R @ mu_s
    return T


def icp(source, target, max_distance, max_iterations, rel_tol, abs_tol, with_scale):
    """v1 rigid_icp loop, with the inner fit optionally allowing uniform scale."""
    T = np.eye(4)
    T[:3, 3] = target.mean(0) - source.mean(0)
    tree = cKDTree(target)
    prev, it, converged = np.inf, 0, False
    for it in range(1, max_iterations + 1):
        moved = source @ T[:3, :3].T + T[:3, 3]
        dist, idx = tree.query(moved, workers=1)
        ok = dist <= max_distance
        if ok.sum() < 3:
            break
        T = fit(moved[ok], target[idx[ok]], with_scale) @ T
        rmse = float(np.sqrt(np.mean(dist[ok] ** 2)))
        if rmse <= abs_tol or (np.isfinite(prev) and abs(prev - rmse) <= rel_tol * max(prev, 1e-15)):
            converged = True
            break
        prev = rmse
    moved = source @ T[:3, :3].T + T[:3, 3]
    dist, _ = tree.query(moved, workers=1)
    ok = dist <= max_distance
    return {'transform': T, 'fitness': float(ok.mean()), 'iterations': it, 'converged': converged,
            'scale': float(np.cbrt(np.linalg.det(T[:3, :3])))}


def evaluate(v1, settings, ref_mesh, cand_mesh, with_scale, max_iterations):
    diag = float(np.linalg.norm(np.ptp(ref_mesh.vertices, axis=0)))
    cutoff = settings.icp_max_correspondence_distance_fraction * diag
    reg_ref = v1.sample_surface(ref_mesh, settings.registration_sample_count, settings.random_seed)
    reg_cand = v1.sample_surface(cand_mesh, settings.registration_sample_count, settings.random_seed)
    reg = icp(reg_cand, reg_ref, cutoff, max_iterations, settings.icp_relative_tolerance,
              settings.icp_absolute_tolerance_fraction * diag, with_scale)
    ref_pts = v1.sample_surface(ref_mesh, settings.evaluation_sample_count, settings.random_seed + 1)
    cand_pts = v1.sample_surface(cand_mesh, settings.evaluation_sample_count, settings.random_seed + 1)
    cand_pts = v1.apply_transform(cand_pts, reg['transform'])
    c2r, _ = cKDTree(ref_pts).query(cand_pts, workers=1)
    r2c, _ = cKDTree(cand_pts).query(ref_pts, workers=1)
    sym = np.concatenate((c2r, r2c))
    far = sym > cutoff
    return {
        'y': float(np.sqrt((np.mean(c2r ** 2) + np.mean(r2c ** 2)) / 2) / diag),
        'fitness': reg['fitness'], 'qc': qc(reg['fitness']), 'scale': reg['scale'],
        'iterations': reg['iterations'], 'converged': reg['converged'],
        'reg_frac_beyond_cutoff': 1 - reg['fitness'],
        'c2r_frac_beyond_cutoff': float((c2r > cutoff).mean()),
        'r2c_frac_beyond_cutoff': float((r2c > cutoff).mean()),
        'far_share_of_sq_error': float((sym[far] ** 2).sum() / (sym ** 2).sum()),
        'y_excluding_far': float(np.sqrt(np.mean(sym[~far] ** 2)) / diag),
        'c2r_rmse_norm': float(np.sqrt(np.mean(c2r ** 2)) / diag),
        'r2c_rmse_norm': float(np.sqrt(np.mean(r2c ** 2)) / diag),
        'transform': reg['transform'].tolist(),
    }


def camera_scale(cand_csv, ref_csv):
    def load(p):
        return {r['label']: np.array([float(r['x']), float(r['y']), float(r['z'])])
                for r in csv.DictReader(open(p, newline='')) if r['x']}
    a, b = load(cand_csv), load(ref_csv)
    k = sorted(set(a) & set(b))
    A, B = np.array([a[i] for i in k]), np.array([b[i] for i in k])
    T = fit(A, B, True)
    resid = np.sqrt((((A @ T[:3, :3].T + T[:3, 3]) - B) ** 2).sum(1).mean()) / np.sqrt(((B - B.mean(0)) ** 2).sum(1).mean())
    return float(np.cbrt(np.linalg.det(T[:3, :3]))), len(k), float(resid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fork', type=Path, required=True)
    ap.add_argument('--cameras', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--sensitivity-iterations', type=int, default=300)
    a = ap.parse_args()
    fork = a.fork.expanduser().resolve()
    label_csv = fork / 'analysis_outputs/fast_dense_geometry_dataset_v1/fast_dense_geometry_modeling_dataset_v1.csv'
    reports = fork / 'analysis_outputs/fast_dense_geometry_dataset_v1/geometry_reports'
    protected = [label_csv, *sorted(reports.glob('*.json'))]
    hashes_before = {str(p): sha256(p) for p in protected}

    v1 = load_v1(fork)
    s1 = v1.GeometryEvaluationSettings()
    s2 = {**asdict(s1), 'evaluator_version': 2,
          'registration_method': 'centroid_initialization_then_similarity_point_to_point_icp_uniform_scale'}
    s2['signature'] = hashlib.sha256(repr(sorted(s2.items())).encode()).hexdigest()[:16]
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'reports').mkdir()

    labels = list(csv.DictReader(label_csv.open(newline='')))
    ref_path = None
    rows, t0 = [], time.time()
    for r in labels:
        rep = json.loads((reports / f"{r['run_id']}.json").read_text())
        ref_path, cand_path = Path(rep['reference']['path']), Path(rep['candidate']['path'])
        if sha256(ref_path) != rep['reference']['sha256'] or sha256(cand_path) != rep['candidate']['sha256']:
            raise SystemExit(f'mesh hash differs from historical report: {r["experiment_id"]}')
        # step 0a: unmodified v1 evaluator
        v1_out = v1.evaluate_geometry(ref_path, cand_path, s1)
        y_v1 = v1_out['symmetric_distances']['primary_target_y_symmetric_rmse_normalized']
        ref_mesh, cand_mesh = v1.load_obj_mesh(ref_path), v1.load_obj_mesh(cand_path)
        rigid = evaluate(v1, s1, ref_mesh, cand_mesh, False, s1.icp_max_iterations)   # step 0b
        sim = evaluate(v1, s1, ref_mesh, cand_mesh, True, s1.icp_max_iterations)      # step 2 (label v2)
        rigid_long = evaluate(v1, s1, ref_mesh, cand_mesh, False, a.sensitivity_iterations)
        sim_long = evaluate(v1, s1, ref_mesh, cand_mesh, True, a.sensitivity_iterations)
        cam_s, n_shared, cam_resid = camera_scale(a.cameras / r['experiment_id'] / 'cameras.csv',
                                                  a.cameras / '_reference' / 'cameras.csv')
        old = float(r['geometry_y_normalized_symmetric_rmse'])
        row = {
            'run_id': r['run_id'], 'experiment_id': r['experiment_id'], 'images_used_count': r['images_used_count'],
            'label_v1_stored': old, 'label_v1_rerun': y_v1, 'label_v1_reimplemented': rigid['y'],
            'qc_v1': r['registration_qc_flag'], 'fitness_v1': rigid['fitness'],
            'label_v2': sim['y'], 'qc_v2': sim['qc'], 'fitness_v2': sim['fitness'],
            'icp_scale_v2': sim['scale'], 'camera_scale': cam_s, 'camera_shared_images': n_shared,
            'camera_shape_residual': cam_resid,
            'scale_check': 'agree' if abs(sim['scale'] / cam_s - 1) <= SCALE_FLAG else 'disagree',
            'iterations_v1': rigid['iterations'], 'converged_v1': rigid['converged'],
            'iterations_v2': sim['iterations'], 'converged_v2': sim['converged'],
        }
        for tag, res in (('v1', rigid), ('v2', sim)):
            for k in ('reg_frac_beyond_cutoff', 'c2r_frac_beyond_cutoff', 'r2c_frac_beyond_cutoff',
                      'far_share_of_sq_error', 'y_excluding_far', 'c2r_rmse_norm', 'r2c_rmse_norm'):
                row[f'{k}_{tag}'] = res[k]
        row |= {'sens_rigid_300_y': rigid_long['y'], 'sens_rigid_300_iterations': rigid_long['iterations'],
                'sens_rigid_300_converged': rigid_long['converged'], 'sens_rigid_300_fitness': rigid_long['fitness'],
                'sens_sim_300_y': sim_long['y'], 'sens_sim_300_scale': sim_long['scale'],
                'sens_sim_300_iterations': sim_long['iterations'], 'sens_sim_300_converged': sim_long['converged'],
                'sens_sim_300_fitness': sim_long['fitness']}
        rows.append(row)
        (a.output / 'reports' / f"{r['run_id']}.json").write_text(json.dumps(
            {'row': row, 'rigid_60': rigid, 'similarity_60': sim, 'rigid_long': rigid_long,
             'similarity_long': sim_long, 'reference': rep['reference'], 'candidate': rep['candidate']}, indent=2))
        print(f"{r['experiment_id']:50s} v1 {old:.5f} (rerun {y_v1:.5f}) -> v2 {sim['y']:.5f}  "
              f"s_icp={sim['scale']:.4f} s_cam={cam_s:.4f} qc {r['registration_qc_flag']}->{sim['qc']}", flush=True)

    with open(a.output / 'labels_v2.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    hashes_after = {str(p): sha256(p) for p in protected}
    (a.output / 'settings.json').write_text(json.dumps({
        'evaluator_v2': s2, 'evaluator_v1_signature': s1.signature(), 'scale_check_tolerance': SCALE_FLAG,
        'qc_rule': 'pass >= 0.95 fitness, caution 0.80-0.95, manual_review < 0.80 (historical METHODS.md)',
        'sensitivity_iterations': a.sensitivity_iterations,
        'reference_mesh': str(ref_path), 'historical_label_csv': str(label_csv),
        'historical_files_unchanged': hashes_before == hashes_after,
        'historical_file_sha256': hashes_after, 'wall_seconds': round(time.time() - t0, 1)}, indent=2))
    print('historical label CSV and reports unchanged:', hashes_before == hashes_after)


if __name__ == '__main__':
    main()
