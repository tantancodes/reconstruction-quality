"""Geometry labels v3: camera-initialised registration, then similarity-ICP refinement.

v1 started ICP from the mesh centroids. For 7 of 16 historical runs that fell into
a wrong local minimum (see results/geometry-labels-v3/NOTES.md). v3 starts from
the similarity transform that maps each run's aligned cameras onto the reference
run's cameras (shared images; the meshes live in the same frames as their
cameras), then refines with point-to-point ICP that allows uniform scale.

Unchanged from v1: mesh loading and area-weighted sampling (imported from the
historical evaluator), seeds, sample counts, 60 iterations, tolerances, 5%
correspondence cutoff, symmetric normalised RMSE target, QC fitness thresholds.

Flags: scale_check = 'disagree' when |icp_scale / camera_scale - 1| > 2.5%.
Also reported (not the label): rigid refinement at fixed camera scale, and the
label with 300 iterations as a convergence check.

No Metashape. Historical labels/reports and the v2 folder are hash-checked
before and after and must be unchanged.

  python3 scripts/rescore_camera_init_icp.py --fork ~/Documents/ChatGPT/bd3d \
      --cameras results/hist16-old-alignment --v2 results/geometry-labels-v2 \
      --output results/geometry-labels-v3
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from rescore_similarity_icp import fit, load_v1, qc, sha256  # noqa: E402

SCALE_TOLERANCE = 0.025
LONG_ITERATIONS = 300


def load_cameras(path):
    return {r['label']: np.array([float(r['x']), float(r['y']), float(r['z'])])
            for r in csv.DictReader(open(path, newline='')) if r['x']}


def camera_transform(cand, ref):
    k = sorted(set(cand) & set(ref))
    A, B = np.array([cand[i] for i in k]), np.array([ref[i] for i in k])
    T = fit(A, B, True)
    resid = np.sqrt((((A @ T[:3, :3].T + T[:3, 3]) - B) ** 2).sum(1).mean()) \
        / np.sqrt(((B - B.mean(0)) ** 2).sum(1).mean())
    return T, len(k), float(resid)


def scale_of(T):
    return float(np.cbrt(np.linalg.det(np.asarray(T)[:3, :3])))


def refine(source, target, T0, max_distance, max_iterations, rel_tol, abs_tol, with_scale):
    """v1's ICP loop, started from T0 instead of a centroid shift."""
    T = np.array(T0, dtype=float)
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
    dist, _ = tree.query(source @ T[:3, :3].T + T[:3, 3], workers=1)
    return {'transform': T, 'fitness': float((dist <= max_distance).mean()),
            'iterations': it, 'converged': converged, 'scale': scale_of(T)}


def score(v1, s, ref_mesh, cand_mesh, T, diag, cutoff):
    ref_pts = v1.sample_surface(ref_mesh, s.evaluation_sample_count, s.random_seed + 1)
    cand_pts = v1.apply_transform(v1.sample_surface(cand_mesh, s.evaluation_sample_count, s.random_seed + 1), T)
    c2r, _ = cKDTree(ref_pts).query(cand_pts, workers=1)
    r2c, _ = cKDTree(cand_pts).query(ref_pts, workers=1)
    sym = np.concatenate((c2r, r2c))
    far = sym > cutoff
    return {
        'y': float(np.sqrt((np.mean(c2r ** 2) + np.mean(r2c ** 2)) / 2) / diag),
        'c2r_rmse_norm': float(np.sqrt(np.mean(c2r ** 2)) / diag),
        'r2c_rmse_norm': float(np.sqrt(np.mean(r2c ** 2)) / diag),
        'c2r_p95_norm': float(np.percentile(c2r, 95) / diag),
        'r2c_p95_norm': float(np.percentile(r2c, 95) / diag),
        'c2r_frac_beyond_cutoff': float((c2r > cutoff).mean()),
        'r2c_frac_beyond_cutoff': float((r2c > cutoff).mean()),
        'far_share_of_sq_error': float((sym[far] ** 2).sum() / (sym ** 2).sum()),
        **{f'{d}_at_{f:g}': float(np.mean(v <= f * diag))
           for f in s.threshold_fractions_of_reference_diagonal
           for d, v in (('accuracy', c2r), ('completeness', r2c))},
    }


def tree_hashes(folder):
    return {str(p): sha256(p) for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fork', type=Path, required=True)
    ap.add_argument('--cameras', type=Path, required=True)
    ap.add_argument('--v2', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    fork = a.fork.expanduser().resolve()
    hist = fork / 'analysis_outputs/fast_dense_geometry_dataset_v1'
    label_csv, reports = hist / 'fast_dense_geometry_modeling_dataset_v1.csv', hist / 'geometry_reports'
    protected_before = {**{str(p): sha256(p) for p in [label_csv, *sorted(reports.glob('*.json'))]},
                        **tree_hashes(a.v2)}

    v1 = load_v1(fork)
    s = v1.GeometryEvaluationSettings()
    settings = {**asdict(s), 'evaluator_version': 3,
                'registration_method': 'camera_similarity_initialization_then_similarity_point_to_point_icp',
                'scale_check_tolerance': SCALE_TOLERANCE}
    settings['signature'] = hashlib.sha256(repr(sorted(settings.items())).encode()).hexdigest()[:16]

    v2 = {r['run_id']: r for r in csv.DictReader(open(a.v2 / 'labels_v2.csv', newline=''))}
    ref_cams = load_cameras(a.cameras / '_reference' / 'cameras.csv')
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'reports').mkdir()
    rows, t0, ref_info = [], time.time(), None
    for r in csv.DictReader(label_csv.open(newline='')):
        rep = json.loads((reports / f"{r['run_id']}.json").read_text())
        ref_path, cand_path = Path(rep['reference']['path']), Path(rep['candidate']['path'])
        if sha256(ref_path) != rep['reference']['sha256'] or sha256(cand_path) != rep['candidate']['sha256']:
            raise SystemExit(f'mesh hash differs from historical report: {r["experiment_id"]}')
        ref_info = rep['reference']
        ref_mesh, cand_mesh = v1.load_obj_mesh(ref_path), v1.load_obj_mesh(cand_path)
        diag = float(np.linalg.norm(np.ptp(ref_mesh.vertices, axis=0)))
        cutoff = s.icp_max_correspondence_distance_fraction * diag
        reg_ref = v1.sample_surface(ref_mesh, s.registration_sample_count, s.random_seed)
        reg_cand = v1.sample_surface(cand_mesh, s.registration_sample_count, s.random_seed)

        T0, n_shared, cam_resid = camera_transform(load_cameras(a.cameras / r['experiment_id'] / 'cameras.csv'), ref_cams)
        cam_scale = scale_of(T0)
        init_d, _ = cKDTree(reg_ref).query(v1.apply_transform(reg_cand, T0), workers=1)
        args = (cutoff, s.icp_max_iterations, s.icp_relative_tolerance, s.icp_absolute_tolerance_fraction * diag)
        sim = refine(reg_cand, reg_ref, T0, *args, with_scale=True)
        rig = refine(reg_cand, reg_ref, T0, *args, with_scale=False)
        long = refine(reg_cand, reg_ref, T0, cutoff, LONG_ITERATIONS, *args[2:], with_scale=True)
        m = score(v1, s, ref_mesh, cand_mesh, sim['transform'], diag, cutoff)
        m_rig = score(v1, s, ref_mesh, cand_mesh, rig['transform'], diag, cutoff)
        m_long = score(v1, s, ref_mesh, cand_mesh, long['transform'], diag, cutoff)
        ratio = sim['scale'] / cam_scale
        row = {
            'run_id': r['run_id'], 'experiment_id': r['experiment_id'], 'images_used_count': r['images_used_count'],
            'label_v1': float(r['geometry_y_normalized_symmetric_rmse']), 'qc_v1': r['registration_qc_flag'],
            'label_v2': float(v2[r['run_id']]['label_v2']),
            'label_v3': m['y'], 'qc_v3': qc(sim['fitness']), 'fitness_v3': sim['fitness'],
            'fitness_at_camera_init': float((init_d <= cutoff).mean()),
            'icp_scale_v3': sim['scale'], 'camera_scale': cam_scale, 'scale_ratio': ratio,
            'scale_check': 'agree' if abs(ratio - 1) <= SCALE_TOLERANCE else 'disagree',
            'iterations_v3': sim['iterations'], 'converged_v3': sim['converged'],
            'camera_shared_images': n_shared, 'camera_shape_residual': cam_resid,
            **{k: v for k, v in m.items() if k != 'y'},
            'label_v3_rigid_camera_scale': m_rig['y'], 'fitness_rigid_camera_scale': rig['fitness'],
            f'label_v3_{LONG_ITERATIONS}it': m_long['y'], f'scale_v3_{LONG_ITERATIONS}it': long['scale'],
            f'converged_v3_{LONG_ITERATIONS}it': long['converged'],
        }
        rows.append(row)
        (a.output / 'reports' / f"{r['run_id']}.json").write_text(json.dumps(
            {'row': row, 'camera_init_transform': T0.tolist(), 'transform_v3': sim['transform'].tolist(),
             'transform_rigid_camera_scale': rig['transform'].tolist(), 'reference': rep['reference'],
             'candidate': rep['candidate']}, indent=2))
        print(f"{r['experiment_id']:50s} v1 {row['label_v1']:.5f} v2 {row['label_v2']:.5f} -> v3 {m['y']:.5f} "
              f"fit {sim['fitness']:.3f} s_icp {sim['scale']:.4f} s_cam {cam_scale:.4f} {row['scale_check']} "
              f"conv {sim['converged']} it {sim['iterations']}", flush=True)

    with open(a.output / 'labels_v3.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    protected_after = {**{str(p): sha256(p) for p in [label_csv, *sorted(reports.glob('*.json'))]},
                       **tree_hashes(a.v2)}
    unchanged = protected_before == protected_after
    (a.output / 'settings.json').write_text(json.dumps({
        'evaluator_v3': settings, 'reference_mesh': ref_info, 'reference_cameras': str(a.cameras / '_reference' / 'cameras.csv'),
        'historical_label_csv': str(label_csv), 'v2_folder': str(a.v2),
        'v1_v2_files_unchanged': unchanged, 'wall_seconds': round(time.time() - t0, 1)}, indent=2))
    print('v1 labels/reports and v2 folder unchanged:', unchanged)
    if not unchanged:
        raise SystemExit('protected files changed')


if __name__ == '__main__':
    main()
