"""Evaluation audit: re-score the 16 historical ARF dog meshes with evaluator v3.

v1 started ICP from the mesh centroids. For 7 of 16 historical runs that fell into a
wrong local minimum (see results/geometry-labels-v3/NOTES.md). v3 starts from the
similarity transform that maps each run's aligned cameras onto the reference run's
cameras, then refines with ICP that allows uniform scale. The method is in
src/bd3d_quality/evaluation.py; this script applies it to the historical runs and keeps
the v1 and v2 labels alongside for comparison.

Inputs (local, not in git):
  --historical  the historical dataset folder holding
                fast_dense_geometry_modeling_dataset_v1.csv (v1 labels) and
                geometry_reports/<run_id>.json (mesh paths and SHA-256 per run)
  --cameras     results/hist16-old-alignment (cameras.csv per run and _reference/)
  --v2          results/geometry-labels-v2 (labels_v2.csv)
The meshes named in the reports must exist with the recorded hashes. The v1 label CSV,
reports and the v2 folder are hash-checked before and after and must be unchanged.

  python scripts/rescore_camera_init_icp.py \
      --historical <bd3d>/analysis_outputs/fast_dense_geometry_dataset_v1 \
      --cameras results/hist16-old-alignment --v2 results/geometry-labels-v2 \
      --output results/geometry-labels-v3
"""
import argparse
import csv
import json
import time
from pathlib import Path

from bd3d_quality.evaluation import Reference, load_cameras, load_obj_mesh, score_candidate, v3_settings
from bd3d_quality.provenance import sha256_file

LABELS_V1 = 'fast_dense_geometry_modeling_dataset_v1.csv'


def tree_hashes(folder):
    return {str(p): sha256_file(p) for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def protected_hashes(label_csv, reports, v2):
    return {**{str(p): sha256_file(p) for p in [label_csv, *sorted(reports.glob('*.json'))]}, **tree_hashes(v2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--historical', type=Path, required=True, help='folder with the v1 label CSV and geometry_reports/')
    ap.add_argument('--cameras', type=Path, required=True)
    ap.add_argument('--v2', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    hist = a.historical.expanduser().resolve()
    label_csv, reports = hist / LABELS_V1, hist / 'geometry_reports'
    before = protected_hashes(label_csv, reports, a.v2)

    v2 = {r['run_id']: r for r in csv.DictReader(open(a.v2 / 'labels_v2.csv', newline=''))}
    ref_cams_path = a.cameras / '_reference' / 'cameras.csv'
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'reports').mkdir()
    rows, t0, refs, ref_info = [], time.time(), {}, None
    for r in csv.DictReader(label_csv.open(newline='')):
        rep = json.loads((reports / f"{r['run_id']}.json").read_text())
        ref_path, cand_path = Path(rep['reference']['path']), Path(rep['candidate']['path'])
        if sha256_file(ref_path) != rep['reference']['sha256'] or sha256_file(cand_path) != rep['candidate']['sha256']:
            raise SystemExit(f'mesh hash differs from historical report: {r["experiment_id"]}')
        ref_info = rep['reference']
        if ref_path not in refs:
            refs[ref_path] = Reference.from_files(ref_path, ref_cams_path)
        scores, transforms = score_candidate(refs[ref_path], load_obj_mesh(cand_path),
                                             load_cameras(a.cameras / r['experiment_id'] / 'cameras.csv'))
        row = {'run_id': r['run_id'], 'experiment_id': r['experiment_id'], 'images_used_count': r['images_used_count'],
               'label_v1': float(r['geometry_y_normalized_symmetric_rmse']), 'qc_v1': r['registration_qc_flag'],
               'label_v2': float(v2[r['run_id']]['label_v2']), **scores}
        rows.append(row)
        (a.output / 'reports' / f"{r['run_id']}.json").write_text(json.dumps(
            {'row': row, **{k: T.tolist() for k, T in transforms.items()},
             'reference': rep['reference'], 'candidate': rep['candidate']}, indent=2))
        print(f"{r['experiment_id']:50s} v1 {row['label_v1']:.5f} v2 {row['label_v2']:.5f} -> v3 {row['label_v3']:.5f} "
              f"fit {row['fitness_v3']:.3f} s_icp {row['icp_scale_v3']:.4f} s_cam {row['camera_scale']:.4f} "
              f"{row['scale_check']} conv {row['converged_v3']} it {row['iterations_v3']}", flush=True)

    with open(a.output / 'labels_v3.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    unchanged = before == protected_hashes(label_csv, reports, a.v2)
    (a.output / 'settings.json').write_text(json.dumps({
        'evaluator_v3': v3_settings(), 'reference_mesh': ref_info, 'reference_cameras': str(ref_cams_path),
        'historical_label_csv': str(label_csv), 'v2_folder': str(a.v2),
        'v1_v2_files_unchanged': unchanged, 'wall_seconds': round(time.time() - t0, 1)}, indent=2))
    print('v1 labels/reports and v2 folder unchanged:', unchanged)
    if not unchanged:
        raise SystemExit('protected files changed')


if __name__ == '__main__':
    main()
