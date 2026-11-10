# scripts/

Research scripts, grouped by pipeline stage. Each script's docstring gives its exact command
line. Scripts marked **Metashape** need the separately licensed Metashape Python module
(set `BD3D_METASHAPE_PYTHON` to an interpreter that can `import Metashape`). Everything else
runs in the project environment.

## 1. Image selection (manifests)

| script | what it does |
|---|---|
| `make_gpu_pilot_manifests.py` | ARF dog GPU pilot: 50-image reference, `random10`, `side40` (40% contiguous side removed). |
| `make_dog_failure_manifests.py` | More ARF dog cases at the same settings: `random15` ×2, `random20` ×2, `side25`. |
| `series_split_from_poses.py` | Assigns reference cameras to capture series by elevation (pose-based k-means) and compares that with the split at EXIF time gaps. Gaps that would leave a series of fewer than `--min-series-size` photos are ignored. |
| `make_pahma_manifests.py` | PAHMA 5-613 coffin candidates from a confirmed series split: `drop_s3`, `drop_s2`, `random30` ×2, `side40_s1`, `random60`. |
| `historical_subset_manifests.py` | Rebuilds the 16 historical ARF dog subsets as current-format manifests. |

## 2. Reconstruction (Metashape)

| script | what it does |
|---|---|
| `run_watched.py` | **Metashape.** General driver: runs manifests in sequence under the resource watchdog. Checks the license before each job and retries license denials. Records alignment failures as results and continues; stops on any watchdog or resource stop. |
| `run_gpu_pilot.py` | **Metashape.** Earlier driver used for the dog GPU pilot. Superseded by `run_watched.py`. |
| `build_image_quality_cache.py` | Per-image quality from a finished reference run (one `analyzeImages` pass), reused by subset runs via `reconstruct --image-quality-cache`. |
| `overnight_chain.py`, `overnight_continue.py` | **Metashape.** The unattended 2026-11-10 run: coffin candidates, dog failure cases, scoring, pilot models, notebook. Kept as the record of how those results were produced. |

## 3. Camera tables from historical projects

| script | what it does |
|---|---|
| `historical_camera_tables.py` | **Metashape.** Opens copies of the 16 historical projects read-only and writes `cameras.csv` per run. |
| `historical_reference_cameras.py` | **Metashape.** The same for the historical 50-image reference. |
| `smoke_check_camera_table.py` | Sanity checks on a run's `cameras.csv` (aligned fraction, missing reasons, plots). |

## 4. Evaluation (labels)

| script | what it does |
|---|---|
| `rescore_similarity_icp.py` | Evaluator v2 (similarity ICP from a centroid start). **Superseded**; kept because v2 labels are part of the audit. |
| `rescore_camera_init_icp.py` | Evaluator v3 applied to the 16 historical runs, producing `results/geometry-labels-v3`. |
| `score_runs_v3.py` | Evaluator v3 for new run folders (`cameras.csv` + `raw-mesh.obj`) against a reference run. |
| `combine_labels.py` | One `labels.csv` per artifact, with missing-surface (reference→candidate) and extra-surface (candidate→reference) breakdown. |
| `recheck_bd3d_ml_claims.py` | Recomputes three earlier claims from the Aphoma `bd3d-ml` branch with v3 labels. |

## 5. Models

| script | what it does |
|---|---|
| `synthetic_coverage_gap.py` | Synthetic benchmark comparing mean, Ridge, PointNet and attention. Two conditions, 5 seeds, TensorBoard logs in `runs/synthetic`. |
| `pilot_set_models.py` | Real-data pilot comparing mean, reprojection RMSE, Ridge, PointNet and attention. Evaluated leave-one-run-out and leave-one-artifact-out. |

## 6. Figures and notebook

| script | what it does |
|---|---|
| `make_figures.py` | Every README figure (`docs/figures/*.png`), plus derived tables and `stats.json` in `docs/figures/data/`. |
| `build_results_notebook.py` | Builds and executes `notebooks/results.ipynb` from saved results only. |
