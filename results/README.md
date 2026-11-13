# results/

Saved outputs behind the README numbers. Photos, Metashape projects, meshes and dense clouds
are never committed; only the small files below are in git.

| path | contents | made by |
|---|---|---|
| `geometry-labels-v3/` | v3 labels, settings and per-run reports for the 16 historical ARF dog runs | `scripts/rescore_camera_init_icp.py` |
| `geometry-labels-v2/` | v2 labels (superseded, kept for the evaluation audit) | `archive/scripts/rescore_similarity_icp.py` |
| `hist16-old-alignment/*/cameras.csv` | camera tables of the 16 historical runs and their reference (`_reference/`) | `scripts/historical_camera_tables.py`, `scripts/historical_reference_cameras.py` |
| `gpu-pilot-d8/`, `gpu-pilot-d8-ext/` | ARF dog GPU runs: `cameras.csv` and `metrics-initial_alignment.json` per run, `geometry-v3/` scores, `labels.csv` | `scripts/run_watched.py`, `scripts/score_runs_v3.py`, `scripts/combine_labels.py` |
| `pahma-5-613-v2/` | PAHMA 5-613 coffin runs: `cameras.csv` per run, `geometry-v3*/` scores, `labels.csv` and `labels-all.csv` | same |
| `synthetic-coverage-gap/` | synthetic benchmark results and summary | `scripts/synthetic_coverage_gap.py` |
| `pilot-real-v3-all/` | real-data pilot: runs, predictions, summary, learning curves | `scripts/pilot_set_models.py` |
| `bd3d-ml-claims-recheck/` | recomputed bd3d-ml claims with v3 labels | `scripts/recheck_bd3d_ml_claims.py` |
| `overnight-2026-11-10/`, `overnight-2026-11-10-summary.md` | logs and summary of the unattended 2026-11-10 run | `archive/scripts/overnight_chain.py` |

The local run folders also keep `raw-mesh.obj`, `run.json`, `manifest.json` and the other
`metrics-*.json` of each scored run, which `score_runs_v3.py` needs. Dense clouds, tie points,
Metashape projects and the earlier-phase result folders were moved out of the repository.
