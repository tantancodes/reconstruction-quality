# results/

The small files behind the README and docs, plus the per-build inputs the pilot needs. Photos,
Metashape projects, meshes, dense clouds and raw logs stay local.

| folder | holds | made by |
|---|---|---|
| `pahma-5-613-v2/` | coffin builds: `cameras.csv` per build, `geometry-v3*/` scores, `labels.csv`, `labels-all.csv` | `scripts/run_watched.py`, `score_runs_v3.py`, `combine_labels.py` |
| `gpu-pilot-d8/`, `gpu-pilot-d8-ext/` | dog GPU builds: `cameras.csv` and `metrics-initial_alignment.json` per build, `geometry-v3/` scores, `labels.csv` | the same |
| `hist16-old-alignment/` | `cameras.csv` of the 16 historical dog builds and their reference (`_reference/`) | `archive/scripts/historical_camera_tables.py`, `historical_reference_cameras.py` |
| `geometry-labels-v3/` | v3 labels, settings and per-build reports for the 16 historical dog builds; `NOTES.md` is the audit write-up | `scripts/rescore_camera_init_icp.py` |
| `geometry-labels-v2/` | v2 labels and reports (superseded, kept for the audit) | `archive/scripts/rescore_similarity_icp.py` |
| `bd3d-ml-claims-recheck/` | three earlier claims recomputed with v1 and v3 labels | `scripts/recheck_bd3d_ml_claims.py` |
| `synthetic-coverage-gap/` | synthetic benchmark results, summary and config | `scripts/synthetic_coverage_gap.py` |
| `pilot-real-v3-all/` | real-data pilot: builds, predictions, summary, learning curves, config | `scripts/pilot_set_models.py` |
| `overnight-2026-11-10/`, `overnight-2026-11-10-summary.md` | the 2026-11-10 unattended run: `chain-status.json` (stage outcomes and times) and a summary; its logs stay local | `archive/scripts/overnight_chain.py` |

`score_runs_v3.py` also needs each build's `raw-mesh.obj`, `run.json` and `manifest.json`, which
stay in the local build folders.
