# scripts/

The scripts that reproduce the current results, in run order. Each docstring gives the exact
command line. Steps 2 and 6 need Metashape (set `BD3D_METASHAPE_PYTHON`, see
[docs/METASHAPE.md](../docs/METASHAPE.md)). Steps 3–5 and 7–9 need the photos or the local build
folders. Steps 11–16 run from files in git. The commands for the coffin builds are in
[docs/PIPELINE.md](../docs/PIPELINE.md).

| step | script | what it does | inputs | outputs |
|---|---|---|---|---|
| 1 | `bd3d-quality manifest create` (package CLI) | manifest of a full capture | photos, `experiments/*/config.json` | parent manifest in `experiments/` |
| 2 | `run_watched.py` | runs manifests in order in Metashape under the resource watchdog; writes the viewer export after each build | manifests, photos | one build folder per manifest, watchdog records, `viewer/data/<build>/` |
| 3 | `build_image_quality_cache.py` | per-photo image quality from a finished reference, reused by its subsets | reference build | `image_quality_cache.csv` |
| 4 | `series_split_from_poses.py` | assigns reference cameras to capture series by elevation and compares with the split at capture-time gaps | reference `cameras.csv`, photos | `series_split/series.csv` |
| 5 | `make_gpu_pilot_manifests.py` | dog manifests: 50-photo reference, `random10`, `side40` | dog parent manifest | `experiments/gpu-pilot-d8/` |
| 5 | `make_pahma_manifests.py` | coffin candidate manifests: `drop_s2`, `drop_s3`, `side40_s1`, random subsets | parent manifest, `series.csv` | `experiments/pahma-5-613/candidates/` |
| 6 | `run_watched.py` | the candidate builds | candidate manifests, photos, cache | candidate build folders |
| 7 | `score_runs_v3.py` | evaluator v3 for candidate builds against a reference | build folders (`raw-mesh.obj`, `cameras.csv`, `run.json`) | `geometry-v3*/labels_v3.csv`, `settings.json` |
| 8 | `combine_labels.py` | one label table per object, with the missing/extra breakdown | `labels_v3.csv`, build folders | `labels.csv`, `labels-all.csv` |
| 9 | `rescore_camera_init_icp.py` | evaluation audit: evaluator v3 on the 16 historical dog builds | historical label table, reports and meshes (local) | `results/geometry-labels-v3/` |
| 10 | `recheck_bd3d_ml_claims.py` | recomputes three earlier `bd3d-ml` claims with v1 and v3 labels | Aphoma modelling table (local), `geometry-labels-v3/labels_v3.csv` | `results/bd3d-ml-claims-recheck/` |
| 11 | `synthetic_coverage_gap.py` | synthetic benchmark: mean, Ridge, PointNet, attention | none | `results/synthetic-coverage-gap/`, `runs/synthetic/` |
| 12 | `pilot_set_models.py` | real-data pilot, leave one build out and leave one object out | label tables, each build's `cameras.csv` | `results/pilot-real-v3-all/` |
| 13 | `build_quality_models.py` | the capture app's experimental quality models, trained on all pilot builds | as step 12 | `models/quality-attention/` (not in git) |
| 14 | `make_figures.py` | every figure in `docs/figures/` and the derived tables | `results/` files in git | `docs/figures/*.png`, `docs/figures/data/` |
| 15 | `build_results_notebook.py` | builds and runs the results notebook | as step 14 | `notebooks/02_results.ipynb` |
| 15 | `build_walkthrough_notebook.py` | builds and runs the walkthrough of one coffin build, manifest to label | `side40_s1` and `ref145b` meshes (local) | `notebooks/01_pipeline_walkthrough.ipynb` |
| 16 | `export_viewer.py` | viewer export of several builds against one reference | build folders | `viewer/data/<object>/` |

`notebooks/03_attention_maps.ipynb` has no builder; it was written by hand and is kept with its
outputs.

Historical and one-off scripts are in `archive/scripts/`: the camera-table extraction from the
historical Metashape projects (`historical_*`), the manifests for the extra dog builds in
`experiments/gpu-pilot-d8-ext/` (`make_dog_failure_manifests.py`), the path scrubber
(`scrub_paths.py`), a camera-table sanity check (`smoke_check_camera_table.py`), evaluator v2,
the first pilot driver and the unattended 2026-11-10 run. See [archive/README.md](../archive/README.md).
