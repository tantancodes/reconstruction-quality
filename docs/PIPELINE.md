# Pipeline

From photos to labels, figures and the two interfaces. What each stage writes is listed beside
it. The method behind the labels is in [METHOD.md](METHOD.md); Metashape setup is in
[METASHAPE.md](METASHAPE.md).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,ml,local]' -c requirements-lock.txt
python -m pytest -q
```

The core dependencies are numpy, Pillow and scipy. The `ml` extra adds torch, scikit-learn,
pandas, matplotlib, TensorBoard and the notebook tools; `local` adds CPU OpenCV for the archived
image-feature code; `test` adds pytest. `requirements-lock.txt` is a full freeze of the
environment that produced the results. It is specific to that interpreter; on another one, use
the ranges in `pyproject.toml` and record a new freeze.

Metashape is installed and licensed separately. Set `BD3D_METASHAPE_PYTHON` to an interpreter
that can `import Metashape` (see `.env.example`). Photos, Metashape projects and meshes are not
in this repository.

## Stages

The commands behind the coffin results. The dog builds use the same steps with the manifests in
`experiments/gpu-pilot-d8*/`. Stages 1–4 need Metashape and the photos, stage 5 needs the local
build folders, and stages 6–7 run from files in git. `<photos>` is the folder of the full capture.

| stage | command | writes |
|---|---|---|
| 1. manifest of the full capture | `bd3d-quality manifest create --root <photos> --dataset-id pahma-5-613-processed-145 --config experiments/pahma-5-613/config.json --output experiments/pahma-5-613/parent-145.json` | manifest: every photo with its SHA-256, the settings, the experiment ID |
| 2. reference build | `python scripts/run_watched.py --manifests experiments/pahma-5-613/ref145b.json --root <photos> --output-dir results/pahma-5-613-v2 --seconds 7200 --swap-growth-mib 6144 --rss-gib 6` | `ref145b/` with `cameras.csv`, `raw-mesh.obj`, `run.json`, `metrics-*.json`; watchdog records beside it |
| 3. image-quality cache, capture series, candidate manifests | `python scripts/build_image_quality_cache.py results/pahma-5-613-v2/ref145b --output results/pahma-5-613-v2/image_quality_cache.csv`, then `python scripts/series_split_from_poses.py --cameras results/pahma-5-613-v2/ref145b/cameras.csv --images <photos> --k 3 --output results/pahma-5-613-v2/series_split`, then `python scripts/make_pahma_manifests.py --parent experiments/pahma-5-613/parent-145.json --root <photos> --series results/pahma-5-613-v2/series_split/series.csv --seeds 20261110 20261111 --output experiments/pahma-5-613/candidates` | `image_quality_cache.csv`, `series_split/series.csv`, `experiments/pahma-5-613/candidates/*.json` |
| 4. candidate builds | `python scripts/run_watched.py --manifests experiments/pahma-5-613/candidates/*.json --root <photos> --output-dir results/pahma-5-613-v2 --image-quality-cache results/pahma-5-613-v2/image_quality_cache.csv` | one build folder per candidate; an alignment failure goes to `<build>-pilot-status.json` and the driver continues |
| 5. labels | `python scripts/score_runs_v3.py --reference results/pahma-5-613-v2/ref145b --candidates results/pahma-5-613-v2/{drop_s3,drop_s2,random30_s20261111,side40_s1,random60_s20261112} --output results/pahma-5-613-v2/geometry-v3-all`, then `python scripts/combine_labels.py --artifact pahma-5-613 --scores results/pahma-5-613-v2/geometry-v3-all --runs-dir results/pahma-5-613-v2 --output results/pahma-5-613-v2/labels-all.csv` | `geometry-v3-all/labels_v3.csv` (label, QC, scale check, distances), `labels-all.csv` (adds the missing/extra breakdown and build context) |
| 6. models | `python scripts/synthetic_coverage_gap.py`; `python scripts/pilot_set_models.py --output results/pilot-real-v3-all --coffin-labels results/pahma-5-613-v2/labels-all.csv` | `results/synthetic-coverage-gap/`, `results/pilot-real-v3-all/`, TensorBoard logs in `runs/synthetic/` |
| 7. figures and notebooks | `python scripts/make_figures.py`; `python scripts/build_results_notebook.py` | `docs/figures/*.png`, `docs/figures/data/` (including `stats.json`), `notebooks/02_results.ipynb` |

`ref145b.json` is a copy of `parent-145.json`. The driver names each build folder after its
manifest file, and the earlier `ref145` attempts were stopped by license denials. The historical
dog labels in `results/geometry-labels-v3/` come from `scripts/rescore_camera_init_icp.py` (see
its docstring). The full script list, in run order, is in [scripts/README.md](../scripts/README.md).

## Manifests and provenance

`bd3d-quality manifest create` selects `.jpg`, `.jpeg`, `.png`, `.tif` and `.tiff` files
recursively, in sorted relative-path order, and fails on an unreadable supported file. The
manifest is the explicit input set: files added to the folder later are not part of it.

A manifest stores the dataset ID, description, ordered portable paths, SHA-256 hashes, byte
counts, the effective settings, the schema version, creation time and software versions. The
experiment ID hashes the schema, dataset ID, ordered photo records, count and all settings, and
excludes absolute roots, descriptions and timestamps, so the same condition has the same ID on
any machine. Order is significant. Changing a setting means a new manifest.

`bd3d-quality reconstruct` verifies every photo against the manifest before running, writes a
new build folder, and refuses to overwrite one. `run.json` records the run ID, status,
timestamps, settings, software versions, stage measurements and artifact hashes. `cameras.csv`
lists each photo with its aligned position and view direction, or the reason it did not align.
`--resume` accepts only a clean, hashed checkpoint with matching identity and settings, and
`--seed-checkpoint` copies a verified alignment-only build into a new folder to continue to the
mesh. A `.run.lock` file prevents two writers on one folder. Commands exit 0 on success and 2 on
validation, dependency or QC failure.

No refinement, cleanup, texturing, marker detection or physical scaling is applied. Raw outputs
are never overwritten. Missing metrics are recorded as null with a reason. Reprojection RMSE is
computed only from valid tie-point observations of aligned cameras.

## Resource watchdog

Each reconstruction runs as a child of `bd3d_quality.resource_monitor`, which samples macOS
memory pressure, swap growth, process RSS, wall time, output size and CPU activity, and stops
its own child when a limit is crossed. Stopped builds keep their files. Because Metashape can
override OpenMP thread requests, the watchdog also runs the child at a 20% duty cycle
(`src/bd3d_quality/resource_monitor.py`). These are sampled safeguards, not kernel limits. Any
limit raised above its default needs a recorded reason, which is stored with the build.

`scripts/run_watched.py` runs a list of manifests in order under the watchdog. Before each one
it checks the Metashape license; it retries license denials in a new folder, records an
alignment failure as a result and continues, and stops on any watchdog stop. After each
successful build it writes the viewer export (below). See its docstring for the full rules.

## Capture app

```bash
python coverage_app.py
```

A desktop app (Tkinter) for one capture, in six steps: choose the photo folder; pick an earlier
build of the same object as the reference (optional); align in Metashape; check camera coverage;
build the mesh (optional); score it against the reference with evaluator v3. "Open builds" lists
every build in `results/` and the app's builds folder (`runs/app/` by default) and shows its
coverage, score and 3D view. Alignment and meshing use the same watched commands as stage 2, with
the limits in `src/bd3d_quality/pipeline_steps.py`. The experimental quality prediction on the
coverage screen comes from models trained on the 28 pilot builds and is labelled as such.

## 3D viewer

```bash
python -m bd3d_quality.viewer_export serve
```

A browser page (`viewer/index.html`) that shows a build's mesh, its cameras and its coverage.
With a reference it also shows the surface the reference has and the build is missing, and the
surface error, after the same v3 registration as the labels. The export
(`bd3d_quality.viewer_export`) is written by `run_watched.py` and the capture app, or by hand:

```bash
python -m bd3d_quality.viewer_export build results/pahma-5-613-v2/side40_s1 --reference results/pahma-5-613-v2/ref145b --open
```

Exports go to `viewer/data/`, which is not in git because it is made from local meshes. Details,
including how to read each layer, are in [viewer/README.md](../viewer/README.md). The app and
the viewer share their colours and font through `viewer/tokens.css`.

## Code layout

| module | role |
|---|---|
| `manifests.py`, `config.py`, `provenance.py` | identity, photo verification, typed settings, atomic writes |
| `experiment_design.py` | deterministic photo subsets with overlap and dependence metadata |
| `metashape_runner.py`, `reconstruction_metrics.py` | the licensed Metashape calls, checkpoints, stage metrics, `cameras.csv` |
| `resource_monitor.py` | the watchdog |
| `evaluation.py` | evaluator v3 |
| `coverage.py` | camera frame and azimuth × elevation coverage cells |
| `builds.py`, `pipeline_steps.py`, `progress.py`, `metashape_locator.py` | the capture app's non-UI parts |
| `quality_model.py` | the experimental quality prediction |
| `viewer_export.py`, `theme.py` | viewer export and local server; shared design tokens |
| `cli.py` | the `manifest` and `reconstruct` commands |
