# Reconstruction viewer

A browser tool for inspecting photogrammetry builds: the camera network, angular coverage,
the reconstructed surface (when you have it locally), and how well the set models predict
reconstruction error. Plain ES modules and three.js from a CDN, so there is no build step.

```
python -m bd3d_quality.viewer_export serve       # serves viewer/ on 127.0.0.1:8000
```

Open it through the server. Browsers block `fetch` from `file://`.

## Data policy

| Data | Where it comes from | Published with the viewer? |
| --- | --- | --- |
| Camera poses (`cameras.csv`) for the 28 builds and 3 references | `viewer/public/`, copied from tracked `results/` by `scripts/export_viewer_public.py` | Yes. Positions, view directions, reprojection error, per-image QC. |
| v3 labels, QC, accuracy/completeness at 1% | label CSVs in `results/` (same script) | Yes |
| Out-of-fold predictions | `results/pilot-real-v3-all/predictions.csv`, `experiments/geometry-aware/results/real/predictions.csv` | Yes, as seed means per build (`viewer/public/predictions.json`) |
| Meshes, photos, Metashape projects | local only | **No.** Load them with *Open folder…*. Files are read in the browser with the File API and are never uploaded. |
| Pipeline exports (`viewer/data/`, with meshes) | `viewer_export` / `run_watched.py` / the capture app | No (gitignored) |

The status bar always says which kind of data is on screen.

## Opening data

- **Public builds.** Listed in the *Datasets* tree, with their v3 label.
- **A local build folder** (*Open folder…* or drag and drop). Any folder containing
  `cameras.csv` is a build. A folder named `ref*` or `_reference` becomes the default
  reference. A `raw-mesh.obj` next to `cameras.csv` is drawn in the reference frame via the
  camera similarity fit. No Python step is needed.
- **A single `cameras.csv`** (*Open cameras.csv…*).
- **Pipeline exports** (`viewer/data/<name>/scene.json`): the mesh, missing surface and surface
  error from the v3 scorer. These are written by the pipeline as before:
  - `scripts/run_watched.py` exports every successful run (`--viewer-reference <run folder>`;
    pass `--no-viewer` to skip it).
  - The capture app's **3D view** button exports the current build and opens
    `?object=<name>`.
  - By hand:

    ```
    python -m bd3d_quality.viewer_export build results/x/run1 [--reference results/x/ref] --open
    python scripts/export_viewer.py results/pahma-5-613-v2 --reference ref145b --runs side40_s1 drop_s2 --open
    ```

URL parameters: `?dataset=<key>&ref=<key>`, or `?object=<export name>` for the capture app.

## What is implemented

| Area | Implemented | Not implemented / proposed |
| --- | --- | --- |
| Camera inspection | Frustums (dashed = reference photo not in this build), optical-axis rays, picking, nearest-neighbour line, look-through-camera, sortable/filterable camera table with CSV export | Image thumbnails (photos are not published; possible for local folders) |
| Coverage | Azimuth × elevation grid (12 × 3, cell missing below 2 cameras) and frame, ported from `coverage.py`. Run-vs-reference comparison with an Umeyama similarity fit. Figures: az–el scatter, largest azimuth gap per band, nearest-neighbour angle histogram (SVG + CSV export) | Coverage from tie points or surface visibility. Only camera directions are used. |
| Surface | Mesh, missing-surface and error colouring from pipeline exports; local OBJ placed by the camera fit | Computing surface error in the browser (needs the v3 scorer) |
| Evaluation | Predicted vs measured scatter from saved out-of-fold predictions, all models per split, MAE identical to the saved `summary.csv` (checked at export), Spearman ρ inside each held-out fold | Running the trained models in the browser (planned: JS forward pass checked against PyTorch) |
| Figures | Viewport PNG at 1/2/4×, every plot as SVG and its data as CSV | Attention maps, ablation figures |

## Units and conventions

- **Frame.** Origin at the point nearest all optical axes; up = normal of the camera ring;
  azimuth 0° = the first aligned camera (`coverage.camera_frame`). Displayed as up = +y.
- **1 display unit** = half the reference camera spread for camera-only data, or the scale of
  the export.
- **Labels and errors** are % of the reference bounding-box diagonal (v3 scorer, `docs/METHOD.md`).
- **Colours and font** come from `viewer/tokens.css`, which the capture app also reads
  (`bd3d_quality.theme`). Red is reserved for problems.

## Tests

```
node --test tests/viewer/geom.test.mjs tests/viewer/compare.test.mjs tests/viewer/data.test.mjs
python tests/viewer/make_fixtures.py     # regenerate fixtures from coverage.py after changing it
```

The JS coverage port is checked against fixtures produced by `coverage.py` on every tracked
build: counts, missing cells, gap descriptions and the similarity fit. Node 18 or newer, with no
npm packages.
