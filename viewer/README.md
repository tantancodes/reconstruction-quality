# Reconstruction viewer

A browser view of reconstruction runs: the mesh, the cameras, and the coverage gaps. Red is used
only for problems: surface the reference recovered and the run did not, surface error, coverage
gaps, and photos that failed to align.

It has two modes:

- **Build** (no reference). One run on its own, in its own frame: mesh (if meshed), cameras and
  coverage cells with fewer than 2 cameras. This is the view for a capture built from scratch.
- **Compare**. One or more runs registered onto a reference run with the v3 scorer (same
  transform as the labels), with per-vertex missing geometry and surface error.

## It is written by the pipeline

- `scripts/run_watched.py` exports every successful run to `viewer/data/<run>/` (the run on
  its own, or against `--viewer-reference <run folder>`). A failed export is logged to
  `<run>-viewer.log` and never stops the batch. Pass `--no-viewer` to skip it.
- The capture app exports when a mesh finishes. The **3D view** button (Coverage step, Mesh
  step, and a build's result screen) exports the current build, against the chosen comparison
  build when both have a mesh, then opens it in the browser.

## By hand

```
python -m bd3d_quality.viewer_export build results/x/run1 [--reference results/x/ref] --open
python scripts/export_viewer.py results/pahma-5-613-v2 --reference ref145b \
    --runs side40_s1 drop_s2 drop_s3 random60_s20261112 random30_s20261111 \
    --title "PAHMA 5-613 coffin" --open
python -m bd3d_quality.viewer_export serve          # serves viewer/ on 127.0.0.1:8000
```

Browsers will not fetch the `.bin` files from `file://`, so always open the viewer through the
server. `?object=<name>` picks an export; the drop-down lists everything in
`viewer/data/index.json`.

## Reading it

- **Missing geometry**: distance from each reference vertex to the nearest point on the
  candidate surface. The slider sets how deep a gap has to be before it is drawn (as a fraction
  of the reference bounding-box diagonal).
- **Surface error**: the candidate coloured by its distance to the reference surface, saturating
  at 2% of the diagonal.
- **Cameras**: dark grey = used, red dashed = removed for this build, solid red = failed to align.
- **Coverage**: azimuth × elevation cells from `coverage.compare` (or `coverage.analyze` with no
  reference). Red cells have fewer than 2 cameras (where the reference has 2 or more).
- **Save PNG** writes the current view at screen resolution for figures.

Colours, font and radius come from `viewer/tokens.css`, which the capture app also reads (`bd3d_quality.theme`).

The exported data (`viewer/data/`) comes from meshes that are not in git, so it is ignored too.
