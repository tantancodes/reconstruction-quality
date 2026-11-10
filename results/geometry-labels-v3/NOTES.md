# Geometry labels v3 — ARF dog, 16 historical FAST_DENSE runs

`labels_v3.csv` (primary column `label_v3`) replaces v1 (`fast_dense_geometry_modeling_dataset_v1.csv`
in the Aphoma fork) and v2 (`results/geometry-labels-v2`), which are left unchanged.
Generated 2026-11-09 by `scripts/rescore_camera_init_icp.py`; no Metashape processing.

## What went wrong in v1

**Cause.** v1 registered each candidate mesh to the 50-image reference mesh with rigid
ICP started from a centroid shift and **zero rotation**. The historical reconstructions
are unscaled local frames (no markers or scale bars), and 7 of the 16 were in frames
rotated **136–167°** from the reference (the other 9: 0.6–16.5°). ICP cannot rotate that
far from an identity start, so those 7 converged to a wrong pose that only partly
overlapped the reference. v1 also did not allow scale, although frames differ in scale
by 0.93–1.17.

**Evidence.**
- The 7 runs with a large rotation are exactly the 7 v1 "caution" runs (fitness 0.81–0.86);
  all 9 small-rotation runs are "pass".
- In those 7 runs, 14–19% of points lay beyond the 5% cutoff in *both* directions
  (candidate→reference and reference→candidate) and carried 87–88% of the squared
  error. Two-sided outliers indicate misplaced geometry, not missing coverage. This is
  what produced the ~0.043 plateau.
- 15 of 16 v1 registrations hit the 60-iteration cap without converging.
- Allowing scale from the same centroid start (v2) did not help: ICP shrank those
  candidates (scale 0.74–0.88 vs 0.93–1.06 from cameras) and errors stayed ≥ 0.043.
- Starting from the camera transform, the same 7 runs fit at 0.996–1.000 *before any
  refinement*.
- Unaffected v1 pass labels were still inflated by the unmodelled scale difference
  (e.g. structured-uniform: scale 1.17, v1 0.0188 → v3 0.0032).

## Fix (evaluator v3)

1. Initial transform = similarity fit (rotation, translation, uniform scale) of each run's
   aligned camera centres onto the reference run's cameras over shared images. Meshes
   share their run's frame, so this is independent of the mesh shapes. Camera shape misfit
   after the fit is ≤ 1e-4 of the camera spread in every run.
2. Refinement = v1's point-to-point ICP loop with uniform scale allowed.
3. Unchanged from v1: mesh loading and sampling code (imported), seeds, 20k/100k samples,
   60 iterations, tolerances, 5% cutoff, symmetric normalised RMSE target, QC thresholds.
4. Check: `scale_check = disagree` if |ICP scale / camera scale − 1| > 2.5%.

## v3 result

- All 16 runs QC pass (fitness 0.976–1.000); all 16 `scale_check = agree` (max deviation 2.3%).
- `label_v3` ranges from 0.0031 to 0.0107 (median 0.0037). 15 runs are within 0.0031–0.0057.
  `image-count-40-seed-20260823-v1` is the outlier (0.0107, fitness 0.976,
  2.1% candidate-side points beyond cutoff, i.e. extra candidate geometry).
- The structured central-gap run scores 0.0033, among the best. The v1 conclusion that
  it was the worst condition was a registration artifact.
- Robustness: 300 iterations change no label by more than 1e-6. Rigid refinement at
  fixed camera scale changes labels by ≤ 0.0009.

## Caveats

- The reference is itself a reconstruction (50 images, same pipeline), not ground truth.
- Labels now vary little (15 of 16 runs within 0.0026 of each other), which limits what a
  model can learn from this dataset. Treat it as one object, 16 related subsets.
- Camera initialisation needs each run's cameras and the reference's in the same image
  naming; a run that shares too few images with the reference cannot be scored this way.
