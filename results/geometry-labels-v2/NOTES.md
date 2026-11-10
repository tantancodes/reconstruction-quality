# Geometry labels v2 — DO NOT USE label_v2 AS A TRAINING TARGET

Produced by scripts/rescore_similarity_icp.py (no Metashape; historical label CSV
and reports hash-verified unchanged). Step 0 passed: the v1 evaluator re-run and
this script's rigid path reproduce all 16 stored v1 labels exactly.

Findings (2026-11-09):
- Historical alignments have no markers/scale bars: unscaled local frames whose
  scale differs between runs by up to ~17% vs the 50-image reference (camera_scale).
- Pass runs: centroid-initialised similarity ICP recovers a scale close to the
  camera scale (within ~2.5%) and label_v2 is valid; v1 overstated their error
  (e.g. structured-uniform 0.0188 -> 0.0033).
- Caution runs: centroid-initialised ICP (rigid or similarity) falls into a wrong
  local minimum: ~15% of points beyond the 5% cutoff in BOTH directions, carrying
  ~88% of squared error. Similarity ICP then shrinks the candidate (scale 0.74-0.88
  vs camera 0.93-1.06). label_v2 is INVALID for these 7 runs.
- diagnostic_camera_init_rigid.json: initialising from the camera similarity
  transform (cameras -> reference cameras, shared images) then rigid refinement
  gives fitness ~1.0 and 0.0031-0.0058 for 15/16 runs; image-count-40-seed-20260823
  stays at 0.0116 with 2.5% candidate-side outliers (likely extra geometry).
- Conclusion: the ~0.043 "caution" plateau was a registration artifact, not surface
  error. A v3 label should use camera-initialised registration.

> Superseded: see `results/geometry-labels-v3/NOTES.md` for the v1 registration-failure write-up and the valid labels (`labels_v3.csv`).
