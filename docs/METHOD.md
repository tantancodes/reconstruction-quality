# Method

How builds are labelled, what the label is relative to, how the evaluator works, and how the
code was checked. The results are in [FINDINGS.md](FINDINGS.md); the pipeline that produces
builds is in [PIPELINE.md](PIPELINE.md).

## Terms

| term | meaning |
|---|---|
| photo | one image file, identified by its SHA-256 in a manifest |
| build | one run folder: a manifest of photos reconstructed once in Metashape (`cameras.csv`, `run.json`, and `raw-mesh.obj` when meshed) |
| camera | a photo that aligned, with a position and view direction in the build's frame |
| reference | the build from the full capture of an object, at the same settings as its candidates |
| label | the v3 geometric error of a candidate build's mesh against the reference mesh |

A condition is a fixed, ordered photo selection with explicit settings. Its experiment ID
hashes the photo records and settings, and excludes absolute paths and timestamps, so it is the
same on any machine. Each execution of a condition gets its own run ID. Nested photo subsets of
one capture share inputs, so their builds are dependent observations; repeated builds of one
condition measure computational variation, not new objects.

## Alignment success

A build counts as aligned when it has the declared camera count, at least 2 aligned cameras, at
least 80% of photos aligned, and at least 1 valid tie point (`src/bd3d_quality/config.py`).
These are operational gates. A build that fails them is recorded as a result, not discarded,
and is reported alongside the scored builds.

## Label

The label is the symmetric RMSE between surface samples of the candidate and reference meshes,
after registration, divided by the reference bounding-box diagonal. A label of 0.01 is an RMSE
of 1% of the object's diagonal. It splits into two parts:

- missing surface: distances from reference samples to the candidate (surface the candidate
  did not reconstruct);
- extra surface: distances from candidate samples to the reference (geometry with no
  counterpart in the reference).

`labels-all.csv` also reports, for each part, the 95th percentile, the fraction beyond the
cutoff, completeness and accuracy at 1% of the diagonal, and which part dominates the squared
error (`scripts/combine_labels.py`).

## Reference

Each label is relative to a reconstruction from the full capture of the same object at the same
settings. The reference has its own errors, so the label measures agreement with that
reconstruction, not physical accuracy. There is no ground-truth scan and no physical scale
calibration. The historical dog builds (CPU) and the GPU dog builds use different references,
so their labels are only roughly comparable.

## Evaluator v3

Code: `src/bd3d_quality/evaluation.py`. Settings: `settings.json` beside each label file.

1. Initial transform. A similarity transform (rotation, translation, uniform scale) is fitted
   from the candidate's camera centres to the reference's camera centres over shared photos.
   Meshes share their build's frame, so this does not depend on the mesh shapes.
2. Refinement. Point-to-point ICP with uniform scale, on 20,000 registration samples per mesh,
   for at most 60 iterations, with a correspondence cutoff of 5% of the reference diagonal.
3. Scoring. Distances on 100,000 separate evaluation samples per mesh, normalised by the
   reference diagonal.
4. Checks. Registration fitness of at least 0.95 is `pass`, at least 0.80 `caution`, below that
   `manual_review`. When ICP scale and camera scale differ by more than 2.5%, `scale_check` is
   `disagree`.

The camera fit needs the candidate and the reference to name photos the same way and to share
enough aligned cameras; a build that shares too few cannot be scored this way
(`results/geometry-labels-v3/NOTES.md`).

## Why v3: the evaluation audit

The first evaluator (v1) started ICP from a centroid shift with no rotation and no scale.
Metashape builds without markers are in arbitrary frames, and 7 of the 16 historical dog builds
were rotated far enough that ICP converged to a wrong pose. v2 allowed scale from the same start
and did not help. v3 starts from the camera fit instead. The audit, the rescored labels and the
claims they overturn are in [FINDINGS.md §1](FINDINGS.md#1-evaluation-audit); the diagnosis
step by step is in `results/geometry-labels-v3/NOTES.md`, and the v2 labels are kept in
`results/geometry-labels-v2/` for comparison.

## Validation

- Unit and regression tests in `tests/` cover manifests and hashing, settings validation,
  Metashape calls (mocked), resume and lock handling, the resource watchdog and its driver,
  the v3 scorer, label tables, camera coverage, build discovery and the viewer export.
  `pytest -m local` rescores the real builds when the local meshes are present.
- `notebooks/01_pipeline_walkthrough.ipynb` recomputes one coffin build (`side40_s1` against
  `ref145b`) from manifest to label and checks it against `results/pahma-5-613-v2/labels-all.csv`.
  Its saved outputs show the check.
- Robustness of v3 on the historical dog builds: 300 ICP iterations change no label by more
  than 1e-6, and rigid refinement at fixed camera scale changes labels by at most 0.0009
  (`results/geometry-labels-v3/NOTES.md`).

The 2026-11-06 implementation record and repository audit, written before the current
pipeline, are kept in `archive/docs/VALIDATION.md` and `archive/docs/AUDIT.md`.

## Reporting rules

Failed builds are reported, not dropped. Results are kept separate by object, reference and
settings. Correlations within one object are not read causally. The real-data model results
are a pilot on two objects and are labelled as such wherever they appear.
