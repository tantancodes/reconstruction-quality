# Archival tabular dataset preparation — 2026-11-06

## Result and scope

Dataset: `results/pytorch-ready-archival-v2/`. **Three archival observations,
two provisional reference-labeled rows, one physical object group.** This is a
PyTorch-compatible numeric format, not a statistically adequate training dataset.
No Metashape process, image decoding, feature detector, geometry computation,
model fit, PyTorch import or train/test split was performed in this preparation.
Only seven tiny array/schema/hash tests were run (7 passed in 0.04 seconds).
Version 1 is an intermediate generated artifact retained untouched; use version 2.

Files:

- `observations.csv` / `observations.json`: all three observations, metadata, predictors, optional later-stage measurements and explicitly masked targets.
- `arrays.npz`: non-pickled float32 predictor/target matrices and boolean finite-value masks.
- `schema.json`: exact column/row order, primary target, grouping and readiness flags.
- `feature_dictionary.json`: field definitions, units, roles and missing-value counts.
- `provenance.json`: source/file hashes, selected images, settings, software and builder-source hash.
- `leakage_audit.json`: actual image overlaps, descriptive correlations and split blockers.
- `validation.json`: explicit successful verification checks and prohibited actions not performed.
- `dataset_files.sha256.json`: hashes of the generated bundle's other files.

All generated data remain ignored by Git. Originals and prior datasets are not
rewritten. Reproduce into a fresh directory:

```bash
.venv/bin/python -m bd3d_quality.tabular_dataset \
  --results results \
  --historical-root <LOCAL>/Documents/BD3D_Test_Output \
  --output results/pytorch-ready-archival-fresh
.venv/bin/python -m pytest -q tests/test_tabular_dataset.py
```

The builder refuses an existing output directory. It uses existing NumPy only;
no ML packages or new dependencies are required. Original archival inputs are
local research data, not committed fixtures.

## Provenance verification

The original archival CSV was matched cell-for-cell against its JSON counterpart.
Unique run IDs, the 50/25/25 input counts, selected image records, source report
IDs and source manifest counts were checked. Audit/geometry/report/manifest/mesh
file hashes were rechecked, as were embedded report/manifest contents and feature
settings signatures. Three small original mesh files were read only for streaming
hashes; no geometry was recomputed. The cached image-feature and ORB-pair files
match their recorded hashes. Images join by filename **and** content hash, with a
complete nonduplicated pair set in the original one-way measurement direction.
Image means and pair means were independently recomputed from those caches and
compared against every original archival CSV row.

Mesh vertex/face counts agree across the source report, geometry record and
Open3D audit. Source surface area, registration diagnostics and Chamfer values
agree with the source CSV. All three runs share one reference hash, reconstruction
profile/settings and geometry evaluator settings. Later sparse metrics explicitly
retain the historical `post_error_reduction_pre_dense` stage.

**Verification limit:** original photographs were hash-verified earlier, but were
not reread or decoded in this lightweight pass. The hash-linked current cache and
historical manifest establish intended input provenance, not independent proof of
which bytes existed when the historical software ran. Visual registration review
has not been performed. Capture session identifiers/dates have not been established
from independent acquisition records: `capture_session_id` is missing, and all
shared-source images are conservatively grouped together.

| Historical run | Input count | Vertices | Faces | Reference-label status |
| --- | ---: | ---: | ---: | --- |
| f09b0233… (50-image reference) | 50 | 33,570 | 67,136 | Self-comparison; all reference targets masked |
| 9ecba7a0… (seed 20260823) | 25 | 31,991 | 63,978 | Numeric QC passed; provisional, visual QC pending |
| dd072dbe… (seed 20260824) | 25 | 31,138 | 62,272 | Numeric QC passed; provisional, visual QC pending |

Dense-point counts and ground-truth accuracy are missing in every row. The legacy
baseline's total runtime conflicts with stage durations; those values are retained
in provenance warnings, not promoted into reliable runtime targets.

## Predictor columns

`X_pre`, shape **(3, 7)**, contains only measurements available from the inputs:

| Column | Definition / units |
| --- | --- |
| `input_image_count` | Selected, validated image-record count |
| `image_brightness_mean` | Mean per-image grayscale brightness, 0–255 |
| `image_contrast_std_mean` | Mean per-image population grayscale standard deviation |
| `image_laplacian_variance_mean` | Mean per-image four-neighbor interior Laplacian variance, intensity² |
| `image_entropy_bits_mean` | Mean per-image 256-bin grayscale Shannon entropy, bits |
| `image_orb_keypoint_count_mean` | Mean cached ORB keypoint count, cap 5,000 |
| `pair_ratio_match_count_mean` | Mean one-way Hamming k=2 ratio-match count over all selected pairs, ratio 0.75; not geometric overlap |

Cached image analysis used Pillow EXIF transpose, RGB-to-L conversion, and a
Lanczos thumbnail with maximum dimension 1,600. ORB used the same maximum dimension.
These statistics are not calibrated radiance or resolution-invariant sharpness.
Exact settings and software are embedded in provenance. No global normalization,
scaling, imputation, PCA or feature selection was fitted.

`X_post_refinement`, shape **(3, 4)**, is a separate optional feature set:

- `post_refinement_alignment_fraction`
- `post_refinement_valid_tie_point_count`
- `post_refinement_reprojection_rmse_px`
- `post_refinement_observations_per_valid_tie_point`

These measurements occur after historical refinement, before dense reconstruction.
They must not appear in an input-only prediction task. In a distinct internal
consistency task, reprojection RMSE could instead be an outcome predicted from
input-only features; then it and all derived versions must be excluded from inputs.
It measures internal image reprojection fit, not physical surface accuracy.

## Continuous target columns

`y_reference`, shape **(3, 4)**:

| Target | Meaning | Preferred direction |
| --- | --- | --- |
| `target_reference_chamfer_mse` | Half the sum of directional nearest-sample mean squared distances; historical export units² | Lower |
| `target_reference_chamfer_rmse_normalized` | sqrt(Chamfer MSE) / reference bounding-box diagonal; dimensionless | Lower |
| `target_reference_completeness_fraction` | Fraction of reference samples within 1% of reference diagonal of candidate samples | Higher |
| `target_candidate_coverage_fraction` | Fraction of candidate samples within 1% of reference diagonal of reference samples | Higher |

Primary provisional regression target: **normalized reference Chamfer RMSE**.
The two subset values are approximately **0.044899** and **0.042442**. Their
reference completeness fractions are **0.335667** and **0.352667**. Measurements
use 3,000 area-sampled points per mesh, deterministic seed, rigid ICP with a
200-iteration cap, no scale fitting and numeric registration QC. The previous
60-iteration rejected results remain separate evidence. Shared-reference fitness
is only approximately 0.827; passing the operational gate does not prove correct
registration. Sampling uncertainty has not been quantified.

These labels describe **agreement with a reconstruction reference**, not true
physical accuracy. Dimensionless normalization permits a common mathematical
scale; it does not establish physical calibration or fix incorrect registration.
MSE and normalized RMSE are deterministic transforms of the same measurement,
not independent target evidence. Do not put one in the predictors for the other.

The reference row's near-zero self-distance is retained solely as
`raw_self_or_reference_chamfer_mse`; all four modeling targets are missing for that
row. `reference_target_eligible` means numerically available non-self labels, not
approved/independent training data. `target_ground_truth_rmse` is null for all rows.
Mesh counts/area describe complexity and geometry extent, not accuracy. Registration
fitness, inlier RMSE and scale ratio are QC diagnostics, not default predictors or
standalone validated reconstruction-quality targets.

## Missing values and tensor contract

CSV blank = JSON null = numeric NaN; none is converted to zero. `arrays.npz` stores
only float32 and boolean arrays, with no pickled objects. Each numeric array has a
matching finite-value mask. `reference_row_mask = [false, true, true]`. The baseline
row is NaN in every reference-target column. A future consumer must index valid
labels before calculating a loss; multiplying a NaN loss by a zero mask is unsafe.
Column order and row IDs come from `schema.json`. IDs/groups remain metadata.

No PyTorch tensors or loaders were instantiated. A future training stage can load
these plain arrays without changing the measurement data, after obtaining adequate
independent data and an approved split/preprocessing protocol.

## Leakage and dependence audit

All rows share the same artifact and source-image pool; conservative object group
`arf-dog` must stay intact across any future split. There is only one group, so a
meaningful independent train/validation/test partition cannot currently exist.
The two subset rows share **14/25 images each** (Jaccard **14/36 = 0.3889**). Each
subset shares all 25 inputs with the 50-image reference. Thus reference errors,
view coverage, material, lighting and acquisition conditions are shared.

The only two non-self labeled rows both contain 25 images: there is no labeled
image-count variation. Post-refinement alignment fraction is constantly 1.0. ORB
means are near the detector cap (approximately 4,953–4,960), limiting interpretation.
The seven input features over three centered observations have rank at most two.
Descriptive n=3 correlations include contrast vs Laplacian variance around 0.997,
but this is not inferential evidence; shared-object and selection effects cannot
be separated with three dependent conditions. With only two labeled rows, nonconstant Pearson
correlations are mechanically ±1. No p-values, coefficients or predictive claims
were fitted. Full descriptive pairs and actual overlaps are in `leakage_audit.json`.

Do not use arbitrary numeric columns from the full CSV as a predictor matrix.
Use the explicit feature whitelist. Output mesh properties, registration/target
metrics, baseline flags, hashes/IDs, selection seeds and reference provenance are
excluded. Fit any future imputation/scaling/selection on training groups only.
These split and preprocessing safeguards follow the primary documentation on
[grouped validation](https://scikit-learn.org/stable/modules/cross_validation.html)
and [leakage prevention](https://scikit-learn.org/stable/common_pitfalls.html).
Those documents were consulted; scikit-learn was not installed or imported.

## Independent labeled-data acquisition recommendation

Move reconstruction execution to a suitable shared lab workstation or remote
machine; keep this laptop for manifest checks and small tabular analysis. Preserve
image manifests/settings/software and transfer only authorized data under the
research project's access rules. Do not resume intensive laptop runs as part of
this dataset preparation.

Start a multi-object feasibility collection spanning materials, texture and shape,
with separately documented acquisition sessions per object. An illustrative planning
matrix is **20 objects × 2 sessions × 3 predeclared image-selection conditions =
120 executions**, before a small explicitly budgeted repeatability subset. This
is an acquisition-planning example, **not a statistical sample-size guarantee or
claim that 120 rows justify a neural network**. It yields at most 20 independent
object groups; multiple rows/session do not increase that count. If one of the
three conditions is only a self-compared reconstruction reference, it contributes
no ordinary agreement label: 120 executions would then supply at most 80 non-self
labels, still only 20 object groups. Determine the
larger study size from measurement repeatability, uncertainty, desired error and
object-held-out learning curves in a separate modeling stage.

Use identical efficient reconstruction settings within each comparison stratum.
Predeclare subset selection and QC; retain failures and capture session/object IDs.
Acquire independent reference scans or other validated geometry with documented
scale/calibration, uncertainty and coordinate transformations when physical accuracy
is the research target. If using full-image reconstructions instead, retain the
weaker reference-agreement interpretation and shared-image dependence. Independently
review registrations and quantify sample-count/seed sensitivity before treating
labels as reliable.

Reserve entire unseen objects, including all their sessions/subsets/repeats, for
final evaluation. A held-out session of a known object answers only within-object
acquisition transfer, not generalization to new artifacts. Keep reference creation
local to each object and avoid cross-partition learned processing. Repeat runs
estimate computational variation rather than new archaeological examples. Collect
many more independent object groups before considering neural-network evaluation;
compare future simple baselines first, under a separately authorized modeling task.

## Exact remaining blockers

1. Only **one object group** and **two provisional non-self labels**; no independent split or meaningful neural-network evaluation is possible.
2. No validated physical ground truth or established physical export units; all quality targets are reference-relative.
3. Visual registration QC and sampling/label repeatability are unmeasured; fitness is marginal and source reference errors are shared.
4. No newly completed dense-cloud/mesh validation or new bounded pilot; no audited separate dense-cloud counts/exports. The prior laptop resource gates remain unresolved, not bypassed here.
5. Historical 38-image subset membership is unresolved. The current three rows do not supply or recreate it.
6. Independent capture-session identifiers/conditions are incomplete; grouping is conservative. Historical execution-time image bytes cannot be independently recovered.
7. Legacy baseline total-runtime inconsistencies and legacy refinement/face-count behavior restrict cross-profile comparisons. Historical final sparse metrics must not be mislabeled initial alignment.
8. No model training, evaluation, hyperparameter tuning, PyTorch environment or predictive result has been produced; those remain separate work.
