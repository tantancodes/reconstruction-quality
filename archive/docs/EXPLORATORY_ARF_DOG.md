# EXPLORATORY / NOT VALIDATED — ARF Dog development data

This 2026-11-07 phase starts at `d345e20` and explicitly permits descriptive
development with the existing Dog observations. It does **not** change scientific
readiness. No new observation, reconstruction, image feature extraction, Ridge
fit, PyTorch installation or model training was performed. The user will implement
the PyTorch model separately. All historical datasets and provenance code remain
unchanged.

## Data and descriptive findings

The read-only source is `results/historical-alignment-recovered-v2/` (40 records).
Selection requires a source-verified numeric pixel-RMSE label, all seven finite
image-only predictors, and artifact group `arf-dog`. This yields **29 run rows**,
**18 distinct photo-content sets**, **50 source-photo hashes**, and **one artifact**.
**Zero rows pass the existing strict gate.** Complete cases are a development
selection, not a readiness upgrade or a representative sample of failed alignments.

Labels are historical `post_error_reduction_pre_dense` residuals, using Metashape
2.2.1 and the recorded legacy valid-tie-point observation RMSE definition. They
are not initial unoptimized alignment labels or physical mesh accuracy. Predictors
are obtainable before alignment; the target's later measurement stage is explicit.

Selected RMSE spans **0.2473299752–0.2767383823 pixels**; mean **0.2669725512**,
median **0.2704989615**, population SD **0.0095366658**. These are descriptive
sample summaries, not model prediction errors.

| Image-only predictor | Observed range | Pearson r with RMSE, rows | Pearson r, set-balanced |
| --- | --- | ---: | ---: |
| input_image_count | 25–50 | 0.892 | 0.844 |
| image_brightness_mean | 161.3473–162.8472 | 0.016 | -0.020 |
| image_contrast_std_mean | 40.5259–41.6548 | -0.500 | -0.363 |
| image_laplacian_variance_mean | 404.0177–455.7121 | -0.066 | 0.044 |
| image_entropy_bits_mean | 6.7066–6.8285 | -0.178 | -0.078 |
| image_orb_keypoint_count_mean | 4936.6667–4964.9600 | -0.421 | -0.373 |
| pair_ratio_match_count_mean | 120.1282–190.9080 | 0.032 | 0.126 |

Photo count has the strongest observed monotonic association: Spearman rho
**0.963** by row and **0.932** set-balanced. This does not establish predictive
performance, causation or that more photographs produce worse geometry. Filtering,
retained observations, image selection and repeated conditions can affect residuals.
Features also correlate with each other: brightness/entropy Pearson r=-0.750,
Laplacian variance/entropy r=0.743, and contrast/ORB count r=0.733. ORB means
vary over a narrow range near the configured 5000-keypoint cap. Such constrained
variation should not be mistaken for a broad study of scene texture or image quality.

Set-balanced statistics give each content set total weight one, shared across
its repeated runs. Weighted Spearman uses weighted midranks with ties. They are
still dependent within-Dog statistics. All tables are pooled descriptive views
across three partial profile strata; colors identify strata in scatter plots.
Weighting does not establish that their unreported settings are equivalent.
No p-values, confidence intervals, fitted lines, preprocessing fit or validation
performance estimates were computed.

## Repeats, overlap and provenance

One 50-image content set has **eight runs**; four other sets (25, 30, 35 and 40
images) have two runs each. The remaining 13 sets appear once. Thus 29 rows contain
11 rows beyond one per content set. Identical sets have equal cached features
within numerical tolerance but may have slightly different labels. The eight-run
50-photo set has descriptive RMSE population SD about **0.0000307 px** across
its recorded profile states; this is not a controlled repeatability estimate
with all settings verified.

All **406 run pairs share photographs**, with shared counts from 10 to 50.
Distinct-set Jaccard overlap spans **0.20–0.90**. The overlap graph has **one
connected component**. No split of these rows can supply unseen-artifact
generalization; even leaving identical image sets together does not produce
photo-disjoint training and testing. Keep artifact, session, source pool and
content-set metadata separate from model inputs.

| Partial historical profile | Plot label | Rows | Distinct content sets |
| --- | --- | ---: | ---: |
| SFM_ONLY | S1 | 10 | 6 |
| FULL_REFERENCE | S2 | 2 | 1 |
| FAST_DENSE | S3 | 17 | 17 |

These content-set counts overlap across profile strata and must not be summed
as independent sets. Exact compatibility hashes and reported settings are in
`provenance-and-selection.json`. They include downstream settings; EDA does not
silently merge them into a common validated alignment condition.

All 29 acquisition-session IDs are missing, all configurations are partially
reported, and keypoint limit, tie-point limit and preselection are missing for
all 29. Source-declared manifests and previous image/cache hashes support the
recorded association, but historical execution bytes are not independently proven.
Original images were not read or rehashed. See PHOTO_PROVENANCE_PROTOCOL.md and
PROVENANCE_READINESS.md for the unchanged stricter evidence requirements.

## Missingness and selection

| Scope | Each of seven predictor fields missing | Pixel-RMSE labels missing |
| --- | ---: | ---: |
| All 40 source records | 5 / 40 (12.5%) | 7 / 40 (17.5%) |
| Selected 29 complete Dog pairs | 0 / 29 | 0 / 29 |

The 11 excluded records comprise six failed Dog snapshots with features but no
label, four verified labels without matched features (three Dog reports and one
5-613 PDF), and the unlabeled 5-1405 project state. The report records exclusions
by ID. The image-count predictor is not backfilled from unmatched report metadata
when the complete image-feature association is missing. No missing value is
imputed to zero; float arrays use NaN plus boolean masks.

## Outputs and reproducible analysis

Use **`results/exploratory-arf-dog-v2/`**. The earlier exploratory-v1 output is
preserved, not an additional dataset. The current bundle has 25 files, about
2.85 MB, including:

- `report.html`: standalone report with embedded plot previews and data/PDF links.
- Six PNG/PDF figure pairs: eight-panel feature/label distributions, seven
  feature-versus-RMSE plots, Pearson and Spearman matrices (row and set-balanced),
  missingness and image-content overlap.
- `summary.csv/json`, `correlations.csv/json`, `missingness.csv/json` and
  `selected-pairs.csv`: explicit descriptive values, status labels and selection.
- `dependence.json`: content-set membership, repeats, pair overlaps and overlap graph.
- `provenance-and-selection.json`: exclusions, profiles, missing metadata and limits.
- `exploratory-arrays.npz`: float32 `(29,7)` inputs and `(29,1)` targets, boolean
  finite-value masks, an all-false strict-ready mask, column names and status.
- `analysis-manifest.json` and `file-hashes.json`: input, code and output hashes,
  software versions, source row indices and explicit no-model/no-gate-change flags.

Every JSON/CSV/NPZ artifact carries **EXPLORATORY / NOT VALIDATED** status; figures
and HTML display it visibly. Generated private analysis outputs remain Git-ignored.

This host already has Matplotlib in the system Python. The analysis used Python
3.14.7, NumPy 2.5.4 and Matplotlib 3.11.2, with the Agg backend. No dependency was
installed. Cached image values were reused, so plotting-environment Pillow/NumPy
versions do not change their original feature definitions. Source JSON float64
values drive descriptive calculations; exported development arrays use float32.

```bash
# Run only descriptive analysis; the output directory must not exist:
PYTHONPATH=src python3 archive/examples/analyze_exploratory_alignment.py \
  results/historical-alignment-recovered-v2 results/exploratory-arf-dog-fresh

# Lightweight tests (no private datasets, baselines or training):
.venv/bin/python -m pytest -q tests/test_alignment_arrays.py \
  tests/test_exploratory_alignment.py tests/test_historical_alignment.py \
  tests/test_photo_provenance.py tests/test_project_metadata.py \
  tests/test_historical_recovery.py tests/test_tabular_dataset.py
```

For another environment, `analysis` is an optional Matplotlib extra in pyproject.toml.
Install `.[analysis,test]` in that environment if needed; PyTorch and scikit-learn
are not required by this implementation. Input/output directory and hash checks
prevent overwriting historical bundles. No network code or image decoding is
part of the analysis.

## NumPy and future PyTorch interface

```python
from bd3d_quality.alignment_arrays import load_alignment_arrays

bundle = "results/historical-alignment-recovered-v2"
strict = load_alignment_arrays(bundle)  # default gate: currently 0 rows
data = load_alignment_arrays(bundle, mode="exploratory", artifact_group="arf-dog")
assert data.X.shape == (29, 7) and data.y.shape == (29, 1)
assert not data.strict_ready_mask.any()

X, y = data.X, data.y  # writable, contiguous float32 copies; no transforms fitted
feature_names = data.feature_names
row_ids = data.record_ids
content_groups, artifact_groups, strata = data.content_groups, data.artifact_groups, data.strata
sample = data[0]  # numeric x/y, masks and strict_ready only; no IDs enter x
```

`paired_only=False` retains incomplete rows with masks for metadata audits;
`dtype="float64"` retains source precision for analysis. `data.take(indices)`
selects explicit unique indices while keeping row IDs, source indices, masks,
groups and strata aligned. It creates no split and cannot promote readiness.
Source files are hash checked, record/feature order is validated, numeric arrays
are cross-checked against records, and expanded NPZ size is bounded. Arrays never
include tie-point counts, aligned-camera counts, settings, poses, mesh statistics
or identifiers as predictors.

The following is a **future interoperability example, not executed**:

```python
import torch
from torch.utils.data import DataLoader

X_tensor = torch.from_numpy(data.X.copy())
y_tensor = torch.from_numpy(data.y.copy())  # shape (N,1), matching a future scalar-output model
loader = DataLoader(data, batch_size=8, shuffle=False, num_workers=0)
```

[PyTorch documents NumPy conversion](https://docs.pytorch.org/docs/stable/generated/torch.from_numpy)
for float32/float64/bool arrays and memory sharing.
[Its DataLoader contract](https://docs.pytorch.org/docs/stable/data.html)
supports map-style indexing and default conversion of NumPy arrays. Runtime
PyTorch interoperability was not tested: PyTorch is not installed, and no neural
network, optimizer or training loop is provided here. Group selection must be
explicit before any later training; random row splitting would mix dependent data.

## Optional baseline script — prepared, not run

`archive/examples/evaluate_exploratory_baselines.py` is deliberately separate from EDA.
Its fitting path was **not executed** and has no performance results. Syntax was
checked; fold-planning tests exercise metadata only, not a fitted model.

Future execution requires `--execute-exploratory-baselines`, a full explicit
`--stratum` compatibility key and a new output directory. It holds each identical
content set together, records shared training/test photo hashes, and refuses
strata with fewer than three content sets (so S2 is ineligible). This is a separate
within-artifact exploration plan; it neither calls nor weakens the strict
artifact-held-out audit.

The prepared methods are a training-fold mean and fixed-alpha Ridge in NumPy.
Each training content set has total weight one. Scaling, centering and target
mean use training rows only; Ridge regularizes standardized coefficients and
leaves the intercept unpenalized. Alpha defaults to 1 and is not tuned. The script
would write per-row predictions and run-weighted/set-balanced MAE/RMSE in pixels,
with all fold parameters and limitations. Those would answer a narrow within-Dog
question involving shared photos, not held-out-artifact generalization.

No baseline output directory, fitted parameters, predictions or performance
numbers were created in this phase. The user's PyTorch implementation remains
the next separate development step.

## Verification

**97 lightweight tests passed in 1.06 seconds.** New tests check explicit
exploratory access, unchanged strict eligibility, hash/array/column consistency,
missing-value masks, row metadata alignment, weighted/tied correlations against
expanded-sample references, repeats, overlap and nonfitting fold plans. Plot
layouts were visually inspected. Original historical/archival bundle hashes and
all analysis output hashes were checked independently. No expensive image or
reconstruction tests were used.
