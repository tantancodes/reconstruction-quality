# Status notes, 2026-11-06 to 2026-11-08 (superseded)

These paragraphs were the "current status" sections of the README before 2026-11-10. They
are kept as a record. Several statements are out of date: the project now trains models
(pilot only, see the README), and labels derived with geometry evaluator v1 were replaced
by v3 (`results/geometry-labels-v3/NOTES.md`).

Current continuation outcome: **resource-blocked; no newly completed dense/mesh
validation or new pilot executions**. The safety gates stopped original-raster and
derivative attempts on swap growth or timeout. A separately labeled archival
stratum now contains three real mesh observations with Open3D checks, image-hash
joins and reference-relative geometry measurements. These do not establish new-run
success or ground-truth accuracy. Latest automated suite: **74 passed**. See the
[actual measurements and remaining prerequisites](VALIDATION.md#resource-safe-pre-ml-continuation--2026-11-06).

## Lightweight archival table (no reconstruction)

The reviewed array-ready dataset is `results/pytorch-ready-archival-v2/`: **3 archival
observations, 2 provisional non-self reference labels, 1 object group**. It includes
a CSV/JSON table, plain float32 NumPy arrays with boolean masks, explicit input-only
and post-refinement feature lists, target definitions, file/row provenance hashes,
and a leakage/dependence audit. Baseline self-comparison targets are masked; physical
accuracy and dense-point counts are missing. No train/test split or model is fitted.
It is format-ready, not statistically training-ready. See
[dataset columns, verification, and remaining blockers](../archive/docs/TABULAR_DATASET.md).

The independently measured dimensional/surface pathway is preserved as a future
extension: see [its protocol, schemas and CLI](../archive/docs/DIMENSIONAL_ACCURACY.md).
It does not change the archival tables or convert reprojection error into physical
accuracy. Current research focus: historical alignment-consistency prediction from
photographs before alignment; no new reconstruction or model training is required
for dataset inventory and metadata preparation.

## Current study: historical alignment-consistency prediction

The active objective is predicting historical Metashape **pixel reprojection RMS
from photographs before alignment**. It is not physical mesh-accuracy prediction.
The current local dataset is `results/historical-alignment-recovered-v2/`:
33 verified historical RMSE labels (1 priority Drive PDF + 32 local archival
reports), including 29 additional labels recovered from ARF Dog. There are 35
manifest/cache-linked feature rows, 29 with numeric labels, all from one artifact.
The 145 candidate PAHMA 5-613 JPEGs still lack exact report/input association.
Small PAHMA 5-1405 metadata links 113 camera labels to candidate TIFF stems but
contains no verified pixel RMSE or photo paths. The 61.6 MB PDF was not downloaded.
No row passes the strict readiness gate; session/settings gaps and dependent
observations prevent meaningful grouped evaluation. No models were trained.
See [recovery findings and exact missing files](../archive/docs/HISTORICAL_ALIGNMENT_RECOVERY.md)
and the [preserved initial inventory and CLI](../archive/docs/HISTORICAL_ALIGNMENT_DATASET.md).
The dimensional/surface extension and all previous datasets remain available.

The [2026-11-07 provenance/readiness audit](../archive/docs/PROVENANCE_READINESS.md) clarifies
the five disconnected 5-1405 alignment components and provides a
[deterministic report-to-photo protocol](../archive/docs/PHOTO_PROVENANCE_PROTOCOL.md).
Counts remain 33 verified historical labels, 29 label-and-feature pairs and zero
strict-ready observations or eligible held-out groups. No Dog-derived examples
were added. The 6-19929 collection is a prospective acquisition lead; the separate
shared-drive archive link currently returns 404 through the connector.

## EXPLORATORY / NOT VALIDATED — Dog development interface

The new [exploratory analysis and loader guide](../archive/docs/EXPLORATORY_ARF_DOG.md) covers
the existing **29 paired runs, 18 overlapping image sets and one ARF Dog artifact**.
Plots, missingness, correlations, repeat audits and float32 development arrays are
in `results/exploratory-arf-dog-v2/`; start with `report.html`.
`load_alignment_arrays` defaults to the unchanged strict gate (zero real rows);
exploratory access is explicit and carries its own status and original flags.
An optional within-Dog mean/Ridge script is prepared but **has not been run**.
No model training, new image processing, reconstruction or large download occurred.
