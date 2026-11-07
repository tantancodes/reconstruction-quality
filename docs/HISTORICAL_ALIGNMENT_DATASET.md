# Historical Metashape reprojection-error dataset — 2026-11-06

**Continuation:** [the label-recovery audit](HISTORICAL_ALIGNMENT_RECOVERY.md)
supersedes the readiness counts below. The current local bundle is
`results/historical-alignment-recovered-v2/`: 33 verified historical labels,
including 29 newly recovered local labels; all added labels are from ARF Dog.
PAHMA 5-1405 now has inspected small project metadata but no verified pixel RMSE.
Strict-ready rows remain zero. The account below documents the preserved
`d2ba5bb` checkpoint; its no-binary-download statement applies to that earlier phase.

## Research question and boundaries

Can image-derived features predict Metashape's reprojection RMS in pixels **before
camera alignment**? This is alignment-consistency prediction, not physical mesh
accuracy. A low reprojection residual can coexist with poor coverage, incomplete
alignment, incorrect calibration, scale error or poor geometry. Keep aligned-camera
fractions and failure states as diagnostics, never predictors. Dimensional/surface
accuracy remains available as a separate future extension in DIMENSIONAL_ACCURACY.md.
No proprietary MeshGate methods are claimed or used.

The saved extension was reviewed, documented and committed as **0597468** after
71 selected lightweight synthetic/compatibility tests passed. Its examples remain
synthetic; no real physical labels or new reconstruction were produced.

Current outputs:

- `results/historical-alignment-discovery-v1/drive-inventory.json`: sanitized remote file/folder metadata, identifiers, sizes and explicit listing limits.
- `results/historical-alignment-discovery-v1/`: small readable report/note snapshots; the report text has its own extraction hash, not a PDF binary hash.
- `results/historical-alignment-reviewed-v2/`: separate final reviewed alignment dataset, CSV/JSON, plain float32 arrays/masks, schema, feature dictionary, source records, overlap audit, readiness and file hashes.

Intermediate metadata-only builds under historical-alignment-v1/v2,
historical-alignment-reviewed-v1 and historical-alignment-dataset-v1 are retained, not extra observations. Use the
reviewed-v2 bundle above. Existing archival/dimensional datasets are unchanged.
No photographs, large PDFs, meshes, project assets or ZIP contents were downloaded.
Only folder/file metadata and bounded readable report/document text were accessed.

## Drive inventory findings

Scope: four requested collections beneath
[BD3D_URAP](https://drive.google.com/drive/folders/1EVxf08YZjMAppzT9mNntF-776TKWjia1).
The inventory is deliberately bounded to relevant inspected subfolders, not a claim
of an exhaustive Drive search. Folder fetches return at most 100 children. TIFF
search pagination did not expose the remaining files; capped listings are lower
bounds. A global PSX search returned no result even though direct folder listing
found a PSX, so negative search results do not prove absence. Checksums requested
through metadata were not returned by the connector: Drive IDs are not image hashes.

| Collection | Processing-report / project evidence | Candidate photographs and limits |
| --- | --- | --- |
| PAHMA 5-613 | One readable 421,753-byte processing PDF; no native project located in inspected collection model folders | Canon-JPGs enumerated in two metadata pages: exactly 145 JPEGs, 420,153,827 bytes. The visible Canon raw folder returned zero entries. |
| PAHMA 6-16015 | Notes describe several alignments/chunk merges; no readable processing report or PSX/PSZ located in inspected model folders | Canon Series1: 29 TIFFs; Series2: at least 100 TIFFs, capped listing; Nikon Series3-rightside: 56 TIFFs. Other Nikon/processed folders also exist. |
| PAHMA 5-1405 | Workshop/Renderings contains a 106-byte PSX plus 23,851-byte chunk.zip and 415-byte project.zip metadata entries. Also a 61,614,677-byte PDF, not read/downloaded; not classified as a processing report without inspection | Canon outer-coffin series: 37+38+31+7 = 113 TIFFs, 6,076,427,378 bytes. Inner-coffin folder: 11 TIFFs plus .DS_Store. Nikon listing capped at 100 entries (50 JPEG + 50 NEF in that page), not 100 independent photographs. |
| PAHMA 5-1405 Inner Lid | Notes say fullsize/websize reports were exported; those PDFs/projects were not located in inspected root/Outputs folders. One GLB output is visible | Notes describe 183 TIFF inputs: MainSeries 141, FaceCloseups 13, MiddleSeries 29. Those actual Canon input folders were not located. This is a note claim, not an enumerated 183-image manifest. |

Related components (5-1405 and Inner Lid) are conservatively one artifact family
until curator-confirmed identity/grouping is available. Four collections do not
establish four independent experimental objects. Source collection metadata names
are evidence of organization, not proof that files were used in a given run.

Additional small metadata checks found app-output ZIP/video files for 5-613, a
23-entry Other Photos folder for 6-16015 with no PDF/PSX/PSZ in that returned listing,
and general-event photos for 5-1405. These are not automatically source-image sets.
The `5-1405 README` says the final model is held in Box; Box was not accessed. Notes
mention a different external Nikon folder for Inner Lid. No external asset download
or new plugin connection was attempted.

## Precisely verified PAHMA 5-613 report

[Source PDF](https://drive.google.com/file/d/1ztQP1C2WAoL8C44kNSqBeKEDcdiF-0ot/view):
`pahma-5-613-v1-report.pdf`, report dated 02 November 2025.

- Images 145; aligned images 145; tie points 346,195 of 407,384.
- **Pixel RMS reprojection error 0.748542**, from Processing Parameters → Tie Points.
- The preceding **0.119078 is the normalized report value**, not pixels.
- The survey summary's 0.749 pix is rounded and is not the selected target.
- Accuracy High; generic preselection Yes; reference preselection Sequential.
- Key point limit 40,000; per-Mpx limit 1,000; tie point limit 10,000.
- Exclude stationary tie points No; guided matching No; adaptive fitting Yes.
- Optimization is recorded for f, k1, k2, p1, p2.
- Software version 2.1.4.19695 / Professional 2.1.4 build 19695.

The stored label is **reported project-state Tie Points RMS with optimization
recorded**, not asserted initial unoptimized alignment RMSE. The notes explicitly
say Optimize Cameras was used. Predicting this label before alignment is a different
endpoint from predicting the residual immediately after Align Photos; future data
must fix the endpoint and keep stages separate. Printed page 5 is inferred from
extracted footer order, not visually checked. File ID, section and readable-text
hash are recorded; this is not a claim to have hashed the original PDF bytes.

The [photogrammetry notes](https://docs.google.com/document/d/1nupDKfqyaTQEvX5UttlmjFdiFwonpm5Rxo1SkYx-sbU/edit)
record photography on 20 October 2025, 145 photographs, input color/brightness edits,
and importing all 145. The session ID is derived from those notes and remains
not independently verified against camera files. The current 145-JPEG folder has
March 2026 metadata, later than the report. It is a **candidate**, not a verified
byte-identical processed input set. Do not substitute RAW images, JPEG derivatives,
enhanced TIFFs, or identically named Canon photos from ARF Dog.

## Dataset schema, available rows and verification levels

One observation denotes one source-identifiable alignment/project state, not a
mesh export. `historical_alignment.validate_run` records:

- artifact ID, conservative artifact-family group, acquisition session ID/status;
- reconstruction run ID/status, source image-set ID and ordered image records;
- image count, Drive IDs or local paths, SHA-256 where known, photo/run association evidence;
- software/version, alignment configuration and its verification status;
- label value/units/definition/stage/status, source file/section/page and evidence hash;
- aligned-camera/tie-point counts as diagnostics only;
- availability, field-level evidence and explicit warnings;
- optional image-only features and their exact source-image-set signature.

Statuses distinguish document-verified, historical-computed-verified, user-supplied,
missing and ambiguous labels; candidate-only vs hash-manifest-backed image
provenance; reported vs incomplete configurations; and unknown/inferred sessions.
A source-derived report ID is not passed off as a native project UUID. Reports
with multiple RMS blocks require manual per-chunk review instead of automatic row
multiplication. Duplicate run IDs, label evidence and explicitly shared underlying
alignment-state IDs are rejected. Mesh/fullsize/websize re-exports are not new
alignment observations merely because their reports/files differ.

| Stratum | Rows | Pixel RMSE values | Linked image-only features |
| --- | ---: | --- | ---: |
| Priority Drive report (5-613, report-state RMS, Metashape 2.1.4.19695) | 1 | 0.748542 | 0 |
| Local archival ARF Dog JSON reports (post_error_reduction_pre_dense, 2.2.1) | 3 | 0.2766747534; 0.2500452950; 0.2473299752 | 3 |

The local supplement uses the three already existing historical runs and cached
hash-joined image features. It creates no new experimental execution or fabricated observation, and performs no feature extraction.
For **reprojection RMSE**, the 50-image baseline has an ordinary nonzero internal
residual; the old mesh-self-comparison exclusion does not apply. The old geometric
table and its masks are preserved. Their archival image association is supported
by historical report manifest hashes and prior cache/content-hash audits; original
execution-time bytes cannot be independently reconstructed. Original photo files
were checked for availability, not re-decoded or rehashed in this phase.

The legacy JSON definition (valid tie-point observation pixel RMSE) is not assumed
equivalent to Metashape's PDF pixel RMS under all versions/weighting settings.
Three local records also lack complete keypoint/preselection settings in their
reports. They remain configuration-partial, not silently promoted from current
code defaults. No mixed-definition/version/stage regression dataset is declared
ready.

## Model input contract

Only these seven columns enter `X_pre`:

`input_image_count`, `image_brightness_mean`, `image_contrast_std_mean`,
`image_laplacian_variance_mean`, `image_entropy_bits_mean`,
`image_orb_keypoint_count_mean`, `pair_ratio_match_count_mean`.

Definitions and cached analysis settings are retained from TABULAR_DATASET.md.
The target is `target_reprojection_rmse_pixels`. Input-count metadata reported by
a PDF is not quietly used as a substitute for a source-verified feature record.
The Drive row has missing features; the three local rows reuse the existing cache.
Aligned cameras, tie-point counts, camera poses, mesh statistics, target values,
run IDs and provenance are excluded from default predictors. Future overlap/coverage
features may use deterministic photograph-only matching or documented capture
plans, never reconstructed poses or label-selected pairs.

Numeric arrays: X_pre (4,7), y_rmse_pixels (4,1), finite-value masks and an explicit
model-input row mask. CSV blank / JSON null / array NaN means unavailable, never
zero. All four labels are source-verified, but **no row passes the current strict
model-ready gate**: Drive lacks matching inputs; local reports lack verified session
identities and complete alignment configuration. Numeric values remain available
for audit, not evidence of a trained model or valid split.

## Readiness report

| Measure | Priority Drive collections | Local archival supplement | Total |
| --- | ---: | ---: | ---: |
| Source-documented historical runs | 1 | 3 | 4 |
| Source-verified pixel-RMSE labels | 1 | 3 | 4 |
| Manifest/hash-backed matching photo sets | 0 | 3 | 3 |
| Ready for image-only feature extraction or cache reuse | 0 | 3 (already cached) | 3 |
| Documented artifact groups with labels | 1 | 1 | 2 |
| Independently verified acquisition-session IDs | 0 | 0 | 0 |
| Strict model-input eligible rows | 0 | 0 | 0 |

There is one notes-documented/inferred session for 5-613, not independently camera-
verified. Only **one artifact group has linked features**. Two names with labels
do not provide an independent train/test artifact pair when one has no verified
features; they also do not resolve uncertain cross-set photo identity. Four Drive
collections represent at most three catalog families before component review.

A training-set mean is mathematically definable, but an independent held-out
performance comparison is not feasible; **no mean predictor was fitted**. Ridge
can be numerically regularized with few rows, but meaningful grouped evaluation
and tuning are not possible here. A PyTorch MLP is not scientifically justified.
No model, preprocessing fit, split or performance result was produced.

## Exact next files and confirmations needed

**5-613:** Supply a run-specific camera/image manifest for the report state, with
exact filenames/paths, processed image version/format, enabled/excluded status,
preprocessing description and content hashes when available. Confirm whether the
145 March-2026 JPEGs are those inputs or later derivatives. Prefer a small exported
manifest over a full multi-GB project. If necessary, provide the PSX/PSZ and only
camera/chunk metadata needed to recover paths and settings—not dense/mesh assets.
The small PDF is already readable; no need to resend its numerical summary.

**6-16015:** Supply processing-parameter pages or report exports for each intended
labeled chunk/run (Canon Series1, combined Canon, Nikon right-side and merged state,
if retained), and map each to the exact image manifest and original/enhanced image
versions. Notes claim 29+132 images yet also a combined 164; later enhanced Series2
notes mention 135. Do not repair that discrepancy by assumption. The capped TIFF
listing cannot resolve it. Report stage must distinguish original vs merged or
optimized states.

**5-1405:** Supply the processing-parameter pages/text from `Pahma_5_1405.pdf`
(61.6 MB) or a new small report export identifying the run. The large PDF was not
read, so its type/content remain unknown. Map `PAHMA_5_1405.psx` to the correct
camera manifest/chunk; the workshop project is not assumed to use the 113 Canon
outer-coffin TIFFs. Confirm whether earlier October attempts, April lid/internal
sets, or the final Box-held model are intended. Folder/notes dates also require
reconciliation (April 22 folder/header versus April 20 in notes).

**Inner Lid:** Supply the fullsize/websize processing reports and v8 camera manifest
plus exact 183-TIFF input IDs/hashes (141+13+29 as notes describe). Identify disabled
images and file versions for any earlier attempts selected for study. Reports for
decimated and fullsize exports may share one alignment state: do not count them
as independent labeled runs. Locate/share the actual Canon TIFF folders; the
external Nikon link is not assumed to contain the final Canon inputs.

Across collections, recover artifact/component grouping, verified session dates,
complete matching/preselection/optimization settings, software versions, label
stage and error definition. Failed/partial alignments remain in the inventory;
missing RMSE is not imputed to zero. Acquire enough independent source-matched
artifacts in comparable strata before model evaluation. Any bulk image/project
transfer requires separate approval. Existing remote JPEG IDs are metadata, not
byte hashes. The large outer-coffin TIFF set is about 6.08 GB and is not appropriate
for automatic download here. If exact sources are confirmed, prefer a lab-side
bounded feature export with hash provenance over moving all images to this laptop.

## Reproducible commands and evaluation plan

```bash
bd3d-quality alignment-report parse REPORT_TEXT.txt --output results/parsed-fresh.json
bd3d-quality alignment-dataset build RECORD.json --output results/alignment-fresh
# Rebuild the reviewed preparation from saved discovery/caches, into a new directory:
.venv/bin/python examples/prepare_historical_alignment.py \
  results/historical-alignment-discovery-v1 results results/alignment-reviewed-fresh
.venv/bin/python -m pytest -q tests/test_historical_alignment.py tests/test_tabular_dataset.py
```

`audit_split(records, assignments)` only checks a proposed split; it does not fit
one. It rejects artifact-family/session leakage, shared image hashes, unknown
source provenance/session identity and incompatible settings/version/definition
strata. Future train/validation/test assignment must keep all related subsets,
repeated runs, sessions and image-overlap connected components together. Capture
multiple independent objects with source-matched labels; repeated reconstructions
are not extra independent objects. Decide whether the endpoint is initial,
post-optimization or another recorded state before pooling. Assess errors within
compatible strata or a separately designed controlled comparison.

After sufficient independent groups exist, compare a training-fold mean, Ridge
and a separately user-implemented small MLP. Fit imputation, normalization, feature
selection and tuning **only within training groups**, with nested group-aware
validation as data permit. Freeze an untouched object-group test set. Report MAE
and RMSE in **pixels**, predicted-versus-observed plots and residuals by artifact,
settings, image count and alignment-failure status. Weight/report object-level
performance as well as run-level error so prolific artifacts do not dominate.
Session-held-out testing within known objects answers a narrower question than
unseen-artifact generalization. Do not report training fit as generalization.

No baselines, PyTorch, image batch, alignment, dense cloud or mesh generation were
executed for this dataset. Empirical model performance and adequate independent
sample size remain unknown.

## Validation for this phase

The preserved extension passed 71 selected lightweight tests in 0.26 seconds.
The new historical parser/schema/array/leakage checks plus archival-array tests
passed 36 tests in 0.05 seconds. Test fixtures are synthetic metadata, never real
observations. No reconstruction or feature-extraction batch was run. The final
reviewed bundle and old archival bundle file hashes were independently checked.
No source images, prior datasets, original Aphoma files or Drive assets were modified.
