# Historical alignment-label recovery — 2026-11-06

See the [2026-11-07 provenance/readiness continuation](PROVENANCE_READINESS.md)
for component-specific interpretation, the deterministic 5-613 evidence gate,
independent-artifact leads and current evidence requests. Dataset counts remain
unchanged; no additional Dog observations were collected in that continuation.

This resource-light continuation starts from `d2ba5bb`. The current local bundle is
`results/historical-alignment-recovered-v2/`. The previous reviewed dataset,
dimensional/surface code, archival geometry dataset, photographs and checkpoints
are preserved. No Metashape API was invoked, reconstruction launched, images
decoded, features extracted or model trained. Only 24,372 bytes of small remote
PSX/ZIP metadata were fetched. The large PDF and photographs were not downloaded.
Generated private evidence remains local and Git-ignored.

## PAHMA 5-1405: what the small project actually contains

The [106-byte PSX](https://drive.google.com/file/d/1IofkWlnUd6SgAm1PA5XKCVmZWOe61UEY/view)
is XML pointing to `{projectname}.files/project.zip`. Its hash does not identify
a particular alignment: identical pointer text can occur in unrelated projects.
The [415-byte project.zip](https://drive.google.com/file/d/1aRLaNWxzSMmjIkArtapbAXo3ocVB-rnr/view)
contains `doc.xml`, listing one chunk, `0/chunk.zip`. It records Metashape
**2.1.3.18946**, original time `2024:12:10 18:48:14`, and last save
`2024:12:10 19:07:50`. Project paths are marked relative.

The [23,851-byte chunk.zip](https://drive.google.com/file/d/1yNoznrBaCwxKQrsGemZFNHfe4qtvOpcY/view)
contains a 70,601-byte `doc.xml`: Chunk 1, Canon EOS Rebel T6, 5184×3456,
adjusted calibration, 113 camera entries, and partial AlignCameras settings.
Camera labels run from `_MG_3783` to `_MG_3895`. All 113 match the filename
stems of the four enumerated outer-coffin TIFF series, with no missing or extra
stems. Those candidate TIFFs total 6,076,427,378 bytes and were not fetched.

There are **72 stored camera transforms across five components** (45, 17, 3, 4,
3 cameras), plus 41 cameras with no stored transform/component. This does not
establish 72 cameras in one coherent alignment or a report-aligned count. The
dataset keeps aligned-camera count null. Its count of 113 describes stored camera
entries, not a proven historical execution input count.

There are **no photo paths or processing-report references** in the three inspected
documents. The chunk points to `0/frame.zip`, expected at
`PAHMA_5_1405.files/0/0/frame.zip`; this companion was not supplied or located in
the inspected listing. The PSX and two ZIPs are not a complete project. No
referenced paths were followed automatically and no ZIP contents extracted to disk.

`OptimizeCameras/sigma0 = 0.13802073798877` is present. Its units and relationship
to the desired pixel RMS are unverified. It remains a raw optimizer diagnostic;
**it is not an RMSE label** and is not converted using an assumed scale factor.
Alignment duration and RAM fields are historical metadata, not current resource
measurements. Complete matching, filtering and optimization settings are absent.

## The 61.6 MB PDF remains unclassified

Metadata identifies [Pahma_5_1405.pdf](https://drive.google.com/file/d/1bWnnOclv3XyG2L_FKVpwRCOnZ62bZAHA/view),
application/pdf, **61,614,677 bytes**, modified `2024-12-10T19:09:35Z`, in the
same workshop Renderings folder as the project metadata. This is close to the
project save time, but filename, folder and timestamp do not prove it is a
Metashape processing report or refers to the same saved alignment state.

Metadata-only fullText index checks for Metashape, reprojection, Processing and
Report returned no matches under filename/parent constraints. Search coverage is
unknown, so those negatives do not establish absence of results. Requested
description/custom properties/checksums were not exposed by the connector.
No PDF binary, text extraction or thumbnail was fetched. Its title page and
whether it contains processing results remain unresolved under the no-download
constraint. A curator-provided first-page preview and processing-parameter pages
or small text export can resolve this without transferring the large file.

## PAHMA 5-613: count agreement is insufficient

The 145 candidate JPEG identifiers have 145 unique names, exactly
`_MG_4748.jpg`–`_MG_4892.jpg`, totaling **420,153,827 bytes**. All carry
`2026-03-06` modified dates; no content hashes were available. These dates are
later than the report dated 2025-11-02 and do not prove capture or input identity.

The verified report has 145 images, 145 aligned cameras, 346,195 tie points and
**0.748542 pixel RMS**. Its text contains no per-camera filename manifest. Notes
describe the 145-photo acquisition and image edits, but no complete run-specific
filename/hash list. Therefore there is no report camera list against which the
candidate names can be compared, and **the exact processed input set remains
unverified**. No features were attached to this label. The label is report-state
RMS with optimization recorded, not a recovered initial-alignment value.

## Additional local labels and stage distinctions

The bounded local search inspected 38 canonical JSON reports under
`<LOCAL>/Documents/BD3D_Test_Output/*/reports/` (166,717 bytes total).
The summary CSV has the same 38 run UUIDs and creates no additional observations.
The source dictionary explicitly says these sparse measurements follow error
reduction. The RESULTS.md value 0.276673 px is a nine-run aggregate mean and is
not added as another label.

| Evidence | Count | Treatment |
| --- | ---: | --- |
| Numeric legacy pixel RMSE at `post_error_reduction_pre_dense` | 32 | Source JSON/hash/section preserved; **29 additional labels** beyond the previous three local records |
| Failed legacy snapshots at `post_alignment_pre_error_reduction` | 6 | RMSE null; not treated as zero or recovered initial RMSE |
| Report-declared, hash-checked manifests and cached image associations | 35 | Seven image-only scalars aggregated from prior caches; no extraction |
| Numeric legacy labels with matched cached features | 29 | All from ARF Dog; sessions unverified and alignment settings partial |
| Successful numeric reports lacking run-specific manifests | 3 | Labels retained; photographs/features remain unlinked |

The 35 manifest-linked records cover 18 distinct image sets. Repeats and overlapping
subsets all depend on the same ARF Dog source pool. Parent-reference IDs, selection
metadata and seeds are retained as excluded metadata. A parent reference does not
automatically mean an inherited alignment; actual separate historical executions
remain repeated, dependent observations. No additional independent artifact was
obtained from this local search. The legacy numeric range is
0.24732997518409391–0.2767383823180734 px.

Separately, `initial-alignment-inventory.json` verifies four existing local pipeline
alignment events against their saved metric-file hashes:

| Existing output | Initial pixel RMSE | Raster condition |
| --- | ---: | --- |
| arf-dog-50-alignment | 0.4772728355651178 | Original 50-image set |
| arf-dog-50-alignment-v2 | 0.4884375710632685 | Original 50-image set |
| arf-dog-50-mesh-validation | 0.4600703386569713 | Original 50-image set; later work stopped |
| arf-dog-derived-648-baseline | 0.36852281222856254 | 648-pixel derivatives; separate stratum |

Two continuation/checkpoint snapshots inherit earlier alignments and are not new
events. These four labels are an auxiliary initial-stage inventory, **not pooled**
into the historical post-refinement table and not new independent objects.
Historical ledger status is not evidence of a live process.

## Current readiness and exact next evidence

The main recovered bundle contains **40 records: 39 execution/report records and
one unlabeled saved project state**. It has **33 verified historical labels**
(32 local plus the one Drive report), seven missing labels, 35 cache-linked feature
rows, and 29 rows with both a numeric label and linked features. There are three
documented artifact groups (ARF Dog, 5-613, 5-1405), but only one with features,
zero independently verified acquisition sessions, and **zero strict-ready rows**.
The nine compatibility keys include partial settings and missing-label states;
they are not nine established scientifically comparable cohorts.

`X_pre` retains only image count, mean brightness, contrast, Laplacian variance,
entropy, ORB keypoint count and pair ratio-match count. Aligned-camera/tie-point
counts, optimizer statistics, RMSE, poses, IDs, stages, run conditions and geometry
remain outside predictors. Null/blank/NaN and masks preserve missing values.
Manifest/cache agreement checks declarations and prior cache identity; original
photos were not rehashed now, and execution-time bytes cannot be proven retroactively.

One **newly project-backed candidate artifact group**, 5-1405, could become usable
with additional evidence. Together with already-labeled but provenance-blocked
5-613, there are **two near-term priority groups to complete**, not two newly
verified independent groups. 6-16015 is another potential catalog family requiring
both labels and provenance. Inner Lid remains grouped with 5-1405 pending curator
confirmation; it is not automatically another independent artifact.

1. **5-1405:** supply `PAHMA_5_1405.files/0/0/frame.zip` or an owner-exported
   camera manifest from the matching saved state, with paths, enabled/excluded
   flags, component/chunk association and exact processed image versions/hashes.
   Supply a small processing report or Tie Points/Processing Parameters/System
   pages establishing pixel RMS, measurement state, software and settings.
   A first-page preview of the large PDF can identify its document type. Confirm
   whether its report state is the inspected chunk; no need to transfer meshes,
   dense assets or the 6.08 GB TIFF set for this decision.
2. **5-613:** supply the report-state project camera/frame metadata or exported
   145-camera manifest, exact input version/format and preprocessing, enabled/
   excluded status and content hashes. Confirm whether the March-2026 JPEGs are
   those historical processed inputs or later derivatives. Session identity needs
   verification. Matching counts alone leave this label blocked.
3. **ARF Dog:** establish acquisition-session identity and recover full alignment,
   tie-point filtering and optimization settings for each intended label cohort.
   Recover run-specific manifests for the three unlinked 50-image reports.
   More repeats cannot supply independent unseen-artifact evaluation.
4. **6-16015 / Inner Lid:** recover run-specific report pages and camera manifests,
   resolve the documented image-count/version ambiguities, and verify component,
   session and artifact-family relationships as detailed in the original inventory.

No baseline was fitted. A mean is numerically definable but independent grouped
evaluation is unavailable; Ridge and a PyTorch MLP remain scientifically unjustified.
The existing grouped-split, MAE/RMSE and predicted-versus-observed evaluation plan
still applies once enough independent, compatible, fully linked groups exist.

## Reproduction and validation

The following command reads saved small metadata, 38 JSON reports/manifests and
existing caches. It has no network access code and refuses an existing output
directory. Local source paths must exist; private evidence is not committed.

```bash
.venv/bin/python examples/recover_historical_alignment.py \
  results/historical-alignment-discovery-v1 results/historical-alignment-reviewed-v2 \
  results <LOCAL>/Documents/BD3D_Test_Output \
  results/historical-alignment-recovery-v1 results/historical-alignment-recovered-fresh
.venv/bin/python -m pytest -q tests/test_project_metadata.py \
  tests/test_historical_recovery.py tests/test_historical_alignment.py \
  tests/test_tabular_dataset.py
```

The inspector bounds compressed/expanded XML, rejects DTD/entities and unsafe ZIP
members, and never follows project references. Recovery rejects manifest/hash/count
mismatches, incompatible cached photo identities, unexpected numeric legacy stages
and duplicate runs. It checks the prior reviewed bundle before use and confirms
the original four labels and image associations are unchanged. Evidence hashes
are in `recovery-file-hashes.json`; original sources remain untouched. The small
intermediate recovered-v1 bundle is preserved and is not additional observations.

Validation completed: **59 lightweight tests passed in 0.08 seconds**. Independent
checks verified all prior reviewed/archival bundle file hashes, all 38 canonical
report hashes, the three small project files, and the recovered bundle's 17 file
hashes. Array checks confirmed `(40, 7)` predictors, `(40, 1)` target, 33 finite
labels, 35 complete predictor rows, 29 paired rows and zero strict-ready rows.
`git diff --check` passed. No reconstruction or feature-extraction tests were run.
