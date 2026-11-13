# Provenance and independent-artifact readiness — 2026-11-07

This continuation starts at clean commit `3dae4a6`. It prioritizes provenance and
independent artifact families; it adds **no reconstruction records, labels or
Dog-derived examples**. The historical dataset remains
`results/historical-alignment-recovered-v2/`. New metadata-only audit outputs are
in `results/alignment-provenance-audit-v2/`; discovery evidence is in
`results/alignment-provenance-evidence-v1/`. Original assets and earlier bundles
are preserved, including the preliminary audit-v1 output.

## A. Meaning of the PAHMA 5-1405 project state

The saved 106-byte PSX, 415-byte project archive and 23,851-byte chunk archive
were re-inspected using bounded XML/ZIP reads. There is one chunk, **not five
chunks or five independent reconstructions**. Its 113 camera entries are labeled
`_MG_3783`–`_MG_3895`. All match the previously enumerated candidate outer-coffin
TIFF stems; labels omit file extension and contain no file paths or hashes.

| Component ID / label | Stored cameras and transforms | Active in saved XML |
| --- | ---: | --- |
| 0 / Component 1 | 45 | Yes, `components active_id="0"` |
| 1 / Component 2 | 17 | No |
| 2 / Component 3 | 3 | No |
| 3 / Component 4 | 4 | No |
| 4 / Component 5 | 3 | No |
| No component | 41 camera entries, zero transforms | Not applicable |

Each of the 72 transformed entries has a finite 16-value stored matrix and a
declared component ID; no reference/transform integrity issues were detected.
This is a saved camera-state count, not independent pose validation. Camera-level
enabled flags are absent, so enabled/excluded status is not recovered from these
entries. Active component 0 contains labels `_MG_3824`–`_MG_3842`,
`_MG_3863`–`_MG_3885`, and `_MG_3893`–`_MG_3895`. Full per-camera component
membership and the 41 entries without transforms are retained in the audit JSON.

[Agisoft's component documentation](https://agisoft.freshdesk.com/support/solutions/articles/31000158896-connected-components-in-camera-alignment-procedure)
describes these as separately aligned image networks that were not connected to
each other. They are alignment components, not mesh components or independent
artifact groups. [Agisoft's API explanation](https://www.agisoft.com/forum/index.php?topic=16025.0)
shows that changing the active component changes the tie-point asset returned.
Therefore any recovered pixel RMSE must identify its component/chunk and camera
inclusion scope; a single error cannot be assigned to all 113 cameras by assumption.

`AlignCameras/reset_alignment=false` is recorded. The metadata does not preserve
the sequence of alignments/optimizations or prove that all transforms came from
one invocation. It cannot recover the initial residual. Original/last-save
software is **2.1.3.18946**, last save `2024:12:10 19:07:50`; this does not
independently date or version each earlier camera alignment operation. XML format
versions 1.2.0/1.7.0 are not Metashape software versions.

Other observed entries:

- AlignCameras metadata includes adaptive fitting false, reset false, min_image 2,
  subdivide true and historical duration/RAM. Matching accuracy/downscale,
  preselection and keypoint/tie-point limits are missing.
- `settings/accuracy_*` values are retained as raw accuracy settings. They do not
  establish the Align Photos quality level or measured physical accuracy.
- Active depth-map/model entries each have ID 0 but no payload or counts in the
  supplied chunk metadata. These declarations do not verify completed geometry.
- `OptimizeCameras/sigma0=0.13802073798877` remains an unknown-definition optimizer
  statistic. **No pixel RMSE is inferred from it.**

The missing explicit reference is `PAHMA_5_1405.files/0/0/frame.zip`.
The current Renderings folder still lists only PSX, project.zip, chunk.zip, OBJ
and the 61,614,677-byte PDF. Bounded name searches did not find a frame/full project;
search negatives do not establish global absence. The PDF binary was not fetched,
and its identity/content remain unverified.

An existing **2,143-byte ARF Dog frame archive** was inspected solely as a format
comparison: its XML joins camera IDs to photo paths, EXIF, dimensions, sizes and
point-cloud references, with no stored pixel RMSE. This is not evidence of the
missing 5-1405 frame's actual contents. The new `inspect_frame` function can read
a supplied frame archive, join camera IDs to chunk labels and flag missing/unknown
IDs without following any path. A frame may recover paths and version hints; it
does not itself prove historical bytes, provide full processing settings or
guarantee pixel-RMSE availability. A small report/state export is still needed.

## B. Deterministic 5-613 provenance check

The [executable evidence procedure](PHOTO_PROVENANCE_PROTOCOL.md) pins the verified
label source and compares candidate identities, exact-case filenames, unique
SHA-256 values, complete camera inclusion and a reviewed binding to the same
report state, component scope, version and measurement stage. Present-day path
hashes alone are not accepted as historical input proof.

Applied to the saved 145 candidate JPEG records, the result is **blocked**:
identifiers are unique and count agreement is true, but all 145 hashes are missing,
the report has no camera filename list, and no report-state camera manifest or
archival input binding exists. Filename differences are null, not 145 measured
mismatches. No additional photos were fetched or processed. The 0.748542 px label
remains valid as a reported value; its association with those JPEG bytes is unknown.

## C. Independent-artifact leads and access limits

The new bounded review covered root collection metadata, relevant student-project
and Collegium folder metadata, the existing four-collection inventory, and two
small native notes texts. Local known report/project metadata continues to identify
ARF Dog. No exhaustive local-disk or Drive inventory is claimed. Student/Collegium
folders and public rendering/model links do not by themselves supply labeled runs.

| Candidate family | Existing evidence | Missing before historical intake |
| --- | --- | --- |
| PAHMA 5-613 | One verified report label/settings; 145 candidate JPEG identifiers | Exact report-state camera manifest, historical processed-version hashes, verified session, same-set features |
| PAHMA 5-1405 | Small chunk/project metadata; 113 matching TIFF stems; disconnected component state | Frame/camera paths, archival byte binding, pixel report/metric definition and scope, complete settings/session |
| PAHMA 6-16015 | Notes describe completed Canon/Nikon and merged models; candidate TIFFs in inventory | Run-specific pixel reports and manifests; original/enhanced versions; resolve 29+132 versus 164 and later 135-photo Series2 claims |
| PAHMA 5-1405 Inner Lid | Notes describe 183 inputs and report exports | Locate actual camera manifests/reports and processed Canon files; distinguish re-exports from independent alignments; keep in 5-1405 family |
| PAHMA 6-19929 | Newly reviewed collection lists scholarly notes; meeting notes plan photography for November 13 | No completed historical run, source-photo manifest or Metashape label located; prospective collection lead only |

The [Fall 2026 notes](https://docs.google.com/document/d/1gkxjlIHik48nzHidDxncrkWcGJyzZqfp5VtFRnXzxe0/edit)
(modified November 6) explicitly plan the November 13 acquisition. They also say
5-1405 accession/component names are still being reconciled, including whether
the so-called cartonnage is the inner coffin. Retain the conservative family
grouping and request curator mapping; model titles do not establish independent
objects. The 6-19929 document contains contextual scholarship/public-image links,
not a run-specific Metashape report or source-photo manifest. Public illustration
images are not assumed to be reconstruction inputs.

Rich-text links recovered from the meeting notes identify a separate
[BD3D shared-drive folder](https://drive.google.com/drive/folders/1yWmMl0x1nETHMZUCVWcox8HyCDf5wcuM).
Metadata search returned no entries and direct connector access returned 404
NOT_FOUND. This can mean access, relocation or deletion; it does not prove an empty
archive. The notes also identify Box as the longer-term archive, with access on an
as-needed basis. Box was not accessed or newly connected. Request a curator's
small report/manifest index from those archives before transferring image sets.

**Additional verified usable independent groups found: zero.** Among existing
completed-model leads, 5-1405 and 6-16015 could add two labeled families beyond
the two currently labeled families if their missing evidence is recovered.
This is a collection priority, not a claim that compatible labeled data already
exist. Inner Lid is related to 5-1405, and 6-19929 is a future acquisition lead.

## D. Readiness and minimum evaluation evidence

| Dimension | Current count | Interpretation |
| --- | ---: | --- |
| Source-verified historical pixel-RMSE labels | 33 | 32 ARF Dog post-refinement plus one 5-613 report-state label |
| Verified labels with matching cached image features | 29 | One artifact family; associations declared in source manifests/hash-matched caches |
| Fully provenance-verified observations | 0 | Label/image association, identified verified session and complete verified settings are not jointly established |
| Strict-ready observations including complete image-only features | 0 | Existing gate unchanged |
| Documented conservative artifact families in the observation table | 3 | ARF Dog, 5-613, 5-1405; last is unlabeled |
| Families with verified labels | 2 | ARF Dog and 5-613; not a usable independent train/test pair |
| Families with label-and-feature pairs | 1 | ARF Dog only |
| Fully verified families / eligible held-out families | 0 / 0 | No certified independent compatible evaluation partition |

The 40-record table is unchanged. A documented family is not proof of independent
acquisition sessions. Repeated reconstructions, photo subsets, component exports
and multiple mesh versions must stay within their artifact family and image-overlap
group. No additional Dog labels were collected in this phase.

Before Ridge evaluation, obtain source-matched photo sets and seven pre-alignment
features for independent artifacts, identify sessions and archival versions,
recover complete alignment/filtering/optimization settings and software, and fix
one compatible pixel-RMSE definition and endpoint. Initial alignment, post-error-
reduction API RMSE and optimized report-state RMS cannot be pooled by name alone.
Report scope must address unaligned cameras and disconnected components. A low
residual measures alignment consistency, not coverage or physical mesh accuracy.

Two fully verified compatible groups would permit only a minimal train/test
partition. A separate tuning/test design requires more groups, and three groups
would still provide very weak generalization evidence. There is no defensible
universal minimum sample count from these data: collect multiple independent
groups in each partition and assess group-level uncertainty and learning curves.
Use a training-fold mean comparator, group-aware Ridge tuning, training-only
preprocessing, MAE/RMSE in pixels and predicted-versus-observed/residual plots.
Weight/report object-level performance so Dog repeats cannot dominate.

A PyTorch MLP is not yet scientifically justified. Establish a defensible Ridge
evaluation and enough independent groups for model selection, uncertainty and a
frozen artifact-held-out test before assessing whether added network capacity is
supported. No model, split or preprocessing fit was produced here.

## Next concrete evidence requests

1. **5-1405 project owner:** provide the matching
   `PAHMA_5_1405.files/0/0/frame.zip`, or a camera manifest with IDs/paths/inclusion
   and component scope. Supply small Tie Points/Processing Parameters/System pages
   or an existing-state report export with pixel RMS, stage, settings and version.
   Identify which of the five components the report describes. Supply historical
   hashes/archive evidence for the exact processed versions. A first-page preview
   can establish whether the 61.6 MB PDF is even the required document.
2. **5-613 report owner:** provide the report-state 145-camera manifest and the
   archival hash/version link to the candidate JPEGs, including preprocessing and
   disabled/excluded state. Confirm whether the March-2026 JPEGs are the November-
   2025 inputs or derivatives. Provide session verification. Current rehashing alone
   would not resolve historical identity.
3. **6-16015 / archive curator:** prioritize small reports and camera-manifest
   exports for the retained Canon, Nikon and merged/enhanced states, resolving
   count/version discrepancies. Provide a small report/project index or correct
   access link for the shared Drive/Box archive and a curator mapping of 5-1405
   components. Transfer no dense assets, meshes or image batches for this decision.

## Validation and preserved work

The selected suite passed **86 lightweight tests in 0.08 seconds**: metadata
component/frame interpretation, provenance gates, historical parser/readiness,
cached scalar recovery and archival array checks. Tests use synthetic metadata.
No Metashape process was launched, large binary downloaded, original image read,
feature batch extracted or model trained. Previous research code/data remain
available; only additive metadata tooling, tests and documentation are committed.
