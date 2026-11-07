# Report-to-photograph provenance protocol

This is a metadata-only procedure for the PAHMA 5-613 report and future historical
intake. It does not run Metashape, read photograph pixels, generate features or
promote records in an existing dataset. Current result: **blocked**, despite
145 candidate JPEGs matching the report's image count.

## Evidence chain and decision

1. Pin the reviewed label source. Retain the report file ID, section, readable-text
   SHA-256 and extraction scope, exact pixel RMS, reported software and measurement
   stage. For 5-613 this is 0.748542 px, Metashape 2.1.4.19695, report-state RMS
   with optimization recorded. Do not substitute the normalized 0.119078 value
   or treat the report as an initial-alignment measurement.
2. Obtain the camera manifest from the **same saved state** as the report. It must
   identify artifact, chunk/component scope, camera IDs, photo paths/extensions,
   enabled/excluded/included state and a source-derived or native run-state ID
   whose origin is clear. A report filename, PSX pointer hash, folder association,
   timestamp proximity or matching count cannot establish this link. Preserve all
   imported cameras; a disabled-camera or partial-component report needs a separately
   reviewed inclusion policy. The initial checker deliberately accepts only a full
   manifest in which every camera is enabled and included.
3. Identify the exact historically processed versions. Distinguish raw sources from
   JPEG/TIFF exports, enhanced images, crops and later replacements. Request
   contemporaneous input hashes or an immutable historical archive tied to that
   state. A curator should identify the archival record and its evidence. Current
   hashes of files at matching paths alone establish only current correspondence,
   not what Metashape processed historically. EXIF, dimensions and file sizes help
   reject mismatches but do not prove content identity.
4. Obtain candidate SHA-256 values from the lab's existing trusted records, or a
   separately approved lab-side inventory. No automatic hashing/downloading of the
   420 MB candidate folder happens here. Maintain Drive IDs alongside exact-case
   filenames. Reject duplicate names/IDs, duplicate content, missing hashes and
   name/byte mismatches. Compare the complete set, not a count or sampled filenames.
5. Review the chain between the label, manifest and archival hash evidence. Record
   reviewer, evidence location/hash, state/scope, version, measurement stage and
   exact pixel RMS. Bind the manifest using its **canonical JSON SHA-256**
   (`bd3d_quality.provenance.digest`), not an unspecified raw-file hash. The receipt
   is a reviewed assertion backed by source evidence, not cryptographic proof of
   historical execution. Its source and authority must be inspected; filling in
   a template or setting `review_status` does not verify an archive.
6. Run `check_photo_provenance`. False gates mean evidence does not establish the
   requirement; they do not assert a measured mismatch when evidence is absent.
   Review contradictions, preserve nulls and retain the original artifacts. Even
   a passing photo association does not establish sessions, complete settings,
   compatible targets or training readiness. Only after intake review may a new
   dataset version attach features from that verified image set.

If no historical archive/hash evidence survives, preserve the weaker
filename/path association and explicitly document that exact execution inputs
cannot be verified. Do not silently relabel it as exact provenance.

## Machine-readable intake contract

`photo_provenance.check_photo_provenance(record, candidates, manifest, receipt)`
takes in-memory JSON metadata only and does not access files or the network.
`record` is a validated historical alignment record; `candidates` is a list of
`{image_id, path, sha256}` records. Full metadata paths in camera manifests can
use Windows or POSIX separators; comparison uses exact-case basenames. Duplicate
basenames require explicit disambiguation before this contract can be used.

The following describes required fields, **not an observed reconstruction**:

| Manifest field | Required meaning |
| --- | --- |
| schema_version / kind | `1` / `report_camera_manifest` |
| artifact_id | Same artifact as the reviewed report record |
| run_state_id | Identifiable saved report state; do not invent a native UUID |
| scope | Nonempty string `chunk_id` and `component_scope` |
| cameras | Complete imported-camera list: unique string `camera_id`, exact `path`, lowercase SHA-256, boolean `enabled` and `included_in_alignment` |

| Review receipt field | Required meaning |
| --- | --- |
| review_status / reviewer / verification_basis | Source-reviewed `verified`, reviewer identity, explanation of the binding evidence |
| review_evidence_location / review_evidence_sha256 | Inspectable archival/binding evidence and its SHA-256 |
| image_identity_basis | `contemporaneous_hashed_inputs` or `immutable_historical_archive`; current path hashes alone are rejected |
| archive_or_hash_record_id | Identifiable supporting archival or historical hash record |
| report_evidence_sha256 | Exact label-source evidence hash from the reviewed record; text-extraction hash is not a PDF binary hash |
| manifest_canonical_sha256 | Canonical JSON digest of the supplied full manifest |
| run_state_id / scope | Match the manifest and reviewed report state exactly |
| metashape_version / label_stage / rmse_pixels | Match the reviewed label source exactly; unknown version/stage is blocked |

Receipt evidence may be a reviewed bundle/index referencing several files. Keep
each underlying original and hash; never present a newly authored receipt as
contemporaneous evidence. No real receipt or camera manifest was fabricated in
this phase.

## Current reproducible check

```bash
.venv/bin/python examples/audit_alignment_provenance.py \
  results/historical-alignment-recovered-v2/records.json \
  results/alignment-provenance-audit-fresh \
  --small-project results/historical-alignment-recovery-v1/small-project
```

The output directory must be new. The command emits readiness dimensions,
the 5-613 missing-evidence gates, optional bounded 5-1405 component inspection,
input/code hashes and output hashes. Existing records and labels are unchanged.
For later reviewed intake, optional `--candidates-path` accepts `{images:[...]}`;
`--manifest-path` and `--receipt-path` accept the contracts above. These are
explicit small metadata files, never original image paths to be opened.

At this checkpoint the candidates have unique identifiers and matching count,
but **145 missing content hashes**, no run-bound camera manifest and no reviewed
historical state binding. Exact filename differences are null because the camera
list is missing. No photo association is verified, and no new feature row is made.
