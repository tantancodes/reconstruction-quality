"""Deterministic, metadata-only provenance checks. No file or network access.

A matching count or a PSX pointer hash cannot bind photographs to a report.
Review receipts are explicit curator/researcher assertions, not cryptographic
proof of historical execution. Their evidence must be inspected before acceptance.
"""
from __future__ import annotations

import itertools
import math
import re

from .historical_alignment import audit_split, validate_run
from bd3d_quality.provenance import digest
from .tabular_dataset import PRE_FEATURES


def valid_hash(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def image_name(image):
    # Windows and POSIX project paths are metadata only; never followed on disk.
    path=image.get('path') or image.get('filename')
    return path.replace('\\','/').rsplit('/',1)[-1] if isinstance(path,str) else None


def check_photo_provenance(record,candidates,manifest=None,receipt=None):
    """Compare exact-case filenames, content hashes and a reviewed report binding.

    ``manifest``: schema_version=1, kind=report_camera_manifest, artifact_id,
    run_state_id, scope={chunk_id, component_scope}, cameras=[{camera_id,path,
    sha256,enabled,included_in_alignment}]. This is a reviewed full imported-camera
    manifest, not a list inferred from counts. ``receipt`` binds its canonical JSON
    digest to the record's label extraction hash and exact state/version/stage.
    No receipt is generated or marked verified by this function.
    """
    validate_run(record)
    candidates=list(candidates)
    cameras=(manifest or {}).get('cameras',[])
    names=[image_name(i) for i in candidates]
    cam_names=[image_name(i) for i in cameras]
    hashes=[i.get('sha256') for i in candidates]
    camera_hashes=[i.get('sha256') for i in cameras]
    candidate_identifiers=[i.get('image_id') or i.get('id') or image_name(i) for i in candidates]
    candidate_unique=bool(candidates) and all(names) and len(names)==len(set(names)) and all(candidate_identifiers) and len(set(candidate_identifiers))==len(candidates)
    camera_ids=[i.get('camera_id') for i in cameras]
    manifest_unique=bool(cameras) and all(cam_names) and len(cam_names)==len(set(cam_names)) and all(isinstance(i,str) and i for i in camera_ids) and len(set(camera_ids))==len(cameras)
    candidate_hashes_valid=bool(hashes) and all(valid_hash(h) for h in hashes) and len(set(hashes))==len(hashes)
    camera_hashes_valid=bool(camera_hashes) and all(valid_hash(h) for h in camera_hashes) and len(set(camera_hashes))==len(camera_hashes)
    scope=(manifest or {}).get('scope') or {}
    manifest_valid=bool(manifest) and manifest.get('schema_version')==1 and manifest.get('kind')=='report_camera_manifest' and manifest.get('artifact_id')==record['artifact_id'] and bool(manifest.get('run_state_id')) and bool(scope.get('chunk_id')) and bool(scope.get('component_scope'))
    label=record['label'];source=label.get('source',{})
    receipt=receipt or {}
    binding_checks=[manifest_valid,receipt.get('review_status')=='verified',bool(receipt.get('reviewer')),
        bool(receipt.get('verification_basis')),valid_hash(receipt.get('review_evidence_sha256')),
        bool(receipt.get('review_evidence_location')),
        receipt.get('image_identity_basis') in {'contemporaneous_hashed_inputs','immutable_historical_archive'},
        bool(receipt.get('archive_or_hash_record_id')),
        valid_hash(source.get('evidence_sha256')),
        receipt.get('report_evidence_sha256')==source.get('evidence_sha256'),
        receipt.get('manifest_canonical_sha256')==digest(manifest),
        receipt.get('run_state_id')==(manifest or {}).get('run_state_id'),receipt.get('scope')==scope,
        bool(record.get('metashape_version')),receipt.get('metashape_version')==record.get('metashape_version'),
        label.get('stage') not in (None,'unknown'),receipt.get('label_stage')==label.get('stage'),
        receipt.get('rmse_pixels')==label.get('rmse_pixels'),
        label.get('status') in {'document_verified','computed_historical_verified'}]
    binding=all(binding_checks)
    count_match=bool(cameras) and len(cameras)==len(candidates)==record['image_count']
    # For this complete-camera contract all cameras must be included. A different
    # report scope/disabled-camera policy requires explicit review, not a guess.
    inclusion=bool(cameras) and all(c.get('enabled') is True and c.get('included_in_alignment') is True for c in cameras)
    filenames=candidate_unique and manifest_unique and set(names)==set(cam_names)
    by_name={image_name(i):i.get('sha256') for i in candidates}
    content=filenames and candidate_hashes_valid and camera_hashes_valid and all(by_name[image_name(c)]==c['sha256'] for c in cameras)
    gates={'candidate_identifiers_unique':bool(candidate_unique),'candidate_sha256_complete_and_unique':bool(candidate_hashes_valid),
        'reviewed_full_camera_manifest_available':bool(manifest_valid and manifest_unique),
        'report_camera_and_candidate_counts_match':bool(count_match),'exact_case_filename_sets_match':bool(filenames),
        'enabled_and_included_camera_scope_known':bool(inclusion),'camera_to_candidate_content_hashes_match':bool(content),
        'reviewed_report_state_binding_matches':bool(binding)}
    return {'schema_version':1,'kind':'photo_provenance_check','record_id':record['record_id'],
        'candidate_count':len(candidates),'report_image_count':record['image_count'],
        'count_agreement_diagnostic_only':len(candidates)==record['image_count'],
        'candidate_metadata_fingerprint':digest(candidates),'gates':gates,
        'exact_report_input_association_verified':all(gates.values()),
        'blocked_by':[k for k,v in gates.items() if not v],
        'candidate_missing_or_invalid_hash_count':sum(not valid_hash(h) for h in hashes),
        'camera_only_filenames':sorted(set(n for n in cam_names if n)-set(n for n in names if n)) if manifest is not None else None,
        'candidate_only_filenames':sorted(set(n for n in names if n)-set(n for n in cam_names if n)) if manifest is not None else None,
        'gate_semantics':'False means evidence does not establish the gate; missing input is not a measured mismatch.',
        'limits':['Pure metadata consistency check; supplied review evidence and its authority require human source review.',
            'Current photo hashes must be measured lab-side or read from trusted receipts; this function never hashes photographs.',
            'frame.zip paths alone cannot establish historical content identity or recover pixel residuals.',
            'Photo association does not by itself verify sessions/settings, attach features or make a row training-ready.']}


def readiness_dimensions(records):
    """Separate provenance, feature and independent-group readiness; fit nothing."""
    labels=[];pairs=[];proven=[];ready=[]
    for r in records:
        validate_run(r)
        verified=r['label']['status'] in {'document_verified','computed_historical_verified'} and r['label']['rmse_pixels'] is not None
        features=r.get('image_features')
        linked=features is not None and all(features.get(k) is not None and math.isfinite(features[k]) for k in PRE_FEATURES)
        full=verified and r['image_provenance_status']=='verified_hash_manifest' and r.get('configuration_status')=='verified' and r.get('session_status')=='verified' and bool(r.get('acquisition_session_id'))
        if verified:labels.append(r)
        if verified and linked:pairs.append(r)
        if full:proven.append(r)
        if full and linked:ready.append(r)
    eligible=set()
    for a,b in itertools.combinations(ready,2):
        try:audit_split([a,b],{a['record_id']:'train',b['record_id']:'test'})
        except ValueError:continue
        eligible.update([a['artifact_group_id'],b['artifact_group_id']])
    def groups(rows):return sorted({r['artifact_group_id'] for r in rows})
    return {'schema_version':1,'documented_records':len(records),'verified_rmse_labels':len(labels),
        'labels_with_matching_image_features':len(pairs),'fully_provenance_verified_observations':len(proven),
        'strict_ready_observations':len(ready),'documented_artifact_groups':groups(records),
        'labeled_artifact_groups':groups(labels),'label_and_feature_artifact_groups':groups(pairs),
        'fully_provenance_verified_artifact_groups':groups(proven),
        'groups_eligible_for_held_out_evaluation':sorted(eligible),
        'independence_limit':'Catalog families are conservative groups, not independently validated sessions; overlapping subsets and component exports stay together.',
        'provenance_definition':'Verified label, hash-manifest association, verified complete configuration and verified identified acquisition session; features not required for this count.',
        'held_out_definition':'At least one distinct fully ready group passes pairwise session/photo/stratum split checks; structural eligibility is not statistical adequacy.',
        'models_trained':False}
