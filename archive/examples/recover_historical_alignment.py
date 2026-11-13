"""Offline recovery from saved small metadata/caches. No network or Metashape API."""
from pathlib import Path
import csv
import sys

from bd3d_legacy.historical_alignment import build_dataset, require
from bd3d_legacy.historical_recovery import bounded_json, recover_legacy_report, initial_alignment_inventory
from bd3d_legacy.project_metadata import inspect_project, compare_stems
from bd3d_quality.provenance import digest, read_json, sha256_file, write_json


def recover(discovery, prior_dataset, results, legacy_root, evidence, output):
    discovery, prior_dataset, results, legacy_root, evidence, output = map(
        Path, (discovery, prior_dataset, results, legacy_root, evidence, output))
    require(not output.exists(), 'recovery output must be a new directory')
    for name, expected in read_json(prior_dataset/'file_hashes.json').items():
        require(sha256_file(prior_dataset/name) == expected, 'prior dataset changed')
    old_records = read_json(prior_dataset/'records.json')['records']
    inventory = read_json(discovery/'drive-inventory.json')
    small = evidence/'small-project'
    for item in read_json(small/'download-provenance.json'):
        require(sha256_file(small/item['filename']) == item['sha256'], 'small metadata hash mismatch')
    inspected = inspect_project(small/'PAHMA_5_1405.psx', small/'project.zip', small/'chunk.zip')
    tiff_folders = {'1n8O7hgkjga3J_Wggn9auLKZeLYG7H-AQ','15WXQNj0THzJllAlGsOm7Pvfo6zwcC3Dd',
                    '1UcCD1JkdSI5bVqjR3FX4gAL3FsDB-Fcs','1AnGWWw8a5s6JkDdWNVR4PXPKYVg3M8Zs'}
    tiffs = [i for f in inventory['folders'] if f['folder_id'] in tiff_folders for i in f['files']]
    comparison = compare_stems(inspected['cameras'], tiffs)
    comparison['candidate_total_bytes'] = sum(i['size_bytes'] for i in tiffs)
    comparison['candidate_files'] = tiffs
    comparison['missing_companion'] = 'PAHMA_5_1405.files/0/0/frame.zip'
    drive = [r for r in old_records if r['source_category']=='priority_drive_report']
    require(len(drive)==1, 'expected one previously verified priority PDF label')
    jpg = drive[0]['source_images']
    source_files = next(f['files'] for f in inventory['folders'] if f['folder_id']=='1lybQarHA0T4l3lLa8sxvFEIaFT5klo0l')
    names = sorted(i['path'] for i in jpg)
    photo_check = {'candidate_count':len(jpg), 'unique_filenames':len(set(names)),
        'first_filename':names[0], 'last_filename':names[-1],
        'contiguous_expected_range':names == [f'_MG_{i}.jpg' for i in range(4748,4893)],
        'total_bytes':sum(i['size_bytes'] for i in source_files),
        'modified_dates':sorted({i['modified_time'][:10] for i in source_files}),
        'candidate_content_hashes_available':sum(bool(i.get('sha256')) for i in source_files),
        'report_image_count':drive[0]['image_count'], 'report_camera_filenames_available':False,
        'report_label_pixels':drive[0]['label']['rmse_pixels'],
        'exact_processed_image_set_verified':False,
        'reason':'Report extraction and session notes do not list run-specific filenames/hashes. Counts and catalog association cannot prove file version or execution inputs.',
        'evidence_report_text_sha256':sha256_file(discovery/'report-1ztQP1C2WAoL8C44kNSqBeKEDcdiF-0ot.txt')}
    fp, mp = results/'arf-dog-50-features.json', results/'arf-dog-50-matches.json'
    features, matches = bounded_json(fp), bounded_json(mp)
    old_cache = read_json(results/'pytorch-ready-archival-v2/provenance.json')
    require(sha256_file(fp)==old_cache['input_feature_file']['sha256'], 'image cache changed')
    require(sha256_file(mp)==old_cache['pair_feature_file']['sha256'], 'pair cache changed')
    cache_source = {'image_cache_path':str(fp), 'image_cache_sha256':sha256_file(fp),
        'pair_cache_path':str(mp), 'pair_cache_sha256':sha256_file(mp),
        'settings_signature':digest({'image':features['settings'],'pair':matches['settings']})}
    records, audits = list(drive), []
    for path in sorted(legacy_root.glob('*/reports/*.json')):
        record, audit = recover_legacy_report(path, features, matches, cache_source)
        require(record['reconstruction_run_id'] not in {r['reconstruction_run_id'] for r in records},
                'duplicate canonical legacy report UUID')
        records.append(record); audits.append(audit)
    summary = legacy_root/'reconstruction_summary.csv'
    require(summary.stat().st_size < 1_000_000, 'summary CSV exceeds metadata budget')
    with summary.open(newline='') as f:
        summary_rows = list(csv.DictReader(f))
    require(len(summary_rows)==len(audits) and {r['run_id'] for r in summary_rows}=={r['run_id'] for r in audits},
            'summary CSV run IDs differ; manual reconciliation required')
    require({r['reconstruction_run_id'] for r in old_records}.issubset(
        {r['reconstruction_run_id'] for r in records}), 'lost a prior observation')
    for old in old_records:
        new = next(r for r in records if r['reconstruction_run_id']==old['reconstruction_run_id'])
        require(new['label']==old['label'], 'prior label/source/stage changed')
        require(new['source_images']==old['source_images'], 'prior image association changed')
        if old.get('image_features'):
            for key, value in old['image_features'].items():
                require(abs(value-new['image_features'][key]) <= 1e-9*max(1,abs(value)), 'cached aggregate differs')
    saved = {'schema_version':1,'kind':'historical_alignment_run',
        'record_id':'drive-project:1IofkWlnUd6SgAm1PA5XKCVmZWOe61UEY/chunk-0',
        'observation_unit':'saved_project_state','source_category':'priority_drive_project',
        'artifact_id':'PAHMA-5-1405','artifact_group_id':'PAHMA-5-1405',
        'acquisition_session_id':None,'session_status':'unknown',
        'reconstruction_run_id':'source-project:1IofkWlnUd6SgAm1PA5XKCVmZWOe61UEY/chunk-0',
        'run_id_status':'source_derived_saved_state_not_native_execution_uuid',
        'source_image_set_id':None,'image_count':inspected['camera_count'],
        'image_count_status':'stored camera entries, not verified historical execution input count',
        'source_images':[{'image_id':i['id'],'path':i['title'],'url':i['url'],'sha256':None} for i in tiffs],
        'image_provenance_status':'candidate_only','image_match_evidence':None,
        'image_features':None,'feature_provenance':None,
        'metashape_version':inspected['project_metadata'].get('Info/LastSavedSoftwareVersion'),
        'alignment_configuration':{k:v for k,v in inspected['chunk_metadata'].items() if k.startswith('AlignCameras/')},
        'configuration_status':'partially_reported',
        'label':{'rmse_pixels':None,'units':'pixels','status':'missing','definition_id':None,'stage':'unknown',
            'source':{'provider':'google_drive','file_id':'1yNoznrBaCwxKQrsGemZFNHfe4qtvOpcY',
                'file_path':str(small/'chunk.zip'),'section':'doc.xml; no verified pixel RMSE',
                'evidence_sha256':sha256_file(small/'chunk.zip')}},
        'diagnostics':{'aligned_camera_count':None,'tie_point_count':None,
            'cameras_with_stored_transform':inspected['cameras_with_stored_transform'],
            'component_count':inspected['component_count'],
            'optimizer_statistics_unknown_units':inspected['candidate_optimizer_statistics']},
        'availability':{'project':'small PSX/project/chunk metadata only; missing frame assets',
            'photographs':'113 matching candidate TIFF stems; no historical paths or byte hashes',
            'report':'61.6 MB PDF identity unverified; binary not downloaded'},
        'warnings':inspected['limitations']}
    records.append(saved)
    scope = {'discovery_inventory_sha256':sha256_file(discovery/'drive-inventory.json'),
        'prior_dataset_path':str(prior_dataset),'prior_records_sha256':sha256_file(prior_dataset/'records.json'),
        'priority_collection_count':4,'conservative_priority_artifact_families':3,
        'unit_limit':'One saved project state is catalogued separately from execution reports; repeated ARF Dog runs are dependent.'}
    initial = initial_alignment_inventory(results)
    document_notes = []
    for filename, conclusion in [
        ('DATA_DICTIONARY.md','Early-SfM residuals are measured after error reduction; they are not initial-alignment labels.'),
        ('RESULTS.md','0.276673 pixel reprojection RMSE is an aggregate repeatability mean, not an additional run-specific observation.')]:
        path = legacy_root.parent/'ChatGPT/bd3d/docs'/filename
        require(path.stat().st_size < 100_000, 'documentation exceeds metadata budget')
        document_notes.append({'path':str(path),'sha256':sha256_file(path),'conclusion':conclusion})
    build_dataset(records, output, inventory_scope=scope)
    write_json(output/'project-inspection.json',inspected)
    write_json(output/'pahma-5-1405-photo-comparison.json',comparison)
    write_json(output/'pahma-5-613-photo-comparison.json',photo_check)
    write_json(output/'drive-metadata-checks.json',read_json(evidence/'drive-metadata-checks.json'))
    write_json(output/'local-report-audit.json',{'reports':audits,
        'canonical_source':'*/reports/*.json; summary CSV duplicates are excluded',
        'prior_report_ids':[r['reconstruction_run_id'] for r in old_records if r['source_category']=='local_archival_report'],
        'additional_verified_legacy_labels':sum(a['rmse_pixels'] is not None and a['run_id'] not in
            {r['reconstruction_run_id'] for r in old_records} for a in audits),
        'additional_independent_artifact_groups':0})
    write_json(output/'initial-alignment-inventory.json',initial)
    write_json(output/'local-documentation-audit.json',{'documents':document_notes,
        'summary_csv':{'path':str(summary),'sha256':sha256_file(summary),'rows':len(summary_rows),
            'all_run_ids_duplicate_canonical_json':True,'additional_observations':0},
        'search_scope':'Local BD3D docs and canonical report JSON, this repository results metadata, and targeted local PDF/project filename search; not a complete filesystem or Drive audit.'})
    write_json(output/'recovery-file-hashes.json',{p.name:sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()})
    return records


if __name__=='__main__':
    recover(*sys.argv[1:])
