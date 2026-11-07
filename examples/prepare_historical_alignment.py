"""Prepare a separate metadata-only historical error dataset; no Drive downloads/jobs."""
from pathlib import Path
import sys
from bd3d_quality.historical_alignment import parse_report_text,build_dataset
from bd3d_quality.provenance import read_json,sha256_file,digest,write_json
from bd3d_quality.tabular_dataset import PRE_FEATURES


def prepare(discovery,results,output):
    discovery=Path(discovery);results=Path(results)
    inventory=read_json(discovery/'drive-inventory.json')
    report_path=discovery/'report-1ztQP1C2WAoL8C44kNSqBeKEDcdiF-0ot.txt'
    parsed=parse_report_text(report_path.read_text())
    assert parsed['image_count']==parsed['aligned_camera_count']==145
    assert parsed['tie_point_count']==346195 and parsed['rmse_pixels']==.748542
    jpg=next(f for f in inventory['folders'] if f['folder_id']=='1lybQarHA0T4l3lLa8sxvFEIaFT5klo0l')
    drive={'schema_version':1,'kind':'historical_alignment_run','record_id':'drive-report-1ztQP1C2WAoL8C44kNSqBeKEDcdiF-0ot-tie-points',
        'source_category':'priority_drive_report','artifact_id':'PAHMA-5-613','artifact_group_id':'PAHMA-5-613',
        'acquisition_session_id':'PAHMA-5-613/2025-10-20','session_status':'documented_in_notes_not_exif_verified',
        'reconstruction_run_id':'source-report:1ztQP1C2WAoL8C44kNSqBeKEDcdiF-0ot',
        'run_id_status':'source_derived_not_native_project_uuid','source_image_set_id':None,'image_count':145,
        'source_images':[{'image_id':i['id'],'path':i['title'],'url':i['url'],'sha256':None} for i in jpg['files']],
        'image_provenance_status':'candidate_only','image_match_evidence':None,
        'image_features':None,'feature_provenance':None,'metashape_version':parsed['metashape_version'],
        'alignment_configuration':parsed['alignment_configuration'],'optimization_configuration':parsed['optimization_configuration'],'configuration_status':'verified',
        'label':{'rmse_pixels':parsed['rmse_pixels'],'units':'pixels','status':'document_verified',
            'definition_id':parsed['definition_id'],'stage':parsed['label_stage'],
            'stage_status':'report_lists_optimization_initial_alignment_value_not_available',
            'source':{'provider':'google_drive','file_id':'1ztQP1C2WAoL8C44kNSqBeKEDcdiF-0ot',
                'url':'https://drive.google.com/file/d/1ztQP1C2WAoL8C44kNSqBeKEDcdiF-0ot/view',
                'section':'Processing Parameters / Tie Points / RMS reprojection error (pixel value in parentheses)',
                'page':5,'page_status':'inferred_from_extracted_printed_page_footer_order',
                'file_path':str(report_path),'evidence_sha256':sha256_file(report_path),
                'hash_scope':'readable text extraction, not original PDF byte hash'}},
        'diagnostics':{'aligned_camera_count':145,'tie_point_count':346195,'normalized_report_rmse_not_pixels':parsed['normalized_report_rmse']},
        'availability':{'report':'readable text verified','project':'not located in inspected collection folders',
            'photographs':'145 candidate remote JPEGs, 420153827 bytes; not downloaded','feature_extraction':'blocked until exact processed input set confirmed'},
        'field_evidence':{'image_count':'reported','aligned_camera_count':'reported','tie_point_count':'reported','rmse_pixels':'reported',
            'acquisition_date':'notes dated 10/20/25; not independently verified from photos',
            'photo_match':'unverified; JPEG files carry 2026-03 metadata after 2025-11 report',
            'label_stage':parsed['stage_limit']},
        'warnings':['Do not substitute the rounded survey summary 0.749 or normalized 0.119078 for the precise 0.748542 pixel RMS.',
            'A matching count does not establish identical image bytes, preprocessing or disabled/excluded-camera state.',
            'Do not join these JPEG names with unrelated ARF Dog photographs with overlapping Canon filenames.']}
    rows=[drive]
    legacy=results/'pytorch-ready-archival-v2'
    for name,h in read_json(legacy/'dataset_files.sha256.json').items():assert sha256_file(legacy/name)==h
    old=read_json(legacy/'observations.json')['rows']
    provenance=read_json(results/'pytorch-ready-archival-v2/provenance.json')
    for item in provenance['lineage']:
        rid=item['run_id'];tab=next(r for r in old if r['run_id']==rid)
        sources=[p for p in (results/'archival-pilot-audit-v2').glob('ARF*.json') if read_json(p)['run_id']==rid]
        assert len(sources)==1
        assert sha256_file(sources[0])==item['audit_sha256']
        audit=read_json(sources[0]);report=audit['source_report'];m=audit['manifest']
        for path,h in [(report['output_status']['report_json_path'],audit['source_report_sha256']),
                       (report['experiment']['manifest_path'],audit['source_manifest_sha256'])]:
            assert sha256_file(Path(path))==h
        aligned=report['alignment'];images=[{'path':i['filename'],'sha256':i['sha256']} for i in m['images']]
        image_set='sha256:'+digest(images)
        features={k:tab[k] for k in PRE_FEATURES}
        assert all((Path(m['source_dataset'])/i['path']).is_file() for i in images)
        rows.append({'schema_version':1,'kind':'historical_alignment_run','record_id':'archival-report:'+rid,
            'source_category':'local_archival_report','artifact_id':m['artifact_id'],'artifact_group_id':m['artifact_id'],
            'acquisition_session_id':None,'session_status':'unknown_shared_source_pool',
            'reconstruction_run_id':rid,'run_id_status':'historical_report_uuid','source_image_set_id':image_set,
            'image_count':len(images),'source_images':images,'image_provenance_status':'verified_hash_manifest',
            'image_match_evidence':{'method':'report manifest SHA-256 plus prior filename/content-hash cache join',
                'manifest_sha256':audit['source_manifest_sha256'],'audit_sha256':sha256_file(sources[0]),
                'limit':'Historical execution bytes cannot be independently recovered. Current original-photo hashes were verified earlier, not rehashed now.'},
            'image_features':features,'feature_provenance':{'stage':'pre_alignment_images_only','source_image_set_id':image_set,
                'cache_file':str(results/'pytorch-ready-archival-v2/observations.json'),
                'cache_sha256':sha256_file(results/'pytorch-ready-archival-v2/observations.json'),
                'settings_signature':digest({'image':provenance['input_feature_file']['settings'],'pair':provenance['pair_feature_file']['settings']}),
                'source':'Previously computed image-only values reused without feature extraction'},
            'metashape_version':report['run']['metashape_api_version'],
            'alignment_configuration':{'sparse_downscale':report['run']['effective_settings']['sparse_downscale'],
                'keypoint_limit':None,'tiepoint_limit':None,'preselection':None,
                'historical_effective_profile':report['run']['effective_settings']},
            'configuration_status':'partially_reported',
            'label':{'rmse_pixels':aligned['reprojection_error_px']['rmse'],'units':'pixels','status':'computed_historical_verified',
                'definition_id':'aphoma_legacy_valid_tie_point_observation_rmse_pixels',
                'stage':aligned['snapshot_stage'],'stage_status':'explicit_historical_snapshot',
                'source':{'provider':'local_archival_report','file_path':report['output_status']['report_json_path'],
                    'section':'alignment.reprojection_error_px.rmse; snapshot_stage='+aligned['snapshot_stage'],
                    'page':None,'evidence_sha256':audit['source_report_sha256']}},
            'diagnostics':{'aligned_camera_count':aligned['aligned_cameras'],'tie_point_count':aligned['valid_tie_point_count']},
            'availability':{'report':'verified local JSON','photographs':'local, manifest-backed; cached image features available',
                'feature_extraction':'no rerun needed; cached seven image-only predictors'},
            'field_evidence':{'source_photo_match':'prior hash/manifest-backed audit','alignment_settings':'partial; do not infer missing settings from current code'},
            'warnings':['These labels are internal post-refinement reprojection RMSE, not geometric surface error.',
                'The 50-image row is valid for this RMSE outcome: its nonzero reprojection residual is not a mesh self-comparison.',
                'Do not pool with GUI PDF RMS until definition/stage/settings equivalence is established.']})
    scope={'drive_inventory_path':str(discovery/'drive-inventory.json'),
        'drive_inventory_sha256':sha256_file(discovery/'drive-inventory.json'),'priority_collection_count':4,
        'conservative_priority_artifact_families':3,
        'note':'PAHMA 5-1405 and its Inner Lid remain one related-artifact family until curator-confirmed grouping is supplied.'}
    return build_dataset(rows,output,inventory_scope=scope)


if __name__=='__main__':prepare(*sys.argv[1:])
