"""Historical alignment-consistency records, conservative provenance gates and no training."""
from __future__ import annotations
import csv
import itertools
import math
from pathlib import Path
import re
import numpy as np
from bd3d_quality.provenance import digest,read_json,write_json,sha256_file
from .tabular_dataset import PRE_FEATURES

LABEL_STATUSES={'document_verified','computed_historical_verified','user_supplied','missing','ambiguous'}


def require(value,message):
    if not value:raise ValueError(message)


def parse_report_text(text):
    """Read the precise pixel RMS, never the normalized first number or rounded summary.

    Only parses a single report-state Tie Points block. Multi-chunk/ambiguous reports
    need explicit source review, not automatic observation multiplication.
    """
    patterns=re.findall(r'RMS\s+reprojection\s+error\s+([\d.]+)\s*\(\s*([\d.]+)\s*(?:pix|pixels?)\s*\)',text,re.I)
    require(len(patterns)==1,'expected one unambiguous normalized/pixel RMS pair; manual review required')
    normalized,pixels=map(float,patterns[0])
    require(math.isfinite(pixels) and pixels>=0,'invalid report pixel RMSE')
    general=re.search(r'Processing Parameters\s+General\s+Images\s+([\d,]+)\s+Aligned images\s+([\d,]+)',text,re.I)
    require(general is not None,'processing-parameter image counts missing')
    count,aligned=(int(x.replace(',','')) for x in general.groups())
    require(count>0 and 0<=aligned<=count,'invalid report camera counts')
    tie=re.search(r'Tie Points\s+Points\s+([\d,]+)\s+of\s+([\d,]+)',text,re.I)
    versions=re.findall(r'Software version\s+(\d+\.\d+\.\d+)(?:\.(\d+)|\s+build\s+(\d+))?',text)
    require(bool(versions),'Metashape version absent')
    require(len({v[0] for v in versions})==1 and len({v[1] or v[2] for v in versions if v[1] or v[2]})<=1,'multiple software versions in report; review per asset')
    build=next((v[1] or v[2] for v in versions if v[1] or v[2]),None)
    version=versions[0][0]+('.'+build if build else '')
    align=re.search(r'Alignment parameters(.*?)Optimization parameters',text,re.S|re.I)
    require(align is not None,'alignment parameter block missing')
    block=align[1]
    def field(name):
        m=re.search(r'^'+re.escape(name)+r'[ \t]+([^\r\n]+)',block,re.M|re.I)
        if m and name=='Key point limit' and m[1].startswith('per Mpx'): return None
        return m[1].strip() if m else None
    settings={k:field(name) for k,name in {
        'accuracy':'Accuracy','generic_preselection':'Generic preselection',
        'reference_preselection':'Reference preselection','keypoint_limit':'Key point limit',
        'keypoint_limit_per_mpx':'Key point limit per Mpx','tiepoint_limit':'Tie point limit',
        'exclude_stationary_tie_points':'Exclude stationary tie points','guided_matching':'Guided image matching',
        'adaptive_camera_model_fitting':'Adaptive camera model fitting'}.items()}
    optimization_block=re.search(r'Optimization parameters(.*?)(?:Date created|Depth Maps)',text,re.S|re.I)
    optimization={}
    if optimization_block:
        for key,name in [('fit_parameters','Parameters'),('adaptive_camera_model_fitting','Adaptive camera model fitting'),('exclude_corners','Exclude corners')]:
            found=re.search(r'^'+re.escape(name)+r'[ \t]+([^\r\n]+)',optimization_block[1],re.M|re.I)
            optimization[key]=found[1].strip() if found else None
    return {'image_count':count,'aligned_camera_count':aligned,'tie_point_count':int(tie[1].replace(',','')) if tie else None,
        'rmse_pixels':pixels,'normalized_report_rmse':normalized,'metashape_version':version,
        'alignment_configuration':settings,'optimization_configuration':optimization,'optimization_recorded':True,
        'definition_id':'metashape_processing_report_tie_points_pixel_rms',
        'label_stage':'reported_project_state_with_optimization_recorded',
        'stage_limit':'The report supplies current Tie Points RMS; initial unoptimized alignment RMS is not recoverable from this report alone.'}


def validate_run(record):
    require(isinstance(record,dict),'historical alignment record must be an object')
    require(record.get('schema_version')==1 and record.get('kind')=='historical_alignment_run','unsupported historical alignment schema')
    for k in ['artifact_id','artifact_group_id','reconstruction_run_id','record_id','source_category']:
        require(isinstance(record.get(k),str) and record[k].strip(),f'missing {k}')
    require(record.get('configuration_status') in {'verified','partially_reported','missing'},'unknown configuration verification status')
    session=record.get('acquisition_session_id')
    require(session is None or (isinstance(session,str) and session.strip()),'invalid session identifier')
    count=record.get('image_count')
    require(type(count) is int and count>0,'positive input image_count required')
    aligned=record.get('diagnostics',{}).get('aligned_camera_count')
    require(aligned is None or (type(aligned) is int and 0<=aligned<=count),'aligned cameras outside input count')
    label=record.get('label',{})
    require(label.get('status') in LABEL_STATUSES,'unknown label verification status')
    require(label.get('units')=='pixels','only pixel RMS labels supported; normalized error is incompatible')
    value=label.get('rmse_pixels')
    require(value is None or (type(value) in (int,float) and math.isfinite(value) and value>=0),'invalid RMSE label')
    if label['status'] in {'document_verified','computed_historical_verified'}:
        require(value is not None,'verified label missing value')
        require(bool(label.get('definition_id')) and bool(label.get('stage')),'verified label needs a definition and stage')
        source=label.get('source',{})
        require(bool(source.get('file_id') or source.get('file_path')),'verified label source missing')
        require(bool(source.get('section')),'source section/measurement location required')
        require(bool(re.fullmatch('[0-9a-f]{64}',source.get('evidence_sha256',''))),'label evidence hash required')
    status=record.get('image_provenance_status')
    require(status in {'verified_hash_manifest','candidate_only','missing'},'unknown image provenance status')
    images=record.get('source_images',[])
    require(len({i.get('image_id') or i.get('path') for i in images})==len(images),'duplicate image identifiers')
    hashes=[i.get('sha256') for i in images if i.get('sha256')]
    require(all(re.fullmatch('[0-9a-f]{64}',h) for h in hashes),'invalid image hash')
    require(len(hashes)==len(set(hashes)),'duplicate image contents')
    if status=='verified_hash_manifest':
        require(len(images)==len(hashes)==count,'verified image manifest/count incomplete')
        require(record.get('source_image_set_id')=='sha256:'+digest(images),'image-set identity differs')
        require(bool(record.get('image_match_evidence')),'verified image/run association needs evidence')
    features=record.get('image_features')
    if features is not None:
        require(set(features)==set(PRE_FEATURES),'leaky feature or incomplete image-only whitelist')
        require(status=='verified_hash_manifest','cannot attach features to ambiguous source photographs')
        source=record.get('feature_provenance',{})
        require(source.get('stage')=='pre_alignment_images_only','leaky feature stage')
        require(source.get('source_image_set_id')==record['source_image_set_id'],'features computed from a different image set')
        require(features['input_image_count']==count,'feature image count mismatch')
        for v in features.values():
            require(v is None or (type(v) in (int,float) and math.isfinite(v)),'invalid feature value')
    return record


def compatibility_key(record):
    return digest({'version':record.get('metashape_version'),'settings':record.get('alignment_configuration'),
                   'optimization':record.get('optimization_configuration'),'settings_status':record.get('configuration_status'),'label_definition':record['label'].get('definition_id'),
                   'label_stage':record['label'].get('stage'),
                   'image_feature_definition':(record.get('feature_provenance') or {}).get('settings_signature')})


def build_dataset(records,output,*,inventory_scope=None):
    require(isinstance(records,list) and bool(records),'at least one historical record required')
    record_ids=set();runs=set();evidence=set();alignment_states=set();rows=[];feature_ready=[]
    for r in records:
        validate_run(r)
        require(r['record_id'] not in record_ids and r['reconstruction_run_id'] not in runs,'duplicate reconstruction record')
        record_ids.add(r['record_id']);runs.add(r['reconstruction_run_id'])
        if r.get('alignment_state_id'):
            require(r['alignment_state_id'] not in alignment_states,'duplicate underlying alignment state; model/report re-exports are not new runs')
            alignment_states.add(r['alignment_state_id'])
        label=r['label'];source=label.get('source',{})
        key=(source.get('evidence_sha256'),source.get('section'),source.get('chunk_id'))
        if key[0]:require(key not in evidence,'duplicate label evidence; duplicated files do not create new runs');evidence.add(key)
        verified=label['status'] in {'document_verified','computed_historical_verified'} and label['rmse_pixels'] is not None
        feature_link=r['image_provenance_status']=='verified_hash_manifest'
        feature_ready.append(feature_link)
        usable=verified and feature_link and r.get('image_features') is not None and all(v is not None for v in r['image_features'].values()) and r.get('configuration_status')=='verified' and r.get('session_status')=='verified'
        rows.append({'record_id':r['record_id'],'observation_unit':r.get('observation_unit','alignment_run_report'),
            'artifact_id':r['artifact_id'],'artifact_group_id':r['artifact_group_id'],
            'acquisition_session_id':r.get('acquisition_session_id'),'session_status':r.get('session_status'),
            'reconstruction_run_id':r['reconstruction_run_id'],'source_image_set_id':r.get('source_image_set_id'),
            'image_count':r['image_count'],'image_provenance_status':r['image_provenance_status'],
            'metashape_version':r.get('metashape_version'),'configuration_status':r.get('configuration_status'),
            'compatibility_stratum':compatibility_key(r),'label_stage':label.get('stage'),
            'label_definition':label.get('definition_id'),'label_status':label['status'],
            'target_reprojection_rmse_pixels':label.get('rmse_pixels'),'diagnostic_aligned_camera_count':r['diagnostics'].get('aligned_camera_count'),
            'diagnostic_tie_point_count':r['diagnostics'].get('tie_point_count'),'model_input_eligible':usable,
            **{k:r['image_features'].get(k) if r.get('image_features') else None for k in PRE_FEATURES}})
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with (output/'observations.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(output/'records.json',{'schema_version':1,'records':records})
    write_json(output/'observations.json',{'rows':rows})
    x=np.array([[r[k] if r[k] is not None else np.nan for k in PRE_FEATURES] for r in rows],dtype=np.float32)
    y=np.array([[r['target_reprojection_rmse_pixels'] if r['label_status'] in {'document_verified','computed_historical_verified'} and r['target_reprojection_rmse_pixels'] is not None else np.nan] for r in rows],dtype=np.float32)
    np.savez_compressed(output/'arrays.npz',X_pre=x,X_pre_mask=np.isfinite(x),y_rmse_pixels=y,y_rmse_pixels_mask=np.isfinite(y),
                        model_input_row_mask=np.array([r['model_input_eligible'] for r in rows],dtype=bool))
    overlaps=[]
    for a,b in itertools.combinations(records,2):
        ah={i['sha256'] for i in a['source_images'] if i.get('sha256')};bh={i['sha256'] for i in b['source_images'] if i.get('sha256')}
        ids_a={i.get('image_id') for i in a['source_images'] if i.get('image_id')};ids_b={i.get('image_id') for i in b['source_images'] if i.get('image_id')}
        overlaps.append({'a':a['record_id'],'b':b['record_id'],'same_artifact_group':a['artifact_group_id']==b['artifact_group_id'],
            'shared_content_hashes':len(ah&bh) if ah and bh else None,'shared_drive_identifiers':len(ids_a&ids_b),
            'overlap_verifiable':bool(ah and bh),'same_stratum':compatibility_key(a)==compatibility_key(b)})
    write_json(output/'leakage_audit.json',{'overlaps':overlaps,'split':None,
        'rules':['Group by artifact family; group sessions and all image-overlap connected components.',
            'Unknown source image provenance blocks certifying a safe split.',
            'Keep Metashape versions/settings/error definitions and measurement stages in separate strata.',
            'No aligned-camera/tie-point counts, camera poses, mesh properties or outcomes enter X_pre.',
            'For reprojection RMSE, the archival 50-image run is a legitimate nonzero label, not a self-comparison target.']})
    def counts(items):
        return {'documented_records':len(items),
            'documented_runs':sum(r.get('observation_unit')!='saved_project_state' for r in items),
            'saved_project_states':sum(r.get('observation_unit')=='saved_project_state' for r in items),
            'verified_rmse_labels':sum(r['label']['status'] in {'document_verified','computed_historical_verified'} for r in items),
            'verified_matching_image_sets':sum(r['image_provenance_status']=='verified_hash_manifest' for r in items),
            'ready_for_feature_extraction_or_cache_reuse':sum(r['image_provenance_status']=='verified_hash_manifest' for r in items),
            'documented_artifact_groups':len({r['artifact_group_id'] for r in items}),
            'verified_acquisition_sessions':len({r['acquisition_session_id'] for r in items if r.get('acquisition_session_id') and r.get('session_status')=='verified'}),
            'documented_or_inferred_sessions':len({r['acquisition_session_id'] for r in items if r.get('acquisition_session_id')})}
    write_json(output/'readiness.json',{'all_historical':counts(records),
        'priority_drive_collections':counts([r for r in records if r['source_category'] in {'priority_drive_report','priority_drive_project'}]),
        'local_archival_supplement':counts([r for r in records if r['source_category']=='local_archival_report']),
        'linked_feature_rows':int(np.isfinite(x).all(axis=1).sum()),'model_input_eligible_rows':sum(r['model_input_eligible'] for r in rows),
        'artifact_groups_among_linked_features':len({r['artifact_group_id'] for r in records if r.get('image_features')}),
        'compatibility_strata':len({compatibility_key(r) for r in records}),
        'independent_evaluation_feasible':False,'mean_baseline':'A mean can be defined, but a grouped held-out evaluation is not currently feasible; none fitted.',
        'ridge_baseline':'Not scientifically evaluable: linked features cover one artifact, settings incomplete and strata differ; none fitted.',
        'pytorch_mlp_justified':False,'missing':['Exact run-specific photo manifest and original processed image versions for Drive report.',
            'RMSE reports/project snapshots for 6-16015, 5-1405 and Inner Lid.',
            'Verified acquisition-session links and complete comparable alignment/optimization settings.',
            'Enough independent artifacts with matching image-only features and compatible labels.']})
    write_json(output/'schema.json',{'schema_version':1,'target':'target_reprojection_rmse_pixels','units':'pixels',
        'interpretation':'alignment consistency, not physical or mesh accuracy','feature_columns':PRE_FEATURES,
        'row_order':[r['record_id'] for r in records],'dtype':'float32','missing':'null/blank/NaN with masks; not zero',
        'split':None,'models_trained':False,'builder_source_sha256':sha256_file(Path(__file__)),'source_records_sha256':[digest(r) for r in records]})
    write_json(output/'feature_dictionary.json',{'image_features':{'columns':PRE_FEATURES,'definitions':'docs/TABULAR_DATASET.md','availability':'photographs before alignment'},
        'target_reprojection_rmse_pixels':'Reported/computed pixel RMS, with a required source definition, stage and verification status; no normalized error substitution.',
        'diagnostics':['diagnostic_aligned_camera_count','diagnostic_tie_point_count'],'never_predictors':['diagnostics','target values','camera poses','mesh metrics','IDs and label provenance'],
        'compatibility_stratum':'Exact hash of version, reported settings/status and label definition/stage; not proof that unreported settings match.'})
    if inventory_scope is not None:write_json(output/'inventory-scope.json',inventory_scope)
    write_json(output/'file_hashes.json',{p.name:sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()})
    return rows


def audit_split(records,assignments):
    """Reject unsafe proposed group splits; does not create or fit a split/model."""
    require(set(assignments)=={r['record_id'] for r in records},'split assignments must cover every unique record')
    for r in records:
        validate_run(r)
        require(r['image_provenance_status']=='verified_hash_manifest','unknown image provenance prevents split verification')
        require(r.get('configuration_status')=='verified','unverified settings prevent compatibility certification')
        require(r.get('acquisition_session_id') and r.get('session_status')=='verified','unknown session provenance prevents split certification')
        require(r['label']['status'] in {'document_verified','computed_historical_verified'},'missing/unverified label prevents supervised split')
    for a,b in itertools.combinations(records,2):
        if assignments[a['record_id']]==assignments[b['record_id']]:continue
        require(a['artifact_group_id']!=b['artifact_group_id'],'artifact leakage across partitions')
        sa=a.get('acquisition_session_id');sb=b.get('acquisition_session_id')
        require(not (sa and sa==sb),'session leakage across partitions')
        ha={i['sha256'] for i in a['source_images']};hb={i['sha256'] for i in b['source_images']}
        require(not ha&hb,'photo-content leakage across partitions')
        require(compatibility_key(a)==compatibility_key(b),'incompatible label/settings strata across evaluation partitions')
    return {'valid':True,'groups':len({r['artifact_group_id'] for r in records})}
