"""Synthetic metadata/parser/array tests only. No API, images, geometry or models."""
import copy
import numpy as np
import pytest
from bd3d_legacy.historical_alignment import parse_report_text,validate_run,build_dataset,audit_split
from bd3d_quality.provenance import digest,read_json,sha256_file
from bd3d_legacy.tabular_dataset import PRE_FEATURES

TEXT='''Processing Parameters
General
Images 145
Aligned images 145
Tie Points
Points 346,195 of 407,384
RMS reprojection error 0.119078 (0.748542 pix)
Alignment parameters
Accuracy High
Generic preselection Yes
Reference preselection Sequential
Key point limit 40,000
Key point limit per Mpx 1,000
Tie point limit 10,000
Optimization parameters
Date created 2025:11:02 09:47:26
Software version 2.1.4.19695
System
Software version 2.1.4 build 19695
'''


@pytest.fixture
def record():
    images=[{'path':f'test-{i}.jpg','sha256':digest(f'synthetic-image-{i}')} for i in range(2)]
    sid='sha256:'+digest(images)
    return {'schema_version':1,'kind':'historical_alignment_run','record_id':'test-record','artifact_id':'test-object',
        'artifact_group_id':'test-object','source_category':'synthetic_test_fixture','acquisition_session_id':'test-session',
        'session_status':'verified','reconstruction_run_id':'test-run','source_image_set_id':sid,'image_count':2,
        'source_images':images,'image_provenance_status':'verified_hash_manifest','image_match_evidence':'SYNTHETIC TEST ONLY',
        'image_features':dict(zip(PRE_FEATURES,[2,100,20,300,5,1000,50])),
        'feature_provenance':{'stage':'pre_alignment_images_only','source_image_set_id':sid,'settings_signature':digest('fixture-settings')},
        'metashape_version':'test-version','configuration_status':'verified','alignment_configuration':{'accuracy':'fixture'},
        'label':{'rmse_pixels':.7,'units':'pixels','status':'computed_historical_verified','definition_id':'test-definition',
                 'stage':'test-stage','source':{'file_path':'synthetic-not-a-real-report','section':'fixture','evidence_sha256':digest('fixture-report')}},
        'diagnostics':{'aligned_camera_count':2,'tie_point_count':100}}


def second_record(record):
    b=copy.deepcopy(record);b['record_id']='test-record-2';b['reconstruction_run_id']='test-run-2'
    b['artifact_id']=b['artifact_group_id']='test-object-2';b['acquisition_session_id']='test-session-2'
    b['source_images']=[{'path':f'other-{i}.jpg','sha256':digest(f'other-image-{i}')} for i in range(2)]
    b['source_image_set_id']='sha256:'+digest(b['source_images']);b['feature_provenance']['source_image_set_id']=b['source_image_set_id']
    b['label']['source']['evidence_sha256']=digest('other-report')
    return b


def test_report_precise_pixel_value_not_normalized_or_summary():
    r=parse_report_text('Reprojection error: 0.749 pix\n'+TEXT)
    assert r['rmse_pixels']==.748542 and r['normalized_report_rmse']==.119078
    assert r['image_count']==r['aligned_camera_count']==145 and r['tie_point_count']==346195
    assert r['metashape_version']=='2.1.4.19695'
    assert r['alignment_configuration']['keypoint_limit']=='40,000'
    assert r['alignment_configuration']['keypoint_limit_per_mpx']=='1,000'
    assert 'optimization' in r['label_stage']


@pytest.mark.parametrize('text',[TEXT+TEXT,TEXT.replace('RMS reprojection error','Unknown error'),TEXT.replace('Aligned images 145','Aligned images 146'),TEXT+'Software version 2.2.1'])
def test_ambiguous_or_invalid_report_rejected(text):
    with pytest.raises(ValueError):parse_report_text(text)


@pytest.mark.parametrize('change',['feature_count','leaky_feature','stage','image_set','missing_hash','label_units','missing_label','duplicate_images','aligned_count'])
def test_bad_record(record,change):
    if change=='feature_count':record['image_features']['input_image_count']=99
    elif change=='leaky_feature':record['image_features']['tie_point_count']=100
    elif change=='stage':record['feature_provenance']['stage']='post_alignment'
    elif change=='image_set':record['feature_provenance']['source_image_set_id']='other'
    elif change=='missing_hash':record['source_images'][0]['sha256']=None
    elif change=='label_units':record['label']['units']='normalized'
    elif change=='missing_label':record['label']['rmse_pixels']=None
    elif change=='duplicate_images':record['source_images'].append(copy.deepcopy(record['source_images'][0]))
    else:record['diagnostics']['aligned_camera_count']=3
    with pytest.raises(ValueError):validate_run(record)


def test_candidate_images_cannot_get_features(record):
    record['image_provenance_status']='candidate_only'
    with pytest.raises(ValueError,match='ambiguous'):validate_run(record)


def test_missing_label_retained_not_zero(record,tmp_path):
    record['label'].update(status='missing',rmse_pixels=None)
    build_dataset([record],tmp_path/'out')
    with np.load(tmp_path/'out/arrays.npz',allow_pickle=False) as a:
        assert np.isnan(a['y_rmse_pixels'][0,0]) and not a['y_rmse_pixels_mask'][0,0]
        assert not a['model_input_row_mask'][0]


def test_exports_and_baseline_not_masked_as_self(record,tmp_path):
    out=tmp_path/'out';build_dataset([record],out)
    with np.load(out/'arrays.npz',allow_pickle=False) as a:
        assert a['X_pre'].shape==(1,7) and a['y_rmse_pixels'][0,0]==pytest.approx(.7)
        assert a['model_input_row_mask'][0]
    assert read_json(out/'schema.json')['feature_columns']==PRE_FEATURES
    for name,h in read_json(out/'file_hashes.json').items():assert sha256_file(out/name)==h


def test_duplicate_records_evidence_and_alignment_state(record,tmp_path):
    with pytest.raises(ValueError,match='duplicate reconstruction'):build_dataset([record,record],tmp_path/'a')
    b=second_record(record);b['label']['source']=copy.deepcopy(record['label']['source'])
    with pytest.raises(ValueError,match='duplicate label evidence'):build_dataset([record,b],tmp_path/'b')
    b=second_record(record);record['alignment_state_id']=b['alignment_state_id']='same-test-state'
    with pytest.raises(ValueError,match='alignment state'):build_dataset([record,b],tmp_path/'c')


@pytest.mark.parametrize('change,match',[('artifact','artifact leakage'),('session','session leakage'),('photos','photo-content leakage'),('version','incompatible'),('settings','incompatible'),('definition','incompatible'),('unknown_session','unknown session')])
def test_split_leakage(record,change,match):
    b=second_record(record)
    if change=='artifact':b['artifact_group_id']=record['artifact_group_id']
    elif change=='session':b['acquisition_session_id']=record['acquisition_session_id']
    elif change=='photos':
        b['source_images']=copy.deepcopy(record['source_images']);b['source_image_set_id']=record['source_image_set_id'];b['feature_provenance']['source_image_set_id']=record['source_image_set_id']
    elif change=='version':b['metashape_version']='different'
    elif change=='settings':b['alignment_configuration']['accuracy']='different'
    elif change=='definition':b['label']['definition_id']='different'
    else:b['acquisition_session_id']=None;b['session_status']='unknown'
    with pytest.raises(ValueError,match=match):audit_split([record,b],{record['record_id']:'train',b['record_id']:'test'})


def test_safe_split_check_without_fitting(record):
    b=second_record(record)
    assert audit_split([record,b],{record['record_id']:'train',b['record_id']:'test'})['valid']


def test_cli_report_parse_is_metadata_only(tmp_path):
    from bd3d_legacy.cli import main
    path=tmp_path/'synthetic-report.txt';path.write_text(TEXT)
    out=tmp_path/'parsed.json'
    assert main(['alignment-report','parse',str(path),'--output',str(out)])==0
    assert read_json(out)['rmse_pixels']==.748542
    assert main(['alignment-report','parse',str(path),'--output',str(out)])==2


def test_partial_configuration_blocks_model_eligibility(record,tmp_path):
    record['configuration_status']='partially_reported'
    build_dataset([record],tmp_path/'out')
    r=read_json(tmp_path/'out/readiness.json')
    assert r['linked_feature_rows']==1 and r['model_input_eligible_rows']==0
    with pytest.raises(ValueError,match='unverified settings'):
        audit_split([record],{record['record_id']:'train'})


def test_optimizer_settings_change_stratum(record):
    b=second_record(record);b['optimization_configuration']={'fit_parameters':'different'}
    with pytest.raises(ValueError,match='incompatible'):
        audit_split([record,b],{record['record_id']:'train',b['record_id']:'test'})


def test_saved_project_state_is_not_counted_as_an_execution_report(record,tmp_path):
    record.update(observation_unit='saved_project_state',source_category='priority_drive_project')
    record['label'].update(rmse_pixels=None,status='missing')
    build_dataset([record],tmp_path/'out')
    r=read_json(tmp_path/'out/readiness.json')
    assert r['all_historical']['documented_records']==1
    assert r['all_historical']['documented_runs']==0
    assert r['priority_drive_collections']['saved_project_states']==1
    assert r['model_input_eligible_rows']==0
