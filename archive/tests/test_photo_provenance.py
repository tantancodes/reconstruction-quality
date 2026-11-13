"""Synthetic metadata tests; never read photos or invoke Metashape/training."""
import copy
import pytest

from bd3d_legacy.photo_provenance import check_photo_provenance,readiness_dimensions
from bd3d_quality.provenance import digest
from bd3d_legacy.tabular_dataset import PRE_FEATURES


@pytest.fixture
def evidence():
    images=[{'path':f'photo-{i}.jpg','image_id':f'candidate-{i}','sha256':digest(str(i))} for i in range(2)]
    r={'schema_version':1,'kind':'historical_alignment_run','record_id':'test-record','source_category':'synthetic',
        'artifact_id':'test-artifact','artifact_group_id':'test-artifact','reconstruction_run_id':'test-run',
        'acquisition_session_id':'test-session','session_status':'verified','configuration_status':'verified',
        'alignment_configuration':{'accuracy':'synthetic'},'metashape_version':'test-version',
        'image_count':2,'source_images':images,'source_image_set_id':'sha256:'+digest(images),
        'image_provenance_status':'verified_hash_manifest','image_match_evidence':'SYNTHETIC ONLY',
        'image_features':dict(zip(PRE_FEATURES,[2,100,20,300,5,1000,50])),
        'feature_provenance':{'stage':'pre_alignment_images_only','source_image_set_id':'sha256:'+digest(images)},
        'label':{'rmse_pixels':.7,'units':'pixels','status':'document_verified','definition_id':'test-rms',
            'stage':'report_state','source':{'file_path':'synthetic-report','section':'test-section','evidence_sha256':digest('report')}},
        'diagnostics':{'aligned_camera_count':2,'tie_point_count':10}}
    manifest={'schema_version':1,'kind':'report_camera_manifest','artifact_id':r['artifact_id'],'run_state_id':'test-state',
        'scope':{'chunk_id':'0','component_scope':'all-imported-cameras'},
        'cameras':[{'camera_id':str(i),'path':'C:\\old\\'+image['path'],'sha256':image['sha256'],
            'enabled':True,'included_in_alignment':True} for i,image in enumerate(images)]}
    receipt={'review_status':'verified','reviewer':'SYNTHETIC FIXTURE ONLY','verification_basis':'SYNTHETIC EVIDENCE',
        'review_evidence_location':'synthetic-not-a-real-file','review_evidence_sha256':digest('receipt-source'),
        'image_identity_basis':'contemporaneous_hashed_inputs','archive_or_hash_record_id':'synthetic-archive',
        'report_evidence_sha256':r['label']['source']['evidence_sha256'],'manifest_canonical_sha256':digest(manifest),
        'run_state_id':manifest['run_state_id'],'scope':manifest['scope'],'metashape_version':r['metashape_version'],
        'label_stage':r['label']['stage'],'rmse_pixels':.7}
    return r,images,manifest,receipt


def test_matching_counts_without_manifest_or_hashes_are_not_proof(evidence):
    r,images,_,_=evidence
    candidates=copy.deepcopy(images)
    for i in candidates:i['sha256']=None
    result=check_photo_provenance(r,candidates)
    assert result['count_agreement_diagnostic_only']
    assert not result['exact_report_input_association_verified']
    assert result['candidate_missing_or_invalid_hash_count']==2
    assert result['candidate_only_filenames'] is None and result['camera_only_filenames'] is None


def test_paths_and_reviewed_hash_binding_pass_only_metadata_contract(evidence):
    result=check_photo_provenance(*evidence)
    assert result['exact_report_input_association_verified'] and not result['blocked_by']
    assert 'human source review' in result['limits'][0]


@pytest.mark.parametrize('change',['candidate_hash','candidate_case','duplicate_name','duplicate_id','duplicate_bytes',
    'missing_camera_hash','disabled','included_unknown','run_state','stage','version','label','report_hash',
    'manifest_hash','unreviewed','scope','artifact','current_hashes_only'])
def test_each_missing_or_conflicting_evidence_blocks(evidence,change):
    r,images,m,receipt=copy.deepcopy(evidence)
    images=copy.deepcopy(images)  # candidate metadata is independent of the source record
    if change=='candidate_hash':images[0]['sha256']=digest('wrong')
    elif change=='candidate_case':images[0]['path']=images[0]['path'].upper()
    elif change=='duplicate_name':images[1]['path']=images[0]['path']
    elif change=='duplicate_id':images[1]['image_id']=images[0]['image_id']
    elif change=='duplicate_bytes':images[1]['sha256']=images[0]['sha256']
    elif change=='missing_camera_hash':m['cameras'][0]['sha256']=None
    elif change=='disabled':m['cameras'][0]['enabled']=False
    elif change=='included_unknown':m['cameras'][0]['included_in_alignment']=None
    elif change=='run_state':receipt['run_state_id']='wrong'
    elif change=='stage':receipt['label_stage']='initial_alignment'
    elif change=='version':receipt['metashape_version']='wrong'
    elif change=='label':receipt['rmse_pixels']=.119078
    elif change=='report_hash':receipt['report_evidence_sha256']=digest('wrong')
    elif change=='manifest_hash':receipt['manifest_canonical_sha256']=digest('wrong')
    elif change=='unreviewed':receipt['review_status']='provisional'
    elif change=='scope':receipt['scope']={'chunk_id':'0','component_scope':'different'}
    elif change=='current_hashes_only':receipt['image_identity_basis']='current_path_hashes_only'
    else:m['artifact_id']='other-artifact'
    if change in {'missing_camera_hash','disabled','included_unknown','artifact'}:
        receipt['manifest_canonical_sha256']=digest(m)
    assert not check_photo_provenance(r,images,m,receipt)['exact_report_input_association_verified']


def test_readiness_provenance_features_and_groups_are_separate(evidence):
    r,*_=copy.deepcopy(evidence)
    r['image_features']=None;r['feature_provenance']=None
    out=readiness_dimensions([r])
    assert out['verified_rmse_labels']==out['fully_provenance_verified_observations']==1
    assert out['labels_with_matching_image_features']==out['strict_ready_observations']==0
    assert out['groups_eligible_for_held_out_evaluation']==[]


def test_partial_session_blocks_strict_provenance(evidence):
    r,*_=copy.deepcopy(evidence);r['session_status']='unknown'
    out=readiness_dimensions([r])
    assert out['labels_with_matching_image_features']==1
    assert out['fully_provenance_verified_observations']==out['strict_ready_observations']==0


def other_group(r):
    b=copy.deepcopy(r)
    b.update(record_id='other-record',reconstruction_run_id='other-run',artifact_id='other',artifact_group_id='other',acquisition_session_id='other-session')
    b['source_images']=[{'path':f'other-{i}.jpg','sha256':digest('other'+str(i))} for i in range(2)]
    b['source_image_set_id']='sha256:'+digest(b['source_images'])
    b['feature_provenance']['source_image_set_id']=b['source_image_set_id']
    b['label']['source']['evidence_sha256']=digest('other-report')
    return b


def test_structural_heldout_requires_independent_compatible_groups(evidence):
    r,*_=evidence;b=other_group(r)
    assert len(readiness_dimensions([r,b])['groups_eligible_for_held_out_evaluation'])==2
    b['artifact_group_id']=r['artifact_group_id']
    assert readiness_dimensions([r,b])['groups_eligible_for_held_out_evaluation']==[]


def test_incompatible_versions_not_eligible_even_when_provenance_complete(evidence):
    r,*_=evidence;b=other_group(r);b['metashape_version']='other-version'
    out=readiness_dimensions([r,b])
    assert out['fully_provenance_verified_observations']==2
    assert out['groups_eligible_for_held_out_evaluation']==[]
