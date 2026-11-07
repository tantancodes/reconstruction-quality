"""Synthetic scalar metadata only: no photographs, Metashape or model training."""
import copy
import itertools

import pytest

from bd3d_quality.historical_recovery import cache_subset_features, recover_legacy_report, initial_alignment_inventory
from bd3d_quality.provenance import digest, write_json, sha256_file


@pytest.fixture
def caches():
    images = [{'path':str(i)+'.jpg','sha256':digest(str(i)),
        'brightness_mean':i,'contrast_std':2*i,'laplacian_variance':3*i,'entropy_bits':4*i} for i in range(3)]
    feature = {'images':images,'settings':{},'settings_signature':digest({})}
    orb = {'images':[{'path':i['path'],'sha256':i['sha256'],'orb_keypoint_count':10*n}
                     for n,i in enumerate(images)],'settings':{},'settings_signature':digest({}),
           'pairs':[{'path_a':a['path'],'path_b':b['path'],'ratio_match_count_a_to_b':n+1}
                    for n,(a,b) in enumerate(itertools.combinations(images,2))]}
    return feature, orb


def selected(caches):
    return [{'path':i['path'],'sha256':i['sha256']} for i in caches[0]['images'][1:]]


def test_cached_subset_uses_hashes_and_original_pair_direction(caches):
    f,m=caches
    result=cache_subset_features(list(reversed(selected(caches))),f,m)
    assert result['input_image_count']==2 and result['image_brightness_mean']==1.5
    assert result['image_orb_keypoint_count_mean']==15 and result['pair_ratio_match_count_mean']==3


@pytest.mark.parametrize('bad',['image_hash','orb_hash','settings','missing_pair','reversed_pair'])
def test_bad_cache_association_rejected(caches,bad):
    f,m=copy.deepcopy(caches);images=selected(caches)
    if bad=='image_hash':images[0]['sha256']=digest('other')
    elif bad=='orb_hash':m['images'][1]['sha256']=digest('other')
    elif bad=='settings':f['settings']={'changed':True}
    elif bad=='missing_pair':m['pairs'].pop()
    else:m['pairs'][0]['path_a'],m['pairs'][0]['path_b']=m['pairs'][0]['path_b'],m['pairs'][0]['path_a']
    with pytest.raises(ValueError):cache_subset_features(images,f,m)


def report():
    return {'run_id':'synthetic-run','run':{'input_image_count':2,'project_name':'ARF_Dog_fixture',
        'effective_settings':{'sparse_downscale':1},'metashape_api_version':'fixture'},
        'alignment':{'snapshot_stage':'post_error_reduction_pre_dense','reprojection_error_px':{'rmse':.3},
            'aligned_cameras':2,'valid_tie_point_count':5},'output_status':{'success':True}}


def test_no_manifest_means_no_association_or_features(tmp_path,caches):
    p=tmp_path/'report.json';write_json(p,report())
    r,a=recover_legacy_report(p,*caches,{})
    assert r['image_features'] is None and r['image_provenance_status']=='missing'
    assert r['label']['rmse_pixels']==.3 and not a['manifest_verified']


def test_old_report_without_effective_settings_keeps_missing(tmp_path,caches):
    d=report();del d['run']['effective_settings']
    p=tmp_path/'old.json';write_json(p,d)
    r,_=recover_legacy_report(p,*caches,{})
    assert r['alignment_configuration']['sparse_downscale'] is None
    assert r['alignment_configuration']['historical_effective_profile'] is None
    assert r['configuration_status']=='partially_reported'


def test_failed_initial_snapshot_keeps_null_not_zero(tmp_path,caches):
    d=report();d['alignment']['snapshot_stage']='post_alignment_pre_error_reduction'
    d['alignment']['reprojection_error_px']['rmse']=None;d['output_status']['success']=False
    p=tmp_path/'failed.json';write_json(p,d)
    r,_=recover_legacy_report(p,*caches,{})
    assert r['label']['rmse_pixels'] is None and r['label']['status']=='missing'
    assert r['label']['stage']=='post_alignment_pre_error_reduction'


@pytest.mark.parametrize('count_key',['images_used_count','image_count'])
def test_manifest_verified_and_mismatched_hash_rejected(tmp_path,caches,count_key):
    m={'artifact_id':'arf-dog','experiment_id':'fixture',count_key:2,
       'images':[{'filename':i['path'],'sha256':i['sha256']} for i in selected(caches)]}
    mp=tmp_path/'manifest.json';write_json(mp,m)
    d=report();d['experiment']={'manifest_path':str(mp),'manifest_sha256':sha256_file(mp),
        'artifact_id':'arf-dog','experiment_id':'fixture','images_used_count':2}
    p=tmp_path/'report.json';write_json(p,d)
    r,a=recover_legacy_report(p,*caches,{})
    assert a['manifest_verified'] and r['image_features']['input_image_count']==2
    d['experiment']['manifest_sha256']=digest('wrong');p2=tmp_path/'wrong.json';write_json(p2,d)
    with pytest.raises(ValueError,match='manifest hash'):recover_legacy_report(p2,*caches,{})


def test_numeric_unknown_stage_requires_review(tmp_path,caches):
    d=report();d['alignment']['snapshot_stage']='unknown'
    p=tmp_path/'report.json';write_json(p,d)
    with pytest.raises(ValueError,match='unreviewed'):recover_legacy_report(p,*caches,{})


def test_inherited_initial_snapshot_not_new_alignment(tmp_path):
    stage={'stage':'initial_alignment','metrics':{'reprojection_rmse_px':.5,'input_image_count':2}}
    for name,parent in [('first',None),('continuation','first')]:
        folder=tmp_path/name;folder.mkdir()
        mp=folder/'metrics-initial_alignment.json';write_json(mp,stage)
        r={'run_id':name,'status':'checkpointed','stages':[stage],'artifacts':[{'path':mp.name,'sha256':sha256_file(mp)}],
            'dataset_id':'fixture','configuration':{},'software':{'Metashape':'fixture'}}
        if parent:r['continuation_of']=parent
        write_json(folder/'run.json',r)
    audit=initial_alignment_inventory(tmp_path)
    assert audit['distinct_initial_alignment_events']==1
    assert audit['inherited_snapshots_not_new_labels']==1
