import copy,json
from pathlib import Path
import numpy as np
import pytest
from bd3d_legacy.measured_dataset import export_measured_dataset,validate_record
from bd3d_quality.provenance import read_json,sha256_file
from bd3d_legacy.tabular_dataset import PRE_FEATURES,validate_arrays


@pytest.fixture
def record():return json.loads(Path('archive/examples/ground_truth/synthetic-measured-experiment.json').read_text())


def test_export_consistency_and_no_leaky_arrays(record,tmp_path):
    out=tmp_path/'test';rows=export_measured_dataset([record],out,allow_synthetic=True)
    assert rows[0]['target_dimensional_relative_rmse_percent']==pytest.approx(2)
    schema=read_json(out/'schema.json')
    assert schema['feature_columns']==PRE_FEATURES and schema['data_origin']=='synthetic_test'
    with np.load(out/'arrays.npz',allow_pickle=False) as a:
        assert a['X_pre'].shape==(1,7) and a['y_dimensional'][0,0]==pytest.approx(2)
        assert np.array_equal(a['y_dimensional_mask'],np.isfinite(a['y_dimensional']))
    for name,h in read_json(out/'file_hashes.json').items():assert sha256_file(out/name)==h


def test_missing_feature_and_check_masks(record,tmp_path):
    record['image_features']['image_entropy_bits_mean']=None
    record['dimensional_input']['reconstruction']['measurements'].pop()
    export_measured_dataset([record],tmp_path/'out',allow_synthetic=True)
    with np.load(tmp_path/'out/arrays.npz',allow_pickle=False) as a:
        assert not a['X_pre_mask'][0,4] and np.isnan(a['X_pre'][0,4])
        assert not a['y_dimensional_mask'][0,0] and np.isnan(a['y_dimensional'][0,0])


@pytest.mark.parametrize('mutation',['extra_feature','stage','image_signature','session','run','object','count','duplicate_hash'])
def test_feature_leakage_and_identity_rejected(record,mutation):
    if mutation=='extra_feature':record['image_features']['mesh_face_count']=10
    elif mutation=='stage':record['feature_provenance']['stage']='post_alignment'
    elif mutation=='image_signature':record['feature_provenance']['input_images_sha256']='0'*64
    elif mutation in {'session','run','object'}:record['dimensional_input'][mutation+'_id']='different'
    elif mutation=='count':record['image_features']['input_image_count']=9
    else:record['images'][1]['sha256']=record['images'][0]['sha256']
    with pytest.raises(ValueError):validate_record(record)


def test_real_synthetic_and_final_holdout_barriers(record,tmp_path):
    with pytest.raises(ValueError,match='allow_synthetic'):export_measured_dataset([record],tmp_path/'a')
    real=copy.deepcopy(record);real['data_origin']='real'
    with pytest.raises(ValueError,match='never be mixed'):export_measured_dataset([real,record],tmp_path/'b',allow_synthetic=True)
    record['evaluation_role']='final_holdout'
    with pytest.raises(ValueError,match='held-out'):export_measured_dataset([record],tmp_path/'c',allow_synthetic=True)
    record['evaluation_role']='development'
    with pytest.raises(ValueError,match='reserved'):export_measured_dataset([record],tmp_path/'d',allow_synthetic=True,reserved_object_ids=[record['object_id']])
    assert not any(p.is_dir() for p in tmp_path.iterdir())


def test_duplicate_run_and_subset_alias(record,tmp_path):
    second=copy.deepcopy(record)
    with pytest.raises(ValueError,match='duplicate run'):export_measured_dataset([record,second],tmp_path/'a',allow_synthetic=True)
    second['run_id']=second['dimensional_input']['run_id']=second['dimensional_input']['reconstruction']['run_id']='second-synthetic-run'
    second['images'][0]['sha256']='a'*64
    from bd3d_quality.provenance import digest
    second['feature_provenance']['input_images_sha256']=digest(second['images'])
    with pytest.raises(ValueError,match='subset ID'):export_measured_dataset([record,second],tmp_path/'b',allow_synthetic=True)


def test_archival_array_contract_unchanged(tmp_path):
    # A tiny synthetic bundle in the existing archival format, not actual observations.
    x=np.zeros((3,7),dtype=np.float32);post=np.zeros((3,4),dtype=np.float32)
    y=np.ones((3,4),dtype=np.float32);y[0]=np.nan
    p=tmp_path/'old-format.npz'
    np.savez_compressed(p,X_pre=x,X_pre_mask=np.isfinite(x),X_post_refinement=post,X_post_refinement_mask=np.isfinite(post),
        y_reference=y,y_reference_mask=np.isfinite(y),reference_row_mask=np.array([False,True,True]))
    validate_arrays(p)
