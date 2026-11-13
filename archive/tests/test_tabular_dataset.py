"""Tiny array/schema tests only; no reconstruction, geometry or model execution."""
import copy
import numpy as np
import pytest
from bd3d_legacy.tabular_dataset import PRE_FEATURES,POST_FEATURES,TARGETS,validate_arrays,verify_file,field_dictionary
from bd3d_quality.provenance import sha256_file


def arrays():
    x=np.ones((3,7),dtype=np.float32);p=np.ones((3,4),dtype=np.float32)
    y=np.ones((3,4),dtype=np.float32);y[0]=np.nan
    return dict(X_pre=x,X_pre_mask=np.isfinite(x),X_post_refinement=p,X_post_refinement_mask=np.isfinite(p),
                y_reference=y,y_reference_mask=np.isfinite(y),reference_row_mask=np.array([False,True,True]))


def test_numeric_bundle_no_pickle(tmp_path):
    path=tmp_path/'arrays.npz';np.savez_compressed(path,**arrays());validate_arrays(path)


@pytest.mark.parametrize('mutation',['mask','self','dtype','shape'])
def test_bad_numeric_bundle(tmp_path,mutation):
    data=arrays()
    if mutation=='mask':data['y_reference_mask'][1,0]=False
    if mutation=='self':data['reference_row_mask'][0]=True
    if mutation=='dtype':data['X_pre']=data['X_pre'].astype(np.float64)
    if mutation=='shape':data['X_pre']=data['X_pre'][:2]
    path=tmp_path/'arrays.npz';np.savez_compressed(path,**data)
    with pytest.raises(ValueError):validate_arrays(path)


def test_hash_corruption_detected(tmp_path):
    path=tmp_path/'evidence.json';path.write_text('{}');expected=sha256_file(path)
    verify_file(path,expected);path.write_text('{"changed":true}')
    with pytest.raises(ValueError):verify_file(path,expected)


def test_feature_roles_separate():
    assert not set(PRE_FEATURES)&set(POST_FEATURES+TARGETS)
    row={k:1 for k in PRE_FEATURES+POST_FEATURES+TARGETS}
    row.update(diagnostic_mesh_surface_area=5,target_ground_truth_rmse=None)
    d=field_dictionary([row])['fields']
    assert all(d[k]['role']=='input_only_predictor' for k in PRE_FEATURES)
    assert all(d[k]['role']=='reference_agreement_target' for k in TARGETS)
    assert d['diagnostic_mesh_surface_area']['units']=='historical export units squared'
    assert d['target_ground_truth_rmse']['missing_count']==1
