"""Checkpoint cloning (bd3d_legacy.checkpoint_clone), moved from tests/test_resources.py."""
import pytest
from bd3d_quality.metashape_runner import run_reconstruction, project_fingerprints
from bd3d_quality.provenance import write_json


def test_clone_rejects_unverified_termination(tmp_path):
    from bd3d_legacy.checkpoint_clone import clone_checkpoint
    parent=tmp_path/'parent';parent.mkdir()
    write_json(tmp_path/'parent-resources.json',{'status':'complete','returncode':0})
    with pytest.raises(ValueError,match='termination'):
        clone_checkpoint(parent,tmp_path/'copy',tmp_path)
    assert not (tmp_path/'copy').exists()


def test_clone_preserves_checkpoint_and_lineage(dataset,tmp_path,fake_api):
    from bd3d_legacy.checkpoint_clone import clone_checkpoint
    root,m=dataset;parent=tmp_path/'parent'
    state=run_reconstruction(m,root,parent,api=fake_api)
    (parent/'project.files').mkdir();(parent/'project.files/doc.xml').write_text('scientific state')
    state['project_files']=project_fingerprints(parent);state['status']='checkpointed'
    write_json(parent/'run.json',state,replace=True)
    (parent/'project.files/lock').write_text('stale')
    (parent/'.run.lock').touch()
    write_json(tmp_path/'parent-resources.json',{'status':'stopped_or_failed','returncode':-15,'pid':2147483647})
    copied=clone_checkpoint(parent,tmp_path/'child',root)
    assert copied['run_id']!=state['run_id']
    assert copied['continuation_of']['run_id']==state['run_id']
    assert (parent/'.run.lock').exists() and not (tmp_path/'child/.run.lock').exists()
    assert project_fingerprints(parent)==project_fingerprints(tmp_path/'child')
