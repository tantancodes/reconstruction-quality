"""Small synthetic XML/ZIP metadata only. Never starts Metashape."""
from pathlib import Path
import zipfile
import pytest
from bd3d_quality.project_metadata import inspect_project,inspect_frame,read_archive_xml,xml_root,compare_stems,MAX_XML


def archive(path,text):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('doc.xml',text)


def test_pointer_chunk_paths_and_sigma_not_a_label(tmp_path):
    p=tmp_path/'test.psx';p.write_text('<document path="{projectname}.files/project.zip"/>')
    pj=tmp_path/'project.zip';archive(pj,'<document><chunks><chunk id="0" path="0/chunk.zip"/></chunks></document>')
    cj=tmp_path/'chunk.zip';archive(cj,'<chunk label="test"><components><component id="0"/><component id="1"/></components><cameras><camera id="0" label="a"><transform>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</transform></camera><camera id="1" label="b"/></cameras><frames><frame path="0/frame.zip"/></frames><meta><property name="OptimizeCameras/sigma0" value="0.123"/></meta></chunk>')
    r=inspect_project(p,pj,cj)
    assert r['camera_count']==2 and r['cameras_with_stored_transform']==1
    assert r['photo_paths']==[] and r['asset_references'][0]['path']=='0/frame.zip'
    assert r['verified_reprojection_rmse_pixels'] is None
    assert r['candidate_optimizer_statistics']['OptimizeCameras/sigma0']['raw_value']=='0.123'
    assert r['report_references']==[]
    m=compare_stems(r['cameras'],[{'title':'a.tif'},{'title':'b.jpg'}])
    assert m['filename_stem_sets_equal'] and not m['exact_processed_image_bytes_verified']


@pytest.mark.parametrize('data',[b'<!DOCTYPE a [<!ENTITY x "abc">]><a>&x;</a>',b'<bad>',b'x'*(MAX_XML+1)])
def test_unsafe_xml_rejected(data):
    with pytest.raises(ValueError):xml_root(data)


@pytest.mark.parametrize('name',['../doc.xml','/doc.xml','folder\\doc.xml'])
def test_archive_traversal_rejected(tmp_path,name):
    p=tmp_path/'bad.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr(name,'<doc/>')
    with pytest.raises(ValueError):read_archive_xml(p)


def test_zip_expansion_budget(tmp_path):
    p=tmp_path/'large.zip';archive(p,'x'*(MAX_XML+1))
    with pytest.raises(ValueError):read_archive_xml(p)


def test_duplicate_member_rejected(tmp_path):
    p=tmp_path/'dup.zip'
    with pytest.warns(UserWarning):
        with zipfile.ZipFile(p,'w') as z:z.writestr('doc.xml','<a/>');z.writestr('doc.xml','<b/>')
    with pytest.raises(ValueError):read_archive_xml(p)


def test_active_component_membership_and_unknown_references(tmp_path):
    p=tmp_path/'p.psx';p.write_text('<document path="p.files/project.zip"/>')
    pj=tmp_path/'project.zip';archive(pj,'<document/>')
    cj=tmp_path/'chunk.zip'
    matrix='1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1'
    archive(cj,f'<chunk><components active_id="1"><component id="0"/><component id="1"/></components><cameras><camera id="0" label="a" component_id="0"><transform>{matrix}</transform></camera><camera id="1" label="b" component_id="1"><transform>{matrix}</transform></camera><camera id="2" label="c"/></cameras></chunk>')
    r=inspect_project(p,pj,cj)
    assert r['active_component_id']=='1'
    assert r['components'][1]['active'] and r['components'][1]['camera_labels']==['b']
    assert r['camera_labels_without_transform']==['c'] and not r['component_integrity_issues']
    archive(cj,'<chunk><components active_id="ghost"/><cameras><camera id="0" component_id="ghost"><transform>NaN</transform></camera></cameras></chunk>')
    assert len(inspect_project(p,pj,cj)['component_integrity_issues'])>=3


def test_frame_paths_join_by_camera_id_without_opening_files(tmp_path):
    p=tmp_path/'frame.zip'
    archive(p,'<frame><cameras><camera camera_id="7"><photo path="../../missing.jpg"><meta><property name="File/ImageWidth" value="100"/></meta></photo></camera></cameras><point_cloud path="missing/cloud.zip"/></frame>')
    r=inspect_frame(p,[{'camera_id':'7','label':'saved-label'},{'camera_id':'8','label':'other'}])
    assert r['photos'][0]['chunk_camera_label']=='saved-label'
    assert r['photos'][0]['photo_path']=='../../missing.jpg'
    assert r['chunk_camera_ids_missing_from_frame']==['8']
    assert not r['historical_content_identity_verified'] and r['verified_reprojection_rmse_pixels'] is None


def test_duplicate_frame_camera_ids_rejected(tmp_path):
    p=tmp_path/'frame.zip';archive(p,'<frame><cameras><camera camera_id="0"/><camera camera_id="0"/></cameras></frame>')
    with pytest.raises(ValueError,match='duplicate frame'):inspect_frame(p,[])
