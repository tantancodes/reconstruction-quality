"""Tiny tetrahedron software fixtures only; no real surface evaluation or reconstruction."""
import copy
import pytest
from bd3d_quality.provenance import sha256_file
from bd3d_legacy.surface_accuracy import compare_surface


@pytest.fixture
def spec(tetra):
    proof={'operator':'SYNTHETIC TEST','date':'2026-11-06','evidence_id':'synthetic-only'}
    candidate={'sha256':sha256_file(tetra),'coordinate_unit':'mm','scale_status':'verified',
               'scale_evidence_id':'synthetic-only','provenance':proof}
    return {'schema_version':1,'candidate':candidate,'reference':{**candidate,'reference_id':'synthetic-tetrahedron',
        'reference_type':'independent_reference_scan','validation':{'status':'verified','acquisition_independent':True,
        'method':'synthetic fixture, not a scan','evidence_id':'synthetic-only','standard_uncertainty_mm':.01}},'tolerance_mm':.1}


def test_independent_scan_surface_metrics(tetra,spec):
    r=compare_surface(tetra,tetra,spec,sample_count=100)
    assert r['reference_type']=='independent_reference_scan' and r['independent_surface_target_eligible']
    assert r['metrics']['sampled_surface_rmse_mm']<1e-12
    assert r['metrics']['sampled_surface_median_mm']<1e-12
    assert r['metrics']['sampled_surface_p95_mm']<1e-12
    assert r['metrics']['reference_completeness_fraction']==1
    assert r['comparison']['registration']['scale_allowed'] is False


def test_photogrammetry_reference_is_never_physical_accuracy(tetra,spec):
    spec['reference']['reference_type']='photogrammetry_reference'
    r=compare_surface(tetra,tetra,spec,sample_count=100)
    assert not r['independent_surface_target_eligible']
    assert r['interpretation']=='photogrammetry_reference_relative_agreement'


def test_scan_validation_pending(tetra,spec):
    spec['reference']['validation']['status']='pending'
    r=compare_surface(tetra,tetra,spec,sample_count=100)
    assert r['metrics'] and not r['independent_surface_target_eligible']


@pytest.mark.parametrize('mutation',['scale','unit','hash','dependent','physical_distances_only'])
def test_reference_provenance_gates(tetra,spec,mutation):
    if mutation=='scale':spec['candidate']['scale_status']='pending'
    elif mutation=='unit':spec['candidate']['coordinate_unit']='m'
    elif mutation=='hash':spec['candidate']['sha256']='0'*64
    elif mutation=='dependent':spec['reference']['validation']['acquisition_independent']=False
    else:spec['reference']['reference_type']='independent_physical_ground_truth'
    with pytest.raises(ValueError):compare_surface(tetra,tetra,spec,sample_count=100)
