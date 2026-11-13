"""Optional explicitly typed, rigid-only surface comparison; dimensional checks need no scan."""
import math
from pathlib import Path
from .dimensional_accuracy import UNIT_MM,require,text,finite,provenance
from bd3d_quality.provenance import sha256_file,digest

REFERENCE_TYPES={'independent_physical_ground_truth','independent_reference_scan','photogrammetry_reference'}


def compare_surface(candidate_path,reference_path,spec,*,sample_count=3000,max_iterations=200):
    require(spec.get('schema_version')==1,'unsupported surface specification')
    ref=spec.get('reference',{});candidate=spec.get('candidate',{})
    kind=ref.get('reference_type');require(kind in REFERENCE_TYPES,'unknown surface reference type')
    text(ref.get('reference_id'),'surface reference_id')
    provenance(ref.get('provenance'),'surface reference');provenance(candidate.get('provenance'),'surface candidate')
    unit=ref.get('coordinate_unit')
    require(unit in UNIT_MM and candidate.get('coordinate_unit')==unit,'surface comparison requires explicitly calibrated matching metric coordinate units; no automatic unit/scale fit')
    for description,record,path in [('reference',ref,reference_path),('candidate',candidate,candidate_path)]:
        require(record.get('scale_status')=='verified',f'{description} scale is not verified')
        text(record.get('scale_evidence_id'),f'{description} scale evidence')
        require(sha256_file(Path(path))==record.get('sha256'),f'{description} geometry hash mismatch')
    tolerance=finite(spec.get('tolerance_mm'),'surface tolerance_mm',positive=True)
    validation=ref.get('validation',{})
    require(validation.get('status') in {'verified','pending'},'reference validation status required')
    if kind!='photogrammetry_reference':
        require(validation.get('acquisition_independent') is True,'independent reference acquisition required')
        text(validation.get('method'),'reference validation method')
        text(validation.get('evidence_id'),'reference validation evidence')
        if validation.get('standard_uncertainty_mm') is not None:finite(validation['standard_uncertainty_mm'],'reference surface uncertainty')
        if kind=='independent_physical_ground_truth':
            require(validation.get('supports_surface_accuracy') is True,'physical distances alone do not define full surface ground truth')
    from .geometry_quality import load_obj_mesh,compare_geometry
    import numpy as np
    ref_mesh=load_obj_mesh(reference_path)
    diagonal=float(np.linalg.norm(np.ptp(ref_mesh.vertices,axis=0)))
    require(diagonal>0,'reference extent is degenerate')
    fraction=tolerance/(UNIT_MM[unit]*diagonal)
    require(0<fraction<=1,'surface tolerance exceeds supported reference-diagonal range')
    result=compare_geometry(candidate_path,reference_path,reference_id=ref['reference_id'],units=unit,
        sample_count=sample_count,max_iterations=max_iterations,threshold_fraction=fraction,
        reference_kind='reconstruction')  # wrapper assigns stricter reference semantics below
    evidence_ok=(kind!='photogrammetry_reference' and validation['status']=='verified'
                 and validation.get('standard_uncertainty_mm') is not None)
    qualified=evidence_ok and result['status']=='complete'
    metrics=None
    if result['metrics'] is not None:
        m=result['metrics'];s=m['symmetric_distances'];factor=UNIT_MM[unit]
        metrics={'sampled_surface_rmse_mm':s['rmse']*factor,'sampled_surface_median_mm':s['median']*factor,
            'sampled_surface_p95_mm':s['p95']*factor,'reference_completeness_fraction':m['reference_completeness_fraction'],
            'candidate_surface_coverage_fraction':m['candidate_coverage_fraction'],'tolerance_mm':tolerance}
    return {'schema_version':1,'kind':'surface_accuracy_report','reference_type':kind,
        'reference_id':ref['reference_id'],'specification_sha256':digest(spec),'specification':spec,
        'comparison_status':result['status'],'measurement_scope':'area-sampled nearest-sample surface agreement',
        'interpretation':'agreement_with_researcher_verified_independent_reference' if qualified else
                         'photogrammetry_reference_relative_agreement' if kind=='photogrammetry_reference' else 'independent_reference_validation_incomplete_or_registration_invalid',
        'independent_surface_target_eligible':qualified,'metrics':metrics,'comparison':result,
        'warnings':['No scale optimization or nonrigid deformation is performed.',
            'Rigid ICP removes pose differences; inspect transformations and fit before interpreting geometry.',
            'Reference validation is a researcher assertion, not independently certified by this software.',
            'Independent scan error, occlusion and sampling noise remain; this is not exact point-to-surface distance.',
            'Physical check distances alone support dimensional accuracy, never full-surface accuracy.']}
