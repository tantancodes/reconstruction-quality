"""Independent check-distance errors, explicit scale and uncertainty; no mesh inference."""
from __future__ import annotations
from datetime import date
import math
import re
from bd3d_quality.provenance import digest

UNIT_MM = {'mm': 1.0, 'cm': 10.0, 'm': 1000.0}
MEASURANDS = {'rectangular_length','rectangular_width','rectangular_height',
              'cylinder_diameter','cylinder_height','landmark_distance'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value, label):
    require(isinstance(value,str) and bool(value.strip()), f'{label} must be a nonempty string')
    return value


def finite(value, label, *, positive=False):
    require(type(value) in (int,float) and math.isfinite(value), f'{label} must be finite numeric data')
    require(value > 0 if positive else value >= 0, f'{label} must be {"positive" if positive else "nonnegative"}')
    return float(value)


def to_mm(value, unit):
    require(unit in UNIT_MM, f'unsupported physical unit {unit!r}; use mm, cm, or m')
    result=finite(value,'dimension',positive=True)*UNIT_MM[unit]
    return finite(result,'converted dimension',positive=True)


def provenance(record, label):
    require(isinstance(record,dict), f'{label} provenance must be an object')
    for key in ['operator','evidence_id','date']:
        text(record.get(key), f'{label} provenance.{key}')
    try:
        date.fromisoformat(record['date'])
    except ValueError as exc:
        raise ValueError(f'{label} provenance.date must be ISO YYYY-MM-DD') from exc
    if record.get('evidence_sha256') is not None:
        require(bool(re.fullmatch('[0-9a-f]{64}',record['evidence_sha256'])), f'{label}: invalid evidence SHA-256')


def standard_uncertainty(record, units, label):
    """Convert an explicitly stated standard or expanded uncertainty; missing != 0."""
    if record is None:
        return None
    require(isinstance(record,dict), f'{label}: uncertainty must be an object or null')
    finite(record.get('value'),f'{label} uncertainty')
    require(record.get('unit') in units, f'{label}: incompatible uncertainty unit')
    text(record.get('method'),f'{label} uncertainty.method')
    kind=record.get('kind')
    require(kind in {'standard','expanded'}, f'{label}: uncertainty kind must be standard or expanded')
    k=1.0
    if kind=='expanded':
        k=finite(record.get('coverage_factor'),f'{label} coverage_factor',positive=True)
        require(k>=1, f'{label}: coverage factor must be >=1')
    elif record.get('coverage_factor') not in (None,1):
        raise ValueError(f'{label}: standard uncertainty cannot have an expanded coverage factor')
    return float(record['value'])*units[record['unit']]/k


def reconstructed_distance(record, geometry_unit):
    require(record.get('unit')==geometry_unit,'reconstructed units must match declared geometry coordinate units')
    method=record.get('method')
    require(method in {'annotated_distance','annotated_landmarks'},'unsupported reconstructed measurement method')
    if method=='annotated_distance':
        require('points' not in record,'provide a distance or points, not both')
        value=record.get('value')
        return None if value is None else finite(value,'reconstructed distance',positive=True)
    require('value' not in record,'provide points or a distance, not both')
    points=record.get('points')
    if points is None:
        return None
    require(isinstance(points,list) and len(points)==2,'exactly two annotated 3D points required')
    for p in points:
        require(isinstance(p,list) and len(p)==3,'point must contain three coordinates')
        require(all(type(v) in (int,float) and math.isfinite(v) for v in p),'nonfinite point coordinates')
    distance=math.dist(*points)
    require(distance>0,'annotated endpoints coincide')
    return distance


def validate_bundle(bundle):
    require(isinstance(bundle,dict),'dimensional input must be an object')
    require(bundle.get('schema_version')==1 and bundle.get('kind')=='dimensional_evaluation_input','unsupported dimensional schema')
    require(bundle.get('data_origin') in {'real','synthetic_test'},'explicit real/synthetic_test data origin required')
    for key in ['object_id','session_id','subset_id','run_id']:
        text(bundle.get(key),key)
    gt=bundle.get('ground_truth',{}); rec=bundle.get('reconstruction',{})
    require(isinstance(gt,dict) and isinstance(rec,dict),'ground_truth and reconstruction must be objects')
    require(gt.get('object_id')==bundle['object_id'],'ground-truth object ID mismatch')
    require(gt.get('reference_type')=='independent_physical_ground_truth','dimensional pathway requires independent physical measurements')
    text(gt.get('reference_id'),'reference_id')
    for key in ['object_id','session_id','subset_id','run_id']:
        require(rec.get(key)==bundle[key],f'reconstruction {key} mismatch')
    require(bool(re.fullmatch('[0-9a-f]{64}',rec.get('geometry_sha256',''))),'geometry SHA-256 required')
    geometry_unit=rec.get('coordinate_unit')
    require(geometry_unit in {'arbitrary',*UNIT_MM},'declare geometry coordinate unit explicitly')
    if rec.get('uncertainty_assumption')=='independent_components':
        text(rec.get('uncertainty_assumption_notes'),'uncertainty independence justification')
    require(rec.get('uncertainty_assumption') in {'independent_components','not_established'},'declare uncertainty/covariance assumption')
    measurements=gt.get('measurements')
    require(isinstance(measurements,list) and bool(measurements),'physical measurements required')
    ids={}; measurands=set(); pairs=set()
    for m in measurements:
        require(isinstance(m,dict),'physical measurement must be an object')
        mid=text(m.get('measurement_id'),'measurement_id')
        require(mid not in ids,'duplicate physical measurement ID'); ids[mid]=m
        measured_object=text(m.get('object_id'),'physical measurement object_id')
        if m.get('role')=='check': require(measured_object==bundle['object_id'],'check measurement object ID mismatch')
        measurand=(measured_object,text(m.get('measurand_id'),'measurand_id'))
        require(measurand not in measurands,'duplicate physical measurand: collapse repeats before evaluation')
        measurands.add(measurand)
        require(m.get('role') in {'calibration','check'},'measurement role must be calibration or check')
        require(m.get('qc_status') in {'verified','pending','rejected'},'physical QC status required')
        definition=m.get('definition',{})
        require(isinstance(definition,dict),'measurand definition must be an object')
        require(definition.get('kind') in MEASURANDS,'unsupported measurand kind')
        text(definition.get('description'),'measurand definition')
        landmarks=definition.get('landmarks')
        require(isinstance(landmarks,list) and len(landmarks)==2 and all(isinstance(v,str) and v.strip() for v in landmarks)
                and landmarks[0]!=landmarks[1],'two distinct landmark/face definitions required')
        pair=(measured_object,*sorted(landmarks))
        require(pair not in pairs,'duplicate endpoint pair: possible calibration/check circularity'); pairs.add(pair)
        require(m.get('unit') in UNIT_MM,'unsupported physical unit')
        if m.get('value') is not None: to_mm(m['value'],m['unit'])
        instrument=m.get('instrument',{})
        require(isinstance(instrument,dict),'instrument must be an object')
        text(instrument.get('instrument_id'),'instrument_id'); text(instrument.get('description'),'instrument description')
        provenance(m.get('provenance'),'physical measurement')
        standard_uncertainty(m.get('uncertainty'),UNIT_MM,'physical measurement')
    reconstructed=rec.get('measurements')
    require(isinstance(reconstructed,list),'reconstructed measurements list required')
    by_id={}
    for m in reconstructed:
        require(isinstance(m,dict),'reconstructed measurement must be an object')
        mid=m.get('measurement_id')
        require(mid in ids and mid not in by_id,'unknown or duplicate reconstructed measurement ID')
        require(m.get('object_id')==ids[mid]['object_id'],'annotation object ID mismatch')
        require(m.get('landmarks')==ids[mid]['definition']['landmarks'],'corresponding landmark definitions/order mismatch')
        require(m.get('qc_status') in {'verified','pending','rejected'},'reconstructed annotation QC required')
        provenance(m.get('provenance'),'reconstructed annotation')
        reconstructed_distance(m,geometry_unit)
        standard_uncertainty(m.get('uncertainty'),{geometry_unit:1},'reconstructed distance')
        by_id[mid]=m
    scale=rec.get('scale',{})
    require(isinstance(scale,dict),'scale must be an object')
    require(scale.get('status') in {'verified','pending'},'scale status required')
    require(scale.get('method') in {'single_calibration_distance','external_metric_scale'},'unsupported scale method')
    provenance(scale.get('provenance'),'scale')
    cal_ids=scale.get('calibration_measurement_ids')
    require(isinstance(cal_ids,list) and len(cal_ids)==len(set(cal_ids)),'explicit unique calibration measurement IDs required')
    for mid in cal_ids:
        require(mid in ids and ids[mid]['role']=='calibration','scale calibration cannot use an independent check measurement')
    if scale['method']=='single_calibration_distance':
        require(len(cal_ids)==1,'single-distance calibration requires exactly one calibration measurement')
        require('standard_uncertainty_mm_per_coordinate_unit' not in scale,'single-distance scale uncertainty is derived from calibration components, not supplied separately')
        mid=cal_ids[0]
        require(mid in by_id,'reconstructed calibration distance missing')
        require(ids[mid]['value'] is not None and reconstructed_distance(by_id[mid],geometry_unit) is not None,'calibration values missing')
        require(ids[mid]['qc_status']==by_id[mid]['qc_status']=='verified','calibration measurement/annotation must be verified')
    else:
        finite(scale.get('mm_per_coordinate_unit'),'external scale',positive=True)
        if scale.get('standard_uncertainty_mm_per_coordinate_unit') is not None:
            finite(scale['standard_uncertainty_mm_per_coordinate_unit'],'scale uncertainty')
    return ids,by_id


def resolve_scale(bundle,physical,reconstructed):
    rec=bundle['reconstruction']; scale=rec['scale']; unit=rec['coordinate_unit']
    warnings=[]
    if scale['method']=='single_calibration_distance':
        mid=scale['calibration_measurement_ids'][0]; p=physical[mid]; r=reconstructed[mid]
        p_mm=to_mm(p['value'],p['unit']); d=reconstructed_distance(r,unit); factor=p_mm/d
        u_p=standard_uncertainty(p.get('uncertainty'),UNIT_MM,'calibration physical')
        u_d=standard_uncertainty(r.get('uncertainty'),{unit:1},'calibration reconstruction')
        u_factor=None
        if u_p is not None and u_d is not None and rec['uncertainty_assumption']=='independent_components':
            u_factor=math.hypot(u_p/d,p_mm*u_d/d**2)
        if scale.get('mm_per_coordinate_unit') is not None:
            supplied=finite(scale['mm_per_coordinate_unit'],'declared calibration scale',positive=True)
            require(math.isclose(supplied,factor,rel_tol=1e-9),'declared scale differs from calibration-only calculation')
        warnings.append('Single-distance scale calibration does not establish shape accuracy or verify other directions.')
    else:
        factor=scale['mm_per_coordinate_unit'];u_factor=scale.get('standard_uncertainty_mm_per_coordinate_unit')
    factor=finite(factor,'resolved scale',positive=True)
    if unit in UNIT_MM:
        require(math.isclose(factor,UNIT_MM[unit],rel_tol=1e-9),'claimed metric coordinate unit inconsistent with calibration scale; declare arbitrary coordinates instead')
    if u_factor is None: warnings.append('Scale standard uncertainty unavailable; no assumption of zero uncertainty.')
    return factor,u_factor,warnings


def evaluate_dimensions(bundle, *, minimum_checks=2, require_uncertainty=True):
    """Unweighted independent-check label; retain partial diagnostics without inventing labels.

    Independence means exclusion from calibration, not statistical independence of
    shared endpoints, instrument bias, scale or repeated acquisitions. Uncertainty
    propagation is first-order and only supplied for an explicit independence
    assertion; aggregate label uncertainty is not inferred.
    """
    require(type(minimum_checks) is int and minimum_checks>=1,'minimum_checks must be a positive integer')
    require(type(require_uncertainty) is bool,'require_uncertainty must be boolean')
    physical,reconstructed=validate_bundle(bundle)
    factor,u_scale,warnings=resolve_scale(bundle,physical,reconstructed)
    rec=bundle['reconstruction']; unit=rec['coordinate_unit']; rows=[]; valid=[]
    checks=[m for m in physical.values() if m['role']=='check']
    for mid,p in physical.items():
        r=reconstructed.get(mid)
        row={'measurement_id':mid,'measurand_id':p['measurand_id'],'object_id':p['object_id'],'role':p['role'],
             'physical_mm':None if p.get('value') is None else to_mm(p['value'],p['unit']),
             'reconstructed_mm':None,'signed_residual_mm':None,'absolute_error_mm':None,
             'relative_error_percent':None,'signed_relative_error_percent':None,
             'physical_standard_uncertainty_mm':standard_uncertainty(p.get('uncertainty'),UNIT_MM,'physical'),
             'reconstructed_standard_uncertainty_mm':None,'residual_standard_uncertainty_mm':None,
             'relative_standard_uncertainty_percent':None,'uncertainty_complete':False,'reasons':[]}
        d=None if r is None else reconstructed_distance(r,unit)
        if d is not None: row['reconstructed_mm']=finite(d*factor,'scaled reconstructed dimension',positive=True)
        if p['role']=='calibration':
            row['status']='excluded_calibration';row['reasons'].append('Calibration distances never enter independent-check aggregates.')
            rows.append(row);continue
        if row['physical_mm'] is None or d is None:
            row['status']='missing';row['reasons'].append('Physical value or corresponding reconstructed annotation is missing.')
            rows.append(row);continue
        if p['qc_status']!='verified' or r['qc_status']!='verified' or rec['scale']['status']!='verified':
            row['status']='qc_not_verified';row['reasons'].append('Physical, annotation or scale QC not verified.')
            rows.append(row);continue
        measured=row['physical_mm'];estimated=row['reconstructed_mm'];residual=estimated-measured
        row.update(status='valid_check',signed_residual_mm=residual,absolute_error_mm=abs(residual),
                   relative_error_percent=100*abs(residual)/measured,signed_relative_error_percent=100*residual/measured)
        u_d=standard_uncertainty(r.get('uncertainty'),{unit:1},'reconstructed')
        u_p=row['physical_standard_uncertainty_mm'];u_r=None
        if u_d is not None and u_scale is not None and rec['uncertainty_assumption']=='independent_components':
            u_r=math.hypot(factor*u_d,d*u_scale)
            row['reconstructed_standard_uncertainty_mm']=u_r
        if u_r is not None and u_p is not None:
            row['residual_standard_uncertainty_mm']=math.hypot(u_r,u_p)
            row['relative_standard_uncertainty_percent']=100*math.hypot(u_r/measured,estimated*u_p/measured**2)
            row['uncertainty_complete']=True
        else:row['reasons'].append('Combined uncertainty unavailable: missing components or independence not established.')
        require(all(math.isfinite(row[k]) for k in ['signed_residual_mm','absolute_error_mm','relative_error_percent','signed_relative_error_percent']), 'derived dimensional residual overflow')
        valid.append(row);rows.append(row)
    n=len(valid);count_ok=n>=minimum_checks;complete=n==len(checks)
    partial={'mean_absolute_error_mm':sum(r['absolute_error_mm'] for r in valid)/n if n else None,
             'mean_absolute_relative_error_percent':sum(r['relative_error_percent'] for r in valid)/n if n else None,
             'relative_rmse_percent':math.sqrt(sum(r['signed_relative_error_percent']**2 for r in valid)/n) if n else None}
    blockers=[]
    if not count_ok:blockers.append('insufficient_independent_checks')
    if not complete:blockers.append('incomplete_or_unverified_check_panel')
    if require_uncertainty and (u_scale is None or any(not r['uncertainty_complete'] for r in valid)):
        blockers.append('uncertainty_incomplete')
    if rec['scale']['status']!='verified':blockers.append('scale_not_verified')
    eligible=not blockers
    warnings += ['Independent checks are independent of scale calibration, not necessarily mutually independent.',
        'Aggregate uncertainty/coverage interval is not estimated: shared scale, endpoints and instrument covariance need a fuller model.',
        'Numeric provenance/QC fields are researcher attestations, not authentication of physical measurements.']
    return {'schema_version':1,'kind':'dimensional_accuracy_report','data_origin':bundle['data_origin'],
        **{k:bundle[k] for k in ['object_id','session_id','subset_id','run_id']},
        'input_sha256':digest(bundle),'ground_truth_reference_id':bundle['ground_truth']['reference_id'],
        'geometry_sha256':rec['geometry_sha256'],'reference_type':'independent_physical_ground_truth',
        'measurement_scope':'annotated independent check distances, not full surface accuracy',
        'resolved_scale':{'mm_per_coordinate_unit':factor,'standard_uncertainty_mm_per_coordinate_unit':u_scale,
                          'calibration_measurement_ids':rec['scale']['calibration_measurement_ids']},
        'policy':{'minimum_checks':minimum_checks,'require_uncertainty':require_uncertainty,'require_complete_panel':True},
        'check_measurement_count':len(checks),'valid_independent_check_count':n,
        'uncertainty_complete_check_count':sum(r['uncertainty_complete'] for r in valid),
        'calibration_measurement_count':sum(p['role']=='calibration' for p in physical.values()),
        'per_measurement':rows,'partial_diagnostics':partial,
        'target_dimensional_relative_rmse_percent':partial['relative_rmse_percent'] if eligible else None,
        'mean_absolute_relative_error_percent':partial['mean_absolute_relative_error_percent'] if eligible else None,
        'mean_absolute_error_mm':partial['mean_absolute_error_mm'] if eligible else None,
        'target_eligible':eligible,'qc_status':'eligible' if eligible else 'blocked','blockers':blockers,
        'warnings':warnings}
