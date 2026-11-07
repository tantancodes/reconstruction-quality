"""Synthetic scalar/landmark fixtures only; no actual reconstruction or physical data."""
import copy,math,json
from pathlib import Path
import pytest
from bd3d_quality.dimensional_accuracy import evaluate_dimensions,to_mm
from bd3d_quality.cli import main
from bd3d_quality.provenance import read_json


@pytest.fixture
def bundle():
    return json.loads(Path('examples/ground_truth/synthetic-box-dimensions.json').read_text())


def test_three_checks_exclude_scale(bundle):
    r=evaluate_dimensions(bundle)
    assert r['valid_independent_check_count']==3 and r['calibration_measurement_count']==1
    assert r['target_dimensional_relative_rmse_percent']==pytest.approx(2)
    assert r['mean_absolute_relative_error_percent']==pytest.approx(2)
    assert r['mean_absolute_error_mm']==pytest.approx((2+1+.5)/3)
    assert r['per_measurement'][0]['status']=='excluded_calibration'
    assert r['per_measurement'][0]['absolute_error_mm'] is None
    assert r['per_measurement'][2]['signed_residual_mm']==-1


@pytest.mark.parametrize('unit,value,expected',[('mm',12,12),('cm',1.2,12),('m',.012,12)])
def test_units(unit,value,expected):assert to_mm(value,unit)==pytest.approx(expected)


@pytest.mark.parametrize('value,unit',[(0,'mm'),(-1,'mm'),(float('nan'),'mm'),(float('inf'),'cm'),(1,'inch'),(True,'mm'),(1e308,'m')])
def test_invalid_units_values(value,unit):
    with pytest.raises(ValueError):to_mm(value,unit)


def test_relative_rmse_is_not_mean_absolute_error(bundle):
    bundle['reconstruction']['measurements'][1]['value']=50 # 0%
    bundle['reconstruction']['measurements'][2]['value']=26 # 4%
    bundle['reconstruction']['measurements'][3]['value']=12.5 # 0%
    r=evaluate_dimensions(bundle)
    assert r['target_dimensional_relative_rmse_percent']==pytest.approx(4/math.sqrt(3))
    assert r['mean_absolute_relative_error_percent']==pytest.approx(4/3)


def test_missing_check_masks_primary(bundle):
    bundle['reconstruction']['measurements'].pop()
    r=evaluate_dimensions(bundle)
    assert r['valid_independent_check_count']==2
    assert r['target_dimensional_relative_rmse_percent'] is None
    assert 'incomplete_or_unverified_check_panel' in r['blockers']
    assert r['partial_diagnostics']['relative_rmse_percent']==pytest.approx(2)


def test_insufficient_checks(bundle):
    r=evaluate_dimensions(bundle,minimum_checks=4)
    assert not r['target_eligible'] and 'insufficient_independent_checks' in r['blockers']


def test_circular_scale_rejected(bundle):
    bundle['reconstruction']['scale']['calibration_measurement_ids']=['length']
    with pytest.raises(ValueError,match='independent check'):evaluate_dimensions(bundle)


def test_endpoint_alias_rejected(bundle):
    original=copy.deepcopy(bundle['ground_truth']['measurements'][1])
    original.update(measurement_id='calibration-alias',measurand_id='another-name',role='calibration')
    original['definition']['landmarks'].reverse()
    bundle['ground_truth']['measurements'].append(original)
    with pytest.raises(ValueError,match='endpoint pair'):evaluate_dimensions(bundle)


def test_expanded_uncertainty_and_propagation(bundle):
    p=bundle['ground_truth']['measurements'][1]
    p['uncertainty'].update(value=.2,kind='expanded',coverage_factor=2)
    r=evaluate_dimensions(bundle);row=r['per_measurement'][1]
    us=math.hypot(.1/50,100*.01/50**2)
    ur=math.hypot(2*.01,51*us)
    assert row['physical_standard_uncertainty_mm']==pytest.approx(.1)
    assert r['resolved_scale']['standard_uncertainty_mm_per_coordinate_unit']==pytest.approx(us)
    assert row['residual_standard_uncertainty_mm']==pytest.approx(math.hypot(ur,.1))
    assert row['relative_standard_uncertainty_percent']==pytest.approx(100*math.hypot(ur/100,102*.1/100**2))


@pytest.mark.parametrize('mutation',['missing','negative','unknown_covariance','no_k','wrong_unit'])
def test_uncertainty_explicit(bundle,mutation):
    u=bundle['ground_truth']['measurements'][1]['uncertainty']
    if mutation=='missing':bundle['ground_truth']['measurements'][1]['uncertainty']=None
    elif mutation=='negative':u['value']=-1
    elif mutation=='unknown_covariance':bundle['reconstruction']['uncertainty_assumption']='not_established'
    elif mutation=='no_k':u['kind']='expanded'
    elif mutation=='wrong_unit':u['unit']='pixels'
    if mutation in {'missing','unknown_covariance'}:
        r=evaluate_dimensions(bundle)
        assert r['target_dimensional_relative_rmse_percent'] is None and 'uncertainty_incomplete' in r['blockers']
    else:
        with pytest.raises(ValueError):evaluate_dimensions(bundle)


def test_annotated_points(bundle):
    m=bundle['reconstruction']['measurements'][1];m.pop('value')
    m.update(method='annotated_landmarks',points=[[0,0,0],[51,0,0]])
    assert evaluate_dimensions(bundle)['per_measurement'][1]['reconstructed_mm']==102
    m['points'][1]=[0,0,0]
    with pytest.raises(ValueError,match='coincide'):evaluate_dimensions(bundle)


@pytest.mark.parametrize('kind',['cylinder_diameter','cylinder_height','landmark_distance'])
def test_measurand_types(bundle,kind):
    bundle['ground_truth']['measurements'][1]['definition']['kind']=kind
    assert evaluate_dimensions(bundle)['target_eligible']


@pytest.mark.parametrize('mutation',['object','run','provenance','scale_unit','landmarks','pending_scale'])
def test_identity_provenance_and_scale(bundle,mutation):
    if mutation=='object':bundle['ground_truth']['object_id']='other'
    elif mutation=='run':bundle['reconstruction']['run_id']='other'
    elif mutation=='provenance':bundle['ground_truth']['measurements'][1]['provenance']['date']='yesterday'
    elif mutation=='landmarks':bundle['reconstruction']['measurements'][1]['landmarks']=['unknown','landmarks']
    elif mutation=='pending_scale':bundle['reconstruction']['scale']['status']='pending'
    else:
        bundle['reconstruction']['coordinate_unit']='mm'
        for m in bundle['reconstruction']['measurements']:m['unit']='mm';m['uncertainty']['unit']='mm'
    if mutation=='pending_scale':assert not evaluate_dimensions(bundle)['target_eligible']
    else:
        with pytest.raises(ValueError):evaluate_dimensions(bundle)


def test_cli_example_and_exclusive_output(tmp_path):
    path=tmp_path/'report.json';args=['dimensions','evaluate','examples/ground_truth/synthetic-box-dimensions.json','--output',str(path)]
    assert main(args)==0
    assert read_json(path)['target_dimensional_relative_rmse_percent']==pytest.approx(2)
    assert main(args)==2
