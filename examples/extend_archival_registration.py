"""Uniform bounded iteration-budget sensitivity check; preserves previous QC results."""
from pathlib import Path
import csv,os,sys
from bd3d_quality.geometry_quality import compare_geometry
from bd3d_quality.provenance import write_json


def evaluate(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    root=Path(os.environ['BD3D_ARCHIVE_ROOT'])  # historical BD3D_Test_Output folder
    names=['ARF_Dog_FAST_DENSE_Geometry_50_v1','ARF_Dog_FD_Random_25_Seed_20260823_v1','ARF_Dog_FD_Random_25_Seed_20260824_v1']
    reference=next((root/names[0]/'analysis_geometry').glob('*.obj'))
    rows=[]
    for name in names:
        mesh=next((root/name/'analysis_geometry').glob('*.obj'))
        result=compare_geometry(mesh,reference,reference_id=names[0],units='historical_shared_export_units_unvalidated',
            sample_count=3000,max_iterations=200)
        write_json(output/(name+'.json'),result)
        rows.append({'name':name,'geometry_status':result['status'],'iterations':result['registration']['iterations'],
            'fitness':result['registration']['fitness'],'scale_radius_ratio':result['scale_radius_ratio'],
            'chamfer_mean_squared':result['metrics']['chamfer_mean_squared'] if result['metrics'] else None,
            'candidate_coverage_fraction':result['metrics']['candidate_coverage_fraction'] if result['metrics'] else None,
            'reference_completeness_fraction':result['metrics']['reference_completeness_fraction'] if result['metrics'] else None,
            'baseline_self_check':name==names[0]})
    write_json(output/'index.json',{'rows':rows,'interpretation':'Archival reference-relative agreement, not validated accuracy.',
        'sensitivity_check':'Same 3000 samples/seed/QC settings as initial audit; iteration budget increased uniformly from 60 to 200.'})
    with (output/'agreement.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__':evaluate(sys.argv[1])
