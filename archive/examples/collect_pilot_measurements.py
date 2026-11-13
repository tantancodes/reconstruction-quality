"""Collect actual image/match, export-integrity and geometric outcomes, preserving nulls."""
from pathlib import Path
import subprocess,sys
from bd3d_quality.provenance import read_json,write_json
from bd3d_legacy.image_features import features_for_manifest
from bd3d_legacy.local_match_features import match_manifest
from bd3d_legacy.geometry_quality import compare_geometry
from bd3d_legacy.reporting import summarize,build_report


def collect(baseline,pilot,root,output,open3d_interpreter):
    baseline=Path(baseline).resolve();pilot=Path(pilot).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    candidates=[baseline]+sorted(p for p in pilot.iterdir() if p.is_dir() and (p/'run.json').exists())
    summaries=[];outcomes=[]
    for run_path in candidates:
        m=read_json(run_path/'manifest.json');run=read_json(run_path/'run.json')
        prefix=run_path.name
        features=features_for_manifest(m,root)
        matches=match_manifest(m,root)
        write_json(output/(prefix+'-features.json'),features)
        write_json(output/(prefix+'-matches.json'),matches)
        geometry=None
        if run['status']=='complete':
            subprocess.run([open3d_interpreter,'-B','archive/examples/validate_exports.py',str(run_path),
                str(output/(prefix+'-exports.json'))],check=True)
            geometry=compare_geometry(run_path/'raw-mesh.obj',baseline/'raw-mesh.obj',
                reference_id='arf-dog-derived-648-baseline',units='arbitrary_shared_assumed_QC_required',sample_count=3000)
            geometry['baseline_self_check']=run_path==baseline
            geometry['internal_diagnostics']=read_json(output/(prefix+'-exports.json'))
            write_json(output/(prefix+'-geometry.json'),geometry)
        summary=summarize(m,features=features,matches=matches,run=run,geometry=geometry)
        path=output/(prefix+'-summary.json');write_json(path,summary);summaries.append(path)
        outcomes.append({'name':prefix,'run_status':run['status'],'geometry_status':geometry['status'] if geometry else None,
                         'geometry_flags':geometry['flags'] if geometry else None,'metrics':run['stages'][-1]['metrics'] if run['stages'] else None})
    build_report(summaries,output/'table')
    write_json(output/'outcomes.json',{'conditions':outcomes,
        'caution':'Baseline self-distance is a numerical check; invalid scale/registration comparisons have missing distance targets. One object/session only.'})


if __name__=='__main__':collect(*sys.argv[1:])
