"""Execute at most three sequential real subset runs after a successful baseline."""
from pathlib import Path
import sys,time
from bd3d_quality.provenance import read_json,write_json
from bd3d_quality.resource_monitor import run_monitored,directory_bytes,system_sample


def pilot(design_path, baseline, root, interpreter, output):
    plan=read_json(design_path);baseline=Path(baseline);output=Path(output)
    run=read_json(baseline/'run.json')
    measured=read_json(baseline.with_name(baseline.name+'-resources.json'))
    if run['status']!='complete' or measured['status']!='complete':
        raise ValueError('baseline must finish all reconstruction stages within watchdog limits')
    output.mkdir(parents=True,exist_ok=False)
    selected=[c for c in plan['conditions'] if c['name'] in plan['pilot_names']]
    inherited_alignment = sum(s['runtime_seconds'] for s in run['stages'] if s['stage']=='initial_alignment') if run.get('continuation_of') else 0
    baseline_full_cost = measured['elapsed_seconds'] + inherited_alignment
    estimates=[{'name':c['name'],'estimated_seconds':baseline_full_cost*(c['manifest']['image_count']/50),
                'estimated_bytes':directory_bytes(baseline)*c['manifest']['image_count']/50} for c in selected]
    budget=plan['pilot_budget']
    projected=sum(e['estimated_seconds'] for e in estimates)
    write_json(output/'budget-before-execution.json',{'baseline_elapsed_seconds':measured['elapsed_seconds'],
        'baseline_output_bytes':directory_bytes(baseline), 'baseline_projected_full_execution_seconds':baseline_full_cost,
        'estimates':estimates,
        'basis':'linear image-count scaling from measured baseline; uncertain, watchdog gates still mandatory',
        'limits':budget})
    if projected>budget['total_additional_seconds_max']:
        write_json(output/'outcome.json',{'status':'budget_projection_exceeded','runs':[]});return
    start=time.monotonic();runs=[]
    for c in selected[:budget['additional_runs_max']]:
        remaining=budget['total_additional_seconds_max']-(time.monotonic()-start)
        if remaining<30 or directory_bytes(output)>=budget['total_additional_output_bytes_max']:break
        if system_sample()['pressure']>=4:break
        manifest=output/(c['name']+'-manifest.json')
        write_json(manifest,c['manifest'])
        destination=output/c['name']
        command=[interpreter,'-B','-m','bd3d_quality.cli','reconstruct',str(manifest),
                 '--root',str(Path(root).resolve()),'--output',str(destination)]
        result=run_monitored(command,destination,seconds=min(300,int(remaining)))
        runs.append({'name':c['name'],'status':result['status'],'elapsed_seconds':result['elapsed_seconds'],
                     'stop_reason':result.get('stop_reason')})
        if result['status']!='complete':break
    write_json(output/'outcome.json',{'status':'complete' if len(runs)==len(selected) and all(r['status']=='complete' for r in runs) else 'bounded_stop',
        'runs':runs,'elapsed_seconds':time.monotonic()-start,
        'unexecuted':[c['name'] for c in selected if c['name'] not in {r['name'] for r in runs}]})


if __name__=='__main__':pilot(*sys.argv[1:])
