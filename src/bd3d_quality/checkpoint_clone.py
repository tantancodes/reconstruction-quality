"""Copy a watchdog-stopped, hash-verified checkpoint; never unlock/alter the source."""
from pathlib import Path
import os
import shutil
import sys
from uuid import uuid4
from .provenance import read_json,write_json,now,sha256_file
from .manifests import require_valid
from .metashape_runner import project_fingerprints


def clone_checkpoint(source,destination,root):
    source=Path(source).resolve();destination=Path(destination).resolve()
    monitor=read_json(source.with_name(source.name+'-resources.json'))
    if monitor.get('status')!='stopped_or_failed' or monitor.get('returncode',0)>=0:
        raise ValueError('source must have a recorded watchdog termination')
    try:os.kill(monitor['pid'],0)
    except ProcessLookupError:pass
    else:raise ValueError('source PID still exists; cannot clone')
    state=read_json(source/'run.json');manifest=read_json(source/'manifest.json')
    require_valid(manifest,root)
    if state['status']!='checkpointed' or state['experiment_id']!=manifest['experiment_id']:
        raise ValueError('source must have a consistent saved checkpoint')
    if state['project_files']!=project_fingerprints(source):raise ValueError('source checkpoint changed')
    for a in state['artifacts']:
        if sha256_file(source/a['path'])!=a['sha256']:raise ValueError('artifact changed')
    destination.mkdir(parents=True,exist_ok=False)
    shutil.copy2(source/'project.psx',destination/'project.psx')
    shutil.copytree(source/'project.files',destination/'project.files',ignore=shutil.ignore_patterns('lock'))
    for a in state['artifacts']:shutil.copy2(source/a['path'],destination/a['path'])
    write_json(destination/'manifest.json',manifest)
    state['continuation_of']={'run_id':state['run_id'],'source_checkpoint_files':state['project_files'],
                              'continued_at':now(),'inherited_stages':[s['stage'] for s in state['stages']]}
    state['run_id']=str(uuid4())
    if project_fingerprints(destination)!=state['project_files']:raise ValueError('copied project mismatch')
    write_json(destination/'run.json',state)
    return state


if __name__=='__main__':clone_checkpoint(*sys.argv[1:])
