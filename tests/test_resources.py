from dataclasses import asdict
import copy
import pytest
from bd3d_quality.config import ResourceProtocol
from bd3d_quality.manifests import experiment_identity, require_valid
from bd3d_quality.metashape_runner import run_reconstruction, verify_seed, project_fingerprints
from bd3d_quality.provenance import write_json
from bd3d_quality.resource_monitor import limit_reason


def test_protocol_identity(dataset):
    root,m=dataset
    old=m['experiment_id']
    m['execution_protocol']=asdict(ResourceProtocol())
    m['experiment_id']=experiment_identity(m)
    assert m['experiment_id']!=old
    require_valid(m,root)
    m['execution_protocol']['max_neighbors']=0
    with pytest.raises(ValueError):require_valid(m,root)


@pytest.mark.parametrize('change,reason', [({'pressure':4},'critical_system_memory_pressure'),
    ({'swap_used_mb':300},'system_swap_growth_over_256_MiB'),
    ({'rss_bytes':4*1024**3},'process_RSS_limit')])
def test_stop_gates(change,reason):
    baseline={'pressure':1,'swap_used_mb':0,'rss_bytes':0}
    assert limit_reason({**baseline,**change},baseline,1,100)==reason


def test_time_and_disk_gates():
    s={'pressure':2,'swap_used_mb':25000,'rss_bytes':100}
    assert limit_reason(s,s,900,0)=='wall_time_limit'
    assert limit_reason(s,s,1,2_000_000_000)=='output_storage_limit'
    assert limit_reason(s,s,1,10) is None  # High stable historic swap is not growth.


def test_seed_reuse_and_raw_protocol(dataset,tmp_path,fake_api):
    root,m=dataset
    parent=tmp_path/'parent'
    state=run_reconstruction(m,root,parent,api=fake_api)
    (parent/'project.files').mkdir()
    (parent/'project.files/doc.xml').write_text('state')
    state['project_files']=project_fingerprints(parent)
    write_json(parent/'run.json',state,replace=True)
    fingerprints=project_fingerprints(parent)
    child=copy.deepcopy(m)
    child['configuration']['profile']='mesh'
    child['configuration']['depth_downscale']=16
    child['execution_protocol']=asdict(ResourceProtocol())
    child['experiment_id']=experiment_identity(child)
    fake_api.CustomFaceCount='CustomFaceCount'
    result=run_reconstruction(child,root,tmp_path/'child',api=fake_api,seed_checkpoint=parent)
    assert result['parent_checkpoint']['run_id']==state['run_id']
    assert result['stages'][0]['runtime_seconds']==0
    assert [s['stage'] for s in result['stages']]==['initial_alignment','depth_maps','dense_reconstruction','mesh_generation']
    assert result['stages'][2]['operation_seconds']['point_cloud']>=0
    assert project_fingerprints(parent)==fingerprints
    child['images'].reverse()
    with pytest.raises(ValueError,match='input provenance'): verify_seed(parent,child,root)


def test_deterministic_experiment_design(dataset):
    from bd3d_quality.experiment_design import subset_manifest
    root,m=dataset
    a=subset_manifest(m,2,'random',71)
    assert a==subset_manifest(m,2,'random',71)
    require_valid(a,root)
    assert len(a['images'])==2 and len({i['path'] for i in a['images']})==2
    assert a['selection']['parent_experiment_id']==m['experiment_id']
    with pytest.raises(ValueError):subset_manifest(m,4,'uniform')


def test_watchdog_collects_lightweight_child(tmp_path,monkeypatch):
    import sys
    import bd3d_quality.resource_monitor as monitor
    monkeypatch.setattr(monitor,'system_sample',lambda pid=None: {'at':'test','pressure':1,'swap_used_mb':0,'rss_bytes':0,'cpu_percent':0})
    monkeypatch.setattr(monitor.subprocess,'check_output',lambda *a,**k:'')
    result=monitor.run_monitored([sys.executable,'-c','print("lightweight child")'],tmp_path/'run',seconds=2)
    assert result['returncode']==0
    assert (tmp_path/'run-resources.json').exists()
    assert not (tmp_path/'.reconstruction-watchdog.lock').exists()


def test_signal_race_is_safe(monkeypatch):
    import signal
    import bd3d_quality.resource_monitor as monitor
    def exited(*args):raise ProcessLookupError()
    monkeypatch.setattr(monitor.os,'killpg',exited)
    monitor.signal_group(123,signal.SIGTERM)
