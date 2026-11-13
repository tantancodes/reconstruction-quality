"""Audit saved alignment metadata into a fresh directory; no network or photo reads."""
import argparse
from pathlib import Path

from bd3d_legacy.historical_alignment import require
from bd3d_legacy.historical_recovery import bounded_json
from bd3d_legacy.photo_provenance import check_photo_provenance, readiness_dimensions
from bd3d_legacy.project_metadata import inspect_project
from bd3d_quality.provenance import sha256_file, write_json


def audit(records_path,output,*,artifact_id='PAHMA-5-613',candidates_path=None,
          manifest_path=None,receipt_path=None,small_project=None):
    records=bounded_json(records_path)['records']
    matches=[r for r in records if r['artifact_id']==artifact_id]
    require(len(matches)==1,'select one identifiable report record; artifact selection ambiguous')
    candidates=bounded_json(candidates_path)['images'] if candidates_path else matches[0]['source_images']
    manifest=bounded_json(manifest_path) if manifest_path else None
    receipt=bounded_json(receipt_path) if receipt_path else None
    checked=check_photo_provenance(matches[0],candidates,manifest,receipt)
    project=None
    if small_project:
        p=Path(small_project)
        project=inspect_project(p/'PAHMA_5_1405.psx',p/'project.zip',p/'chunk.zip')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'photo-provenance-check.json',checked)
    write_json(output/'readiness.json',readiness_dimensions(records))
    if project is not None:write_json(output/'project-inspection.json',project)
    inputs={k:{'path':str(p),'sha256':sha256_file(Path(p))} if p else None for k,p in
            [('records',records_path),('candidates',candidates_path),('manifest',manifest_path),('receipt',receipt_path)]}
    write_json(output/'audit-sources.json',{'inputs':inputs,
        'checker_source_sha256':sha256_file(Path(__file__).parents[1]/'src/bd3d_quality/photo_provenance.py'),
        'scope':'Metadata only; existing dataset unchanged; no features extracted or readiness status promoted.'})
    write_json(output/'file-hashes.json',{p.name:sha256_file(p) for p in output.iterdir() if p.is_file()})
    return checked


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('records_path');p.add_argument('output')
    p.add_argument('--artifact-id',default='PAHMA-5-613')
    for name in ['candidates-path','manifest-path','receipt-path','small-project']:p.add_argument('--'+name)
    audit(**vars(p.parse_args()))
