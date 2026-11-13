"""Read-only audit of three existing real meshes; never claim a new reconstruction."""
from pathlib import Path
import csv,os,sys
import numpy as np
import open3d as o3d
from bd3d_quality.provenance import read_json,write_json,sha256_file
from bd3d_quality.manifests import require_valid
from bd3d_legacy.geometry_quality import compare_geometry


def audit(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    root=Path(os.environ['BD3D_ARCHIVE_ROOT'])  # historical BD3D_Test_Output folder
    baseline=read_json('results/arf-dog-50-manifest.json')
    require_valid(baseline,os.environ['BD3D_DOG_PHOTOS']  # ARF dog photo folder)
    original_hashes={e['path']:e['sha256'] for e in baseline['images']}
    names=['ARF_Dog_FAST_DENSE_Geometry_50_v1','ARF_Dog_FD_Random_25_Seed_20260823_v1','ARF_Dog_FD_Random_25_Seed_20260824_v1']
    reference=next((root/names[0]/'analysis_geometry').glob('*.obj'))
    rows=[]
    for name in names:
        folder=root/name
        mesh_path=next((folder/'analysis_geometry').glob('*.obj'))
        report_path=next((folder/'reports').glob('*.json'))
        report=read_json(report_path)
        manifest_path=Path(report['experiment']['manifest_path'])
        manifest=read_json(manifest_path)
        if sha256_file(manifest_path)!=report['experiment']['manifest_sha256']:raise ValueError('historical manifest hash mismatch')
        if any(original_hashes.get(e['filename'])!=e['sha256'] for e in manifest['images']):raise ValueError('historical input hash mismatch')
        mesh=o3d.io.read_triangle_mesh(str(mesh_path))
        v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles)
        if not len(f) or not np.isfinite(v).all():raise ValueError('invalid archival mesh')
        diagnostics={'Open3D':o3d.__version__,'vertex_count':len(v),'face_count':len(f),
            'surface_area':float(mesh.get_surface_area()),'bounds_min':v.min(axis=0).tolist(),'bounds_max':v.max(axis=0).tolist(),
            'watertight':mesh.is_watertight(),'edge_manifold_with_boundary':mesh.is_edge_manifold(allow_boundary_edges=True)}
        del mesh,v,f
        geometry=compare_geometry(mesh_path,reference,reference_id=names[0],units='historical_shared_export_units_unvalidated',sample_count=3000)
        timing=report['timing_seconds']
        warnings=['Archival observation, not a new reconstruction or validation of the new runner.',
            'Manifest hash and current input hashes verified; historical image bytes at execution are not independently recoverable.',
            'Legacy refinement/orientation and High face-count settings differ from the new undecimated protocol.',
            'No separately exported dense cloud was found for this run; dense point count remains missing.']
        if (timing.get('total_runtime') or 0)<sum(timing.get(k) or 0 for k in ['photo_matching_alignment','depth_maps','model_building']):
            warnings.append('Historical total runtime is inconsistent with stage durations; retain source values without repair.')
        record={'kind':'archival_experiment_audit','run_id':report['run_id'],'source_report_sha256':sha256_file(report_path),
            'source_manifest_sha256':sha256_file(manifest_path),'source_mesh_sha256':sha256_file(mesh_path),
            'manifest':manifest,'source_report':report,'export_validation':diagnostics,'geometry':geometry,'warnings':warnings}
        write_json(output/(name+'.json'),record)
        rows.append({'source':'archival','run_id':report['run_id'],'image_count':len(manifest['images']),
            'random_seed':manifest.get('random_seed'),'mesh_vertices':diagnostics['vertex_count'],'mesh_faces':diagnostics['face_count'],
            'mesh_surface_area':diagnostics['surface_area'],'dense_points':None,'geometry_status':geometry['status'],
            'scale_radius_ratio':geometry['scale_radius_ratio'],'registration_fitness':geometry['registration']['fitness'],
            'chamfer_mean_squared':geometry['metrics']['chamfer_mean_squared'] if geometry['metrics'] else None,
            'baseline_self_check':name==names[0]})
    with (output/'archival_observations.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(output/'index.json',{'rows':rows,'experimental_unit':'archival run, one shared object/session; overlapping subsets',
        'prohibited_interpretation':'Not independent ground truth, not new pilot executions, not original-resolution validation of the new runner.',
        'frameworks_imported':[m for m in ['torch','tensorflow','sklearn','xgboost'] if m in sys.modules]})


if __name__=='__main__':audit(sys.argv[1])
