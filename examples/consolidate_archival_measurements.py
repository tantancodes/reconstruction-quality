"""Explicit input-hash joins of existing image features and audited archival geometry."""
from pathlib import Path
import csv,sys
import numpy as np
from bd3d_quality.provenance import read_json,write_json,sha256_file


def consolidate(audit_dir,geometry_dir,image_features,pair_features,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    features=read_json(image_features);matches=read_json(pair_features)
    lookup={(e['path'],e['sha256']):e for e in features['images']}
    match_inputs={e['path']:e['sha256'] for e in matches['images']}
    rows=[];sources=[]
    for path in sorted(Path(audit_dir).glob('ARF*.json')):
        record=read_json(path);images=record['manifest']['images']
        selected={e['filename'] for e in images}
        image_rows=[lookup[(e['filename'],e['sha256'])] for e in images]
        if any(match_inputs.get(e['filename'])!=e['sha256'] for e in images):raise ValueError('pair input hash mismatch')
        pairs=[p for p in matches['pairs'] if p['path_a'] in selected and p['path_b'] in selected]
        if len(pairs)!=len(images)*(len(images)-1)//2:raise ValueError('incomplete pair set')
        geometry_path=Path(geometry_dir)/path.name;geometry=read_json(geometry_path)
        if geometry['candidate']['sha256']!=record['source_mesh_sha256']:raise ValueError('geometry hash mismatch')
        r={'source':'archival','run_id':record['run_id'],'image_count':len(images),
           'selection_seed':record['manifest'].get('random_seed'),
           'mesh_vertices':record['export_validation']['vertex_count'],'mesh_faces':record['export_validation']['face_count'],
           'mesh_surface_area':record['export_validation']['surface_area'],'dense_points':None,
           'geometry_status':geometry['status'],'registration_fitness':geometry['registration']['fitness'],
           'scale_radius_ratio':geometry['scale_radius_ratio'],
           'reference_agreement_chamfer_mean_squared':geometry['metrics']['chamfer_mean_squared'] if geometry['metrics'] else None,
           'baseline_self_check':geometry['candidate']['sha256']==geometry['reference']['sha256'],
           'pair_ratio_matches_mean':float(np.mean([p['ratio_match_count_a_to_b'] for p in pairs]))}
        for key in ['brightness_mean','contrast_std','laplacian_variance','entropy_bits']:
            r['image_'+key+'_mean']=float(np.mean([e[key] for e in image_rows]))
        rows.append(r)
        sources.append({'audit_sha256':sha256_file(path),'geometry_sha256':sha256_file(geometry_path),
                        'run_id':record['run_id'],'warnings':record['warnings']})
    write_json(output/'observations.json',{'rows':rows,'sources':sources,
        'image_features_sha256':sha256_file(Path(image_features)),'pair_features_sha256':sha256_file(Path(pair_features)),
        'image_feature_settings':features['settings'],'pair_feature_settings':matches['settings'],
        'experimental_unit':'archival execution; all share one object and acquisition',
        'limitations':['Baseline self-distance must not be used as an independent quality target.',
            'Overlapping subsets are dependent. No ground-truth accuracy or external validation.',
            'Dense points unavailable. No newly completed dense/mesh pilot executions.',
            'Current image measurements joined explicitly by filename and original content hash.',
            'Historical final sparse measurements are not assumed to be initial-alignment measurements.']})
    with (output/'observations.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(output/'feature_dictionary.json',{'experimental_unit':'one archival run',
        'missing':'JSON null / CSV empty; not zero','reference_agreement_chamfer_mean_squared':'Symmetric sampled distance squared in unvalidated historical units; not physical accuracy.',
        'image_aggregates':'Means across hash-matched image features; settings and definitions retained in observations.json and docs/FEATURE_DICTIONARY.md.',
        'pair_ratio_matches_mean':'Mean of complete one-way ORB ratio-match pair set in canonical measurement order.',
        'dense_points':'Unavailable in audited legacy exports; missing.'})


if __name__=='__main__':consolidate(*sys.argv[1:])
