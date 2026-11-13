"""Lightweight archival tabulation: hash joins, target masks, no model/library training."""
from __future__ import annotations
import argparse
import csv
import itertools
import math
import os
from pathlib import Path
import numpy as np
from bd3d_quality.provenance import read_json, write_json, sha256_file, digest

PRE_FEATURES = ['input_image_count', 'image_brightness_mean', 'image_contrast_std_mean',
                'image_laplacian_variance_mean', 'image_entropy_bits_mean',
                'image_orb_keypoint_count_mean', 'pair_ratio_match_count_mean']
POST_FEATURES = ['post_refinement_alignment_fraction', 'post_refinement_valid_tie_point_count',
                 'post_refinement_reprojection_rmse_px', 'post_refinement_observations_per_valid_tie_point']
TARGETS = ['target_reference_chamfer_mse', 'target_reference_chamfer_rmse_normalized',
           'target_reference_completeness_fraction', 'target_candidate_coverage_fraction']


def expect(value, message):
    if not value:
        raise ValueError(message)


def equal_number(a, b):
    return a == b if a is None or b is None else math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12)


def verify_file(path, expected):
    expect(sha256_file(Path(path)) == expected, f'provenance hash mismatch: {path}')


def numeric(value):
    return float(value) if value not in (None, '') else None


def field_dictionary(rows):
    pre = {
        'input_image_count': ('images', 'Number of distinct validated selected image records'),
        'image_brightness_mean': ('intensity 0..255', 'Mean of cached per-image Pillow grayscale means'),
        'image_contrast_std_mean': ('intensity', 'Mean of cached per-image population grayscale standard deviations'),
        'image_laplacian_variance_mean': ('intensity squared', 'Mean cached interior four-neighbor Laplacian variance; resolution-dependent'),
        'image_entropy_bits_mean': ('bits', 'Mean cached 256-bin grayscale Shannon entropy'),
        'image_orb_keypoint_count_mean': ('keypoints', 'Mean cached ORB count; cap 5000 at max dimension 1600'),
        'pair_ratio_match_count_mean': ('descriptor matches', 'Mean cached one-way ORB/Hamming k=2 ratio matches (ratio .75), complete unordered pair set in original measurement direction; not verified overlap')}
    target = {
        TARGETS[0]: ('historical export units squared', 'Half the sum of directional nearest-sample mean squared distances; lower agreement error is better'),
        TARGETS[1]: ('dimensionless', 'sqrt(symmetric squared Chamfer) / reference bounding-box diagonal; primary recommended reference-agreement target'),
        TARGETS[2]: ('fraction', 'Reference samples within 1% of reference diagonal of candidate samples; higher is more complete relative to reference'),
        TARGETS[3]: ('fraction', 'Candidate samples within 1% of reference diagonal of reference samples; higher proximity/coverage is better')}
    post = {
        POST_FEATURES[0]: ('fraction', 'Aligned / input cameras at historical post_error_reduction_pre_dense snapshot'),
        POST_FEATURES[1]: ('tie points', 'Valid tie-point count at that historical snapshot'),
        POST_FEATURES[2]: ('pixels', 'Historical RMS reprojection residual at that snapshot; internal consistency, not physical accuracy'),
        POST_FEATURES[3]: ('observations / tie point', 'Valid tie-point observations / valid tie-point count at that snapshot')}
    diagnostics = {
        'diagnostic_mesh_vertex_count': ('vertices', 'Audited Open3D vertex count, matched to source report; not a predictor'),
        'diagnostic_mesh_face_count': ('triangles', 'Audited Open3D face count, matched to source report; not a quality label'),
        'diagnostic_mesh_surface_area': ('historical export units squared', 'Audited triangle surface area; no physical calibration'),
        'diagnostic_registration_fitness': ('fraction', 'Accepted nearest-sample registration correspondence fraction; QC only'),
        'diagnostic_registration_inlier_rmse': ('historical export units', 'RMS residual of accepted ICP inliers only, not all-surface error'),
        'diagnostic_scale_radius_ratio': ('dimensionless', 'Candidate/reference centered surface-sample RMS radius; scale/coverage QC only'),
        'raw_self_or_reference_chamfer_mse': ('historical export units squared', 'Unmasked source measurement, including baseline self-check; excluded from features and training targets')}
    fields = {}
    for name in rows[0]:
        if name in pre:
            units, definition = pre[name]; role = 'input_only_predictor'
        elif name in target:
            units, definition = target[name]; role = 'reference_agreement_target'
        elif name in post:
            units, definition = post[name]; role = 'optional_post_refinement_predictor'
        elif name == 'target_ground_truth_rmse':
            units, definition, role = 'unknown physical units', 'Unavailable: no independently validated ground truth or scale calibration', 'unavailable_target'
        else:
            units, definition, role = 'not applicable', 'Provenance/group/QC metadata or output diagnostic; excluded from default predictor arrays', 'excluded_metadata_or_diagnostic'
        if name in diagnostics: units, definition = diagnostics[name]
        if name == 'diagnostic_dense_point_count': definition = 'No separate dense-cloud export/count in the audited source records; unavailable, not zero'
        if name == 'registration_visual_qc_status': definition = 'Not performed in this tabulation; numeric registration QC does not establish visually correct alignment'
        fields[name] = {'role': role, 'units': units, 'definition': definition,
                        'missing_count': sum(r[name] is None for r in rows),
                        'dtype': 'boolean' if isinstance(rows[0][name], bool) else 'number_or_null' if name in pre or name in post or name in target or name.startswith('diagnostic_') or name == 'target_ground_truth_rmse' else 'metadata',
                        'missing_semantics': 'JSON null / CSV empty / numeric tensor NaN; never imputed here'}
    return {'version': 1, 'fields': fields,
            'tensor_contract': {'X_pre': PRE_FEATURES, 'X_post_refinement': POST_FEATURES,
                                'y_reference': TARGETS, 'primary_target': TARGETS[1],
                                'missing': 'NaN with boolean finite-value masks; apply selected target mask before any future loss',
                                'transforms': 'No scaling, standardization, imputation, split, or training performed'},
            'preprocessing_reference': 'docs/FEATURE_DICTIONARY.md; exact settings/signatures/software in provenance.json',
            'target_cautions': ['Baseline self-comparison targets are all masked, not presented as zero-error training labels.',
                'MSE and normalized RMSE are deterministic transforms of one error measurement, not independent outcomes.',
                'Only 3000 sampled points per mesh; sampling noise and local ICP bias are unquantified.',
                'No verified physical-accuracy label is available.']}


def write_csv(path, rows):
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def prepare(results, historical_root, output):
    """Read existing records; writes only to a new, exclusive dataset directory.

    Rehashes three small mesh files and metadata, not original photographs. Original
    byte hashes were verified earlier; this report makes that temporal limit explicit.
    """
    results, historical_root, output = map(Path, (results, historical_root, output))
    expect(not output.exists(), 'dataset output must be a new directory')
    source_dir = results / 'archival-pre-ml-observations'
    source = read_json(source_dir / 'observations.json')
    with (source_dir / 'observations.csv').open(newline='') as f: csv_rows = list(csv.DictReader(f))
    expect(len(source['rows']) == len(csv_rows) == 3, 'expected exactly three archival rows')
    expect(len({r['run_id'] for r in source['rows']}) == 3, 'duplicate source run')
    expect({r['run_id'] for r in csv_rows} == {r['run_id'] for r in source['rows']}, 'CSV run set differs')
    for r in source['rows']:
        c = next(c for c in csv_rows if c['run_id'] == r['run_id'])
        expect(set(c) == set(r), 'CSV/JSON columns differ')
        for k, value in r.items():
            expected = '' if value is None else str(value)
            expect(c[k] == expected, f'CSV/JSON value mismatch: {r["run_id"]}/{k}')
    fp = results / 'arf-dog-50-features.json'; mp = results / 'arf-dog-50-matches.json'
    verify_file(fp, source['image_features_sha256']); verify_file(mp, source['pair_features_sha256'])
    features, matches = read_json(fp), read_json(mp)
    for record in (features, matches):
        expect(record['settings_signature'] == digest(record['settings']), 'feature settings signature differs')
    inputs = {(e['path'], e['sha256']): e for e in features['images']}
    expect(len(inputs) == len(features['images']) == 50, 'duplicate or missing cached image features')
    match_inputs = {e['path']: e for e in matches['images']}
    expect(len(match_inputs) == len(matches['images']) == 50, 'duplicate or missing cached ORB images')
    pairs = {(e['path_a'], e['path_b']): e for e in matches['pairs']}
    expect(len(pairs) == len(matches['pairs']) == 1225, 'duplicate or missing cached pair features')
    expect(set(pairs) == set(itertools.combinations(match_inputs, 2)), 'cached pairs not complete in measurement order')
    audits = {read_json(p)['run_id']: (p, read_json(p)) for p in (results/'archival-pilot-audit-v2').glob('ARF*.json')}
    rows, lineage, image_sets, checked_files = [], [], {}, []
    signatures = set(); reference_hashes = set(); settings = set()
    for old in source['rows']:
        rid = old['run_id']; audit_path, audit = audits[rid]
        src = next(s for s in source['sources'] if s['run_id'] == rid)
        verify_file(audit_path, src['audit_sha256'])
        geom_path = results/'archival-registration-200'/audit_path.name
        verify_file(geom_path, src['geometry_sha256']); geom = read_json(geom_path)
        manifest, report = audit['manifest'], audit['source_report']
        verify_file(report['experiment']['manifest_path'], audit['source_manifest_sha256'])
        verify_file(report['output_status']['report_json_path'], audit['source_report_sha256'])
        expect(read_json(report['experiment']['manifest_path']) == manifest, 'embedded and external manifest differ')
        expect(read_json(report['output_status']['report_json_path']) == report, 'embedded and external report differ')
        mesh = historical_root/audit_path.stem/'analysis_geometry'/geom['candidate']['filename']
        verify_file(mesh, audit['source_mesh_sha256'])
        expect(geom['candidate']['sha256'] == audit['source_mesh_sha256'], 'candidate geometry hash differs')
        n = len(manifest['images'])
        expect(n == old['image_count'] == manifest['images_used_count'] == report['run']['input_image_count'] == report['experiment']['images_used_count'], 'input count differs')
        selected = [(e['filename'], e['sha256']) for e in manifest['images']]
        expect(len(set(selected)) == n, 'duplicate selected image')
        expect(set(selected).issubset(inputs), 'source input is not in hash-matched feature cache')
        for name, sha in selected: expect(match_inputs[name]['sha256'] == sha, 'ORB image hash differs')
        image_sets[rid] = {sha for _, sha in selected}
        signatures.add(digest(report['run']['effective_settings'])); reference_hashes.add(geom['reference']['sha256'])
        expect(geom['settings_signature'] == digest(geom['settings']), 'geometry settings signature differs')
        settings.add(geom['settings_signature'])
        expect(geom['status'] == 'complete' and not geom['flags'], 'source numeric geometry QC failed')
        for k, diagnostic, metric in [('mesh_vertices', 'vertex_count', 'model_vertex_count'), ('mesh_faces', 'face_count', 'model_face_count')]:
            expect(old[k] == audit['export_validation'][diagnostic] == geom['candidate'][diagnostic] == report['dense_model'][metric], 'mesh count differs')
        expect(equal_number(old['mesh_surface_area'], audit['export_validation']['surface_area']), 'mesh surface area differs')
        expect(old['dense_points'] is None, 'unexpected dense count; review provenance')
        expect(report['output_status']['success'] is True and report['dense_model']['model_succeeded'] is True, 'historical run failed')
        self_check = geom['candidate']['sha256'] == geom['reference']['sha256']
        expect(self_check == old['baseline_self_check'], 'self-reference flag differs')
        for a,b in [('registration_fitness', geom['registration']['fitness']), ('scale_radius_ratio', geom['scale_radius_ratio']),
                    ('reference_agreement_chamfer_mean_squared', geom['metrics']['chamfer_mean_squared'])]:
            expect(equal_number(old[a], b), f'archival metric differs: {a}')
        ordered_names = [name for name in match_inputs if any(name == x[0] for x in selected)]
        counts = [pairs[p]['ratio_match_count_a_to_b'] for p in itertools.combinations(ordered_names, 2)]
        means = {key: float(np.mean([inputs[i][key] for i in selected])) for key in ['brightness_mean','contrast_std','laplacian_variance','entropy_bits']}
        for key,value in means.items(): expect(equal_number(value, old['image_'+key+'_mean']), 'cached image mean differs')
        expect(equal_number(float(np.mean(counts)), old['pair_ratio_matches_mean']), 'pair aggregate differs')
        a = report['alignment']
        expect(a['snapshot_stage'] == 'post_error_reduction_pre_dense', 'unexpected sparse measurement stage')
        r = {'run_id':rid,'source':'archival','object_group_id':'arf-dog','source_pool_group_id':'arf-dog/shared-source-pool-50',
             'capture_session_id':None,'historical_experiment_id':report['experiment']['experiment_id'],
             'reference_mesh_sha256':geom['reference']['sha256'],'candidate_mesh_sha256':geom['candidate']['sha256'],
             'baseline_self_check':self_check,'reference_target_eligible':not self_check,
             'registration_visual_qc_status':'not_performed','physical_accuracy_validated':False,
             'input_image_count':n,**dict(zip(PRE_FEATURES[1:5], [means[k] for k in ['brightness_mean','contrast_std','laplacian_variance','entropy_bits']])),
             'image_orb_keypoint_count_mean':float(np.mean([match_inputs[name]['orb_keypoint_count'] for name,_ in selected])),
             'pair_ratio_match_count_mean':float(np.mean(counts)),
             POST_FEATURES[0]:a['alignment_fraction'],POST_FEATURES[1]:a['valid_tie_point_count'],
             POST_FEATURES[2]:a['reprojection_error_px']['rmse'],
             POST_FEATURES[3]:a['valid_tie_point_observation_count']/a['valid_tie_point_count'] if a['valid_tie_point_count'] else None,
             'diagnostic_mesh_vertex_count':old['mesh_vertices'],'diagnostic_mesh_face_count':old['mesh_faces'],
             'diagnostic_mesh_surface_area':old['mesh_surface_area'],'diagnostic_dense_point_count':None,
             'diagnostic_registration_fitness':geom['registration']['fitness'],
             'diagnostic_registration_inlier_rmse':geom['registration']['inlier_rmse'],
             'diagnostic_scale_radius_ratio':geom['scale_radius_ratio'],
             'raw_self_or_reference_chamfer_mse':geom['metrics']['chamfer_mean_squared'],
             'target_ground_truth_rmse':None}
        mse = geom['metrics']['chamfer_mean_squared']; diagonal = geom['metrics']['reference_diagonal']
        labels = [mse, math.sqrt(mse)/diagonal, geom['metrics']['reference_completeness_fraction'], geom['metrics']['candidate_coverage_fraction']]
        r.update(dict(zip(TARGETS, [None]*len(TARGETS) if self_check else labels)))
        for key in PRE_FEATURES + POST_FEATURES + TARGETS:
            expect(r[key] is None or math.isfinite(float(r[key])), 'nonfinite numeric measurement')
        rows.append(r)
        lineage.append({'run_id':rid,'selected_images':manifest['images'],'audit_sha256':sha256_file(audit_path),
            'geometry_sha256':sha256_file(geom_path),'report_sha256':audit['source_report_sha256'],
            'manifest_sha256':audit['source_manifest_sha256'],'mesh_sha256':audit['source_mesh_sha256'],
            'reconstruction_settings':report['run']['effective_settings'],'software':report['run']['metashape_api_version'],
            'geometry_settings':geom['settings'],'geometry_software':geom['software'],'source_warnings':audit['warnings'],
            'input_verification_scope':'manifest/file/cache/mesh hashes rechecked; original images not rehashed this pass; no proof of bytes present at historical execution'})
        checked_files += [{'path':str(p),'sha256':sha256_file(Path(p))} for p in [audit_path,geom_path,mesh,report['experiment']['manifest_path'],report['output_status']['report_json_path']]]
    expect(len(reference_hashes) == len(signatures) == len(settings) == 1, 'incompatible archival settings/reference')
    expect(sum(r['baseline_self_check'] for r in rows) == 1, 'expected one baseline self-check')
    expect(next(iter(reference_hashes)) == next(r['candidate_mesh_sha256'] for r in rows if r['baseline_self_check']), 'reference differs from audited baseline')
    overlap=[]
    for a,b in itertools.combinations(rows,2):
        sa,sb=image_sets[a['run_id']],image_sets[b['run_id']]
        overlap.append({'run_a':a['run_id'],'run_b':b['run_id'],'shared_images':len(sa&sb),
                        'union_images':len(sa|sb),'jaccard':len(sa&sb)/len(sa|sb)})
    x = np.array([[r[c] for c in PRE_FEATURES] for r in rows],dtype=np.float32)
    post = np.array([[r[c] if r[c] is not None else np.nan for c in POST_FEATURES] for r in rows],dtype=np.float32)
    y = np.array([[r[c] if r[c] is not None else np.nan for c in TARGETS] for r in rows],dtype=np.float32)
    corr = np.corrcoef(x.astype(float),rowvar=False)
    correlation_pairs=[{'a':PRE_FEATURES[i],'b':PRE_FEATURES[j],'pearson_r_descriptive_only':float(corr[i,j]) if np.isfinite(corr[i,j]) else None}
                       for i in range(len(PRE_FEATURES)) for j in range(i+1,len(PRE_FEATURES))]
    output.mkdir(parents=True,exist_ok=False)
    write_csv(output/'observations.csv',rows)
    write_json(output/'observations.json',{'rows':rows,'schema_version':1,'experimental_unit':'one archival run; one object group',
        'missing':'JSON null / CSV blank / float32 tensor NaN with boolean masks'})
    np.savez_compressed(output/'arrays.npz',X_pre=x,X_pre_mask=np.isfinite(x),X_post_refinement=post,
        X_post_refinement_mask=np.isfinite(post),y_reference=y,y_reference_mask=np.isfinite(y),
        reference_row_mask=np.array([r['reference_target_eligible'] for r in rows],dtype=bool))
    validate_arrays(output/'arrays.npz')
    write_json(output/'schema.json',{'schema_version':1,'row_order':[r['run_id'] for r in rows],
        'pre_feature_columns':PRE_FEATURES,'post_refinement_feature_columns':POST_FEATURES,
        'reference_target_columns':TARGETS,'primary_target':TARGETS[1],'numeric_dtype':'float32',
        'reference_labeled_rows':int(np.isfinite(y).all(axis=1).sum()),'object_groups':1,
        'split':None,'statistically_training_ready':False,'array_format_ready':True,
        'self_target_policy':'baseline self-comparison masked; raw self statistic retained as diagnostic'})
    write_json(output/'feature_dictionary.json',field_dictionary(rows))
    write_json(output/'provenance.json',{'builder_source_sha256':sha256_file(Path(__file__)), 'lineage':lineage,'checked_files':checked_files,
        'input_feature_file':{'sha256':sha256_file(fp),'settings':features['settings'],'software':features['software']},
        'pair_feature_file':{'sha256':sha256_file(mp),'settings':matches['settings'],'software':matches['software']},
        'original_observations_csv_sha256':sha256_file(source_dir/'observations.csv'),
        'original_observations_json_sha256':sha256_file(source_dir/'observations.json')})
    write_json(output/'leakage_audit.json',{'sample_count':3,'reference_labeled_rows':2,'object_groups':1,
        'capture_session_id':'unknown; conservatively group all shared-source images together',
        'overlap':overlap,'all_same_reference':True,'all_same_reconstruction_profile':True,
        'constant_post_columns':[c for c in POST_FEATURES if len({r[c] for r in rows})==1],
        'pre_feature_correlation_rank_bound':2,'descriptive_correlations_n3':correlation_pairs,
        'correlation_warning':'n=3 shared-object dependent rows: no inferential correlation claims. With only two labeled rows, nonconstant Pearson correlations are mechanically +/-1.',
        'split_valid':False,'split_blocker':'One object group cannot populate independent train/validation/test partitions.',
        'excluded_from_default_features':['all targets','mesh counts/surface area','registration fitness/inlier RMSE/scale ratio',
            'self-reference status','IDs, hashes, seeds','all post-refinement features'],
        'leakage_risks':['Reference mesh uses all 50 images, including both subset input sets.',
            'The baseline is both reference and self-check; never use its zero distance as an ordinary label.',
            'Post-refinement metrics unavailable at input-only prediction time.',
            'No global scaling/imputation/feature selection fitted; future transforms must be learned on training groups only.',
            'Shared source views, acquisition and artifact confound apparent feature-target associations.']})
    write_json(output/'validation.json',{'valid':True,'rows':3,'reference_labeled_rows':2,'object_groups':1,
        'original_csv_json_consistent':True,'source_hashes_verified':True,'image_cache_hash_joins_verified':True,
        'mesh_counts_verified_against_source_reports':True,'tensor_masks_verified':True,
        'metashape_started':False,'geometry_recomputed':False,'models_trained':False})
    write_json(output/'dataset_files.sha256.json', {p.name:sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()})
    return rows


def validate_arrays(path):
    with np.load(path, allow_pickle=False) as arrays:
        for key, shape in [('X_pre',(3,7)),('X_post_refinement',(3,4)),('y_reference',(3,4))]:
            values=arrays[key]
            expect(values.shape==shape and values.dtype==np.float32, 'incorrect numeric array shape or type')
            expect(np.array_equal(arrays[key+'_mask'],np.isfinite(values)), 'incorrect missing-value mask')
            expect(arrays[key+'_mask'].dtype==np.bool_, 'mask must be boolean')
        expect(np.array_equal(arrays['reference_row_mask'],[False,True,True]), 'self-comparison mask incorrect')
        expect(np.isnan(arrays['y_reference'][0]).all(), 'self-targets must remain missing')
        expect(np.isfinite(arrays['y_reference'][1:]).all(), 'missing provisional subset targets')
        expect(not np.isinf(arrays['X_pre']).any(), 'infinite predictor')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results'))
    p.add_argument('--historical-root',type=Path,required='BD3D_ARCHIVE_ROOT' not in os.environ,
                   default=os.environ.get('BD3D_ARCHIVE_ROOT'), help='historical BD3D_Test_Output folder (or set BD3D_ARCHIVE_ROOT)')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); rows=prepare(a.results,a.historical_root,a.output)
    print(f'Prepared {len(rows)} observations, 2 reference-labeled rows, 1 object group: {a.output}')


if __name__=='__main__':main()
