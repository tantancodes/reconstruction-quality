"""Versioned measured-experiment joins and input-only arrays; no training or imputation."""
from __future__ import annotations
import csv
from pathlib import Path,PurePosixPath
import re
import numpy as np
from .dimensional_accuracy import evaluate_dimensions,require,text,finite,provenance
from .provenance import digest,read_json,write_json,sha256_file
from .tabular_dataset import PRE_FEATURES

PRIMARY_TARGET='target_dimensional_relative_rmse_percent'


def validate_record(record):
    require(isinstance(record,dict),'measured experiment must be an object')
    require(record.get('schema_version')==2 and record.get('kind')=='measured_experiment','unsupported measured-experiment schema')
    require(record.get('data_origin') in {'real','synthetic_test'},'data origin required')
    for k in ['object_id','session_id','subset_id','run_id']:text(record.get(k),k)
    require(record.get('evaluation_role') in {'development','final_holdout'},'explicit development/final_holdout role required')
    images=record.get('images')
    require(isinstance(images,list) and bool(images),'nonempty image manifest required')
    seen=set();seen_hashes=set()
    for e in images:
        name=text(e.get('path'),'image path');p=PurePosixPath(name)
        require(not p.is_absolute() and '..' not in p.parts and str(p)==name and '\\' not in name,'portable image path required')
        require(name.casefold() not in seen,'duplicate image path');seen.add(name.casefold())
        require(bool(re.fullmatch('[0-9a-f]{64}',e.get('sha256',''))),'invalid image SHA-256')
        require(e['sha256'] not in seen_hashes,'duplicate image content hash')
        seen_hashes.add(e['sha256'])
    features=record.get('image_features',{})
    require(set(features)==set(PRE_FEATURES),'input feature whitelist mismatch: reconstruction/target metrics are forbidden')
    for k,v in features.items():
        if v is not None:finite(v,f'input feature {k}')
    require(type(features['input_image_count']) is int and features['input_image_count']==len(images),'input count differs from manifest')
    feature_source=record.get('feature_provenance',{})
    require(feature_source.get('stage')=='pre_reconstruction' and feature_source.get('source_kind')=='images_only','leaky or unsupported feature stage/source')
    require(feature_source.get('input_images_sha256')==digest(images),'feature/input image signature mismatch')
    for k in ['extractor_version','settings_signature']:text(feature_source.get(k),f'feature {k}')
    require(bool(re.fullmatch('[0-9a-f]{64}',feature_source['settings_signature'])),'feature settings SHA-256 required')
    provenance(feature_source.get('provenance'),'feature cache')
    require(isinstance(record.get('reconstruction_configuration'),dict) and bool(record['reconstruction_configuration']),'reconstruction configuration required')
    provenance(record.get('provenance'),'experiment')
    bundle=record.get('dimensional_input')
    require(isinstance(bundle,dict),'embedded dimensional input required')
    for k in ['object_id','session_id','subset_id','run_id','data_origin']:
        require(bundle.get(k)==record[k],f'dimensional input {k} mismatch')
    return evaluate_dimensions(bundle)


def export_measured_dataset(records, output, *, allow_synthetic=False, reserved_object_ids=()):
    require(isinstance(records,list) and bool(records),'at least one measured record required')
    require(all(isinstance(r,dict) for r in records),'records must be objects')
    origins={r.get('data_origin') for r in records}
    require(len(origins)==1,'synthetic fixtures and real observations must never be mixed')
    require(origins=={'real'} or (allow_synthetic and origins=={'synthetic_test'}),'synthetic examples require explicit allow_synthetic=True')
    require(not any(r.get('object_id') in reserved_object_ids for r in records),'reserved final-test object is forbidden in development export')
    require(all(r.get('evaluation_role')!='final_holdout' for r in records),'final held-out objects must remain outside development exports')
    require(len({r.get('run_id') for r in records})==len(records),'duplicate run IDs')
    # A subset ID denotes one ordered input selection within an object/session.
    selections={};profiles=set();feature_settings=set();evaluations=[];panels={}
    for r in records:
        e=validate_record(r);evaluations.append(e)
        panel=digest(sorted((m['measurand_id'],digest(m['definition'])) for m in r['dimensional_input']['ground_truth']['measurements'] if m['role']=='check'))
        require(r['object_id'] not in panels or panels[r['object_id']]==panel,'object check panel changed across conditions; compare separate panels explicitly')
        panels[r['object_id']]=panel
        key=(r['object_id'],r['session_id'],r['subset_id']);signature=digest(r['images'])
        require(key not in selections or selections[key]==signature,'subset ID reused for different image inputs')
        selections[key]=signature
        profiles.add(digest(r['reconstruction_configuration']))
        feature_settings.add((r['feature_provenance']['extractor_version'],r['feature_provenance']['settings_signature']))
    require(len(profiles)==1 and len(feature_settings)==1,'incompatible configurations/feature definitions; export separate study strata')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for r,e in zip(records,evaluations):
        rows.append({**{k:r[k] for k in ['object_id','session_id','subset_id','run_id','data_origin','evaluation_role']},
            **r['image_features'],PRIMARY_TARGET:e[PRIMARY_TARGET],
            'valid_independent_check_count':e['valid_independent_check_count'],'target_eligible':e['target_eligible'],
            'qc_status':e['qc_status'],'dimensional_input_sha256':digest(r['dimensional_input']),
            'feature_inputs_sha256':r['feature_provenance']['input_images_sha256']})
    x=np.array([[r[k] if r[k] is not None else np.nan for k in PRE_FEATURES] for r in rows],dtype=np.float32)
    y=np.array([[r[PRIMARY_TARGET] if r[PRIMARY_TARGET] is not None else np.nan] for r in rows],dtype=np.float32)
    with (output/'observations.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(output/'observations.json',{'schema_version':2,'data_origin':next(iter(origins)),'rows':rows})
    np.savez_compressed(output/'arrays.npz',X_pre=x,X_pre_mask=np.isfinite(x),y_dimensional=y,y_dimensional_mask=np.isfinite(y))
    groups=sorted({r['object_id'] for r in rows})
    write_json(output/'schema.json',{'schema_version':2,'data_origin':next(iter(origins)),
        'feature_columns':PRE_FEATURES,'target_columns':[PRIMARY_TARGET],'row_order':[r['run_id'] for r in rows],
        'object_groups':groups,'grouping_key':'object_id','split':None,'models_trained':False,
        'synthetic_software_example_only':origins=={'synthetic_test'},
        'statistical_readiness':'not established; requires independent objects, label QC and a locked evaluation protocol',
        'missing':'CSV empty / JSON null / array NaN with finite boolean masks; no imputation or normalization',
        'target_units':'percent','target_formula':'100 * sqrt(mean(((reconstructed_mm - measured_mm) / measured_mm)**2))'})
    write_json(output/'evaluations.json',{'reports':evaluations})
    write_json(output/'provenance.json',{'records':records,'record_sha256':[digest(r) for r in records],
        'builder_sha256':sha256_file(Path(__file__)),'dimensional_evaluator_sha256':sha256_file(Path(__file__).with_name('dimensional_accuracy.py')),
        'evidence_scope':'provenance/QC assertions validated structurally; software cannot authenticate measurements or manual endpoints'})
    write_json(output/'feature_dictionary.json',{
        'input_features':{'columns':PRE_FEATURES,'role':'pre-reconstruction image-only predictors','definitions':'docs/TABULAR_DATASET.md'},
        'targets':{PRIMARY_TARGET:{'unit':'percent','role':'independent annotated check-distance error, not full surface accuracy',
            'formula':'100 * sqrt(mean(((reconstructed_mm - measured_mm) / measured_mm)**2))'}},
        'excluded_metadata':['object_id','session_id','subset_id','run_id','data_origin','evaluation_role',
            'valid_independent_check_count','target_eligible','qc_status','dimensional_input_sha256','feature_inputs_sha256'],
        'missing':'null/blank/NaN with mask; no zero filling'})
    write_json(output/'file_hashes.json',{p.name:sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()})
    return rows


def export_files(paths,output,*,allow_synthetic=False,reserved_object_ids=()):
    return export_measured_dataset([read_json(p) for p in paths],output,allow_synthetic=allow_synthetic,reserved_object_ids=reserved_object_ids)
