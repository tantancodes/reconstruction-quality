"""Bounded, read-only Metashape XML/ZIP inspection. No Metashape import or extraction."""
from __future__ import annotations
from collections import Counter
import math
from pathlib import Path,PurePosixPath
import re
import xml.etree.ElementTree as ET
import zipfile
from .provenance import sha256_file

MAX_XML=1_000_000
MAX_ARCHIVE=1_000_000


def xml_root(data):
    if len(data)>MAX_XML:raise ValueError('XML exceeds metadata byte budget')
    text=data.decode('utf-8-sig')
    if re.search(r'<!\s*(DOCTYPE|ENTITY)\b',text,re.I):raise ValueError('DTD/entities are forbidden')
    try:return ET.fromstring(text)
    except ET.ParseError as exc:raise ValueError('invalid metadata XML') from exc


def read_archive_xml(path):
    path=Path(path)
    if path.stat().st_size>MAX_ARCHIVE:raise ValueError('archive exceeds metadata byte budget')
    with zipfile.ZipFile(path) as z:
        entries=z.infolist()
        if len(entries)>64 or sum(e.file_size for e in entries)>2*MAX_XML:
            raise ValueError('archive expansion exceeds metadata budget')
        names=[e.filename for e in entries]
        if len(set(names))!=len(names):raise ValueError('duplicate archive members')
        for e in entries:
            p=PurePosixPath(e.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in e.filename:raise ValueError('unsafe archive member path')
            if e.flag_bits&1:raise ValueError('encrypted metadata archive')
        if 'doc.xml' not in names:raise ValueError('metadata archive has no doc.xml')
        entry=z.getinfo('doc.xml')
        if entry.file_size>MAX_XML:raise ValueError('doc.xml exceeds metadata byte budget')
        with z.open(entry) as f:data=f.read(MAX_XML+1)
    return xml_root(data)


def properties(root):
    return {p.get('name'):p.get('value') for p in root.findall('./meta/property')}


def inspect_project(psx,project_zip,chunk_zip):
    psx=Path(psx)
    if psx.stat().st_size>MAX_XML:raise ValueError('PSX exceeds metadata byte budget')
    pointer=xml_root(psx.read_bytes());project=read_archive_xml(project_zip);chunk=read_archive_xml(chunk_zip)
    cameras=[]
    for c in chunk.findall('./cameras/camera'):
        matrix=(c.findtext('transform') or '').split()
        valid_matrix=len(matrix)==16
        if matrix:
            try:valid_matrix=valid_matrix and all(math.isfinite(float(v)) for v in matrix)
            except ValueError:valid_matrix=False
        cameras.append({'camera_id':c.get('id'),'sensor_id':c.get('sensor_id'),'label':c.get('label'),
            'component_id':c.get('component_id'),'enabled_attribute':c.get('enabled'),
            'has_stored_transform':valid_matrix,'has_transform_element':c.find('transform') is not None})
    refs=[{'element':e.tag,'path':e.get('path')} for e in chunk.iter() if e.get('path')]
    photo_paths=[p.get('path') for p in chunk.findall('.//photo') if p.get('path')]
    meta=properties(chunk)
    report_refs=[]
    for root in (pointer,project,chunk):
        for e in root.iter():
            for v in [*e.attrib.values(),e.text or '']:
                if re.search(r'\.pdf(?:$|[?#\s])',v,re.I):report_refs.append(v)
    possible={k:{'raw_value':v,'definition_status':'not_verified_pixel_RMSE'} for k,v in meta.items()
              if re.search(r'rmse|reprojection|sigma0',k,re.I)}
    return {'schema_version':1,'kind':'metashape_metadata_inspection','psx_pointer':pointer.attrib,
        'project_metadata':properties(project),'project_chunks':[e.attrib for e in project.findall('./chunks/chunk')],
        'chunk_label':chunk.get('label'),'chunk_metadata':meta,'camera_count':len(cameras),
        'cameras_with_stored_transform':sum(c['has_stored_transform'] for c in cameras),
        'camera_components':dict(Counter(c['component_id'] if c['component_id'] is not None else 'not_stated' for c in cameras)),
        'component_count':len(chunk.findall('./components/component')),'cameras':cameras,
        'photo_paths':photo_paths,'asset_references':refs,'report_references':report_refs,
        'candidate_optimizer_statistics':possible,'verified_reprojection_rmse_pixels':None,
        'source_files':[{'path':str(p),'size_bytes':Path(p).stat().st_size,'sha256':sha256_file(Path(p))} for p in [psx,project_zip,chunk_zip]],
        'limitations':['No referenced files were opened automatically and no ZIP entries extracted to disk.',
            'Stored transforms may span disconnected components; their count is not a verified report-aligned-camera count.',
            'Camera labels are not photo paths, processed-file versions or content hashes.',
            'OptimizeCameras/sigma0 is not accepted as pixel reprojection RMSE without a verified definition.',
            'The PSX is a pointer, not a complete project. Missing frame assets prevent full provenance recovery.']}


def compare_stems(cameras,files):
    labels=[Path(c['label']).stem.casefold() for c in cameras if c.get('label')]
    names=[Path(f['title']).stem.casefold() for f in files]
    return {'camera_count':len(cameras),'candidate_file_count':len(files),
        'unique_camera_labels':len(set(labels)),'unique_candidate_stems':len(set(names)),
        'matching_stems':len(set(labels)&set(names)),
        'camera_only':sorted(set(labels)-set(names)),'candidate_only':sorted(set(names)-set(labels)),
        'filename_stem_sets_equal':set(labels)==set(names),
        'exact_processed_image_bytes_verified':False,
        'reason':'Stem agreement alone does not verify file extensions, preprocessing versions or historical content hashes.'}
