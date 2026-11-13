"""Argparse CLI; all scientific outputs are explicit files, never implicit reuse."""
import argparse
import json
import sys
from pathlib import Path
from bd3d_quality.config import ReconstructionConfig
from bd3d_quality.manifests import create_manifest, validate_manifest
from bd3d_quality.provenance import read_json, write_json, sha256_file


def parser():
    p = argparse.ArgumentParser(prog='bd3d-quality', description='Reproducible photogrammetry research (no predictive modeling)')
    sub = p.add_subparsers(dest='command', required=True)
    m = sub.add_parser('manifest', help='Create, inspect or validate a portable image manifest')
    ms = m.add_subparsers(dest='action', required=True)
    c = ms.add_parser('create', help='Recursively hash and verify supported images, sorted by relative path')
    c.add_argument('--root', required=True, type=Path)
    c.add_argument('--dataset-id', required=True)
    c.add_argument('--description', default='')
    c.add_argument('--config', type=Path, help='JSON effective reconstruction settings')
    c.add_argument('--output', required=True, type=Path)
    for action in ['inspect', 'validate']:
        v = ms.add_parser(action)
        v.add_argument('manifest', type=Path)
        if action == 'validate':
            v.add_argument('--root', required=True, type=Path, help='Dataset root for hash and image decoding validation')
            v.add_argument('--output', type=Path, help='Optional validation JSON')
    f = sub.add_parser('features', help='CPU image and pair descriptors')
    fs = f.add_subparsers(dest='action', required=True)
    for action in ['extract', 'match']:
        e = fs.add_parser(action)
        e.add_argument('manifest', type=Path)
        e.add_argument('--root', required=True, type=Path)
        e.add_argument('--output', required=True, type=Path)
        e.add_argument('--max-dimension', type=int, default=1600)
    r = sub.add_parser('reconstruct', help='Run licensed Metashape 2.2.x with fresh project or verified resume')
    r.add_argument('manifest', type=Path)
    r.add_argument('--root', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True, help='New run directory (existing only with --resume)')
    r.add_argument('--resume', action='store_true')
    r.add_argument('--seed-checkpoint', type=Path, help='Copy verified alignment checkpoint into a fresh run')
    r.add_argument('--image-quality-cache', type=Path,
                   help='CSV of per-image Image/Quality (by SHA-256) to reuse instead of analyzeImages; recorded in run.json')
    g = sub.add_parser('geometry', help='Reference-relative mesh agreement')
    gs = g.add_subparsers(dest='action', required=True)
    gc = gs.add_parser('compare', help='OBJ triangle surfaces: rigid ICP, scale diagnostics and QC')
    gc.add_argument('candidate', type=Path)
    gc.add_argument('reference', type=Path)
    gc.add_argument('--reference-id', required=True)
    gc.add_argument('--units', required=True, help='Explicit common coordinate units (e.g. mm or arbitrary_shared)')
    gc.add_argument('--sample-count', type=int, default=10000)
    gc.add_argument('--output', required=True, type=Path)
    dimensional = sub.add_parser('dimensions', help='Independent annotated dimensional error, with explicit scale and uncertainty')
    ds = dimensional.add_subparsers(dest='action', required=True)
    de = ds.add_parser('evaluate')
    de.add_argument('input', type=Path)
    de.add_argument('--minimum-checks', type=int, default=2)
    de.add_argument('--allow-missing-uncertainty', action='store_true', help='Diagnostic point labels only; uncertainty incompleteness remains documented')
    de.add_argument('--output', required=True, type=Path)
    dataset = sub.add_parser('measured-dataset', help='Export image-only predictors and independent dimensional labels')
    dataset.add_argument('records', nargs='+', type=Path)
    dataset.add_argument('--reserved-object-id', action='append', default=[], help='Object reserved for final testing; reject it from development export')
    dataset.add_argument('--allow-synthetic', action='store_true', help='Synthetic software-example export only; never mixed with real data')
    dataset.add_argument('--output', required=True, type=Path)
    surface = sub.add_parser('surface', help='Optional typed metric surface-reference comparison')
    ss = surface.add_subparsers(dest='action', required=True)
    se = ss.add_parser('compare')
    se.add_argument('candidate', type=Path)
    se.add_argument('reference', type=Path)
    se.add_argument('--specification', required=True, type=Path)
    se.add_argument('--sample-count', type=int, default=3000)
    se.add_argument('--output', required=True, type=Path)
    alignment_report = sub.add_parser('alignment-report', help='Parse historical Metashape processing-report text; no alignment job')
    ar = alignment_report.add_subparsers(dest='action', required=True)
    arp = ar.add_parser('parse')
    arp.add_argument('text_file', type=Path)
    arp.add_argument('--output', required=True, type=Path)
    alignment_dataset = sub.add_parser('alignment-dataset', help='Historical image-only / pixel-RMSE records and provenance checks')
    ads = alignment_dataset.add_subparsers(dest='action', required=True)
    adb = ads.add_parser('build')
    adb.add_argument('records', nargs='+', type=Path, help='Individual record JSON or a JSON object containing records')
    adb.add_argument('--output', required=True, type=Path)
    report = sub.add_parser('report', help='Summarize a condition and consolidate compatible summaries')
    rs = report.add_subparsers(dest='action', required=True)
    summary = rs.add_parser('summarize')
    summary.add_argument('manifest', type=Path)
    for key in ['features', 'matches', 'run', 'geometry']:
        summary.add_argument('--' + key, type=Path)
    summary.add_argument('--output', required=True, type=Path)
    build = rs.add_parser('build')
    build.add_argument('summaries', nargs='+', type=Path)
    build.add_argument('--output', required=True, type=Path, help='New report directory')
    return p


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    try:
        if a.command == 'manifest':
            if a.action == 'create':
                config = ReconstructionConfig(**read_json(a.config)) if a.config else ReconstructionConfig()
                result = create_manifest(a.root, a.dataset_id, config, a.description)
                write_json(a.output, result)
            elif a.action == 'inspect':
                result = read_json(a.manifest)
            else:
                result = validate_manifest(read_json(a.manifest), a.root)
                if a.output:
                    write_json(a.output, result)
                print(json.dumps(result, indent=2))
                return 0 if result['valid'] else 2
        elif a.command == 'features':
            if a.action == 'extract':
                from .image_features import features_for_manifest
                result = features_for_manifest(read_json(a.manifest), a.root, a.max_dimension)
            else:
                from .local_match_features import match_manifest
                result = match_manifest(read_json(a.manifest), a.root, a.max_dimension)
            write_json(a.output, result)
        elif a.command == 'dimensions':
            from .dimensional_accuracy import evaluate_dimensions
            result = evaluate_dimensions(read_json(a.input), minimum_checks=a.minimum_checks,
                require_uncertainty=not a.allow_missing_uncertainty)
            write_json(a.output, result)
            print(json.dumps({'qc_status': result['qc_status'], 'valid_checks': result['valid_independent_check_count'], 'output': str(a.output)}))
            return 0 if result['target_eligible'] else 2
        elif a.command == 'measured-dataset':
            from .measured_dataset import export_files
            result = export_files(a.records, a.output, allow_synthetic=a.allow_synthetic, reserved_object_ids=a.reserved_object_id)
        elif a.command == 'surface':
            from .surface_accuracy import compare_surface
            result = compare_surface(a.candidate, a.reference, read_json(a.specification), sample_count=a.sample_count)
            write_json(a.output, result)
            print(json.dumps({'comparison_status': result['comparison_status'], 'output': str(a.output)}))
            return 0 if result['comparison_status'] == 'complete' else 2
        elif a.command == 'alignment-report':
            from .historical_alignment import parse_report_text
            result = parse_report_text(a.text_file.read_text())
            result['source'] = {'file_path': str(a.text_file), 'extracted_text_sha256': sha256_file(a.text_file)}
            result['interpretation'] = 'Historical source values only; alignment consistency, not physical accuracy'
            write_json(a.output, result)
        elif a.command == 'alignment-dataset':
            from .historical_alignment import build_dataset
            records = []
            for path in a.records:
                value = read_json(path)
                records.extend(value['records'] if 'records' in value else [value])
            result = build_dataset(records, a.output)
        elif a.command == 'reconstruct':
            from bd3d_quality.metashape_runner import run_reconstruction
            result = run_reconstruction(read_json(a.manifest), a.root, a.output, resume=a.resume, seed_checkpoint=a.seed_checkpoint,
                                        image_quality_cache=a.image_quality_cache)
        elif a.command == 'geometry':
            from .geometry_quality import compare_geometry
            result = compare_geometry(a.candidate, a.reference, reference_id=a.reference_id,
                                      units=a.units, sample_count=a.sample_count)
            write_json(a.output, result)
            print(json.dumps({'status': result['status'], 'output': str(a.output)}))
            return 0 if result['status'] == 'complete' else 2
        elif a.action == 'summarize':
            from .reporting import summarize
            records, sources = {}, []
            for key in ['features', 'matches', 'run', 'geometry']:
                path = getattr(a, key)
                if path:
                    records[key] = read_json(path)
                    sources.append({'role': key, 'filename': path.name, 'sha256': sha256_file(path)})
            result = summarize(read_json(a.manifest), sources=sources, **records)
            write_json(a.output, result)
        else:
            from .reporting import build_report
            result = build_report(a.summaries, a.output)
        print(json.dumps({'status': 'ok', 'output': str(getattr(a, 'output', '')), 'result': result if a.command == 'manifest' and a.action == 'inspect' else None}, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as exc:
        print(f'bd3d-quality: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
