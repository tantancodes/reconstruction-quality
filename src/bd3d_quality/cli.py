"""Command line: create/validate image manifests and run a Metashape reconstruction.

All outputs are explicit files, never implicit reuse. The earlier-phase subcommands
(features, geometry compare, dimensions, ...) are in archive/src/bd3d_legacy/cli.py.
"""
import argparse
import json
import sys
from pathlib import Path
from .config import ReconstructionConfig
from .manifests import create_manifest, validate_manifest
from .provenance import read_json, write_json


def parser():
    p = argparse.ArgumentParser(prog='bd3d-quality', description='Image manifests and watched Metashape reconstruction runs')
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
    r = sub.add_parser('reconstruct', help='Run licensed Metashape 2.2.x with fresh project or verified resume')
    r.add_argument('manifest', type=Path)
    r.add_argument('--root', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True, help='New run directory (existing only with --resume)')
    r.add_argument('--resume', action='store_true')
    r.add_argument('--seed-checkpoint', type=Path, help='Copy verified alignment checkpoint into a fresh run')
    r.add_argument('--image-quality-cache', type=Path,
                   help='CSV of per-image Image/Quality (by SHA-256) to reuse instead of analyzeImages; recorded in run.json')
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
        elif a.command == 'reconstruct':
            from .metashape_runner import run_reconstruction
            result = run_reconstruction(read_json(a.manifest), a.root, a.output, resume=a.resume, seed_checkpoint=a.seed_checkpoint,
                                        image_quality_cache=a.image_quality_cache)
        print(json.dumps({'status': 'ok', 'output': str(getattr(a, 'output', '')), 'result': result if a.command == 'manifest' and a.action == 'inspect' else None}, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as exc:
        print(f'bd3d-quality: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
