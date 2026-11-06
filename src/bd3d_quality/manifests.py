"""Portable ordered image manifests. Root locations never enter experiment identity."""
from pathlib import Path, PurePosixPath
import re
from PIL import Image
from .config import ReconstructionConfig
from .provenance import digest, now, sha256_file, software_versions

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.tif', '.tiff', '.png'}


def identity_payload(m):
    return {k: m[k] for k in ['schema_version', 'dataset_id', 'images', 'image_count', 'configuration']}


def experiment_identity(m):
    return 'bd3d-' + digest(identity_payload(m))


def safe_path(root, name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('image path must be a nonempty POSIX relative path')
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or str(p) != name:
        raise ValueError(f'nonportable image path: {name}')
    root = Path(root).resolve()
    target = (root / name).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f'image escapes dataset root: {name}')
    return target


def create_manifest(root, dataset_id, config=None, description=''):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('dataset root must be a directory')
    images = []
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            name = path.relative_to(root).as_posix()
            safe_path(root, name)
            with Image.open(path) as image:
                image.verify()
            images.append({'path': name, 'sha256': sha256_file(path), 'size_bytes': path.stat().st_size})
    if not images or not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError('nonempty dataset ID and at least one readable image required')
    m = {'schema_version': 1, 'dataset_id': dataset_id, 'description': description,
         'images': images, 'image_count': len(images),
         'configuration': (config or ReconstructionConfig()).to_dict(),
         'created_at': now(), 'software': software_versions(), 'status': 'prepared',
         'warnings': [], 'failures': [], 'artifacts': {}}
    m['experiment_id'] = experiment_identity(m)
    return m


def validate_manifest(m, root=None):
    errors = []
    try:
        if m['schema_version'] != 1 or not isinstance(m['dataset_id'], str) or not m['dataset_id'].strip():
            raise ValueError('invalid schema or dataset ID')
        config = ReconstructionConfig(**m['configuration'])
        if config.to_dict() != m['configuration']:
            raise ValueError('configuration must include all effective settings')
        if not m['images'] or type(m['image_count']) is not int or m['image_count'] != len(m['images']):
            raise ValueError('image count mismatch or empty image set')
        seen = set()
        for e in m['images']:
            name = e['path']
            safe_path(root or '.', name)
            if name.casefold() in seen:
                raise ValueError(f'duplicate image path (case insensitive): {name}')
            seen.add(name.casefold())
            if not re.fullmatch('[0-9a-f]{64}', e['sha256']):
                raise ValueError(f'invalid SHA-256: {name}')
            if type(e['size_bytes']) is not int or e['size_bytes'] <= 0:
                raise ValueError(f'invalid file size: {name}')
            if root is not None:
                p = safe_path(root, name)
                if sha256_file(p) != e['sha256'] or p.stat().st_size != e['size_bytes']:
                    raise ValueError(f'input hash/size mismatch: {name}')
                with Image.open(p) as image:
                    image.verify()
        if m['experiment_id'] != experiment_identity(m):
            raise ValueError('experiment identity mismatch')
    except (KeyError, TypeError, ValueError, OSError) as exc:
        errors.append(str(exc))
    return {'valid': not errors, 'experiment_id': m.get('experiment_id'),
            'files_verified': root is not None and not errors, 'errors': errors}


def require_valid(m, root=None):
    result = validate_manifest(m, root)
    if not result['valid']:
        raise ValueError('; '.join(result['errors']))
    return result
