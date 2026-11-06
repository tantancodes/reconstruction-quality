"""Canonical identity, streaming hashes and atomic, exclusive result creation."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, *, replace=False):
    """Publish complete JSON atomically; default refuses overwriting evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    fd, name = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if replace:
            os.replace(name, path)
        else:
            os.link(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def software_versions():
    out = {'python': platform.python_version(), 'platform': platform.platform()}
    for name in ['bd3d-quality', 'numpy', 'Pillow', 'scipy', 'opencv-python-headless']:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out
