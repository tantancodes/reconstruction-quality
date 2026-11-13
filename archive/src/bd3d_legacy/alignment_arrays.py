"""EXPLORATORY / NOT VALIDATED array interface; no PyTorch dependency or training.

Strict mode is the default. Explicit exploratory access exposes source-verified
labels and pre-alignment features without changing any readiness status.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import zipfile

import numpy as np

from .historical_alignment import compatibility_key, require, validate_run
from .historical_recovery import bounded_json
from bd3d_quality.provenance import digest, sha256_file
from .tabular_dataset import PRE_FEATURES

STATUS = 'EXPLORATORY / NOT VALIDATED'
TARGET = 'target_reprojection_rmse_pixels'


def content_set_id(record):
    """Order/path-independent content group, never an independent artifact ID."""
    hashes = [i.get('sha256') for i in record['source_images']]
    if len(hashes) != record['image_count'] or not all(hashes):
        return None
    return 'sha256:' + digest(sorted(hashes))


def strict_ready(record):
    # Mirrors the existing historical builder gate; never promotes source records.
    return (record['label']['status'] in {'document_verified', 'computed_historical_verified'}
            and record['label']['rmse_pixels'] is not None
            and record['image_provenance_status'] == 'verified_hash_manifest'
            and record.get('image_features') is not None
            and all(v is not None for v in record['image_features'].values())
            and record['configuration_status'] == 'verified'
            and record.get('session_status') == 'verified')


@dataclass
class AlignmentArrays:
    """X[N,7], y[N,1]; float arrays and boolean masks, metadata kept separate.

    ``__getitem__`` returns numeric arrays/masks only, suitable for a future
    PyTorch DataLoader's default collation. No torch import is performed.
    """
    X: np.ndarray
    y: np.ndarray
    X_mask: np.ndarray
    y_mask: np.ndarray
    strict_ready_mask: np.ndarray
    records: tuple
    source_records: tuple
    source_indices: np.ndarray
    source_hashes: dict
    mode: str
    feature_names: tuple = tuple(PRE_FEATURES)
    target_name: str = TARGET
    status: str = STATUS

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return {'x': self.X[index].copy(), 'y': self.y[index].copy(),
                'x_mask': self.X_mask[index].copy(), 'y_mask': self.y_mask[index].copy(),
                'strict_ready': self.strict_ready_mask[index].copy()}

    def take(self, indices):
        """Explicit unique-row selection, preserving metadata and gate flags."""
        indices = np.asarray(indices)
        if not indices.size:
            indices = np.asarray([], dtype=np.int64)
        require(indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer), 'one-dimensional integer indices required')
        require(len(set(indices.tolist())) == len(indices), 'duplicate subset indices')
        require(((indices >= 0) & (indices < len(self))).all(), 'subset index out of range')
        return replace(self, X=np.ascontiguousarray(self.X[indices]), y=np.ascontiguousarray(self.y[indices]),
            X_mask=self.X_mask[indices].copy(), y_mask=self.y_mask[indices].copy(),
            strict_ready_mask=self.strict_ready_mask[indices].copy(),
            records=tuple(self.records[i] for i in indices), source_indices=self.source_indices[indices].copy())

    @property
    def record_ids(self):
        return tuple(r['record_id'] for r in self.records)

    @property
    def artifact_groups(self):
        return tuple(r['artifact_group_id'] for r in self.records)

    @property
    def content_groups(self):
        return tuple(content_set_id(r) for r in self.records)

    @property
    def strata(self):
        return tuple(compatibility_key(r) for r in self.records)


def load_alignment_arrays(bundle, *, mode='strict', artifact_group=None,
                          paired_only=True, dtype=np.float32):
    """Load an immutable historical bundle; metadata/arrays only, never photos.

    ``mode='strict'`` retains the existing gate (zero real rows currently).
    ``mode='exploratory'`` must be explicit. ``paired_only=False`` exposes missing
    values as NaN/masks for audits, without imputation. dtype is float32 or float64.
    """
    require(mode in {'strict', 'exploratory'}, 'mode must be strict or exploratory')
    dtype = np.dtype(dtype)
    require(dtype in (np.dtype('float32'), np.dtype('float64')), 'float32/float64 required')
    bundle = Path(bundle)
    manifest_path = next((bundle/n for n in ('recovery-file-hashes.json', 'file_hashes.json')
                          if (bundle/n).is_file()), None)
    require(manifest_path is not None, 'bundle hash manifest missing')
    hashes = bounded_json(manifest_path)
    require(len(hashes) <= 64, 'bundle file count exceeds metadata budget')
    required = {'records.json', 'observations.json', 'schema.json', 'arrays.npz'}
    require(required.issubset(hashes), 'bundle hash manifest incomplete')
    for name, expected in hashes.items():
        require(isinstance(name, str) and '/' not in name and '\\' not in name and name not in {'.', '..'},
                'unsafe bundle hash entry')
        p = bundle/name
        require(p.stat().st_size <= 2_000_000, 'bundle file exceeds lightweight byte budget')
        require(sha256_file(p) == expected, 'bundle hash mismatch: '+name)
    records = bounded_json(bundle/'records.json')['records']
    require(len(records) <= 500, 'bundle row count exceeds exploratory budget')
    for r in records:
        validate_run(r)
    require(len({r['record_id'] for r in records}) == len(records), 'duplicate record ID')
    require(len({r['reconstruction_run_id'] for r in records}) == len(records), 'duplicate run ID')
    schema = bounded_json(bundle/'schema.json')
    require(schema['feature_columns'] == PRE_FEATURES and schema['target'] == TARGET,
            'unexpected feature/target contract; possible leakage')
    require(schema['row_order'] == [r['record_id'] for r in records], 'bundle row-order mismatch')
    require(schema['source_records_sha256'] == [digest(r) for r in records], 'source-record digest mismatch')
    X = np.asarray([[r['image_features'].get(k) if r.get('image_features') else np.nan
                     for k in PRE_FEATURES] for r in records], dtype=np.float64).reshape(-1, len(PRE_FEATURES))
    y = np.asarray([[r['label']['rmse_pixels'] if r['label']['status'] in
                     {'document_verified', 'computed_historical_verified'} else np.nan]
                    for r in records], dtype=np.float64).reshape(-1, 1)
    ready = np.asarray([strict_ready(r) for r in records], dtype=bool)
    with zipfile.ZipFile(bundle/'arrays.npz') as archive:
        require(sum(i.file_size for i in archive.infolist()) <= 2_000_000, 'array expansion exceeds lightweight budget')
    with np.load(bundle/'arrays.npz', allow_pickle=False) as arrays:
        expected = {'X_pre': X.astype(np.float32), 'y_rmse_pixels': y.astype(np.float32),
                    'X_pre_mask': np.isfinite(X), 'y_rmse_pixels_mask': np.isfinite(y),
                    'model_input_row_mask': ready}
        for key, value in expected.items():
            require(np.array_equal(arrays[key], value, equal_nan=True), 'array/record mismatch: '+key)
    rows = bounded_json(bundle/'observations.json')['rows']
    require([r['record_id'] for r in rows] == schema['row_order'], 'observation order mismatch')
    require([r['model_input_eligible'] for r in rows] == ready.tolist(), 'strict readiness flag mismatch')
    selected = np.ones(len(records), dtype=bool)
    if mode == 'strict':
        selected &= ready
    if paired_only:
        selected &= np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
    if artifact_group is not None:
        selected &= np.asarray([r['artifact_group_id'] == artifact_group for r in records])
    indices = np.flatnonzero(selected)
    return AlignmentArrays(np.ascontiguousarray(X[selected], dtype=dtype),
        np.ascontiguousarray(y[selected], dtype=dtype), np.isfinite(X[selected]), np.isfinite(y[selected]),
        ready[selected].copy(), tuple(records[i] for i in indices), tuple(records), indices,
        {'hash_manifest_sha256': sha256_file(manifest_path), **hashes}, mode)
