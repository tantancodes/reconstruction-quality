"""Version 1 image descriptors; see docs/FEATURE_DICTIONARY.md."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from bd3d_quality.manifests import require_valid, safe_path
from bd3d_quality.provenance import sha256_file, software_versions, digest


def load_gray(path, max_dimension=1600):
    if type(max_dimension) is not int or max_dimension < 8:
        raise ValueError('max_dimension must be an integer >= 8')
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert('RGB')
        width, height = rgb.size
        rgb.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        gray = np.asarray(rgb.convert('L'), dtype=np.uint8)
    return gray, width, height


def extract_image_features(path, max_dimension=1600):
    gray, w, h = load_gray(path, max_dimension)
    a = gray.astype(float)
    probabilities = np.bincount(gray.ravel(), minlength=256) / gray.size
    p = probabilities[probabilities > 0]
    lap = a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] - 4 * a[1:-1, 1:-1]
    q05, median, q95 = np.percentile(a, [5, 50, 95])
    return {'sha256': sha256_file(path), 'width_px': w, 'height_px': h,
            'aspect_ratio': w / h, 'size_bytes': Path(path).stat().st_size,
            'analysis_width_px': gray.shape[1], 'analysis_height_px': gray.shape[0],
            'brightness_mean': float(a.mean()), 'contrast_std': float(a.std()),
            'gray_p05': float(q05), 'gray_median': float(median), 'gray_p95': float(q95),
            'contrast_p95_p05': float(q95 - q05),
            'laplacian_variance': float(lap.var()) if lap.size else None,
            'entropy_bits': float(-(p * np.log2(p)).sum()),
            'missing_reasons': {} if lap.size else {'laplacian_variance': 'image smaller than 3x3'}}


def features_for_manifest(m, root, max_dimension=1600):
    require_valid(m, root)
    rows = []
    for entry in m['images']:
        row = extract_image_features(safe_path(root, entry['path']), max_dimension)
        if row['sha256'] != entry['sha256']:
            raise ValueError('input changed during extraction')
        rows.append({'path': entry['path'], 'status': 'complete', **row})
    settings = {'feature_version': 1, 'max_dimension': max_dimension,
                'grayscale': 'Pillow RGB-to-L, EXIF transpose, Lanczos thumbnail'}
    return {'schema_version': 1, 'kind': 'image_features', 'experiment_id': m['experiment_id'],
            'settings': settings, 'settings_signature': digest(settings),
            'software': software_versions(), 'images': rows}
