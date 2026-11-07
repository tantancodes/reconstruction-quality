"""Rigid reference agreement. Low-level sampling/ICP adapted from Aphoma (CC0).
See docs/ATTRIBUTION.md. No geometry is modified on disk.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from .provenance import digest, software_versions, sha256_file
@dataclass(frozen=True)
class TriangleMesh:
    vertices: np.ndarray
    triangles: np.ndarray


def load_obj_mesh(path) -> TriangleMesh:
    """Load OBJ geometry only; materials, normals, and textures are ignored."""
    path = Path(path).expanduser().resolve()
    vertices, triangles = [], []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                values = line.split()
                vertices.append(tuple(float(value) for value in values[1:4]))
            elif line.startswith("f "):
                indices = [int(item.split("/")[0]) for item in line.split()[1:]]
                if 0 in indices:
                    raise ValueError("OBJ indices cannot be zero")
                resolved = [index - 1 if index > 0 else len(vertices) + index for index in indices]
                for offset in range(1, len(resolved) - 1):
                    triangles.append((resolved[0], resolved[offset], resolved[offset + 1]))
    mesh = TriangleMesh(np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64))
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("nonfinite mesh coordinates")
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"OBJ has no usable triangle mesh: {path}")
    if mesh.triangles.min() < 0 or mesh.triangles.max() >= len(mesh.vertices):
        raise ValueError(f"OBJ contains an invalid face index: {path}")
    return mesh


def sample_surface(mesh: TriangleMesh, count: int, seed: int) -> np.ndarray:
    """Sample triangle surfaces in proportion to area, deterministically."""
    triangle_vertices = mesh.vertices[mesh.triangles]
    cross = np.cross(
        triangle_vertices[:, 1] - triangle_vertices[:, 0],
        triangle_vertices[:, 2] - triangle_vertices[:, 0],
    )
    areas = np.linalg.norm(cross, axis=1) * 0.5
    usable = areas > np.finfo(float).eps
    if not np.any(usable):
        raise ValueError("mesh has no nondegenerate triangles")
    triangle_vertices, areas = triangle_vertices[usable], areas[usable]
    rng = np.random.default_rng(seed)
    choices = rng.choice(len(areas), size=count, p=areas / areas.sum())
    chosen = triangle_vertices[choices]
    first = np.sqrt(rng.random(count))
    second = rng.random(count)
    return (
        (1 - first)[:, None] * chosen[:, 0]
        + (first * (1 - second))[:, None] * chosen[:, 1]
        + (first * second)[:, None] * chosen[:, 2]
    )


def _rigid_transform(source: np.ndarray, target: np.ndarray):
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    covariance = np.einsum(
        "ni,nj->ij", source - source_center, target - target_center,
        optimize=False,
    )
    left, _, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        rotation = right.T @ left.T
    translation = target_center - np.einsum("ij,j->i", rotation, source_center)
    return rotation, translation


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.einsum("ni,ji->nj", points, transform[:3, :3]) + transform[:3, 3]


def rigid_icp(
    source, target, max_distance, max_iterations, relative_tolerance,
    absolute_tolerance=0.0,
):
    """Centroid-initialized, correspondence-trimmed point-to-point rigid ICP."""
    transform = np.eye(4)
    transform[:3, 3] = target.mean(axis=0) - source.mean(axis=0)
    target_tree = cKDTree(target)
    previous_rmse = np.inf
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        transformed = apply_transform(source, transform)
        distances, indices = target_tree.query(transformed, workers=1)
        accepted = distances <= max_distance
        if accepted.sum() < 3:
            break
        rotation, translation = _rigid_transform(transformed[accepted], target[indices[accepted]])
        update = np.eye(4)
        update[:3, :3], update[:3, 3] = rotation, translation
        transform = update @ transform
        rmse = float(np.sqrt(np.mean(np.square(distances[accepted]))))
        if rmse <= absolute_tolerance or (
            np.isfinite(previous_rmse)
            and abs(previous_rmse - rmse) <= relative_tolerance * max(previous_rmse, 1e-15)
        ):
            break
        previous_rmse = rmse
    transformed = apply_transform(source, transform)
    distances, _ = target_tree.query(transformed, workers=1)
    accepted = distances <= max_distance
    return {
        "transform": transform,
        "fitness": float(accepted.mean()),
        "inlier_rmse": float(np.sqrt(np.mean(np.square(distances[accepted])))) if accepted.any() else None,
        "iterations": iterations,
        "correspondence_count": int(accepted.sum()),
    }



def compare_geometry(candidate_path, reference_path, *, reference_id, units,
                     sample_count=10000, seed=20260822, max_iterations=60,
                     correspondence_fraction=0.05, min_fitness=0.8, scale_tolerance=0.25,
                     threshold_fraction=0.01, reference_kind='reconstruction',
                     ground_truth_evidence=None):
    """Return agreement only after registration and scale QC pass.

    RMS radius ratio is rotation invariant, but also responds to missing coverage.
    It is a diagnostic, never an estimated correction to apply.
    """
    import math
    if (not reference_id or not units or type(sample_count) is not int or sample_count < 100
        or type(max_iterations) is not int or max_iterations < 1
        or type(seed) is not int or seed < 0):
        raise ValueError('reference ID, units, >=100 samples, nonnegative seed and positive iterations required')
    values = [correspondence_fraction, min_fitness, scale_tolerance, threshold_fraction]
    if not all(math.isfinite(v) for v in values) or not (0 < min_fitness <= 1 and
            0 < correspondence_fraction <= 1 and 0 < scale_tolerance < 1 and 0 < threshold_fraction <= 1):
        raise ValueError('invalid geometry thresholds')
    if reference_kind not in {'reconstruction', 'validated_ground_truth'}:
        raise ValueError('invalid reference kind')
    if reference_kind == 'validated_ground_truth' and not ground_truth_evidence:
        raise ValueError('ground-truth claim requires validation evidence')
    candidate, reference = load_obj_mesh(candidate_path), load_obj_mesh(reference_path)
    x = sample_surface(candidate, sample_count, seed)
    y = sample_surface(reference, sample_count, seed)
    radius = lambda a: float(np.sqrt(np.mean(np.sum((a - a.mean(axis=0)) ** 2, axis=1))))
    diagonal = float(np.linalg.norm(np.ptp(reference.vertices, axis=0)))
    ratio = radius(x) / radius(y)
    registration = rigid_icp(x, y, diagonal * correspondence_fraction, max_iterations, 1e-7, 1e-12 * diagonal)
    flags = []
    if not 1 - scale_tolerance <= ratio <= 1 + scale_tolerance:
        flags.append('major_scale_or_coverage_inconsistency')
    if registration['fitness'] < min_fitness or registration['correspondence_count'] < 3:
        flags.append('registration_failed')
    elif registration['iterations'] >= max_iterations:
        flags.append('registration_iteration_limit')
    transform = registration.pop('transform')
    out = {'schema_version': 1, 'kind': 'geometry_comparison', 'status': 'invalid' if flags else 'complete',
           'interpretation': 'reference_agreement' if reference_kind == 'reconstruction' else 'agreement_with_declared_validated_ground_truth',
           'reference_id': reference_id, 'reference_kind': reference_kind,
           'ground_truth_evidence': ground_truth_evidence, 'units': units,
           'candidate': {'filename': Path(candidate_path).name, 'sha256': sha256_file(Path(candidate_path)),
                         'vertex_count': len(candidate.vertices), 'face_count': len(candidate.triangles)},
           'reference': {'filename': Path(reference_path).name, 'sha256': sha256_file(Path(reference_path)),
                         'vertex_count': len(reference.vertices), 'face_count': len(reference.triangles)},
           'settings': {'version': 1, 'sample_count': sample_count, 'seed': seed,
                        'max_iterations': max_iterations, 'correspondence_fraction': correspondence_fraction,
                        'min_fitness': min_fitness, 'scale_tolerance': scale_tolerance,
                        'threshold_fraction': threshold_fraction, 'method': 'centroid then rigid point-to-point ICP'},
           'software': software_versions(), 'scale_radius_ratio': ratio,
           'registration': {**registration, 'transform_candidate_to_reference': transform.tolist(), 'scale_allowed': False},
           'flags': flags, 'metrics': None,
           'warnings': ['Sample-to-sample distances approximate surface agreement.',
                        'ICP is local: symmetry and poor initial orientation can yield misleading fitness.',
                        'RMS radius ratio responds to coverage; inspect geometry before interpretation.']}
    out['settings_signature'] = digest(out['settings'])
    if not flags:
        # Independent evaluation stream; same seed on both meshes gives a paired self-check.
        xe = apply_transform(sample_surface(candidate, sample_count, seed + 1), transform)
        ye = sample_surface(reference, sample_count, seed + 1)
        dxy = cKDTree(ye).query(xe, workers=1)[0]
        dyx = cKDTree(xe).query(ye, workers=1)[0]
        def stats(d):
            return {'mean': float(d.mean()), 'median': float(np.median(d)),
                    'p95': float(np.percentile(d, 95)), 'max': float(d.max()),
                    'rmse': float(np.sqrt(np.mean(d**2)))}
        out['metrics'] = {'symmetric_distances': stats(np.concatenate((dxy,dyx))), 'candidate_to_reference': stats(dxy), 'reference_to_candidate': stats(dyx),
                          'chamfer_mean_squared': float((np.mean(dxy**2) + np.mean(dyx**2)) / 2),
                          'candidate_coverage_fraction': float(np.mean(dxy <= threshold_fraction * diagonal)),
                          'reference_completeness_fraction': float(np.mean(dyx <= threshold_fraction * diagonal)),
                          'threshold_distance': threshold_fraction * diagonal,
                          'reference_diagonal': diagonal}
    return out
