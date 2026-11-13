"""Geometry evaluator v3: how far a candidate mesh is from a reference mesh.

Registration starts from the similarity transform (rotation, translation, uniform scale)
that maps the candidate run's aligned cameras onto the reference run's cameras over shared
images, then refines it with point-to-point ICP that also allows uniform scale. The label
is the symmetric RMSE between surface samples, divided by the reference bounding-box
diagonal. Runs where the ICP scale and the camera scale differ by more than
SCALE_TOLERANCE are flagged scale_check = 'disagree'.

Mesh loading, area-weighted sampling, seeds, sample counts, tolerances and the QC rule are
those of the historical Aphoma evaluator (CC0, see docs/ATTRIBUTION.md), so labels match
results/geometry-labels-v3 exactly. The same ICP loop with a centroid start is the v2
evaluator (archive/scripts/rescore_similarity_icp.py).
"""
import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SCALE_TOLERANCE = 0.025
LONG_ITERATIONS = 300
QC = [(0.95, 'pass'), (0.80, 'caution'), (-1, 'manual_review')]  # registration fitness thresholds
REGISTRATION_METHOD = 'camera_similarity_initialization_then_similarity_point_to_point_icp'


@dataclass(frozen=True)
class GeometryEvaluationSettings:
    """Sampling and ICP settings, unchanged from the historical evaluator (version 1)."""
    evaluator_version: int = 1
    schema_version: str = "1.0"
    random_seed: int = 20260822
    evaluation_sample_count: int = 100_000
    registration_sample_count: int = 20_000
    registration_method: str = "centroid_initialization_then_rigid_point_to_point_icp"
    icp_max_iterations: int = 60
    icp_relative_tolerance: float = 1e-7
    icp_absolute_tolerance_fraction: float = 1e-12
    icp_max_correspondence_distance_fraction: float = 0.05
    threshold_fractions_of_reference_diagonal: tuple[float, ...] = (0.0025, 0.005, 0.01, 0.02)

    def signature(self) -> str:
        return hashlib.sha256(repr(sorted(asdict(self).items())).encode("utf-8")).hexdigest()[:16]


SETTINGS = GeometryEvaluationSettings()


def v3_settings(s=SETTINGS):
    """Settings record written next to v3 labels; its signature identifies the method."""
    out = {**asdict(s), 'evaluator_version': 3, 'registration_method': REGISTRATION_METHOD,
           'scale_check_tolerance': SCALE_TOLERANCE}
    out['signature'] = hashlib.sha256(repr(sorted(out.items())).encode()).hexdigest()[:16]
    return out


# ---------------------------------------------------------------- meshes

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


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.einsum("ni,ji->nj", points, transform[:3, :3]) + transform[:3, 3]


def bbox_diagonal(mesh: TriangleMesh) -> float:
    return float(np.linalg.norm(np.ptp(mesh.vertices, axis=0)))


# ---------------------------------------------------------------- registration

def similarity_fit(source, target, with_scale):
    """Least-squares rotation + translation (+ uniform scale) mapping source -> target (Umeyama)."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    A, B = source - mu_s, target - mu_t
    U, S, Vt = np.linalg.svd(np.einsum('ni,nj->ij', A, B, optimize=False))
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    s = float((S * np.diag(D)).sum() / (A ** 2).sum()) if with_scale else 1.0
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = mu_t - s * R @ mu_s
    return T


def scale_of(T):
    return float(np.cbrt(np.linalg.det(np.asarray(T)[:3, :3])))


def centroid_start(source, target):
    """v1/v2 initial transform: translate the source centroid onto the target centroid."""
    T = np.eye(4)
    T[:3, 3] = target.mean(0) - source.mean(0)
    return T


def icp(source, target, T0, max_distance, max_iterations, rel_tol, abs_tol, with_scale):
    """Point-to-point ICP from T0; with_scale also solves one uniform scale per iteration."""
    T = np.array(T0, dtype=float)
    tree = cKDTree(target)
    prev, it, converged = np.inf, 0, False
    for it in range(1, max_iterations + 1):
        moved = source @ T[:3, :3].T + T[:3, 3]
        dist, idx = tree.query(moved, workers=1)
        ok = dist <= max_distance
        if ok.sum() < 3:
            break
        T = similarity_fit(moved[ok], target[idx[ok]], with_scale) @ T
        rmse = float(np.sqrt(np.mean(dist[ok] ** 2)))
        if rmse <= abs_tol or (np.isfinite(prev) and abs(prev - rmse) <= rel_tol * max(prev, 1e-15)):
            converged = True
            break
        prev = rmse
    dist, _ = tree.query(source @ T[:3, :3].T + T[:3, 3], workers=1)
    return {'transform': T, 'fitness': float((dist <= max_distance).mean()),
            'iterations': it, 'converged': converged, 'scale': scale_of(T)}


def qc(fitness):
    return next(name for lo, name in QC if fitness >= lo)


# ---------------------------------------------------------------- cameras

def load_cameras(path):
    """Aligned camera centres from a cameras.csv, keyed by image label."""
    with open(path, newline='') as f:
        return {r['label']: np.array([float(r['x']), float(r['y']), float(r['z'])])
                for r in csv.DictReader(f) if r['x']}


def camera_transform(cand, ref):
    """Similarity transform of candidate cameras onto reference cameras (shared labels).

    Returns (transform, shared image count, residual RMS / reference camera spread).
    """
    k = sorted(set(cand) & set(ref))
    A, B = np.array([cand[i] for i in k]), np.array([ref[i] for i in k])
    T = similarity_fit(A, B, True)
    resid = np.sqrt((((A @ T[:3, :3].T + T[:3, 3]) - B) ** 2).sum(1).mean()) \
        / np.sqrt(((B - B.mean(0)) ** 2).sum(1).mean())
    return T, len(k), float(resid)


# ---------------------------------------------------------------- scoring

def surface_distances(ref_pts, cand_pts, T, diag, cutoff, s=SETTINGS):
    """Distance summaries after moving candidate samples by T; 'y' is the label."""
    cand_pts = apply_transform(cand_pts, T)
    c2r, _ = cKDTree(ref_pts).query(cand_pts, workers=1)
    r2c, _ = cKDTree(cand_pts).query(ref_pts, workers=1)
    sym = np.concatenate((c2r, r2c))
    far = sym > cutoff
    return {
        'y': float(np.sqrt((np.mean(c2r ** 2) + np.mean(r2c ** 2)) / 2) / diag),
        'c2r_rmse_norm': float(np.sqrt(np.mean(c2r ** 2)) / diag),
        'r2c_rmse_norm': float(np.sqrt(np.mean(r2c ** 2)) / diag),
        'c2r_p95_norm': float(np.percentile(c2r, 95) / diag),
        'r2c_p95_norm': float(np.percentile(r2c, 95) / diag),
        'c2r_frac_beyond_cutoff': float((c2r > cutoff).mean()),
        'r2c_frac_beyond_cutoff': float((r2c > cutoff).mean()),
        'far_share_of_sq_error': float((sym[far] ** 2).sum() / (sym ** 2).sum()),
        **{f'{d}_at_{f:g}': float(np.mean(v <= f * diag))
           for f in s.threshold_fractions_of_reference_diagonal
           for d, v in (('accuracy', c2r), ('completeness', r2c))},
    }


class Reference:
    """A reference mesh and its cameras, with the samples every candidate is scored against."""

    def __init__(self, mesh, cameras, s=SETTINGS):
        self.mesh, self.cameras, self.s = mesh, cameras, s
        self.diag = bbox_diagonal(mesh)
        self.cutoff = s.icp_max_correspondence_distance_fraction * self.diag
        self.reg_pts = sample_surface(mesh, s.registration_sample_count, s.random_seed)
        self.eval_pts = sample_surface(mesh, s.evaluation_sample_count, s.random_seed + 1)

    @classmethod
    def from_files(cls, mesh_path, cameras_path, s=SETTINGS):
        return cls(load_obj_mesh(mesh_path), load_cameras(cameras_path), s)


def score_candidate(ref: Reference, cand_mesh, cand_cameras):
    """v3 label and diagnostics for one candidate.

    Returns (row, transforms). The row holds the label ('label_v3'), registration QC, the
    camera/ICP scale check, distance summaries, and two checks that are not the label:
    rigid refinement at the camera scale, and LONG_ITERATIONS of similarity ICP.
    """
    s = ref.s
    reg_cand = sample_surface(cand_mesh, s.registration_sample_count, s.random_seed)
    eval_cand = sample_surface(cand_mesh, s.evaluation_sample_count, s.random_seed + 1)
    T0, n_shared, resid = camera_transform(cand_cameras, ref.cameras)
    init_d, _ = cKDTree(ref.reg_pts).query(apply_transform(reg_cand, T0), workers=1)
    args = (ref.cutoff, s.icp_max_iterations, s.icp_relative_tolerance, s.icp_absolute_tolerance_fraction * ref.diag)
    sim = icp(reg_cand, ref.reg_pts, T0, *args, with_scale=True)
    rig = icp(reg_cand, ref.reg_pts, T0, *args, with_scale=False)
    long = icp(reg_cand, ref.reg_pts, T0, ref.cutoff, LONG_ITERATIONS, *args[2:], with_scale=True)
    m = surface_distances(ref.eval_pts, eval_cand, sim['transform'], ref.diag, ref.cutoff, s)
    m_rig = surface_distances(ref.eval_pts, eval_cand, rig['transform'], ref.diag, ref.cutoff, s)
    m_long = surface_distances(ref.eval_pts, eval_cand, long['transform'], ref.diag, ref.cutoff, s)
    cam_scale = scale_of(T0)
    ratio = sim['scale'] / cam_scale
    row = {
        'label_v3': m['y'], 'qc_v3': qc(sim['fitness']), 'fitness_v3': sim['fitness'],
        'fitness_at_camera_init': float((init_d <= ref.cutoff).mean()),
        'icp_scale_v3': sim['scale'], 'camera_scale': cam_scale, 'scale_ratio': ratio,
        'scale_check': 'agree' if abs(ratio - 1) <= SCALE_TOLERANCE else 'disagree',
        'iterations_v3': sim['iterations'], 'converged_v3': sim['converged'],
        'camera_shared_images': n_shared, 'camera_shape_residual': resid,
        **{k: v for k, v in m.items() if k != 'y'},
        'label_v3_rigid_camera_scale': m_rig['y'], 'fitness_rigid_camera_scale': rig['fitness'],
        f'label_v3_{LONG_ITERATIONS}it': m_long['y'], f'scale_v3_{LONG_ITERATIONS}it': long['scale'],
        f'converged_v3_{LONG_ITERATIONS}it': long['converged'],
    }
    transforms = {'camera_init_transform': T0, 'transform_v3': sim['transform'],
                  'transform_rigid_camera_scale': rig['transform']}
    return row, transforms
