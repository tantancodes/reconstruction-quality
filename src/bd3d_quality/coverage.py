"""Measured camera coverage of one aligned run (cameras.csv), before dense reconstruction.

The frame follows notebooks/pipeline_walkthrough.ipynb: the centre is the point closest to all
viewing rays and "up" is the direction in which the camera directions spread least (the normal of
the camera rings), signed toward the cameras, then refined so a gap in one ring does not tilt it
(see camera_frame). Azimuth is measured about up, counter-clockwise seen
from above, with 0 deg at the first aligned camera (the Metashape frame has no natural origin).
Elevation is the angle above the plane through the centre perpendicular to up.

compare() measures a run against a reference run instead: the run's cameras are moved into the
reference frame with the v3 scorer's camera similarity fit (evaluation.camera_transform), and a
cell is missing only where the reference has 2+ cameras and the run does not.
"""
import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .evaluation import camera_transform

REQUIRED = ['label', 'aligned', 'x', 'y', 'z', 'view_x', 'view_y', 'view_z']
BAND_NAMES = {1: ['all'], 2: ['low', 'high'], 3: ['low', 'mid', 'high']}


@dataclass
class Coverage:
    labels: list            # aligned camera labels
    failed: list            # labels of cameras that did not align (no pose)
    positions: np.ndarray   # (n, 3) aligned camera centres in the up frame (e1, e2, up), centre at 0
    views: np.ndarray       # (n, 3) view directions in the same frame
    azimuth: np.ndarray     # (n,) degrees in [0, 360)
    elevation: np.ndarray   # (n,) degrees
    counts: np.ndarray      # (n_el, n_az) cameras per cell, row 0 = lowest band
    az_edges: np.ndarray
    el_edges: np.ndarray
    min_cameras: int
    missing: np.ndarray     # (n_el, n_az) bool: cell drawn red
    never: np.ndarray       # (n_el, n_az) bool: reference never covered it (grey); all False without a reference
    summary: str            # one line describing the missing regions
    reference: object = None  # comparison only: the reference Coverage (same frame and grid)
    fit: tuple = None         # comparison only: (shared cameras, residual RMS / reference camera spread)


def load_cameras(path):
    """Return (labels, aligned mask, positions, view directions); rows without a pose get NaN."""
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'{path} has no camera rows')
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise ValueError(f'{path} is missing columns: {", ".join(missing)}')
    labels = [r['label'] for r in rows]
    aligned = np.array([r['aligned'].strip() == 'True' for r in rows])

    def num(r, keys):
        return [float(r[k]) if r[k] not in ('', None) else np.nan for k in keys]
    P = np.array([num(r, ['x', 'y', 'z']) for r in rows])
    V = np.array([num(r, ['view_x', 'view_y', 'view_z']) for r in rows])
    aligned &= np.isfinite(P).all(axis=1) & np.isfinite(V).all(axis=1)
    return labels, aligned, P, V


def camera_frame(P, V):
    """Centre nearest all viewing rays and an orthonormal basis (rows e1, e2, up).

    How "up" is found, in plain terms. Seen from the object, each camera is a direction (a point on
    a unit sphere). A capture is a few horizontal rings of such points, so "up" is the axis along
    which the points vary least: they spread widely sideways but only a little in height. PCA picks
    that axis, which is what the walkthrough notebook does.

    PCA alone goes wrong when one ring has a gap on one side. Take the low ring with its western
    half removed: the remaining low cameras sit to the east, while the higher rings are complete.
    So "low" now goes together with "east" and "high" with "west", and an axis tilted from the
    low-east side to the high-west side sees less spread than the true vertical. PCA picks that
    tilted axis, about 10 degrees off for side40_s1, which is exactly the gap we want to measure.

    The fix: split the cameras into low/mid/high bands using the current "up", and measure each
    camera relative to the average of its own band rather than the average of all cameras. A
    lopsided band then cannot shift itself sideways relative to the other bands, so the
    "low goes with east" link disappears. Within a ring the height is constant, so the least-spread
    axis of what remains is the ring normal. A few rounds of re-banding and re-fitting settle it.
    """
    if len(P) < 3:
        raise ValueError(f'need at least 3 aligned cameras, got {len(P)}')
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    A, b = np.zeros((3, 3)), np.zeros(3)
    for p, v in zip(P, V):
        M = np.eye(3) - np.outer(v, v)
        A += M
        b += M @ p
    centre = np.linalg.lstsq(A, b, rcond=None)[0]
    D = P - centre
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    up = np.linalg.eigh(np.cov(D.T))[1][:, 0]
    up = up if (D @ up).mean() > 0 else -up
    for _ in range(5):  # refinement: centre per elevation band, refit (see docstring)
        h = D @ up
        band = np.digitize(h, np.linspace(h.min(), h.max(), 4)[1:-1])
        R = D.copy()
        for b in np.unique(band):
            R[band == b] -= R[band == b].mean(axis=0)
        u = np.linalg.eigh(R.T @ R)[1][:, 0]
        up = u if (D @ u).mean() > 0 else -u
    first = D[0] - (D[0] @ up) * up            # azimuth origin: the first aligned camera
    if np.linalg.norm(first) < 1e-9:           # first camera straight above: fall back to any horizontal axis
        first = np.linalg.svd(np.eye(3) - np.outer(up, up))[0][:, 0]
    e1 = first / np.linalg.norm(first)
    return centre, np.stack([e1, np.cross(up, e1), up])


def azimuth_elevation(Q):
    """Degrees for points Q already expressed in the (e1, e2, up) frame about the centre."""
    az = np.degrees(np.arctan2(Q[:, 1], Q[:, 0])) % 360
    el = np.degrees(np.arcsin(np.clip(Q[:, 2] / np.linalg.norm(Q, axis=1), -1, 1)))
    return az, el


def coverage_grid(az, el, n_az=12, n_el=3, el_edges=None):
    """Camera counts per azimuth x elevation cell; elevation bands split the captured range evenly
    unless el_edges is given (cameras outside it are not counted)."""
    if el_edges is None:
        lo, hi = float(el.min()), float(el.max())
        if hi - lo < 1.0:
            lo, hi = lo - 0.5, hi + 0.5
        el_edges = np.linspace(lo, hi, n_el + 1)
    else:  # cameras just outside given bands (fit residual, rounding) still count for the edge band
        lo, hi = el_edges[0], el_edges[-1]
        el = np.where((el < lo) & (el > lo - 1), lo, np.where((el > hi) & (el < hi + 1), hi, el))
    az_edges = np.linspace(0, 360, n_az + 1)
    counts, _, _ = np.histogram2d(el, az, bins=[el_edges, az_edges])
    return counts.astype(int), az_edges, el_edges


def _runs(empty):
    """Circular runs of True as (start, length) over a 1-D boolean array."""
    n = len(empty)
    if empty.all():
        return [(0, n)]
    start = int(np.argmin(empty))              # begin scanning at a covered cell so runs don't wrap
    runs, i = [], 0
    while i < n:
        j = (start + i) % n
        if empty[j]:
            k = i
            while k < n and empty[(start + k) % n]:
                k += 1
            runs.append((j, k - i))
            i = k
        else:
            i += 1
    return sorted(runs)


def describe_gaps(missing, az_edges):
    """Azimuth ranges per elevation band where missing is True, as text (None if nothing is missing)."""
    names = BAND_NAMES.get(len(missing), [f'band {i + 1}' for i in range(len(missing))])
    step = az_edges[1] - az_edges[0]
    parts = []
    for name, row in zip(names, missing):
        runs = _runs(np.asarray(row, bool))
        if not runs:
            continue
        if runs == [(0, len(row))]:
            parts.append(f'{name} cameras at every azimuth')
            continue
        spans = [f'{az_edges[s]:.0f}° to {(az_edges[s] + n * step) % 360 or 360:.0f}°' for s, n in runs]
        parts.append(f'{name} cameras from ' + ', '.join(spans))
    return '; '.join(parts) if parts else None


def _failed_note(labels, failed):
    return f' ({len(failed)} of {len(labels)} cameras failed to align)' if failed else ''


def _in_frame(P, V, centre, B, n_az, n_el, el_edges=None):
    Q, W = (P - centre) @ B.T, V @ B.T
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    az, el = azimuth_elevation(Q)
    counts, az_edges, el_edges = coverage_grid(az, el, n_az, n_el, el_edges)
    return Q, W, az, el, counts, az_edges, el_edges


def analyze(path, n_az=12, n_el=3, min_cameras=2):
    """Coverage of one run in its own frame: a cell is missing when it has fewer than min_cameras."""
    labels, aligned, P, V = load_cameras(path)
    centre, B = camera_frame(P[aligned], V[aligned])
    Q, W, az, el, counts, az_edges, el_edges = _in_frame(P[aligned], V[aligned], centre, B, n_az, n_el)
    failed = [l for l, a in zip(labels, aligned) if not a]
    missing = counts < min_cameras
    gaps = describe_gaps(missing, az_edges)
    summary = (f'Missing: {gaps}' if gaps else f'No gaps: every cell has {min_cameras}+ cameras.')
    return Coverage([l for l, a in zip(labels, aligned) if a], failed, Q, W, az, el, counts, az_edges,
                    el_edges, min_cameras, missing, np.zeros_like(missing), summary + _failed_note(labels, failed))


def compare(path, reference_path, n_az=12, n_el=3, min_cameras=2):
    """Coverage of a run measured in a reference run's frame (its centre, up, azimuth origin and
    elevation bands). Missing (red) = reference has min_cameras+ in the cell, this run has fewer;
    never (grey) = the reference has fewer too."""
    ref = analyze(reference_path, n_az, n_el, min_cameras)
    r_labels, r_aligned, rP, rV = load_cameras(reference_path)
    centre, B = camera_frame(rP[r_aligned], rV[r_aligned])
    labels, aligned, P, V = load_cameras(path)
    P, V = P[aligned], V[aligned]
    cand = {l: p for l, p in zip(np.array(labels)[aligned], P)}
    refd = {l: p for l, p in zip(np.array(r_labels)[r_aligned], rP[r_aligned])}
    if len(set(cand) & set(refd)) < 3:
        raise ValueError('run and reference share fewer than 3 aligned camera labels; cannot align them')
    T, shared, resid = camera_transform(cand, refd)
    sR = T[:3, :3]
    P_ref, V_ref = P @ sR.T + T[:3, 3], V @ sR.T          # view directions: rotation (+scale, normalised later)
    Q, W, az, el, counts, az_edges, el_edges = _in_frame(P_ref, V_ref, centre, B, n_az, n_el, ref.el_edges)
    never = ref.counts < min_cameras
    missing = ~never & (counts < min_cameras)
    failed = [l for l, a in zip(labels, aligned) if not a]
    gaps = describe_gaps(missing, az_edges)
    summary = (f'Missing vs reference: {gaps} (reference angles)' if gaps
               else 'Covers every cell the reference covers.')
    summary += _failed_note(labels, failed)
    return Coverage(list(np.array(labels)[aligned]), failed, Q, W, az, el, counts, az_edges, el_edges,
                    min_cameras, missing, never, summary, reference=ref, fit=(shared, resid))


def default_run():
    return Path(__file__).resolve().parents[2] / 'results' / 'pahma-5-613-v2' / 'side40_s1'


def default_reference():
    return default_run().parent / 'ref145b'
