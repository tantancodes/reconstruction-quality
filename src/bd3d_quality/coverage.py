"""Measured camera coverage of one aligned run (cameras.csv), before dense reconstruction.

The frame follows notebooks/pipeline_walkthrough.ipynb: the centre is the point closest to all
viewing rays and "up" is the direction in which the camera directions spread least (the normal of
the camera rings), signed toward the cameras, then refined so a gap in one ring does not tilt it
(see camera_frame). Azimuth is measured about up, counter-clockwise seen
from above, with 0 deg at the first aligned camera (the Metashape frame has no natural origin).
Elevation is the angle above the plane through the centre perpendicular to up.
"""
import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np

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
    summary: str            # one line describing the missing regions


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
    """Centre nearest all viewing rays and an orthonormal basis (rows e1, e2, up)."""
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
    # Refinement: a one-sided gap in one ring couples height with horizontal position and tilts the
    # global PCA normal toward the gap (~10 deg on side40_s1). Centring the directions per elevation
    # band removes that coupling; the smallest-variance axis of the residuals is the ring normal.
    for _ in range(5):
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


def coverage_grid(az, el, n_az=12, n_el=3):
    """Camera counts per azimuth x elevation cell; elevation bands split the captured range evenly."""
    lo, hi = float(el.min()), float(el.max())
    if hi - lo < 1.0:
        lo, hi = lo - 0.5, hi + 0.5
    az_edges = np.linspace(0, 360, n_az + 1)
    el_edges = np.linspace(lo, hi, n_el + 1)
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


def describe_gaps(counts, az_edges, min_cameras=2):
    """One line naming the azimuth ranges, per elevation band, with fewer than min_cameras cameras."""
    names = BAND_NAMES.get(len(counts), [f'band {i + 1}' for i in range(len(counts))])
    step = az_edges[1] - az_edges[0]
    parts = []
    for name, row in zip(names, counts):
        runs = _runs(row < min_cameras)
        if not runs:
            continue
        if runs == [(0, len(row))]:
            parts.append(f'{name} cameras at every azimuth')
            continue
        spans = [f'{az_edges[s]:.0f}° to {(az_edges[s] + n * step) % 360 or 360:.0f}°' for s, n in runs]
        parts.append(f'{name} cameras from ' + ', '.join(spans))
    if not parts:
        return f'No gaps: every cell has {min_cameras}+ cameras.'
    return 'Missing: ' + '; '.join(parts)


def analyze(path, n_az=12, n_el=3, min_cameras=2):
    labels, aligned, P, V = load_cameras(path)
    centre, B = camera_frame(P[aligned], V[aligned])
    Q, W = (P[aligned] - centre) @ B.T, V[aligned] @ B.T
    az, el = azimuth_elevation(Q)
    counts, az_edges, el_edges = coverage_grid(az, el, n_az, n_el)
    failed = [l for l, a in zip(labels, aligned) if not a]
    summary = describe_gaps(counts, az_edges, min_cameras)
    if failed:
        summary += f' ({len(failed)} of {len(labels)} cameras failed to align)'
    return Coverage([l for l, a in zip(labels, aligned) if a], failed, Q, W, az, el,
                    counts, az_edges, el_edges, min_cameras, summary)


def default_run():
    return Path(__file__).resolve().parents[2] / 'results' / 'pahma-5-613-v2' / 'side40_s1'
