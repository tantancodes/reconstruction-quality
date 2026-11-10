"""Assign reference cameras to capture series from their poses, and compare with capture-time gaps.

Pose method (frame-independent):
  1. object centre c = least-squares point closest to all camera viewing rays;
  2. d_i = unit vector from c to camera i;
  3. up axis u = direction of least spread of the d_i (smallest PCA eigenvector),
     sign chosen so that the mean elevation is positive (cameras mostly above);
  4. elevation_i = asin(d_i . u); 1-D k-means (k clusters, quantile init) on elevation.
Capture-time method: split capture order at the (k-1) largest EXIF time gaps.
Clusters are relabelled 1..k by increasing mean elevation.

  python3 scripts/series_split_from_poses.py --cameras results/pahma-5-613-v1/ref145/cameras.csv \
      --images ~/Documents/bd3d-data/PAHMA-5-613 --k 3 --output results/pahma-5-613-v1/series_split
"""
import argparse
import csv
import datetime
import json
from pathlib import Path

import numpy as np


def load(path):
    rows = [r for r in csv.DictReader(open(path, newline='')) if r['x']]
    P = np.array([[float(r[k]) for k in 'xyz'] for r in rows])
    V = np.array([[float(r['view_' + k]) for k in 'xyz'] for r in rows])
    return [r['label'] for r in rows], P, V


def ray_centre(P, V):
    A, b = np.zeros((3, 3)), np.zeros(3)
    for p, v in zip(P, V):
        M = np.eye(3) - np.outer(v, v)
        A += M
        b += M @ p
    return np.linalg.solve(A, b)


def kmeans_1d(x, k, iters=100):
    c = np.quantile(x, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        lab = np.argmin(np.abs(x[:, None] - c[None]), 1)
        new = np.array([x[lab == j].mean() if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    order = np.argsort(c)
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    return remap[lab] + 1, c[order]


def exif_time(path):
    from PIL import Image
    e = Image.open(path).getexif()
    t = e.get_ifd(0x8769).get(36867) or e.get(306)
    return datetime.datetime.strptime(t, '%Y:%m:%d %H:%M:%S') if t else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cameras', type=Path, required=True)
    ap.add_argument('--images', type=Path)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    labels, P, V = load(a.cameras)
    c = ray_centre(P, V)
    D = P - c
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    w, U = np.linalg.eigh(np.cov(D.T))
    u = U[:, 0]
    if (D @ u).mean() < 0:
        u = -u
    elev = np.degrees(np.arcsin(np.clip(D @ u, -1, 1)))
    # Azimuth only for reporting: angle in the plane orthogonal to u.
    e1 = U[:, 2] - (U[:, 2] @ u) * u
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    az = np.degrees(np.arctan2(D @ e2, D @ e1)) % 360
    pose_series, centres = kmeans_1d(elev, a.k)

    time_series, gaps = None, []
    if a.images:
        ts = [exif_time(a.images.expanduser() / f'{l}.jpg') for l in labels]
        g = [((ts[i + 1] - ts[i]).total_seconds(), i + 1) for i in range(len(ts) - 1)]
        cuts = sorted(i for _, i in sorted(g, reverse=True)[:a.k - 1])
        gaps = [dict(after=labels[i - 1], before=labels[i], seconds=s) for s, i in sorted(g, reverse=True)[:a.k - 1]]
        time_series = np.searchsorted(cuts, np.arange(len(labels)), side='right') + 1
        # relabel capture-time groups by mean elevation so labels are comparable
        means = [elev[time_series == j].mean() for j in range(1, a.k + 1)]
        rank = {j + 1: r + 1 for r, j in enumerate(np.argsort(means))}
        time_series = np.array([rank[s] for s in time_series])

    a.output.mkdir(parents=True, exist_ok=False)
    with open(a.output / 'series.csv', 'w', newline='') as f:
        out = csv.writer(f)
        out.writerow(['capture_index', 'label', 'elevation_deg', 'azimuth_deg', 'series_pose', 'series_capture_time'])
        for i, l in enumerate(labels):
            out.writerow([i, l, round(float(elev[i]), 2), round(float(az[i]), 2), int(pose_series[i]),
                        '' if time_series is None else int(time_series[i])])
    summary = {'k': a.k, 'centre': c.tolist(), 'up_axis': u.tolist(), 'pca_eigenvalues': w.tolist(),
               'cluster_mean_elevation_deg': centres.round(2).tolist(),
               'pose_counts': {int(j): int((pose_series == j).sum()) for j in range(1, a.k + 1)},
               'pose_elevation_range_deg': {int(j): [round(float(elev[pose_series == j].min()), 1),
                                                     round(float(elev[pose_series == j].max()), 1)]
                                            for j in range(1, a.k + 1)},
               'pose_series_contiguous_in_capture_order': bool(all(
                   np.all(np.diff(np.where(pose_series == j)[0]) == 1) for j in range(1, a.k + 1)))}
    if time_series is not None:
        dis = [labels[i] for i in range(len(labels)) if pose_series[i] != time_series[i]]
        summary |= {'capture_time_counts': {int(j): int((time_series == j).sum()) for j in range(1, a.k + 1)},
                    'capture_time_gaps': gaps, 'disagreements': dis, 'agree': not dis}
    (a.output / 'series.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
