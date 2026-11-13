"""Synthetic run folders for evaluator v3 tests (no Metashape, no private data).

The reference is a lumpy closed surface seen by a ring of cameras. The candidate is the
same surface with one patch missing and a small bump added, placed in a different frame
by a known similarity transform (150 degree rotation, scale 1.3, offset), the way an
unmarked Metashape reconstruction lands in an arbitrary local frame.
"""
import csv
import json

import numpy as np

ROTATION_DEG = 150.0
SCALE = 1.3
OFFSET = np.array([4.0, -2.0, 7.5])


def rotation(deg, axis=(0.3, -0.5, 0.8)):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def surface(n_lat=40, n_lon=64, bump=0.0):
    """Asymmetric closed surface: r depends on direction, so the registration is unique."""
    lat = np.linspace(0, np.pi, n_lat)
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    T, P = np.meshgrid(lat, lon, indexing='ij')
    d = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1)
    r = 1 + 0.25 * np.cos(2 * P) * np.sin(T) + 0.15 * np.cos(T) + 0.1 * np.sin(3 * P) * np.sin(T) ** 2
    r = r + bump * np.exp(-((T - 1.0) ** 2 + (P - 4.0) ** 2) / 0.05)
    v = (d * (r[..., None] * np.array([1.6, 1.0, 0.7]))).reshape(-1, 3)
    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            a, b = i * n_lon + j, i * n_lon + (j + 1) % n_lon
            c, e = a + n_lon, b + n_lon
            faces += [(a, c, b), (b, c, e)]
    return v, np.array(faces)


def cameras(n=24, radius=4.0):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.stack([radius * np.cos(ang), radius * np.sin(ang), 0.8 * np.sin(2 * ang)], 1)
    return [f'IMG_{i:03d}' for i in range(n)], pos


def write_obj(path, v, f):
    with open(path, 'w') as h:
        h.writelines(f'v {float(x)!r} {float(y)!r} {float(z)!r}\n' for x, y, z in v)
        h.writelines(f'f {a + 1} {b + 1} {c + 1}\n' for a, b, c in f)


def write_cameras(path, labels, pos, aligned):
    with open(path, 'w', newline='') as h:
        w = csv.writer(h)
        w.writerow(['label', 'aligned', 'x', 'y', 'z'])
        for lab, p, ok in zip(labels, pos, aligned):
            w.writerow([lab, ok, *(repr(float(c)) for c in p)] if ok else [lab, ok, '', '', ''])


def make_runs(root, scale=SCALE, camera_scale=None, drop_patch=True, bump=0.25):
    """Write root/ref and root/cand run folders; return their paths.

    camera_scale, when given, places the candidate cameras at a different scale from its
    mesh, so the camera-derived and ICP-derived scales disagree.
    """
    camera_scale = scale if camera_scale is None else camera_scale
    R = rotation(ROTATION_DEG)
    v, f = surface()
    labels, pos = cameras()
    ref, cand = root / 'ref', root / 'cand'
    for d, rid in ((ref, 'run-ref'), (cand, 'run-cand')):
        d.mkdir(parents=True)
        (d / 'run.json').write_text(json.dumps({'run_id': rid, 'experiment_id': f'bd3d-{rid}'}))
    write_obj(ref / 'raw-mesh.obj', v, f)
    write_cameras(ref / 'cameras.csv', labels, pos, [True] * len(labels))

    vc, fc = surface(bump=bump)
    if drop_patch:  # remove a patch on the +x side
        centre = vc[fc].mean(1)
        fc = fc[~((centre[:, 0] > 0.9) & (np.abs(centre[:, 2]) < 0.35))]
    write_obj(cand / 'raw-mesh.obj', scale * vc @ R.T + OFFSET, fc)
    aligned = [i % 4 != 0 for i in range(len(labels))]  # a quarter of the cameras not aligned
    write_cameras(cand / 'cameras.csv', labels, camera_scale * pos @ R.T + OFFSET, aligned)
    return ref, cand
