"""Export reconstruction runs for the web viewer (viewer/index.html), and serve it locally.

Two kinds of export, both written to viewer/data/<name>/ (scene.json plus one .bin per mesh):

  build    One run on its own: its mesh, its cameras and its coverage gaps, in its own frame.
           Needs no reference, so it works for a capture built from scratch. With --reference it
           becomes a one-run comparison (below). An alignment-only run (no raw-mesh.obj) still
           exports its cameras and coverage.
  compare  Several runs of one object against a reference run. Each candidate is registered onto
           the reference with the v3 scorer (the same transform as the labels), then per vertex:
             missing  reference vertex -> nearest candidate surface   (where geometry was lost)
             extra    candidate vertex -> nearest reference surface    (where geometry is wrong)

Distances are fractions of the reference bounding-box diagonal, quantised to one byte over
[0, CUTOFF] (the scorer's ICP cutoff, 5% of the diagonal). Meshes go out in a display frame built
from the cameras (coverage.camera_frame): centre at the origin, "up" along +y, one unit = the
(reference) mesh's bounding-box diagonal.

The pipeline calls export_build after every mesh run (scripts/run_watched.py and the capture app);
by hand:

    python -m bd3d_quality.viewer_export build results/x/run1 [--reference results/x/ref] --open
    python -m bd3d_quality.viewer_export compare results/pahma-5-613-v2 --reference ref145b \\
        --runs side40_s1 drop_s2 --title "PAHMA 5-613 coffin" --open
    python -m bd3d_quality.viewer_export serve
"""
import argparse
import csv
import json
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import numpy as np

from . import coverage
from .evaluation import (Reference, TriangleMesh, bbox_diagonal, load_cameras as load_camera_centres,
                         load_obj_mesh, sample_surface, score_candidate)

REPO = Path(__file__).resolve().parents[2]
VIEWER = REPO / 'viewer'
DEFAULT_OUT = VIEWER / 'data'
CUTOFF = 0.05          # distances above this fraction of the diagonal saturate
SURFACE_SAMPLES = 600_000
FORMAT = 2


# ---------------------------------------------------------------- geometry helpers

def surface_points(mesh, seed):
    """Dense surface samples plus the vertices, so distances to a surface are not to a sparse cloud."""
    return np.vstack([sample_surface(mesh, SURFACE_SAMPLES, seed), mesh.vertices])


def distance_bytes(points, surface, diag):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(surface).query(points, workers=-1)
    return np.round(np.clip(d / diag / CUTOFF, 0, 1) * 255).astype(np.uint8)


def to_display(P, centre, B, diag):
    """Camera frame -> display frame: (e1, e2, up) becomes (x, -z, y), scaled by the diagonal."""
    Q = (P - centre) @ B.T / diag
    return np.stack([Q[:, 0], Q[:, 2], -Q[:, 1]], axis=1)


def dirs_to_display(V, B):
    W = V @ B.T
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    return np.stack([W[:, 0], W[:, 2], -W[:, 1]], axis=1)


def quantise(X, lo, hi):
    return np.round((np.clip(X, lo, hi) - lo) / (hi - lo) * 65535).astype(np.uint16)


def display_bounds(*arrays):
    allv = np.vstack(arrays)
    return np.percentile(allv, 0.01, axis=0) - 0.05, np.percentile(allv, 99.99, axis=0) + 0.05


class Packer:
    """Concatenates typed arrays into one .bin, 4-byte aligned, and records where each one is."""

    def __init__(self):
        self.parts, self.offset, self.index = [], 0, {}

    def add(self, name, arr):
        b = np.ascontiguousarray(arr).tobytes()
        self.index[name] = {'offset': self.offset, 'count': int(arr.size), 'type': str(arr.dtype)}
        self.parts.append(b)
        self.offset += len(b)
        pad = (-self.offset) % 4
        if pad:
            self.parts.append(b'\0' * pad)
            self.offset += pad

    def write(self, path):
        path.write_bytes(b''.join(self.parts))
        return self.index


# ---------------------------------------------------------------- run metadata

def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def run_facts(run_dir, label_row=None):
    """Reprojection RMSE and build time: from labels-all.csv when given, else from the run's own files."""
    run_dir = Path(run_dir)
    label_row = label_row or {}
    metrics = _read_json(run_dir / 'metrics-initial_alignment.json').get('metrics', {})
    res = _read_json(run_dir.with_name(run_dir.name + '-resources.json'))
    return {
        'reprojection_px': _num(label_row.get('run_reprojection_rmse_px')) or _num(metrics.get('reprojection_rmse_px')),
        'build_seconds': _num(label_row.get('elapsed_seconds')) or _num(res.get('elapsed_seconds')),
        'tie_points': _num(metrics.get('valid_tie_point_count')),
    }


def read_labels(folder):
    path = Path(folder) / 'labels-all.csv'
    if not path.is_file():
        return {}
    with open(path, newline='') as f:
        return {r['run']: r for r in csv.DictReader(f)}


def coverage_json(cov):
    return {'counts': cov.counts.tolist(), 'missing': cov.missing.tolist(),
            'never': cov.never.tolist(), 'summary': cov.summary}


# ---------------------------------------------------------------- output

def _finish(out_root, name, scene, log):
    """Write scene.json last (atomically), drop stale .bin files, and list the export in index.json."""
    out = out_root / name
    tmp = out / 'scene.json.tmp'
    tmp.write_text(json.dumps(scene, separators=(',', ':')))
    tmp.replace(out / 'scene.json')
    keep = {r['file'] for r in scene['runs'] if r.get('file')}
    if scene.get('reference_mesh'):
        keep.add(scene['reference_mesh']['file'])
    for f in out.glob('*.bin'):
        if f.name not in keep:
            f.unlink()
    listing_path = out_root / 'index.json'
    try:
        listing = [o for o in json.loads(listing_path.read_text()) if o['name'] != name]
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        listing = []
    listing.insert(0, {'name': name, 'title': scene['title'], 'mode': scene['mode'],
                       'runs': len(scene['runs']), 'updated': datetime.now(timezone.utc).isoformat(timespec='seconds')})
    tmp = out_root / 'index.json.tmp'
    tmp.write_text(json.dumps(listing, indent=1))
    tmp.replace(listing_path)
    log(f'viewer: wrote {out}')
    return out


def export_build(run_dir, reference_dir=None, *, out=DEFAULT_OUT, name=None, title=None, log=print):
    """Export one run. Without a reference: the run alone in its own frame. Returns the output folder."""
    run_dir = Path(run_dir)
    if reference_dir is not None:
        return export_comparison([run_dir], reference_dir, out=out, name=name or run_dir.name,
                                 title=title or f'{run_dir.name} vs {Path(reference_dir).name}', log=log)
    name = name or run_dir.name
    out = Path(out)
    (out / name).mkdir(parents=True, exist_ok=True)

    labels, aligned, P, V = coverage.load_cameras(run_dir / 'cameras.csv')
    if aligned.sum() < 3:
        raise ValueError(f'{run_dir}: fewer than 3 aligned cameras, nothing to show')
    centre, B = coverage.camera_frame(P[aligned], V[aligned])
    mesh_path = run_dir / 'raw-mesh.obj'
    mesh = load_obj_mesh(mesh_path) if mesh_path.is_file() else None
    if mesh is not None:
        diag, unit = bbox_diagonal(mesh), 'mesh bounding-box diagonal'
    else:   # no mesh yet: scale so the cameras sit about where they would around a mesh
        diag, unit = 0.5 * float(np.linalg.norm(np.ptp(P[aligned], axis=0))), 'half the camera spread'

    cams = to_display(P[aligned], centre, B, diag)
    run = {'id': run_dir.name, 'file': None, 'buffers': None,
           'images': len(labels), 'aligned': int(aligned.sum()),
           'cameras': ['used'] * int(aligned.sum()),
           'failed': [l for l, ok in zip(labels, aligned) if not ok],
           'label': None, 'qc': None, 'completeness': None, 'accuracy': None,
           'missing_rmse': None, 'extra_rmse': None, **run_facts(run_dir)}
    if mesh is not None:
        V_disp = to_display(mesh.vertices, centre, B, diag)
        lo, hi = display_bounds(V_disp)
        pk = Packer()
        pk.add('positions', quantise(V_disp, lo, hi))
        pk.add('indices', mesh.triangles.astype(np.uint32))
        run.update(file=f'{run_dir.name}.bin', buffers=pk.write(out / name / f'{run_dir.name}.bin'),
                   vertices=int(len(mesh.vertices)), faces=int(len(mesh.triangles)))
    else:
        lo, hi = display_bounds(cams)

    cov = coverage.analyze(run_dir / 'cameras.csv')
    run['coverage'] = coverage_json(cov)
    scene = {
        'format': FORMAT, 'mode': 'build', 'name': name, 'title': title or run_dir.name,
        'reference': None, 'cutoff': CUTOFF, 'unit': unit, 'bounds': [lo.tolist(), hi.tolist()],
        'reference_mesh': None, 'reference_images': len(labels),
        'camera_labels': [l for l, ok in zip(labels, aligned) if ok],
        'camera_positions': np.round(cams, 5).tolist(),
        'camera_views': np.round(dirs_to_display(V[aligned], B), 5).tolist(),
        'grid': {'az_edges': cov.az_edges.tolist(), 'el_edges': cov.el_edges.tolist(), 'counts': cov.counts.tolist()},
        'runs': [run],
    }
    return _finish(out, name, scene, log)


def export_comparison(run_dirs, reference_dir, *, out=DEFAULT_OUT, name, title=None, log=print):
    """Export candidate runs registered onto a reference run. Returns the output folder."""
    run_dirs, ref_dir, out = [Path(d) for d in run_dirs], Path(reference_dir), Path(out)
    (out / name).mkdir(parents=True, exist_ok=True)
    labels = {}
    for d in {d.parent for d in run_dirs}:
        labels.update(read_labels(d))

    t0 = time.time()
    log(f'viewer: reference {ref_dir.name}: loading mesh')
    ref = Reference.from_files(ref_dir / 'raw-mesh.obj', ref_dir / 'cameras.csv')
    r_labels, r_aligned, rP, rV = coverage.load_cameras(ref_dir / 'cameras.csv')
    centre, B = coverage.camera_frame(rP[r_aligned], rV[r_aligned])
    ref_surface = surface_points(ref.mesh, 1)

    # Candidates first: their registered meshes set the shared quantisation box.
    runs, meshes = [], {}
    for d in run_dirs:
        log(f'viewer: {d.name}: registering ({time.time() - t0:.0f}s)')
        mesh = load_obj_mesh(d / 'raw-mesh.obj')
        row, transforms = score_candidate(ref, mesh, load_camera_centres(d / 'cameras.csv'))
        T = transforms['transform_v3']
        meshes[d] = (mesh.vertices @ T[:3, :3].T + T[:3, 3], mesh.triangles)
        runs.append((d, row))

    disp_ref = to_display(ref.mesh.vertices, centre, B, ref.diag)
    lo, hi = display_bounds(disp_ref, *[to_display(v, centre, B, ref.diag) for v, _ in meshes.values()])

    pk = Packer()
    pk.add('positions', quantise(disp_ref, lo, hi))
    pk.add('indices', ref.mesh.triangles.astype(np.uint32))
    ref_index = pk.write(out / name / 'reference.bin')

    scene_runs = []
    ref_cams = [l for l, ok in zip(r_labels, r_aligned) if ok]
    for d, row in runs:
        log(f'viewer: {d.name}: distances ({time.time() - t0:.0f}s)')
        V_ref, tris = meshes[d]
        cand_surface = surface_points(TriangleMesh(V_ref, tris), 2)
        pk = Packer()
        pk.add('positions', quantise(to_display(V_ref, centre, B, ref.diag), lo, hi))
        pk.add('indices', tris.astype(np.uint32))
        pk.add('extra', distance_bytes(V_ref, ref_surface, ref.diag))
        pk.add('missing', distance_bytes(ref.mesh.vertices, cand_surface, ref.diag))
        index = pk.write(out / name / f'{d.name}.bin')

        c_labels, c_aligned, _, _ = coverage.load_cameras(d / 'cameras.csv')
        ok_set = {l for l, ok in zip(c_labels, c_aligned) if ok}
        bad_set = {l for l, ok in zip(c_labels, c_aligned) if not ok}
        state = ['used' if l in ok_set else 'unaligned' if l in bad_set else 'removed' for l in ref_cams]
        cov = coverage.compare(d / 'cameras.csv', ref_dir / 'cameras.csv')
        scene_runs.append({
            'id': d.name, 'file': f'{d.name}.bin', 'buffers': index,
            'images': len(c_labels), 'aligned': len(ok_set), 'cameras': state, 'failed': sorted(bad_set),
            'label': row['label_v3'], 'qc': row['qc_v3'],
            'completeness': row.get('completeness_at_0.01'), 'accuracy': row.get('accuracy_at_0.01'),
            'missing_rmse': row['r2c_rmse_norm'], 'extra_rmse': row['c2r_rmse_norm'],
            'vertices': int(len(V_ref)), 'faces': int(len(tris)),
            **run_facts(d, labels.get(d.name)),
            'coverage': coverage_json(cov),
        })
        log(f'viewer: {d.name}: label {row["label_v3"]:.4f}, {len(ok_set)} cameras')

    ref_cov = coverage.analyze(ref_dir / 'cameras.csv')
    scene = {
        'format': FORMAT, 'mode': 'compare', 'name': name, 'title': title or name, 'reference': ref_dir.name,
        'cutoff': CUTOFF, 'unit': 'reference bounding-box diagonal', 'bounds': [lo.tolist(), hi.tolist()],
        'reference_mesh': {'file': 'reference.bin', 'buffers': ref_index,
                           'vertices': int(len(ref.mesh.vertices)), 'faces': int(len(ref.mesh.triangles))},
        'reference_images': len(r_labels),
        'camera_labels': ref_cams,
        'camera_positions': np.round(to_display(rP[r_aligned], centre, B, ref.diag), 5).tolist(),
        'camera_views': np.round(dirs_to_display(rV[r_aligned], B), 5).tolist(),
        'grid': {'az_edges': ref_cov.az_edges.tolist(), 'el_edges': ref_cov.el_edges.tolist(),
                 'counts': ref_cov.counts.tolist()},
        'runs': sorted(scene_runs, key=lambda r: r['label']),
    }
    log(f'viewer: done in {time.time() - t0:.0f}s')
    return _finish(out, name, scene, log)


# ---------------------------------------------------------------- local server

class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')   # re-exports show up on reload
        super().end_headers()


_server = None
_server_lock = threading.Lock()


def viewer_url(name=None, port=0):
    """URL of the viewer, starting a local server for viewer/ (127.0.0.1 only) on first use."""
    global _server
    with _server_lock:
        if _server is None:
            _server = ThreadingHTTPServer(('127.0.0.1', port), partial(_QuietHandler, directory=str(VIEWER)))
            threading.Thread(target=_server.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{_server.server_address[1]}/'
    return url + (f'?object={quote(name)}' if name else '')


def open_viewer(name=None):
    url = viewer_url(name)
    webbrowser.open(url)
    return url


# ---------------------------------------------------------------- command line

def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m bd3d_quality.viewer_export', description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build', help='one run, with or without a reference')
    b.add_argument('run', type=Path, help='run folder (cameras.csv, and raw-mesh.obj if meshed)')
    b.add_argument('--reference', type=Path, help='reference run folder (optional)')
    c = sub.add_parser('compare', help='several runs of one object against a reference run')
    c.add_argument('folder', type=Path, help='experiment results folder holding the run folders')
    c.add_argument('--reference', required=True, help='reference run folder name')
    c.add_argument('--runs', nargs='+', required=True, help='candidate run folder names')
    for p in (b, c):
        p.add_argument('--name', help='output name')
        p.add_argument('--title', default='', help='display title, e.g. "PAHMA 5-613 coffin"')
        p.add_argument('--out', type=Path, default=DEFAULT_OUT)
        p.add_argument('--open', action='store_true', help='serve the viewer and open it in the browser')
    s = sub.add_parser('serve', help='serve viewer/ on 127.0.0.1 and open it')
    s.add_argument('--port', type=int, default=8000)
    s.add_argument('--object', help='export name to open')
    a = ap.parse_args(argv)

    if a.command == 'serve':
        url = viewer_url(a.object, a.port)
        print(f'viewer at {url}  (Ctrl-C to stop)')
        webbrowser.open(url)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            return 0
    if a.command == 'build':
        needs = [(a.run, 'cameras.csv')]
        if a.reference:
            needs += [(a.run, 'raw-mesh.obj'), (a.reference, 'cameras.csv'), (a.reference, 'raw-mesh.obj')]
    else:
        needs = [(a.folder / r, f) for r in [a.reference, *a.runs] for f in ('cameras.csv', 'raw-mesh.obj')]
    missing = [str(Path(d) / f) for d, f in needs if not (Path(d) / f).is_file()]
    if missing:
        print('not found (give a run folder that holds these files):\n  ' + '\n  '.join(missing), file=sys.stderr)
        return 2
    if a.command == 'build':
        out = export_build(a.run, a.reference, out=a.out, name=a.name, title=a.title or None)
    else:
        name = a.name or next((r['artifact'] for r in read_labels(a.folder).values()), a.folder.name)
        out = export_comparison([a.folder / r for r in a.runs], a.folder / a.reference, out=a.out,
                                name=name, title=a.title or name)
    if a.open:
        if a.out.resolve() != DEFAULT_OUT.resolve():
            print('--open serves viewer/data only; this export went elsewhere', file=sys.stderr)
            return 0
        print(f'viewer at {open_viewer(out.name)}  (Ctrl-C to stop)')
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
