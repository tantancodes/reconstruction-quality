"""Smoke-check a run's cameras.csv against its metrics and, optionally, real Metashape.

Usage:
  python scripts/smoke_check_camera_table.py RUN_DIR [--plot PATH] [--no-metashape]

Part 1 (any Python with matplotlib): first rows, aligned fraction, missing-reason
counts, column ranges, x-vs-y / x-vs-z plot, and per-camera RMSE pooled against
collect_metrics' run-level RMSE.
Part 2 (Metashape interpreter, PYTHONPATH=src): opens RUN_DIR/project.psx
read-only and checks Matrix.row(), camera.meta missing-key behaviour and the
analyzeImages signature, recomputes the table, and verifies project files are
unchanged afterwards.

Exits 1 if any check fails.
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

NUMERIC = ['x', 'y', 'z', 'view_x', 'view_y', 'view_z',
           'n_projections', 'reprojection_rmse_px', 'image_quality']


def num(v):
    return float(v) if v != '' else None


def summarize(run, plot_path, failures):
    rows = list(csv.DictReader((run / 'cameras.csv').open(newline='')))
    print(','.join(rows[0]))
    for r in rows[:10]:
        print(','.join(r.values()))
    n = len(rows)
    aligned = sum(r['aligned'] == 'True' for r in rows)
    print(f'\naligned fraction: {aligned}/{n} = {aligned / n:.3f}')
    reasons = Counter(part for r in rows for part in filter(None, r['missing_reasons'].split('; ')))
    print('missing_reasons:', dict(reasons) or 'none (all rows complete)')
    for c in NUMERIC:
        v = [num(r[c]) for r in rows]
        vv = [x for x in v if x is not None]
        rng = f'min={min(vv):.4g} max={max(vv):.4g}' if vv else 'all blank'
        print(f'{c:22s} {rng} blank={v.count(None)}')
    norms = {round(math.dist((0, 0, 0), [num(r[k]) for k in ['view_x', 'view_y', 'view_z']]), 9)
             for r in rows if r['view_x']}
    print('view axis norms:', norms)

    # Per-camera RMSE pooled by n_projections must reproduce collect_metrics.
    m = json.load(open(run / 'metrics-initial_alignment.json'))['metrics']
    ok = [r for r in rows if r['reprojection_rmse_px']]
    nobs = sum(int(r['n_projections']) for r in ok)
    if nobs and m['reprojection_rmse_px'] is not None:
        pooled = math.sqrt(sum(int(r['n_projections']) * float(r['reprojection_rmse_px']) ** 2
                               for r in ok) / nobs)
        rel = abs(pooled - m['reprojection_rmse_px']) / m['reprojection_rmse_px']
        print(f'\npooled from cameras.csv: rmse={pooled!r} obs={nobs}')
        print(f"collect_metrics:          rmse={m['reprojection_rmse_px']!r} obs={m['reprojection_observation_count']}")
        print(f'relative diff: {rel:.1e}')
        if rel > 1e-9 or nobs != m['reprojection_observation_count']:
            failures.append('pooled per-camera RMSE does not match collect_metrics')
    else:
        print('\nRMSE consistency: skipped (no per-camera or run-level RMSE)')

    placed = [r for r in rows if r['x']]
    if placed:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        x, y, z = ([float(r[k]) for r in placed] for k in 'xyz')
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, other, name in [(axes[0], y, 'y'), (axes[1], z, 'z')]:
            ax.scatter(x, other, c=range(len(placed)), cmap='viridis', s=40)
            for xi, oi, r in zip(x, other, placed):
                ax.annotate(Path(r['label']).stem.replace('_MG_', ''), (xi, oi),
                            fontsize=7, xytext=(3, 3), textcoords='offset points')
            ax.set_xlabel('x (world, arbitrary scale)')
            ax.set_ylabel(f'{name} (world, arbitrary scale)')
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_title(f'camera x vs {name}')
            ax.grid(alpha=.3)
        fig.suptitle(f'{run.name}: camera centres (colour = manifest order)')
        fig.tight_layout()
        fig.savefig(plot_path, dpi=130)
        print('saved', plot_path)


def probe_metashape(run, failures):
    import numpy as np
    import Metashape
    from bd3d_quality.metashape_runner import project_fingerprints
    from bd3d_quality.reconstruction_metrics import collect_camera_table, _matrix4

    before = project_fingerprints(run)
    if before != json.load(open(run / 'run.json'))['project_files']:
        failures.append('project files already differ from run.json ledger; probe not run')
        return
    print('\nMetashape', Metashape.app.version, 'activated', Metashape.app.activated)
    doc = Metashape.Document()
    doc.open(str(run / 'project.psx'), read_only=True)
    chunk = doc.chunks[0]
    cam = next((c for c in chunk.cameras if c.transform is not None), chunk.cameras[0])

    print('\n[1] Matrix.row()')
    M = chunk.transform.matrix
    print(' chunk.transform.matrix type:', type(M).__name__, 'size:', M.size)
    r0 = list(M.row(0))
    print(' row(0):', r0, 'len', len(r0))
    if len(r0) != 4:
        failures.append('Matrix.row(0) did not return 4 values')
    try:
        print(' np.array(Matrix) directly ->', np.array(M, dtype=float).shape)
    except Exception as e:
        print(' np.array(Matrix) directly fails:', type(e).__name__, e)
    try:
        print(' Matrix @ Matrix ->', type(M @ cam.transform).__name__)
    except Exception as e:
        print(' Matrix @ Matrix fails:', type(e).__name__, e)
    prod_diff = pos_diff = 0.0
    for c in chunk.cameras:
        if c.transform is None:
            continue
        ours = _matrix4(M) @ _matrix4(c.transform)
        prod_diff = max(prod_diff, float(np.abs(ours - _matrix4(M * c.transform)).max()))
        pos_diff = max(pos_diff, float(np.abs(ours[:3, 3] - np.array(list(M.mulp(c.center)))).max()))
    print(f' numpy product vs Metashape M*T: max abs diff {prod_diff:.3g}; '
          f'position vs M.mulp(camera.center): {pos_diff:.3g}')
    if prod_diff > 1e-9 or pos_diff > 1e-9:
        failures.append('numpy pose product disagrees with Metashape')

    print('\n[2] camera.meta missing-key behaviour')
    print(' meta type:', type(cam.meta).__name__)
    q = cam.meta['Image/Quality']
    print(" meta['Image/Quality'] ->", repr(q), type(q).__name__)
    try:
        print(" meta['Not/AKey'] -> returns", repr(cam.meta['Not/AKey']))
    except Exception as e:
        print(" meta['Not/AKey'] -> raises", type(e).__name__, repr(e))
    try:
        print(' keys():', list(cam.meta.keys()))
    except Exception as e:
        print(' keys() fails:', type(e).__name__, e)

    print('\n[3] analyzeImages signature')
    print(' ', (Metashape.Chunk.analyzeImages.__doc__ or '').strip().replace('\n', '\n  '))
    print(' run.json warnings:', json.load(open(run / 'run.json'))['warnings'])

    print('\n[4] recompute table from reopened project vs cameras.csv')
    fresh = collect_camera_table(chunk)
    saved = list(csv.DictReader((run / 'cameras.csv').open(newline='')))
    if len(fresh) != len(saved):
        failures.append('recomputed table has a different row count')
    worst = {}
    for a, b in zip(fresh, saved):
        if a['label'] != b['label'] or str(a['aligned']) != b['aligned']:
            failures.append(f"label/aligned mismatch at {b['label']}")
        for k in NUMERIC:
            if (a[k] is None) != (b[k] == ''):
                failures.append(f"blank/value mismatch at {b['label']} {k}")
            elif a[k] is not None:
                # Re-reading the project gives last-digit float noise, so compare with a tolerance.
                rel = abs(a[k] - float(b[k])) / max(abs(float(b[k])), 1e-12)
                worst[k] = max(worst.get(k, 0.0), rel)
    print(' max relative diff:', {k: f'{v:.1e}' for k, v in worst.items()})
    if any(v > 1e-9 for v in worst.values()):
        failures.append('recomputed table differs from cameras.csv beyond float noise')

    del chunk, doc
    after = project_fingerprints(run)
    lock = (run / 'project.files' / 'lock').exists()
    print('\nproject files unchanged by read-only probe:', before == after, '| lock left behind:', lock)
    if before != after or lock:
        failures.append('read-only probe changed project files or left a lock')


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('run', type=Path, help='Run directory containing cameras.csv and run.json')
    p.add_argument('--plot', type=Path, help='Plot output path (default: RUN/cameras-xy-xz.png)')
    p.add_argument('--no-metashape', action='store_true', help='Skip the read-only Metashape probe')
    args = p.parse_args()
    run = args.run.resolve()
    failures = []
    summarize(run, args.plot or run / 'cameras-xy-xz.png', failures)
    if not args.no_metashape:
        probe_metashape(run, failures)
    print('\nRESULT:', 'PASS' if not failures else 'FAIL')
    for f in failures:
        print(' -', f)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
