"""cameras.csv from the 16 historical FAST_DENSE alignments, joined to their ICP labels.

Features and labels then come from the same alignment. For each label row:
  1. fingerprint the original .psx + .files (SHA-256 of every file);
  2. copy both into OUTPUT/<experiment_id>/ (original basename kept, because the
     .psx refers to its .files folder by name) and check the copy is identical;
  3. open the COPY with read_only=True and run collect_camera_table;
  4. write OUTPUT/<experiment_id>/cameras.csv = label keys + camera table + target;
  5. record the chunk frame (transform scale, markers, scale bars, CRS);
  6. re-fingerprint original and copy: both must be unchanged.
The original projects are only ever read. No alignment, no dense, no saving.

Needs the Metashape interpreter:
  PYTHONPATH=src /Users/tristan/Documents/ChatGPT/bd3d/venv312/bin/python -B \
      scripts/historical_camera_tables.py --labels <label csv> --output results/hist16-old-alignment
"""
import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from bd3d_quality.provenance import sha256_file
from bd3d_quality.reconstruction_metrics import CAMERA_TABLE_COLUMNS, collect_camera_table

KEYS = ['run_id', 'experiment_id', 'images_used_count']
# label-CSV column -> output column (run-level RMSE renamed so it can't clobber the per-camera one)
LABELS = {'geometry_y_normalized_symmetric_rmse': 'geometry_y_normalized_symmetric_rmse',
          'registration_qc_flag': 'registration_qc_flag',
          'registration_fitness': 'registration_fitness',
          'registration_rmse_normalized': 'registration_rmse_normalized',
          'reprojection_rmse_px': 'run_reprojection_rmse_px'}


def fingerprints(psx):
    psx = Path(psx)
    files = [psx] + sorted(p for p in psx.with_suffix('.files').rglob('*')
                           if p.is_file() and p.name != 'lock')
    return {p.relative_to(psx.parent).as_posix(): sha256_file(p) for p in files}


def frame_info(chunk, Metashape):
    t = chunk.transform
    info = {'transform_scale': None, 'transform_matrix_is_identity': None,
            'n_markers': len(chunk.markers),
            'n_markers_with_position': sum(m.position is not None for m in chunk.markers),
            'n_scalebars': len(chunk.scalebars),
            'n_scalebars_with_distance': sum(s.reference.distance is not None for s in chunk.scalebars),
            'crs': str(chunk.crs) if chunk.crs is not None else None}
    try:
        info['transform_scale'] = None if t.scale is None else float(t.scale)
        if t.matrix is not None:
            info['transform_matrix_is_identity'] = bool(t.matrix == Metashape.Matrix.Diag([1, 1, 1, 1]))
    except Exception as exc:  # record, never fatal
        info['transform_error'] = f'{type(exc).__name__}: {exc}'
    info['marker_scaled'] = info['n_scalebars_with_distance'] > 0 and info['transform_scale'] not in (None, 1.0)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()

    import Metashape
    print('Metashape', Metashape.app.version, 'activated', Metashape.app.activated)
    if not Metashape.app.activated:
        sys.exit('license unavailable or checkout denied; nothing copied')

    rows = list(csv.DictReader(a.labels.open(newline='')))
    a.output.mkdir(parents=True, exist_ok=False)
    summary = []
    for r in rows:
        src = Path(r['project_path'])
        dst_dir = a.output / r['experiment_id']
        dst = dst_dir / src.name
        rec = {k: r[k] for k in KEYS} | {'source_project': str(src)}
        before = fingerprints(src)
        dst_dir.mkdir()
        shutil.copy2(src, dst)
        shutil.copytree(src.with_suffix('.files'), dst.with_suffix('.files'),
                        ignore=shutil.ignore_patterns('lock'))
        if fingerprints(dst) != before:
            sys.exit(f'copy differs from original: {src}')

        doc = Metashape.Document()
        doc.open(str(dst), read_only=True)
        n = int(r['images_used_count'])
        chunks = [c for c in doc.chunks if len(c.cameras) == n]
        rec['n_chunks'] = len(doc.chunks)
        if len(chunks) != 1:
            sys.exit(f'{src}: expected exactly one chunk with {n} cameras, found {len(chunks)}')
        chunk = chunks[0]
        table = collect_camera_table(chunk)
        rec['frame'] = frame_info(chunk, Metashape)
        doc = None  # release the document before re-hashing

        with open(dst_dir / 'cameras.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=KEYS + CAMERA_TABLE_COLUMNS + list(LABELS.values()))
            w.writeheader()
            for row in table:
                w.writerow({k: r[k] for k in KEYS} | row | {out: r[k] for k, out in LABELS.items()})

        rec['original_unchanged'] = fingerprints(src) == before
        rec['copy_unchanged'] = fingerprints(dst) == before
        rec['rows'] = len(table)
        rec['aligned'] = sum(bool(x['aligned']) for x in table)
        rec['missing_reasons'] = dict(Counter(part.split(':')[0] for x in table
                                              for part in (x['missing_reasons'] or '').split('; ') if part))
        summary.append(rec)
        fr = rec['frame']
        print(f"{r['experiment_id']:52s} rows={rec['rows']:2d} aligned={rec['aligned']:2d} "
              f"missing={rec['missing_reasons'] or '-'} scale={fr['transform_scale']} "
              f"scalebars={fr['n_scalebars_with_distance']} markers={fr['n_markers']} "
              f"orig_ok={rec['original_unchanged']} copy_ok={rec['copy_unchanged']}", flush=True)
        if not (rec['original_unchanged'] and rec['copy_unchanged']):
            sys.exit('project files changed during read-only extraction; stopping')

    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with open(a.output / 'cameras_all.csv', 'w', newline='') as out:
        w = None
        for rec in summary:
            for row in csv.DictReader((a.output / rec['experiment_id'] / 'cameras.csv').open(newline='')):
                w = w or csv.DictWriter(out, fieldnames=list(row))
                if out.tell() == 0:
                    w.writeheader()
                w.writerow(row)
    print(f'\n{len(summary)} runs -> {a.output}')


if __name__ == '__main__':
    main()
