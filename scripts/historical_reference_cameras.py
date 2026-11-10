"""cameras.csv for the historical 50-image FAST_DENSE reference, from a read-only copy.

Same procedure as historical_camera_tables.py: fingerprint, copy, open the copy
read_only, collect_camera_table, re-fingerprint both. No processing, no saving.

  PYTHONPATH=src /Users/tristan/Documents/ChatGPT/bd3d/venv312/bin/python -B \
      scripts/historical_reference_cameras.py \
      --project ~/Documents/BD3D_Test_Output/ARF_Dog_FAST_DENSE_Geometry_50_v1/ARF_Dog_FAST_DENSE_Geometry_50_v1.psx \
      --output results/hist16-old-alignment/_reference
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from bd3d_quality.metashape_runner import write_camera_table
from bd3d_quality.reconstruction_metrics import collect_camera_table

sys.path.insert(0, str(Path(__file__).parent))
from historical_camera_tables import fingerprints, frame_info  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    import Metashape
    print('Metashape', Metashape.app.version, 'activated', Metashape.app.activated)
    if not Metashape.app.activated:
        sys.exit('license unavailable or checkout denied; nothing copied')
    src = a.project.expanduser().resolve()
    before = fingerprints(src)
    a.output.mkdir(parents=True, exist_ok=False)
    dst = a.output / src.name
    shutil.copy2(src, dst)
    shutil.copytree(src.with_suffix('.files'), dst.with_suffix('.files'), ignore=shutil.ignore_patterns('lock'))
    if fingerprints(dst) != before:
        sys.exit('copy differs from original')
    doc = Metashape.Document()
    doc.open(str(dst), read_only=True)
    if len(doc.chunks) != 1:
        sys.exit(f'expected one chunk, found {len(doc.chunks)}')
    chunk = doc.chunks[0]
    table = collect_camera_table(chunk)
    frame = frame_info(chunk, Metashape)
    doc = None
    write_camera_table(a.output / 'cameras.csv', table)
    ok = {'original_unchanged': fingerprints(src) == before, 'copy_unchanged': fingerprints(dst) == before}
    summary = {'source_project': str(src), 'rows': len(table),
               'aligned': sum(bool(r['aligned']) for r in table), 'frame': frame} | ok
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if not all(ok.values()):
        sys.exit('project files changed during read-only extraction')


if __name__ == '__main__':
    main()
