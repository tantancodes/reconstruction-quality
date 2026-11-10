"""Build a per-image Image/Quality cache from a completed reference run.

Reads RUN/cameras.csv (Image/Quality written by analyzeImages) and RUN/manifest.json,
and writes one row per image keyed by SHA-256, with the source run identity. Subset
runs pass it to `reconstruct --image-quality-cache` instead of re-running analyzeImages.
Image quality is computed per image, so the value does not depend on which other
images are in the chunk.

  python3 scripts/build_image_quality_cache.py results/pahma-5-613-v1/ref145b \
      --output results/pahma-5-613-v1/image_quality_cache.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bd3d_quality.metashape_runner import IMAGE_QUALITY_CACHE_COLUMNS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    run = json.loads((a.run / 'run.json').read_text())
    manifest = json.loads((a.run / 'manifest.json').read_text())
    if run['status'] != 'complete':
        raise SystemExit('source run is not complete')
    if run.get('image_quality_source', {}).get('method') == 'cache':
        raise SystemExit('source run itself used a cache; build from a run that ran analyzeImages')
    rows = list(csv.DictReader((a.run / 'cameras.csv').open(newline='')))
    if len(rows) != len(manifest['images']):
        raise SystemExit('cameras.csv rows differ from manifest images')
    out = []
    for row, entry in zip(rows, manifest['images']):
        if Path(entry['path']).stem != row['label']:
            raise SystemExit(f'order mismatch: {entry["path"]} vs {row["label"]}')
        if row['image_quality'] == '':
            raise SystemExit(f'no image quality for {row["label"]}: {row["missing_reasons"]}')
        out.append({'path': entry['path'], 'sha256': entry['sha256'], 'image_quality': row['image_quality'],
                    'source_run_id': run['run_id'], 'source_experiment_id': run['experiment_id'],
                    'metashape_version': run['software'].get('Metashape', '')})
    if a.output.exists():
        raise SystemExit(f'{a.output} exists; refusing to overwrite')
    with a.output.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=IMAGE_QUALITY_CACHE_COLUMNS)
        w.writeheader()
        w.writerows(out)
    print(f'wrote {len(out)} rows to {a.output}')


if __name__ == '__main__':
    main()
