"""Unattended overnight chain (2026-11-10). Run under `caffeinate -i`.

  0. wait for the already-running ref145b driver (--wait-pid) to exit
  1. coffin: image-quality cache -> pose series split (vs capture time) -> manifests ->
     a drop_s3, b drop_s2, d random30 x2, c side40_s1, e random60 (cache reused)
  2. dog (GPU d8): random15 x2, random20 x2, side25
  3. v3 scoring + one labels.csv per artifact
  4. pilot model comparison + results notebook (no Metashape; always runs)

Metashape jobs go through scripts/run_watched.py, which preflights the license before
each job (exit 3 = "license unavailable (check VPN)" -> all remaining Metashape jobs are
skipped) and stops a part on any watchdog/resource stop or failure (exit 2 -> next part).
Every step's outcome goes to <log-dir>/chain-status.json. Nothing is deleted.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENV = str(REPO / '.venv/bin/python')
# Local inputs, from the environment (see .env.example)
FORK = Path(os.environ.get('BD3D_FORK', 'bd3d')).expanduser()
COFFIN_ROOT = Path(os.environ.get('BD3D_COFFIN_PHOTOS', 'PAHMA-5-613')).expanduser()
DOG_ROOT = Path(os.environ.get('BD3D_DOG_PHOTOS', 'Jpg-processed')).expanduser()
V2 = REPO / 'results/pahma-5-613-v2'
REF = V2 / 'ref145b'
DOG_REF = REPO / 'results/gpu-pilot-d8/ref50'
DOG_OUT = REPO / 'results/gpu-pilot-d8-ext'
COFFIN_MEM = ['--swap-growth-mib', '6144', '--rss-gib', '6',
              '--memory-note', '145-image coffin reference on 16 GB Mac; slowdown acceptable']
COFFIN_ORDER = [  # (manifest stem, cap seconds)
    ('drop_s3', 5400), ('drop_s2', 5400), ('random30_s20261110', 1800), ('random30_s20261111', 1800),
    ('side40_s1', 6000), ('random60_s20261112', 3000)]
COFFIN_NOTE = ('5-613 coffin subsets at GPU d8; caps ~1.5-2x of image-count-scaled estimate from the '
               '145-image reference (drop/side runs 95-122 images, est 40-70 min; random30 ~10 min; random60 ~25 min)')
DOG_ORDER = [('random15_s20261110', 900), ('random15_s20261111', 900), ('random20_s20261110', 900),
             ('random20_s20261111', 900), ('side25', 1500)]
DOG_NOTE = 'dog side25 (38 images) at GPU d8; side40 (30 images) took ~7 min, ref50 ~15 min'
DOG_MEM = ['--swap-growth-mib', '2048', '--memory-note', 'GPU d8 dog pilot settings (2 GB swap growth, approved 2026-11-09)']

DRY = False
status = {'started_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'steps': []}


def log(msg):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}', flush=True)


def save():
    (LOG_DIR / 'chain-status.json').write_text(json.dumps(status, indent=2) + '\n')


def step(name, cmd, log_name=None, env_extra=None):
    """Run one command, logging to <log-dir>/<log_name>.log; record and return its exit code."""
    t = time.time()
    log(f'step {name}: {" ".join(map(str, cmd))[:300]}')
    if DRY:
        status['steps'].append({'step': name, 'dry_run': True})
        return 0
    env = {**os.environ, 'PYTHONPATH': str(REPO / 'src'), **(env_extra or {})}
    with open(LOG_DIR / f'{log_name or name}.log', 'a') as f:
        try:
            code = subprocess.run([str(c) for c in cmd], cwd=REPO, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        except Exception:
            f.write(traceback.format_exc())
            code = -1
    status['steps'].append({'step': name, 'exit_code': code, 'seconds': round(time.time() - t, 1),
                            'finished_at': time.strftime('%H:%M:%S')})
    save()
    log(f'step {name}: exit {code} ({time.time() - t:.0f}s)')
    return code


def note(key, value):
    status[key] = value
    save()
    log(f'{key}: {value}')


def complete(run_dir):
    p = run_dir / 'run.json'
    return p.exists() and json.loads(p.read_text()).get('status') == 'complete' and (run_dir / 'raw-mesh.obj').exists()


def completed_runs(folder, stems):
    """Final completed directory per manifest stem (base or -retryN)."""
    out = []
    for stem in stems:
        for d in [folder / stem] + [folder / f'{stem}-retry{n}' for n in (1, 2, 3)]:
            if complete(d):
                out.append(d)
                break
    return out


def run_part(name, manifests, caps, root, out_dir, extra):
    note(f'{name}_manifests', [m.stem for m in manifests])
    code = step(f'{name}_metashape', [sys.executable, 'scripts/run_watched.py', '--manifests', *manifests,
                                      '--root', root, '--output-dir', out_dir, '--seconds', *caps, *extra])
    return {0: 'complete', 3: 'license_unavailable'}.get(code, 'stopped')


def main():
    global LOG_DIR, DRY
    ap = argparse.ArgumentParser()
    ap.add_argument('--wait-pid', type=int)
    ap.add_argument('--log-dir', type=Path, default=REPO / 'results/overnight-2026-11-10')
    ap.add_argument('--dry-run', action='store_true', help='print the steps only (assumes ref145b complete)')
    a = ap.parse_args()
    LOG_DIR, DRY = a.log_dir, a.dry_run
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    save()
    metashape_allowed = True

    # 0. wait for ref145b
    if a.wait_pid:
        log(f'waiting for ref145b driver pid {a.wait_pid}')
        while True:
            try:
                os.kill(a.wait_pid, 0)
            except ProcessLookupError:
                break
            except PermissionError:
                pass
            time.sleep(30)
    ps = V2 / 'ref145b-pilot-status.json'
    note('ref145b_status', json.loads(ps.read_text()) if ps.exists() else 'no pilot-status file')
    pre = V2 / 'license-preflight.log'
    if pre.exists() and 'license unavailable' in (pre.read_text().splitlines() or [''])[-1]:
        metashape_allowed = False
    ref_ok = complete(REF) or DRY
    note('ref145b_complete', ref_ok)

    # 1. coffin
    part1 = 'not_run'
    if ref_ok:
        cache = V2 / 'image_quality_cache.csv'
        split = V2 / 'series_split'
        cands = REPO / 'experiments/pahma-5-613/candidates'
        ok = step('p1_cache', [sys.executable, 'scripts/build_image_quality_cache.py', REF, '--output', cache]) == 0
        ok = ok and step('p1_series_split', [sys.executable, 'scripts/series_split_from_poses.py', '--cameras',
                                             REF / 'cameras.csv', '--images', COFFIN_ROOT, '--k', '3',
                                             '--output', split]) == 0
        if ok and (split / 'series.csv').exists():
            rows = list(csv.DictReader(open(split / 'series.csv', newline='')))
            agree = sum(r['series_pose'] == r['series_capture_time'] for r in rows)
            sizes = {s: sum(r['series_pose'] == s for r in rows) for s in '123'}
            note('series_split', {'pose_vs_capture_time_agree': f'{agree}/{len(rows)}', 'pose_series_sizes': sizes,
                                  'used': 'series_pose'})
        ok = ok and step('p1_manifests', [sys.executable, 'scripts/make_pahma_manifests.py', '--parent',
                                          'experiments/pahma-5-613/parent-145.json', '--root', COFFIN_ROOT,
                                          '--series', split / 'series.csv', '--column', 'series_pose',
                                          '--seeds', '20261110', '20261111', '--output', cands]) == 0
        if not ok:
            part1 = 'setup_failed'
        elif not metashape_allowed:
            part1 = 'skipped_license_unavailable'
        else:
            part1 = run_part('p1', [cands / f'{m}.json' for m in dict(COFFIN_ORDER)],
                             [str(c) for _, c in COFFIN_ORDER], COFFIN_ROOT, V2,
                             ['--seconds-note', COFFIN_NOTE, '--image-quality-cache', cache, *COFFIN_MEM])
    else:
        part1 = 'skipped_reference_incomplete'
    if part1 == 'license_unavailable':
        metashape_allowed = False
    note('part1', part1)

    # 2. dog
    if metashape_allowed:
        part2 = run_part('p2', [REPO / f'experiments/gpu-pilot-d8-ext/{m}.json' for m, _ in DOG_ORDER],
                         [str(c) for _, c in DOG_ORDER], DOG_ROOT, DOG_OUT, ['--seconds-note', DOG_NOTE, *DOG_MEM])
    else:
        part2 = 'skipped_license_unavailable'
    note('part2', part2)

    # 3. scoring (always attempted on whatever completed)
    coffin_runs = completed_runs(V2, [m for m, _ in COFFIN_ORDER]) if ref_ok else []
    if coffin_runs:
        if step('p3_score_coffin', [VENV, 'scripts/score_runs_v3.py', '--fork', FORK, '--reference', REF,
                                    '--candidates', *coffin_runs, '--output', V2 / 'geometry-v3']) == 0:
            step('p3_labels_coffin', [sys.executable, 'scripts/combine_labels.py', '--artifact', 'pahma-5-613',
                                      '--scores', V2 / 'geometry-v3', '--runs-dir', V2, '--output', V2 / 'labels.csv'])
    dog_runs = [DOG_REF.parent / 'random10', DOG_REF.parent / 'side40'] + completed_runs(DOG_OUT, [m for m, _ in DOG_ORDER])
    if not DRY:
        DOG_OUT.mkdir(exist_ok=True)
    if step('p3_score_dog', [VENV, 'scripts/score_runs_v3.py', '--fork', FORK, '--reference', DOG_REF,
                             '--candidates', *dog_runs, '--output', DOG_OUT / 'geometry-v3']) == 0:
        step('p3_labels_dog', [sys.executable, 'scripts/combine_labels.py', '--artifact', 'arf-dog',
                               '--scores', DOG_OUT / 'geometry-v3', '--runs-dir', DOG_REF.parent, DOG_OUT,
                               '--output', DOG_OUT / 'labels.csv'])
    note('scored', {'coffin': [d.name for d in coffin_runs], 'dog': [d.name for d in dog_runs]})

    # 4. no Metashape
    step('p4a_pilot_models', [sys.executable, 'scripts/pilot_set_models.py', '--output', REPO / 'results/pilot-real-v3'])
    step('p4b_results_notebook', [sys.executable, 'scripts/build_results_notebook.py'])

    left = subprocess.run(['pgrep', '-fl', 'bd3d_quality|run_watched'], capture_output=True, text=True).stdout.strip()
    note('leftover_processes', left or 'none')
    status['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    save()
    log('CHAIN DONE')


if __name__ == '__main__':
    main()
