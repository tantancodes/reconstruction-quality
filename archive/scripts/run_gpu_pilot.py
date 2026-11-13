"""Run the GPU dense pilot (random10, side40, ref50) under the watchdog.

Stop rules:
  - license denied   -> retry up to 3 times, 5 minutes apart, each in a NEW directory
                        (<name>-retryN); then stop.
  - watchdog stop     -> stop everything and report.
  - watchdog vanished (no <run>-resources.json):
                        continue only if Metashape's run.json says 'complete';
                        recorded as 'resources unrecorded' in <run>-pilot-status.json.
  - any other failure -> stop and report.
The watchdog's own stdout/stderr go to <run>-watchdog.log, opened before launch.
Outputs are never deleted. Limit overrides are recorded by the watchdog.

  python3 scripts/run_gpu_pilot.py                       # all three
  python3 scripts/run_gpu_pilot.py --only ref50 --seconds 1800 \
      --seconds-note "50-image reference at GPU d8, est ~13 min"
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = os.environ.get('BD3D_METASHAPE_PYTHON', 'python3')  # interpreter with the Metashape module
ROOT = str(Path(os.environ.get('BD3D_DOG_PHOTOS', 'Jpg-processed')).expanduser())  # ARF dog photos
MANIFESTS = REPO / 'experiments' / 'gpu-pilot-d8'
OUT = REPO / 'results' / 'gpu-pilot-d8'
ORDER = ['random10', 'side40', 'ref50']
SWAP_MIB = 2048
SWAP_NOTE = ('User-approved 2026-11-09: swap-growth limit raised 256 -> 2048 MiB so Chrome can stay open; '
             'GPU (Metal) instead of CPU-only; depth_downscale 8; time-constrained pilot.')
RETRIES, WAIT = 3, 300


def stamp():
    return time.strftime('%H:%M:%S')


def attempt(name, out, seconds, seconds_note):
    cmd = [PY, '-B', '-m', 'bd3d_quality.resource_monitor', '--output', str(out), '--seconds', str(seconds),
           '--swap-growth-mib', str(SWAP_MIB), '--limit-note', SWAP_NOTE]
    if seconds_note:
        cmd += ['--seconds-note', seconds_note]
    cmd += ['--', PY, '-B', '-m', 'bd3d_quality.cli', 'reconstruct', str(MANIFESTS / f'{name}.json'),
            '--root', ROOT, '--output', str(out)]
    wd_log = out.with_name(out.name + '-watchdog.log')
    print(f'[{stamp()}] start {out.name} (watchdog log: {wd_log.name})', flush=True)
    with wd_log.open('x') as log:  # opened before launch so nothing the watchdog prints is lost
        log.write(f'command: {cmd}\n')
        log.flush()
        proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                              env={'PYTHONPATH': str(REPO / 'src'), 'PATH': '/usr/bin:/bin:/usr/sbin:/sbin'})
        log.write(f'\nwatchdog exit code: {proc.returncode}\n')
    res_path = out.with_name(out.name + '-resources.json')
    res = json.loads(res_path.read_text()) if res_path.exists() else None
    run = json.loads((out / 'run.json').read_text()) if (out / 'run.json').exists() else {}
    msg = ' | '.join(f.get('message', '') for f in run.get('failures', []))
    status = {'run': out.name, 'watchdog_exit_code': proc.returncode, 'metashape_status': run.get('status'),
              'resources_recorded': res is not None,
              'watchdog_status': res and res['status'], 'stop_reason': res and res.get('stop_reason'),
              'elapsed_seconds': res and round(res['elapsed_seconds'], 1), 'failures': msg or None}
    if res is None:
        status['note'] = 'resources unrecorded'
    (out.with_name(out.name + '-pilot-status.json')).write_text(json.dumps(status, indent=2) + '\n')
    print(f'[{stamp()}] {json.dumps(status)}', flush=True)
    if res is None:
        return 'ok_unrecorded' if run.get('status') == 'complete' else 'watchdog_vanished'
    if res['status'] == 'complete' and run.get('status') == 'complete':
        return 'ok'
    if res.get('stop_reason') or res['status'] == 'watchdog_interrupted':
        return 'watchdog_stop'
    if 'license' in msg.lower():
        return 'license'
    return 'failed'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='+', choices=ORDER)
    ap.add_argument('--seconds', type=int, default=900)
    ap.add_argument('--seconds-note')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in a.only or ORDER:
        for n in range(RETRIES + 1):
            out = OUT / (name if n == 0 else f'{name}-retry{n}')
            result = attempt(name, out, a.seconds, a.seconds_note)
            if result in ('ok', 'ok_unrecorded'):
                break
            if result == 'license' and n < RETRIES:
                print(f'license denied; waiting {WAIT}s before retry {n + 1}/{RETRIES}', flush=True)
                time.sleep(WAIT)
                continue
            print(f'STOP: {name} -> {result}', flush=True)
            sys.exit(2)
    print('ALL COMPLETE', flush=True)


if __name__ == '__main__':
    main()
