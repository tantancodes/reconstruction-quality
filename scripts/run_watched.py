"""Run one or more manifests sequentially under the watchdog (general form of run_gpu_pilot.py).

Stop rules:
  - license denied   -> retry up to 3 times, 5 minutes apart, each in a NEW directory
                        (<name>-retryN); then stop.
  - watchdog stop     -> stop and report.
  - watchdog vanished (no <run>-resources.json): continue only if Metashape's run.json
                        says 'complete'; recorded as 'resources unrecorded'.
  - any other failure -> stop and report.
The watchdog's stdout/stderr go to <run>-watchdog.log, opened before launch.
Outputs are never deleted. Limit overrides and their reasons are recorded by the watchdog.

  python3 scripts/run_watched.py --manifests experiments/pahma-5-613/ref145.json \
      --root ~/Documents/bd3d-data/PAHMA-5-613 --output-dir results/pahma-5-613-v1 \
      --seconds 4800 --seconds-note "..." --swap-growth-mib 6144 --rss-gib 6 --memory-note "..."
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = '/Users/tristan/Documents/ChatGPT/bd3d/venv312/bin/python'
RETRIES, WAIT = 3, 300


def stamp():
    return time.strftime('%H:%M:%S')


def attempt(manifest, out, a):
    cmd = [PY, '-B', '-m', 'bd3d_quality.resource_monitor', '--output', str(out), '--seconds', str(a.seconds)]
    if a.seconds_note:
        cmd += ['--seconds-note', a.seconds_note]
    if a.swap_growth_mib != 256:
        cmd += ['--swap-growth-mib', str(a.swap_growth_mib), '--limit-note', a.memory_note]
    if a.rss_gib != 3:
        cmd += ['--rss-gib', str(a.rss_gib), '--rss-note', a.memory_note]
    cmd += ['--', PY, '-B', '-m', 'bd3d_quality.cli', 'reconstruct', str(manifest), '--root', str(a.root),
            '--output', str(out)]
    wd_log = out.with_name(out.name + '-watchdog.log')
    print(f'[{stamp()}] start {out.name}', flush=True)
    with wd_log.open('x') as log:
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
              'resources_recorded': res is not None, 'watchdog_status': res and res['status'],
              'stop_reason': res and res.get('stop_reason'),
              'elapsed_seconds': res and round(res['elapsed_seconds'], 1), 'failures': msg or None}
    if res is None:
        status['note'] = 'resources unrecorded'
    out.with_name(out.name + '-pilot-status.json').write_text(json.dumps(status, indent=2) + '\n')
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
    ap.add_argument('--manifests', type=Path, nargs='+', required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--seconds', type=int, nargs='+', default=[900],
                    help='one cap for all manifests, or one per manifest')
    ap.add_argument('--seconds-note')
    ap.add_argument('--swap-growth-mib', type=float, default=256)
    ap.add_argument('--rss-gib', type=float, default=3)
    ap.add_argument('--memory-note')
    a = ap.parse_args()
    caps = a.seconds * len(a.manifests) if len(a.seconds) == 1 else a.seconds
    if len(caps) != len(a.manifests):
        ap.error('give one --seconds value, or one per manifest')
    a.root = a.root.expanduser()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for manifest, cap in zip(a.manifests, caps):
        a.seconds = cap
        name = manifest.stem
        for n in range(RETRIES + 1):
            out = (a.output_dir / (name if n == 0 else f'{name}-retry{n}')).resolve()
            result = attempt(manifest.resolve(), out, a)
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
