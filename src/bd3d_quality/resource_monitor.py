"""macOS watchdog for a single child reconstruction; no deletion on cancellation.
RSS is not physical footprint; pressure and swap growth are independent gates.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from .provenance import now, write_json, read_json


def signal_group(pid, signum):
    """Child may exit between poll and signal; never signal another PID.

    macOS raises PermissionError (EPERM) rather than ProcessLookupError when the
    group has already exited or only zombies remain. Treat both as 'gone' so the
    cleanup path still writes the resources JSON and removes the lock.
    """
    try:
        os.killpg(pid, signum)
    except (ProcessLookupError, PermissionError):
        pass


def system_sample(pid=None):
    def query(*args):
        return subprocess.check_output(args, text=True).strip()
    swap = query('sysctl', '-n', 'vm.swapusage')
    sample = {'at': now(), 'pressure': int(query('sysctl', '-n', 'kern.memorystatus_vm_pressure_level')),
              'swap_used_mb': float(re.search(r'used =\s*([\d.]+)M', swap)[1]),
              'rss_bytes': 0, 'cpu_percent': 0.0}
    if pid:
        values = subprocess.run(['ps', '-p', str(pid), '-o', 'rss=,pcpu='], capture_output=True, text=True).stdout.split()
        if values:
            sample['rss_bytes'] = int(values[0]) * 1024
            sample['cpu_percent'] = float(values[1])
    return sample


def limit_reason(sample, baseline, elapsed, disk_bytes, *, seconds=900, max_bytes=2_000_000_000, rss_bytes=3*1024**3,
                 swap_growth_mib=256):
    if sample['pressure'] >= 4: return 'critical_system_memory_pressure'
    if sample['swap_used_mb'] - baseline['swap_used_mb'] > swap_growth_mib:
        return f'system_swap_growth_over_{swap_growth_mib:g}_MiB'
    if sample['rss_bytes'] > rss_bytes: return 'process_RSS_limit'
    if elapsed >= seconds: return 'wall_time_limit'
    if disk_bytes >= max_bytes: return 'output_storage_limit'
    return None


def directory_bytes(path):
    if not path.exists(): return 0
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def run_monitored(command, output, *, seconds=900, allow_cloned_checkpoint=False, swap_growth_mib=256,
                  limit_note=None, seconds_note=None, rss_gib=3, rss_note=None):
    output = Path(output).resolve()
    if output.exists():
        state = read_json(output / 'run.json') if allow_cloned_checkpoint else {}
        if not state.get('continuation_of') or state.get('status') != 'checkpointed' or '--resume' not in command:
            raise ValueError('watchdog requires a fresh output directory or an explicit cloned checkpoint')
    baseline = system_sample()
    if baseline['pressure'] >= 4: raise RuntimeError('critical system pressure: reconstruction not launched')
    # The lock serializes watchdog invocations; additionally detect standalone CLI runs.
    lock = output.parent / '.reconstruction-watchdog.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    child = None
    result = {'status': 'not_started', 'started_at': now(), 'baseline': baseline, 'samples': [], 'command': command,
              'limits': {'seconds': seconds, 'output_bytes': 2_000_000_000, 'RSS_bytes': int(rss_gib*1024**3),
                         'swap_growth_MiB': swap_growth_mib, 'critical_pressure_level': 4,
                         'OpenMP_requested_thread_limit': 2, 'CPU_run_fraction': 0.2, 'CPU_percent_sustained_limit': 300}}
    overrides = {}
    if swap_growth_mib != 256:
        overrides['swap_growth_MiB'] = {'default': 256, 'used': swap_growth_mib, 'note': limit_note}
    if seconds != 900:
        overrides['seconds'] = {'default': 900, 'used': seconds, 'note': seconds_note}
    if rss_gib != 3:
        overrides['RSS_GiB'] = {'default': 3, 'used': rss_gib, 'note': rss_note}
    if overrides:
        result['limit_overrides'] = overrides
    start = time.monotonic()
    log_path = output.with_name(output.name + '-console.log')
    result_path = output.with_name(output.name + '-resources.json')
    try:
        processes = subprocess.check_output(['ps', '-Ao', 'pid=,comm=,args='], text=True).splitlines()
        if any(int(row.split()[0]) != os.getpid() and ('python' in row.split()[1].lower())
               and 'bd3d_quality.cli reconstruct' in row for row in processes):
            raise RuntimeError('another reconstruction CLI is active')
        if shutil.disk_usage(output.parent).free < 5_000_000_000:
            raise RuntimeError('less than 5 GB free storage')
        env = {**os.environ, 'OMP_NUM_THREADS': '2', 'OMP_THREAD_LIMIT': '2', 'OPENBLAS_NUM_THREADS': '2',
               'MKL_NUM_THREADS': '2', 'VECLIB_MAXIMUM_THREADS': '2', 'PYTHONDONTWRITEBYTECODE': '1'}
        high_cpu_samples = 0
        with log_path.open('x') as log:
            child = subprocess.Popen(['nice', '-n', '15', *command], stdout=log, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
            result['pid'] = child.pid
            while child.poll() is None:
                sample = system_sample(child.pid)
                sample['elapsed_seconds'] = time.monotonic() - start
                sample['output_bytes'] = directory_bytes(output)
                result['samples'].append(sample)
                reason = limit_reason(sample, baseline, sample['elapsed_seconds'], sample['output_bytes'], seconds=seconds,
                                      swap_growth_mib=swap_growth_mib, rss_bytes=int(rss_gib*1024**3))
                high_cpu_samples = high_cpu_samples + 1 if sample['cpu_percent'] > 300 else 0
                if high_cpu_samples >= 5: reason = 'CPU_concurrency_limit_not_respected'
                if reason:
                    result['stop_reason'] = reason
                    signal_group(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        signal_group(child.pid, signal.SIGKILL)
                        child.wait()
                    break
                # Metashape may override OpenMP settings. Enforce a 20% run duty
                # cycle for this child process group, rather than promise a thread cap.
                for _ in range(5):
                    if child.poll() is not None: break
                    time.sleep(0.08)
                    if child.poll() is not None: break
                    signal_group(child.pid, signal.SIGSTOP)
                    try: time.sleep(0.32)
                    finally:
                        if child.poll() is None: signal_group(child.pid, signal.SIGCONT)
            result['returncode'] = child.wait()
        result['status'] = 'complete' if result['returncode'] == 0 else 'stopped_or_failed'
    finally:
        # If the watchdog itself is interrupted, never leave a child behind.
        if child is not None and child.poll() is None:
            signal_group(child.pid, signal.SIGCONT)
            signal_group(child.pid, signal.SIGTERM)
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                signal_group(child.pid, signal.SIGKILL)
                child.wait()
            result['status'] = 'watchdog_interrupted'
            result['returncode'] = child.returncode
        result['elapsed_seconds'] = time.monotonic() - start
        result['finished_at'] = now()
        write_json(result_path, result)
        lock.unlink(missing_ok=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seconds', type=int, default=900)
    p.add_argument('--allow-cloned-checkpoint', action='store_true')
    p.add_argument('--swap-growth-mib', type=float, default=256,
                   help='system swap growth that stops the run (default 256); overrides are recorded')
    p.add_argument('--limit-note', help='reason for the swap-growth override, stored in the resources JSON')
    p.add_argument('--rss-gib', type=float, default=3, help='process RSS limit in GiB (default 3); overrides are recorded')
    p.add_argument('--rss-note', help='reason for the RSS override (required for it)')
    p.add_argument('--seconds-note', help='reason for a wall-time cap above 900 s (required for it), stored in the resources JSON')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command or not 0 < a.seconds <= 7200: p.error('command and timeout 1..7200 seconds required')
    if a.rss_gib != 3 and not a.rss_note: p.error('an RSS override needs --rss-note')
    if a.seconds > 900 and not a.seconds_note: p.error('a wall-time cap above 900 s needs --seconds-note')
    def interrupted(signum, frame):
        raise KeyboardInterrupt('watchdog received termination signal')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        result = run_monitored(command, a.output, seconds=a.seconds, allow_cloned_checkpoint=a.allow_cloned_checkpoint,
                               swap_growth_mib=a.swap_growth_mib, limit_note=a.limit_note, seconds_note=a.seconds_note,
                               rss_gib=a.rss_gib, rss_note=a.rss_note)
    except KeyboardInterrupt:
        # Cancel (SIGTERM) or Ctrl-C: run_monitored has already stopped the child and written the
        # watchdog_interrupted record; report it in one line instead of a traceback.
        out = a.output.resolve()
        print(f'cancelled by user; outputs kept in {out} (record: {out.name}-resources.json)', flush=True)
        return 2
    print(result['status'])
    return 0 if result['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
