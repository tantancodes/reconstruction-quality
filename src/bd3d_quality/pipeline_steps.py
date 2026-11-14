"""Non-UI building blocks for the coverage app's pipeline steps.

Everything here reuses the existing pipeline: manifests.create_manifest for the manifest,
`bd3d_quality.cli reconstruct` under `bd3d_quality.resource_monitor` (the same watched command
scripts/run_watched.py builds) for alignment and meshing, the existing seed-checkpoint path to
continue an alignment-only run to the mesh, and evaluation.score_candidate for the v3 label.

Settings match the pilot runs (ref50, ref145b): profile mesh, Metal GPU (cpu_only False),
depth_downscale 8, all other settings at their defaults. The alignment run uses the same
settings with profile 'alignment', so the mesh run can be seeded from it.
"""
import copy
import errno
import json
import os
import signal
import stat
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .config import ReconstructionConfig
from .evaluation import Reference, load_cameras, load_obj_mesh, score_candidate
from .manifests import create_manifest, experiment_identity
from .provenance import now

REPO = Path(__file__).resolve().parents[2]
CHILD_PATH = '/usr/bin:/bin:/usr/sbin:/sbin'   # same minimal PATH as scripts/run_watched.py


def pilot_config(profile):
    """Pilot reconstruction settings (as used for ref50 and ref145b) with the given profile."""
    return ReconstructionConfig(profile=profile, cpu_only=False, depth_downscale=8)


def settings_text():
    c = pilot_config('mesh')
    return (f'profile {c.profile} · {"CPU only" if c.cpu_only else "Metal GPU"} · depth_downscale {c.depth_downscale}'
            f' · {c.depth_filter} · keypoints {c.keypoint_limit} · tie points {c.tiepoint_limit}')


@dataclass(frozen=True)
class WatchdogLimits:
    seconds: int
    rss_gib: float = 3
    swap_growth_mib: float = 256
    note: str = ''

    def args(self):
        """resource_monitor arguments; every override above its default carries the note."""
        a = ['--seconds', str(self.seconds)]
        if self.seconds > 900:
            a += ['--seconds-note', self.note]
        if self.swap_growth_mib != 256:
            a += ['--swap-growth-mib', f'{self.swap_growth_mib:g}', '--limit-note', self.note]
        if self.rss_gib != 3:
            a += ['--rss-gib', f'{self.rss_gib:g}', '--rss-note', self.note]
        return a

    def text(self):
        return f'{self.rss_gib:g} GiB RSS · {self.swap_growth_mib:g} MiB swap growth · {self.seconds} s'


ALIGN_LIMITS = WatchdogLimits(1800, note='coverage app: alignment-only step, wall-time cap 1800 s (user-approved 2026-11-14)')
MESH_LIMITS = WatchdogLimits(7200, rss_gib=6, swap_growth_mib=6144,
                             note='coverage app: mesh step with the pilot limits (6 GiB RSS, 6144 MiB swap, '
                                  '7200 s; user-approved 2026-11-14)')


# ---------------------------------------------------------------- paths

def inspect_folder(path, need=()):
    """(ok, resolved path, reason). Distinguishes a missing folder from one macOS will not let us read.

    Path.exists() returns False for any OS error (Python 3.14), so a privacy denial on ~/Documents
    would otherwise be reported as "does not exist".
    """
    p = Path(os.path.expanduser(str(path)))
    try:
        p = p.resolve(strict=False)
    except OSError:
        pass
    try:
        st = os.stat(p)
    except FileNotFoundError:
        parent = next((q for q in p.parents if q.exists()), None)
        return False, p, f'does not exist (nearest existing parent: {parent})'
    except PermissionError:
        return False, p, ('permission denied by macOS; allow the app that launched Python (Terminal, VS Code) '
                          'under System Settings > Privacy & Security > Files and Folders')
    except OSError as exc:
        return False, p, f'cannot be read: {errno.errorcode.get(exc.errno, exc.errno)} {exc.strerror}'
    if not stat.S_ISDIR(st.st_mode):
        return False, p, 'is a file, not a folder'
    try:
        os.listdir(p)
    except PermissionError:
        return False, p, ('folder exists but listing it is denied by macOS; allow the launching app under '
                          'System Settings > Privacy & Security > Files and Folders')
    for name in need:
        f = p / name
        try:
            os.stat(f)
        except FileNotFoundError:
            return False, p, f'has no {name}'
        except PermissionError:
            return False, p, f'{name} exists but reading it is denied by macOS'
    return True, p, 'ok'


@dataclass(frozen=True)
class CapturePaths:
    output: Path
    name: str

    @property
    def manifests(self):
        return self.output / 'manifests'

    @property
    def align_manifest(self):
        return self.manifests / f'{self.name}-align.json'

    @property
    def mesh_manifest(self):
        return self.manifests / f'{self.name}-mesh.json'

    @property
    def align_run(self):
        return self.output / f'{self.name}-align'

    @property
    def mesh_run(self):
        return self.output / f'{self.name}-mesh'


RUN_FILES = ('', '-watchdog.log', '-console.log', '-resources.json', '-pilot-status.json')


def run_name_taken(output, run_name):
    """True when a run folder, any of its sibling logs, or its manifest already exists."""
    output = Path(output)
    return (any((output / f'{run_name}{s}').exists() for s in RUN_FILES)
            or (output / 'manifests' / f'{run_name}.json').exists())


def unused_run_name(output, base):
    """base, or base-2, base-3, ... : the first run name with nothing on disk (outputs are never reused)."""
    name, n = base, 1
    while run_name_taken(output, name):
        n += 1
        name = f'{base}-{n}'
    return name


def capture_name(photos, output):
    """Capture name from the photo folder name such that neither <name>-align nor <name>-mesh is taken."""
    base = ''.join(c if c.isalnum() or c in '-_' else '-' for c in Path(photos).name) or 'capture'
    name, n = base, 1
    while run_name_taken(output, f'{name}-align') or run_name_taken(output, f'{name}-mesh'):
        n += 1
        name = f'{base}-{n}'
    return name


# ---------------------------------------------------------------- manifests

def alignment_manifest(photos, dataset_id, description=''):
    """Manifest of every image under photos, with the pilot settings and profile 'alignment'."""
    return create_manifest(photos, dataset_id, pilot_config('alignment'),
                           description or 'coverage app: alignment-only capture check')


def mesh_manifest(align):
    """Same images and alignment settings as the alignment manifest, profile 'mesh' (seedable from it)."""
    m = copy.deepcopy(align)
    m['configuration'] = pilot_config('mesh').to_dict()
    m['description'] = 'coverage app: mesh continued from ' + align['experiment_id']
    m['created_at'] = now()
    m['status'] = 'prepared'
    m['experiment_id'] = experiment_identity(m)
    return m


def write_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:              # never overwrite an existing manifest
        json.dump(manifest, f, indent=2)
        f.write('\n')
    return path


# ---------------------------------------------------------------- watched commands

def metashape_python():
    return os.environ.get('BD3D_METASHAPE_PYTHON', '')


def child_env():
    return {'PYTHONPATH': str(REPO / 'src'), 'PATH': CHILD_PATH}


def license_command(python):
    return [python, '-B', '-c', 'import Metashape; print(bool(Metashape.app.activated))']


def license_ok(python, timeout=120):
    """Same activation check scripts/run_watched.py runs before each job: (ok, detail)."""
    if not python:
        return False, 'BD3D_METASHAPE_PYTHON is not set (an interpreter that can import Metashape)'
    try:
        proc = subprocess.run(license_command(python), capture_output=True, text=True, timeout=timeout,
                              env={'PATH': CHILD_PATH})
    except subprocess.TimeoutExpired:
        return False, f'activation check timed out after {timeout} s'
    except OSError as exc:
        return False, f'cannot start {python}: {exc}'
    ok = proc.returncode == 0 and proc.stdout.strip().splitlines()[-1:] == ['True']
    detail = (proc.stdout + proc.stderr).strip()[-300:]
    return ok, 'activated' if ok else f'license unavailable (check VPN) | {detail}'


def watched_command(python, manifest, photos, run_dir, limits, seed=None, image_quality_cache=None):
    """resource_monitor wrapping `bd3d_quality.cli reconstruct`, as in scripts/run_watched.py."""
    cmd = [python, '-B', '-m', 'bd3d_quality.resource_monitor', '--output', str(run_dir), *limits.args(),
           '--', python, '-B', '-m', 'bd3d_quality.cli', 'reconstruct', str(manifest),
           '--root', str(photos), '--output', str(run_dir)]
    if seed is not None:
        cmd += ['--seed-checkpoint', str(seed)]
    if image_quality_cache is not None:
        cmd += ['--image-quality-cache', str(Path(image_quality_cache).resolve())]
    return cmd


class WatchedProcess:
    """Run a watched command in its own process group, streaming its output and the run's console log.

    on_line(text) is called from background threads; the caller must hand it to the UI thread.
    cancel() sends SIGTERM to the watchdog's process group: the watchdog stops the reconstruction
    child, writes <run>-resources.json and exits. Nothing is deleted.
    """

    def __init__(self, cmd, run_dir, on_line, env=None, cwd=REPO):
        self.cmd, self.run_dir, self.on_line = cmd, Path(run_dir), on_line
        self.env, self.cwd = env if env is not None else child_env(), cwd
        self.proc, self.cancelled = None, False

    def run(self, poll=0.5):
        wd_log = self.run_dir.with_name(self.run_dir.name + '-watchdog.log')
        console = self.run_dir.with_name(self.run_dir.name + '-console.log')
        with wd_log.open('x') as log:
            log.write(f'command: {self.cmd}\n')
            log.flush()
            self.proc = subprocess.Popen(self.cmd, cwd=self.cwd, env=self.env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            if self.cancelled:                     # Cancel pressed before the process existed
                self.cancel()

            def pump():
                for line in self.proc.stdout:
                    log.write(line)
                    log.flush()
                    self.on_line(line.rstrip('\n'))
            reader = threading.Thread(target=pump, daemon=True)
            reader.start()
            offset = 0
            while True:
                done = self.proc.poll() is not None
                offset = self._tail(console, offset)
                if done:
                    break
                time.sleep(poll)
            reader.join(timeout=5)
            self._tail(console, offset)
            log.write(f'\nwatchdog exit code: {self.proc.returncode}\n')
        return self.proc.returncode

    def _tail(self, path, offset):
        try:
            with open(path) as f:
                f.seek(offset)
                chunk = f.read()
                offset = f.tell()
        except FileNotFoundError:
            return offset
        for line in chunk.splitlines():
            self.on_line('  | ' + line)
        return offset

    def cancel(self):
        self.cancelled = True
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass


# ---------------------------------------------------------------- status

def run_outcome(run_dir):
    """(state, detail) of a watched run from disk: not started / done / failed / cancelled / running.

    Mirrors scripts/run_watched.classify: the watchdog record (<run>-resources.json) and run.json.
    """
    run_dir = Path(run_dir)
    res_path = run_dir.with_name(run_dir.name + '-resources.json')
    if not run_dir.exists() and not res_path.exists():
        return 'not started', ''
    res = json.loads(res_path.read_text()) if res_path.exists() else None
    run = json.loads((run_dir / 'run.json').read_text()) if (run_dir / 'run.json').exists() else {}
    msg = ' | '.join(f.get('message', '') for f in run.get('failures', []))
    if res is None:
        return ('done', 'resources unrecorded') if run.get('status') == 'complete' else ('running', '')
    if res['status'] == 'complete' and run.get('status') == 'complete':
        return 'done', f'{res.get("elapsed_seconds", 0):.0f} s'
    if res['status'] == 'watchdog_interrupted':
        return 'cancelled', 'watchdog interrupted; outputs kept'
    if res.get('stop_reason'):
        return 'failed', f'watchdog stop: {res["stop_reason"]}'
    if 'license' in msg.lower():
        return 'failed', 'Metashape license unavailable'
    if 'alignment failed' in msg.lower():
        return 'failed', 'alignment failed the camera/tie-point criteria'
    return 'failed', msg or f'watchdog status {res["status"]}'


def is_alignment_checkpoint(run_dir):
    """True for a complete alignment-only run (what --seed-checkpoint accepts)."""
    try:
        state = json.loads((Path(run_dir) / 'run.json').read_text())
    except (OSError, ValueError):
        return False
    return state.get('status') == 'complete' and [s['stage'] for s in state.get('stages', [])] == ['initial_alignment']


# ---------------------------------------------------------------- image-quality cache

CACHE_NAME = 'image_quality_cache.csv'


def find_quality_cache(reference_dir):
    """The reference run's image_quality_cache.csv, or one beside it (e.g. results/pahma-5-613-v2/)."""
    if reference_dir is None:
        return None
    reference_dir = Path(reference_dir)
    for p in (reference_dir / CACHE_NAME, reference_dir.parent / CACHE_NAME):
        if p.is_file():
            return p
    return None


def cache_coverage(cache_path, manifest):
    """(usable, reason). Mirrors metashape_runner.apply_image_quality_cache, which fails the run unless
    every image has a finite cached value (matched by SHA-256) from a single source run."""
    import csv
    import math
    try:
        with open(cache_path, newline='') as f:
            rows = {r['sha256']: r for r in csv.DictReader(f)}
    except (OSError, KeyError, csv.Error) as exc:
        return False, f'cannot read cache: {exc}'
    covered, sources = 0, set()
    for e in manifest['images']:
        r = rows.get(e['sha256'])
        try:
            ok = r is not None and math.isfinite(float(r['image_quality']))
        except (TypeError, ValueError, KeyError):
            ok = False
        if ok:
            covered += 1
            sources.add((r.get('source_run_id'), r.get('source_experiment_id'), r.get('metashape_version')))
    total = len(manifest['images'])
    if covered < total:
        return False, f'cache covers {covered} of {total} photos (matched by SHA-256)'
    if len(sources) != 1:
        return False, 'cache rows come from more than one source run'
    return True, f'cache covers all {total} photos'


def image_quality_source(run_dir):
    """How the run got Image/Quality, from its run.json: 'cache …', 'analyzeImages' or None."""
    try:
        src = json.loads((Path(run_dir) / 'run.json').read_text()).get('image_quality_source') or {}
    except (OSError, ValueError):
        return None
    if src.get('method') == 'cache':
        return f'cache {src.get("cache_path")} ({src.get("images")} images)'
    return src.get('method')


# ---------------------------------------------------------------- scoring

def score_run(run_dir, reference_dir):
    """v3 label of run_dir's raw mesh against reference_dir's (evaluation.score_candidate)."""
    run_dir, reference_dir = Path(run_dir), Path(reference_dir)
    for d in (run_dir, reference_dir):
        for name in ('raw-mesh.obj', 'cameras.csv'):
            if not (d / name).is_file():
                raise FileNotFoundError(f'{d / name} not found (scoring needs both runs\' mesh and cameras)')
    ref = Reference.from_files(reference_dir / 'raw-mesh.obj', reference_dir / 'cameras.csv')
    row, _ = score_candidate(ref, load_obj_mesh(run_dir / 'raw-mesh.obj'), load_cameras(run_dir / 'cameras.csv'))
    return row


def score_text(row):
    return (f'v3 label {row["label_v3"]:.4f} (QC {row["qc_v3"]}, scale {row["scale_check"]}) · '
            f'missing RMSE {row["r2c_rmse_norm"]:.4f} · extra RMSE {row["c2r_rmse_norm"]:.4f} '
            f'(fractions of the reference bounding-box diagonal; {row["camera_shared_images"]} shared cameras)')
