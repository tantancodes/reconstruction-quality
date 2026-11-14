"""Find a Python interpreter that can `import Metashape`, for the capture app.

Candidates, in order (duplicates dropped, first working one wins):
  1. BD3D_METASHAPE_PYTHON from the environment, then from the repo's .env
  2. the path saved last time in the local settings file (.coverage-app.json, gitignored)
  3. interpreters recorded in watchdog logs: results/gpu-pilot-d8/ref50-watchdog.log first,
     then other *-watchdog.log files under results/ and runs/, newest first
  4. likely locations: ~/.venvs/*/bin/python, the repo's own .venv*/venv*, and the Python
     bundled inside /Applications/Metashape*.app
Each candidate is tested with `<python> -c "import Metashape; print(Metashape.app.version)"`.
This does not check the license; the Align and Mesh steps still run the activation check.
"""
import ast
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SETTINGS_FILE = REPO / '.coverage-app.json'
PRIMARY_LOG = Path('results/gpu-pilot-d8/ref50-watchdog.log')
PROBE = 'import Metashape; print(Metashape.app.version)'
PROBE_PATH = '/usr/bin:/bin:/usr/sbin:/sbin'


@dataclass
class Found:
    python: str
    version: str
    source: str


@dataclass
class Search:
    found: Found = None
    tried: list = field(default_factory=list)   # (source, python, reason) for every failed candidate


# ---------------------------------------------------------------- sources

def read_dotenv(path):
    """KEY=VALUE pairs from a .env file (comments, blank lines and `export ` ignored)."""
    values = {}
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.removeprefix('export ').split('=', 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_saved(path=SETTINGS_FILE):
    try:
        return json.loads(Path(path).read_text()).get('metashape_python') or None
    except (OSError, ValueError, AttributeError):
        return None


def save_choice(python, path=SETTINGS_FILE):
    """Remember the interpreter; other keys in the settings file are kept."""
    path = Path(path)
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data['metashape_python'] = str(python)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def python_from_watchdog_log(path):
    """First element of the `command: [...]` line a watchdog log starts with, or None."""
    try:
        with open(path) as f:
            first = f.readline()
    except OSError:
        return None
    if not first.startswith('command: '):
        return None
    try:
        cmd = ast.literal_eval(first[len('command: '):].strip())
    except (ValueError, SyntaxError):
        return None
    return cmd[0] if isinstance(cmd, list) and cmd and isinstance(cmd[0], str) else None


def watchdog_log_pythons(repo=REPO):
    logs = []
    for folder in ('results', 'runs'):
        if (repo / folder).is_dir():
            logs += (repo / folder).rglob('*-watchdog.log')

    def mtime(p):
        try:
            return p.stat().st_mtime
        except OSError:
            return 0
    logs = sorted(set(logs), key=mtime, reverse=True)
    primary = repo / PRIMARY_LOG
    logs = ([primary] if primary.exists() else []) + [p for p in logs if p != primary]
    return [(f'watchdog log {p.relative_to(repo)}', py) for p in logs if (py := python_from_watchdog_log(p))]


def likely_locations(home=None, repo=REPO, applications=Path('/Applications')):
    home = Path(home) if home else Path.home()
    out = [('~/.venvs', p) for p in sorted(home.glob('.venvs/*/bin/python'))]
    out += [('repo venv', p) for pattern in ('.venv*/bin/python', 'venv*/bin/python')
            for p in sorted(repo.glob(pattern))]
    bundled = sorted(applications.glob('Metashape*.app/Contents/Frameworks/Python.framework/Versions/*/bin/python3*'))
    out += [('Metashape app', p) for p in bundled if re.fullmatch(r'python3(\.\d+)?', p.name)]
    return [(s, str(p)) for s, p in out]


def candidates(env=None, repo=REPO, home=None, applications=Path('/Applications'), settings=SETTINGS_FILE):
    """(source, python) in search order, without duplicates (same path or same resolved file)."""
    env = os.environ if env is None else env
    raw = []
    if env.get('BD3D_METASHAPE_PYTHON'):
        raw.append(('BD3D_METASHAPE_PYTHON (environment)', env['BD3D_METASHAPE_PYTHON']))
    dotenv = read_dotenv(repo / '.env').get('BD3D_METASHAPE_PYTHON')
    if dotenv:
        raw.append(('BD3D_METASHAPE_PYTHON (.env)', dotenv))
    saved = load_saved(settings)
    if saved:
        raw.append(('saved choice', saved))
    raw += watchdog_log_pythons(repo)
    raw += likely_locations(home, repo, applications)
    seen, out = set(), []
    for source, python in raw:
        python = os.path.expanduser(str(python))
        # venv interpreters are symlinks; keep the venv path but dedupe on (path, target)
        keys = {python, os.path.realpath(python)} if not os.path.islink(python) else {python}
        if keys & seen:
            continue
        seen |= keys
        out.append((source, python))
    return out


# ---------------------------------------------------------------- testing candidates

def probe(python, timeout=20):
    """(ok, Metashape version or reason it failed)."""
    if not os.path.isfile(python):
        return False, 'not found'
    if not os.access(python, os.X_OK):
        return False, 'not executable'
    try:
        proc = subprocess.run([python, '-B', '-c', PROBE], capture_output=True, text=True, timeout=timeout,
                              env={'PATH': PROBE_PATH})
    except subprocess.TimeoutExpired:
        return False, f'timed out after {timeout} s'
    except OSError as exc:
        return False, f'cannot run: {exc}'
    out = proc.stdout.strip().splitlines()
    if proc.returncode == 0 and out:
        return True, out[-1]
    err = (proc.stderr.strip().splitlines() or ['no output'])[-1]
    return False, err[-200:]


def find_metashape_python(cands=None, test=probe, on_try=None):
    """First candidate that imports Metashape. on_try(source, python, ok, detail) is called per candidate."""
    search = Search()
    for source, python in (candidates() if cands is None else cands):
        ok, detail = test(python)
        if on_try:
            on_try(source, python, ok, detail)
        if ok:
            search.found = Found(python, detail, source)
            return search
        search.tried.append((source, python, detail))
    return search
