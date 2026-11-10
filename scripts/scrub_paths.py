"""List (default) or scrub absolute /Users/<name> paths in files tracked by git.

  python3 scripts/scrub_paths.py                      # list tracked files with /Users/ paths
  python3 scripts/scrub_paths.py --include-untracked  # also check untracked, non-ignored files
  python3 scripts/scrub_paths.py --apply              # replace /Users/<name> with <LOCAL> in place
  python3 scripts/scrub_paths.py --apply docs/A.md     # only the named files (must be tracked)

--apply keeps the rest of each path, so /Users/alice/Documents/x becomes <LOCAL>/Documents/x.
It rewrites files in place, so run it on a clean working tree and review `git diff` before
committing. Binary files (containing NUL bytes) are skipped. Exit code: 1 if any path was
found in list mode, else 0.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r'/Users/[^/\s"\'<>`\\]+')
PLACEHOLDER = '<LOCAL>'


def files(include_untracked):
    cmd = ['git', 'ls-files', '-z']
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, check=True).stdout.split(b'\0')
    if include_untracked:
        out += subprocess.run(cmd + ['--others', '--exclude-standard'], cwd=REPO, capture_output=True,
                              check=True).stdout.split(b'\0')
    return sorted({p.decode() for p in out if p})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true', help=f'replace /Users/<name> with {PLACEHOLDER} in place')
    ap.add_argument('--include-untracked', action='store_true')
    ap.add_argument('paths', nargs='*', help='limit to these repo-relative files (default: all)')
    a = ap.parse_args()
    known = files(a.include_untracked)
    missing = [x for x in a.paths if x not in known]
    if missing:
        ap.error(f'not tracked (or not found): {missing}')
    hits = []
    for rel in (a.paths or known):
        path = REPO / rel
        if path.resolve() == Path(__file__).resolve() or not path.is_file():  # its own docs/regex
            continue
        data = path.read_bytes()
        if b'\0' in data:
            continue
        text = data.decode('utf-8', errors='surrogateescape')
        found = PATTERN.findall(text)
        if not found:
            continue
        lines = [i + 1 for i, line in enumerate(text.splitlines()) if PATTERN.search(line)]
        hits.append((rel, len(found), lines, sorted(set(found))))
        if a.apply:
            path.write_bytes(PATTERN.sub(PLACEHOLDER, text).encode('utf-8', errors='surrogateescape'))
    for rel, n, lines, prefixes in hits:
        shown = ','.join(map(str, lines[:8])) + (',...' if len(lines) > 8 else '')
        print(f'{rel}: {n} occurrence(s), line(s) {shown}  [{", ".join(prefixes)}]')
    total = sum(n for _, n, _, _ in hits)
    verb = 'replaced' if a.apply else 'found'
    print(f'\n{len(hits)} file(s), {total} occurrence(s) {verb}')
    return 1 if hits and not a.apply else 0


if __name__ == '__main__':
    sys.exit(main())
