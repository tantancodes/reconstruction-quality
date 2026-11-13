"""List (default) or scrub local absolute paths in files tracked by git.

  python scripts/scrub_paths.py                      # list tracked files with local paths
  python scripts/scrub_paths.py --include-untracked  # also check untracked, non-ignored files
  python scripts/scrub_paths.py --apply              # rewrite them in place
  python scripts/scrub_paths.py --apply docs/A.md    # only the named files (must be tracked)

Rewrites, in order (RULES):
  /Users/<name>/Documents/bd3d-reconstruction-quality/x  ->  x  (repo-relative)
  /Users/<name>/Documents/ChatGPT/bd3d, ~/..., <LOCAL>/...  ->  <bd3d>  (the Aphoma bd3d fork)
  .../Documents/BD3D_Test_Output   ->  <BD3D_Test_Output>  (historical Metashape outputs)
  .../Documents/bd3d-data          ->  <bd3d-data>         (coffin photos)
  .../Downloads/Jpg-processed      ->  <dog-photos>        (ARF dog photos)
  any other /Users/<name>          ->  <LOCAL>

Only path text changes; no other content. It rewrites files in place, so run it on a clean
working tree and review `git diff` before committing. Binary files (containing NUL bytes)
are skipped. Exit code: 1 if any path was found in list mode, else 0.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HOME = r'(?:/Users/[^/\s"\'<>`\\]+|~|<LOCAL>)'
RULES = [
    (re.compile(r'/Users/[^/\s"\'<>`\\]+/Documents/bd3d-reconstruction-quality/'), ''),
    (re.compile(r'/Users/[^/\s"\'<>`\\]+/Documents/bd3d-reconstruction-quality(?![\w-])'), '.'),
    (re.compile(HOME + r'/Documents/ChatGPT/bd3d(?![\w-])'), '<bd3d>'),
    (re.compile(HOME + r'/Documents/BD3D_Test_Output(?![\w-])'), '<BD3D_Test_Output>'),
    (re.compile(HOME + r'/Documents/bd3d-data(?![\w-])'), '<bd3d-data>'),
    (re.compile(HOME + r'/Downloads/Jpg-processed(?![\w-])'), '<dog-photos>'),
    (re.compile(r'/Users/[^/\s"\'<>`\\]+'), '<LOCAL>'),
]
FIND = re.compile('|'.join(f'(?:{p.pattern})' for p, _ in RULES)
                  + r'|~/(?:Documents|Downloads)/|ChatGPT/bd3d')


def scrub(text):
    for pattern, new in RULES:
        text = pattern.sub(new, text)
    return text


def files(include_untracked):
    cmd = ['git', 'ls-files', '-z']
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, check=True).stdout.split(b'\0')
    if include_untracked:
        out += subprocess.run(cmd + ['--others', '--exclude-standard'], cwd=REPO, capture_output=True,
                              check=True).stdout.split(b'\0')
    return sorted({p.decode() for p in out if p})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true', help='rewrite local paths in place')
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
        found = FIND.findall(text)
        if not found:
            continue
        lines = [i + 1 for i, line in enumerate(text.splitlines()) if FIND.search(line)]
        hits.append((rel, len(found), lines))
        if a.apply:
            path.write_bytes(scrub(text).encode('utf-8', errors='surrogateescape'))
    for rel, n, lines in hits:
        shown = ','.join(map(str, lines[:8])) + (',...' if len(lines) > 8 else '')
        print(f'{rel}: {n} occurrence(s), line(s) {shown}')
    total = sum(n for _, n, _ in hits)
    print(f"\n{len(hits)} file(s), {total} occurrence(s) {'rewritten' if a.apply else 'found'}")
    return 1 if hits and not a.apply else 0


if __name__ == '__main__':
    sys.exit(main())
