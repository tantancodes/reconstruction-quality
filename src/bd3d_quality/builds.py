"""Photo folders and existing builds, for the capture app (no UI code).

A "build" is a run folder with cameras.csv. An alignment-only run and the mesh run seeded from
it (run.json parent_checkpoint) are shown as one build. Two builds are compatible when they
share at least MIN_SHARED photos with the same content (SHA-256 from manifest.json) and the same
file name: the camera fit (evaluation.camera_transform) pairs cameras by label, which Metashape
takes from the file name. Builds without a manifest.json are listed but never compatible.
"""
import csv
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .manifests import IMAGE_SUFFIXES
from .pipeline_steps import CACHE_NAME, cache_coverage, find_quality_cache, inspect_folder
from .provenance import sha256_file

MIN_PHOTOS = 10      # fewer than this: warn
MIN_SHARED = 3       # photos two builds must share to be compared


# ---------------------------------------------------------------- photo folders

@dataclass
class PhotoFolder:
    root: Path
    photos: list                                  # relative paths, in manifest order (sorted)
    ignored: dict = field(default_factory=dict)   # suffix -> count of files that are not photos we use
    subfolders: list = field(default_factory=list)  # relative folders (other than the root) holding photos
    hidden: int = 0                               # photos with a name or folder starting with '.'
    warnings: list = field(default_factory=list)

    @property
    def ok(self):
        return bool(self.photos)

    @property
    def summary(self):
        n = len(self.photos)
        return f'{n} photo{"s" * (n != 1)} found' if n else 'No photos found'


def scan_photo_folder(path, min_photos=MIN_PHOTOS):
    """Count the photos a manifest of this folder would contain (manifests.create_manifest takes every
    file with a supported suffix, in subfolders too) and collect plain-language warnings.
    Raises ValueError with the reason when the folder cannot be read."""
    ok, root, reason = inspect_folder(path)
    if not ok:
        raise ValueError(f'This folder cannot be used: it {reason}.')
    photos, ignored, folders, hidden = [], {}, set(), 0
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            suffix = p.suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                rel = p.relative_to(root)
                if any(part.startswith('.') for part in rel.parts):
                    hidden += 1
                photos.append(rel)
                if rel.parent != Path('.'):
                    folders.add(rel.parent.as_posix())
            elif not name.startswith('.'):
                key = suffix or '(no extension)'
                ignored[key] = ignored.get(key, 0) + 1
    photos.sort(key=lambda r: r.as_posix())   # same order as create_manifest (sorted rglob)
    pf = PhotoFolder(root, photos, ignored, sorted(folders), hidden)
    supported = ', '.join(sorted(IMAGE_SUFFIXES))
    if not photos:
        pf.warnings.append(f'No supported photos here ({supported}).')
    if ignored:
        n = sum(ignored.values())
        kinds = ', '.join(sorted(ignored, key=lambda k: (k.startswith('('), k)))
        pf.warnings.append(f'{n} file{"s" * (n != 1)} ignored ({kinds}: not a supported photo type).')
    if folders:
        k = len(folders)
        pf.warnings.append(f'Photos are inside subfolders ({k} folder{"s" * (k != 1)}); they will be included.')
    if hidden:
        pf.warnings.append(f'{hidden} hidden photo file{"s" * (hidden != 1)} (name or folder starting with ".") '
                           f'will be included; remove {"them" if hidden != 1 else "it"} if not a real photo.')
    if 0 < len(photos) < min_photos:
        pf.warnings.append(f'Only {len(photos)} photos: at least {min_photos} are recommended.')
    return pf


def photo_fingerprints(folder, on_progress=None):
    """{SHA-256: label} for a PhotoFolder; label = file name without extension (the Metashape camera label).
    on_progress(done, total) is called after each photo."""
    out, total = {}, len(folder.photos)
    for i, rel in enumerate(folder.photos, 1):
        out[sha256_file(folder.root / rel)] = rel.stem
        if on_progress:
            on_progress(i, total)
    return out


# ---------------------------------------------------------------- builds

@dataclass
class Build:
    name: str
    group: str            # parent folder name, to tell builds with the same name apart
    path: Path            # folder whose cameras.csv is shown (the mesh run when there is one)
    photo_count: int
    mesh: Path = None     # folder with raw-mesh.obj, or None
    date: datetime = None
    photos: dict = None   # {SHA-256: label} from manifest.json, or None (not comparable)
    run_id: str = None
    parent_run_id: str = None
    paths: tuple = ()     # every folder that makes up this build

    @property
    def has_mesh(self):
        return self.mesh is not None

    @property
    def key(self):
        return str(self.path)


def display_name(folder_name):
    """Folder name without the app's -align / -mesh(-N) suffix."""
    return re.sub(r'-(align|mesh)(-\d+)?$', '', folder_name) or folder_name


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def _date(run, folder):
    for key in ('finished_at', 'started_at'):
        try:
            d = datetime.fromisoformat(run[key])
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except (KeyError, TypeError, ValueError):
            pass
    try:
        return datetime.fromtimestamp((folder / 'cameras.csv').stat().st_mtime, timezone.utc)
    except OSError:
        return None


def _camera_rows(path):
    try:
        with open(path, newline='') as f:
            return sum(1 for _ in csv.DictReader(f))
    except (OSError, csv.Error):
        return 0


def read_build(folder):
    """Build for one run folder with cameras.csv, or None."""
    folder = Path(folder)
    if not (folder / 'cameras.csv').is_file():
        return None
    run = _read_json(folder / 'run.json')
    manifest = _read_json(folder / 'manifest.json')
    photos = None
    if isinstance(manifest.get('images'), list) and manifest['images']:
        try:
            photos = {e['sha256']: Path(e['path']).stem for e in manifest['images']}
        except (KeyError, TypeError):
            photos = None
    count = manifest.get('image_count') if isinstance(manifest.get('image_count'), int) else _camera_rows(folder / 'cameras.csv')
    return Build(display_name(folder.name), folder.parent.name, folder, count,
                 folder if (folder / 'raw-mesh.obj').is_file() else None, _date(run, folder), photos,
                 run.get('run_id'), (run.get('parent_checkpoint') or {}).get('run_id'), (folder,))


def find_builds(roots):
    """Builds under the given folders (missing folders are skipped), newest first.
    An alignment run whose mesh run (seeded from it) is also found is merged into that mesh build."""
    folders = set()
    for root in roots:
        root = Path(root)
        if root.is_dir():
            folders |= {p.parent for p in root.rglob('cameras.csv')}
    builds = [b for b in map(read_build, sorted(folders)) if b is not None]
    by_id = {b.run_id: b for b in builds if b.run_id}
    merged = set()
    for b in builds:
        parent = by_id.get(b.parent_run_id)
        if parent is not None and parent is not b:
            merged.add(id(parent))
            b.paths = (parent.path, *b.paths)
    out = [b for b in builds if id(b) not in merged]
    old = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(out, key=lambda b: (b.date or old, b.name), reverse=True)


def shared_photos(a, b):
    """Photos in both {SHA-256: label} maps with the same content and the same label."""
    if not a or not b:
        return 0
    return sum(1 for h, label in a.items() if b.get(h) == label)


def compatible_builds(photos, builds, min_shared=MIN_SHARED, exclude=None):
    """[(build, shared photos)] for builds sharing at least min_shared photos, most shared first."""
    out = []
    for b in builds:
        if exclude is not None and b.key == exclude.key:
            continue
        n = shared_photos(photos, b.photos)
        if n >= min_shared:
            out.append((b, n))
    return sorted(out, key=lambda t: (-t[1], -t[0].photo_count, t[0].name))


def default_reference(build, builds):
    """The build to compare an opened build against by default: the compatible build with a mesh and
    more photos than this one, the largest first (e.g. side40 -> ref50, side40_s1 -> ref145b); None if none."""
    options = [(b, n) for b, n in compatible_builds(build.photos, builds, exclude=build)
               if b.has_mesh and b.photo_count > build.photo_count]
    if not options:
        return None
    return max(options, key=lambda t: (t[0].photo_count, t[1]))[0]


# ---------------------------------------------------------------- image-quality cache and time estimate

def find_usable_quality_cache(photos, preferred=None, roots=()):
    """(cache path, reason) for an image_quality_cache.csv that covers every photo in {SHA-256: label}.
    Looks in or beside the preferred build first, then anywhere under roots. (None, reason) if none does."""
    candidates = []
    if preferred is not None:
        for p in getattr(preferred, 'paths', ()) or (Path(preferred),):
            c = find_quality_cache(p)
            if c is not None:
                candidates.append(c)
    for root in roots:
        if Path(root).is_dir():
            candidates += sorted(Path(root).rglob(CACHE_NAME))
    manifest = {'images': [{'sha256': h} for h in photos]}
    seen, best = set(), None
    for c in candidates:
        c = Path(c).resolve()
        if c in seen:
            continue
        seen.add(c)
        usable, why = cache_coverage(c, manifest)
        if usable:
            return c, why
        best = best or why
    return None, best or 'no image-quality cache found'


def mesh_time_estimate(photo_count):
    """Plain text from the pilot runs: 0.2-0.4 min per photo (ref50 15 min, side40_s1 36 min, ref145b 57 min)."""
    lo, hi = max(3, round(0.2 * photo_count)), max(5, round(0.4 * photo_count))
    if hi < 60:
        return f'about {lo}–{hi} minutes'
    if lo < 60:
        return f'about {lo} minutes to {hi / 60:.1f} hours'.replace('.0 hours', ' hours').replace(' 1 hours', ' 1 hour')
    return f'about {lo / 60:.1f}–{hi / 60:.1f} hours'.replace('.0', '')
