import hashlib
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from bd3d_quality import builds as B
from bd3d_quality.manifests import create_manifest

REPO = Path(__file__).resolve().parents[1]


def touch(path, data=b'x'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def sha(data):
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- photo folders

def test_scan_counts_photos_and_warns(tmp_path):
    for i in range(12):
        touch(tmp_path / f'IMG_{i:02d}.JPG', bytes([i]))
    touch(tmp_path / 'IMG_99.heic')
    touch(tmp_path / 'IMG_98.CR2')
    touch(tmp_path / 'notes')
    touch(tmp_path / '.DS_Store')                      # hidden non-photo: silently ignored
    touch(tmp_path / 'extra/side/IMG_50.png')
    touch(tmp_path / 'extra/IMG_51.tif')
    touch(tmp_path / '._IMG_00.JPG')                   # macOS resource fork that looks like a photo
    touch(tmp_path / '.hidden_dir/IMG_77.jpg')         # included by create_manifest too: counted, flagged
    pf = B.scan_photo_folder(tmp_path)
    assert pf.ok and len(pf.photos) == 16 and pf.summary == '16 photos found'
    assert pf.ignored == {'.heic': 1, '.cr2': 1, '(no extension)': 1}
    assert pf.subfolders == ['.hidden_dir', 'extra', 'extra/side'] and pf.hidden == 2
    text = ' '.join(pf.warnings)
    assert '3 files ignored (.cr2, .heic, (no extension)' in text
    assert 'inside subfolders (3 folders)' in text and '2 hidden photo files' in text
    assert 'recommended' not in text


def test_scan_warns_about_few_photos_and_empty_folders(tmp_path):
    for i in range(4):
        touch(tmp_path / 'few' / f'{i}.jpg')
    pf = B.scan_photo_folder(tmp_path / 'few')
    assert pf.ok and pf.warnings == ['Only 4 photos: at least 10 are recommended.']
    (tmp_path / 'empty').mkdir()
    pf = B.scan_photo_folder(tmp_path / 'empty')
    assert not pf.ok and pf.summary == 'No photos found' and 'No supported photos' in pf.warnings[0]


def test_scan_reports_unreadable_folders(tmp_path):
    with pytest.raises(ValueError, match='does not exist'):
        B.scan_photo_folder(tmp_path / 'missing')
    touch(tmp_path / 'file.jpg')
    with pytest.raises(ValueError, match='is a file'):
        B.scan_photo_folder(tmp_path / 'file.jpg')


@pytest.mark.skipif(os.geteuid() == 0, reason='root ignores permissions')
def test_scan_reports_permission_denied(tmp_path):
    locked = tmp_path / 'locked'
    touch(locked / 'a.jpg')
    locked.chmod(0)
    try:
        with pytest.raises(ValueError, match='denied'):
            B.scan_photo_folder(locked)
    finally:
        locked.chmod(0o755)


def test_scan_order_and_fingerprints_match_the_manifest(tmp_path):
    for i, name in enumerate(['b.jpg', 'a.png', 'sub/c.jpg', 'A2.tif', '.h/d.jpg']):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (8, 8), (i * 40, 0, 0)).save(tmp_path / name)
    pf = B.scan_photo_folder(tmp_path)
    manifest = create_manifest(tmp_path, 'x')
    assert [p.as_posix() for p in pf.photos] == [e['path'] for e in manifest['images']]
    seen = []
    fp = B.photo_fingerprints(pf, on_progress=lambda d, t: seen.append((d, t)))
    assert fp == {e['sha256']: Path(e['path']).stem for e in manifest['images']}
    assert seen[-1] == (5, 5)


# ---------------------------------------------------------------- builds

def write_build(folder, photos, mesh=False, run_id=None, parent=None, finished='2026-11-10T06:00:00+00:00',
                manifest=True, cameras=None):
    """photos: {label: bytes}."""
    folder.mkdir(parents=True)
    rows = ['label,aligned,x,y,z,view_x,view_y,view_z'] + [f'{l},True,1,0,0,-1,0,0' for l in (cameras or photos)]
    (folder / 'cameras.csv').write_text('\n'.join(rows) + '\n')
    if manifest:
        images = [{'path': f'{l}.jpg', 'sha256': sha(d), 'size_bytes': len(d)} for l, d in photos.items()]
        (folder / 'manifest.json').write_text(json.dumps({'images': images, 'image_count': len(images)}))
    run = {'run_id': run_id or folder.name, 'finished_at': finished}
    if parent:
        run['parent_checkpoint'] = {'run_id': parent}
    (folder / 'run.json').write_text(json.dumps(run))
    if mesh:
        (folder / 'raw-mesh.obj').write_text('v 0 0 0\n')
    return folder


def photo_set(n, start=0):
    return {f'IMG_{i:03d}': f'photo {i}'.encode() for i in range(start, start + n)}


def test_find_builds_lists_runs_with_cameras_and_merges_seeded_mesh_runs(tmp_path):
    write_build(tmp_path / 'results/obj/ref', photo_set(50), mesh=True, finished='2026-11-09T10:00:00+00:00')
    write_build(tmp_path / 'runs/app/dog-align', photo_set(30), run_id='A1')
    write_build(tmp_path / 'runs/app/dog-mesh', photo_set(30), mesh=True, run_id='M1', parent='A1',
                finished='2026-11-12T10:00:00+00:00')
    write_build(tmp_path / 'runs/app/cat-align', photo_set(20), run_id='A2', finished='2026-11-11T10:00:00+00:00')
    write_build(tmp_path / 'results/old/legacy', {}, manifest=False, cameras=['p1', 'p2', 'p3'],
                finished='2026-08-01T00:00:00+00:00')
    (tmp_path / 'results/obj/not-a-build').mkdir()
    (tmp_path / 'results/obj/not-a-build/run.json').write_text('{}')
    got = B.find_builds([tmp_path / 'results', tmp_path / 'runs/app', tmp_path / 'nowhere'])
    assert [(b.name, b.group, b.photo_count, b.has_mesh) for b in got] == [
        ('dog', 'app', 30, True), ('cat', 'app', 20, False), ('ref', 'obj', 50, True), ('legacy', 'old', 3, False)]
    dog = got[0]
    assert dog.path.name == 'dog-mesh' and [p.name for p in dog.paths] == ['dog-align', 'dog-mesh']
    assert got[-1].photos is None                       # no manifest: listed, never comparable


def test_compatible_builds_match_content_and_name(tmp_path):
    ref = B.read_build(write_build(tmp_path / 'ref', photo_set(50), mesh=True))
    part = B.read_build(write_build(tmp_path / 'part', photo_set(20, start=40)))           # shares 10
    two = B.read_build(write_build(tmp_path / 'two', photo_set(2)))                         # shares 2 < 3
    renamed = {f'X_{i}': d for i, d in enumerate(photo_set(50).values())}                  # same content, other names
    other = B.read_build(write_build(tmp_path / 'renamed', renamed, mesh=True))
    legacy = B.read_build(write_build(tmp_path / 'legacy', photo_set(50), manifest=False))
    builds = [ref, part, two, other, legacy]
    mine = {sha(d): l for l, d in photo_set(45).items()}
    assert [(b.name, n) for b, n in B.compatible_builds(mine, builds)] == [('ref', 45), ('part', 5)]
    assert [(b.name, n) for b, n in B.compatible_builds(ref.photos, builds, exclude=ref)] == [('part', 10)]
    assert B.compatible_builds(None, builds) == []


def test_default_reference_is_the_largest_compatible_build_with_a_mesh(tmp_path):
    big = B.read_build(write_build(tmp_path / 'big', photo_set(60), mesh=True))
    mid = B.read_build(write_build(tmp_path / 'mid', photo_set(50), mesh=True))
    huge_no_mesh = B.read_build(write_build(tmp_path / 'nomesh', photo_set(80)))
    small = B.read_build(write_build(tmp_path / 'small', photo_set(30)))
    builds = [big, mid, huge_no_mesh, small]
    assert B.default_reference(small, builds) is big
    assert B.default_reference(big, builds) is None          # nothing larger with a mesh


def write_cache(path, shas, source='run1'):
    rows = ['path,sha256,image_quality,source_run_id,source_experiment_id,metashape_version']
    rows += [f'p{i}.jpg,{h},0.8,{source},exp,2.2.1' for i, h in enumerate(shas)]
    touch(path, ('\n'.join(rows) + '\n').encode())
    return path


def test_find_usable_quality_cache(tmp_path):
    photos = {sha(d): l for l, d in photo_set(10).items()}
    ref = B.read_build(write_build(tmp_path / 'results/obj/ref', photo_set(10), mesh=True))
    beside = write_cache(tmp_path / 'results/obj/image_quality_cache.csv', list(photos))
    assert B.find_usable_quality_cache(photos, ref) == (beside.resolve(), 'cache covers all 10 photos')
    beside.unlink()
    write_cache(tmp_path / 'results/obj/ref/image_quality_cache.csv', list(photos)[:7])   # partial
    elsewhere = write_cache(tmp_path / 'results/zzz/image_quality_cache.csv', list(photos))
    assert B.find_usable_quality_cache(photos, ref, roots=[tmp_path / 'results'])[0] == elsewhere.resolve()
    elsewhere.unlink()
    assert B.find_usable_quality_cache(photos, ref, roots=[tmp_path / 'results']) == \
        (None, 'cache covers 7 of 10 photos (matched by SHA-256)')
    assert B.find_usable_quality_cache(photos) == (None, 'no image-quality cache found')


def test_mesh_time_estimate_scales_with_photos():
    assert B.mesh_time_estimate(5) == 'about 3–5 minutes'
    assert B.mesh_time_estimate(50) == 'about 10–20 minutes'           # ref50 took 15 min
    assert B.mesh_time_estimate(145) == 'about 29–58 minutes'          # ref145b took 57 min
    assert B.mesh_time_estimate(200) == 'about 40 minutes to 1.3 hours'
    assert B.mesh_time_estimate(300) == 'about 1–2 hours'


@pytest.mark.skipif(not (REPO / 'results/gpu-pilot-d8/ref50/manifest.json').exists(), reason='pilot results absent')
def test_real_builds_pick_the_pilot_references():
    builds = B.find_builds([REPO / 'results'])
    by = {(b.group, b.name): b for b in builds}
    assert B.default_reference(by['gpu-pilot-d8', 'side40'], builds).name == 'ref50'
    assert B.default_reference(by['pahma-5-613-v2', 'side40_s1'], builds).name == 'ref145b'
    names = {b.name for b, _ in B.compatible_builds(by['gpu-pilot-d8', 'ref50'].photos, builds)}
    assert 'side40' in names and not any(n.startswith('arf-dog-image-count') for n in names)  # no manifests
