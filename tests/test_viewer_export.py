"""Web viewer export of a single build with no reference (built from scratch)."""
import json

import numpy as np

from bd3d_quality import viewer_export as vx
from test_coverage import write_rings


def write_box_obj(path, half=0.3):
    corners = np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)])
    faces = [(1, 2, 4), (1, 4, 3), (5, 7, 8), (5, 8, 6), (1, 5, 6), (1, 6, 2),
             (3, 4, 8), (3, 8, 7), (1, 3, 7), (1, 7, 5), (2, 6, 8), (2, 8, 4)]
    path.write_text(''.join(f'v {x} {y} {z}\n' for x, y, z in corners) +
                    ''.join(f'f {a} {b} {c}\n' for a, b, c in faces))


def test_build_without_reference_exports_mesh_cameras_and_gaps(tmp_path):
    run = tmp_path / 'capture-mesh'
    run.mkdir()
    full = np.arange(0, 360, 10)
    low = [a for a in full if not 150 < a < 270]
    write_rings(run / 'cameras.csv', [(5, low), (30, full), (55, full)], failed=['bad1'])
    write_box_obj(run / 'raw-mesh.obj')

    out = vx.export_build(run, out=tmp_path / 'data', log=lambda t: None)
    scene = json.loads((out / 'scene.json').read_text())
    r = scene['runs'][0]
    assert scene['mode'] == 'build' and scene['reference_mesh'] is None
    assert r['aligned'] == len(low) + 72 and r['failed'] == ['bad1']
    assert r['cameras'] == ['used'] * r['aligned'] == ['used'] * len(scene['camera_positions'])
    assert r['coverage']['summary'].startswith('Missing: low cameras from 150° to 270°')
    assert (out / r['file']).stat().st_size > 0 and r['faces'] == 12
    listing = json.loads((tmp_path / 'data' / 'index.json').read_text())
    assert listing[0]['name'] == 'capture-mesh'


def test_alignment_only_run_exports_cameras(tmp_path):
    run = tmp_path / 'capture-align'
    run.mkdir()
    write_rings(run / 'cameras.csv', [(10, np.arange(0, 360, 15)), (40, np.arange(0, 360, 15))])
    out = vx.export_build(run, out=tmp_path / 'data', log=lambda t: None)
    scene = json.loads((out / 'scene.json').read_text())
    assert scene['runs'][0]['file'] is None and scene['unit'] == 'half the camera spread'
    assert not list(out.glob('*.bin'))
