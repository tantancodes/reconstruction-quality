"""Minimal stand-in for the licensed Metashape module, for subprocess tests of the app pipeline.

Cameras are placed on rings around the origin (two rings, looking at the centre) in photo order,
so cameras.csv has real poses. The mesh is an octahedron. Projects are JSON photo lists, so a
seeded run can reopen them. FAKE_METASHAPE_SLEEP (seconds) slows alignment, for cancel tests;
FAKE_METASHAPE_ACTIVATED=0 simulates a missing license.
"""
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np


class _App(NS):
    pass


app = _App(version='2.2.1', activated=os.environ.get('FAKE_METASHAPE_ACTIVATED', '1') == '1',
           gpu_mask=1, cpu_enable=False)
for _name in ['TiePointsData', 'PointCloudData', 'PointCloudFormatPLY', 'DepthMapsData', 'Arbitrary',
              'EnabledInterpolation', 'HighFaceCount', 'CustomFaceCount', 'ModelFormatOBJ', 'NoFiltering',
              'MildFiltering', 'ModerateFiltering', 'AggressiveFiltering']:
    globals()[_name] = _name


def _pose(i, n):
    ring, k = i % 2, i // 2
    per = math.ceil(n / 2)
    az, el = 2 * math.pi * k / per, math.radians(10 if ring == 0 else 40)
    d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    z = -d                                          # camera +Z looks at the centre
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, 3 * d
    return m


class Chunk:
    def __init__(self):
        self.cameras, self.tie_points, self.point_cloud, self.model = [], None, None, None
        self.transform = NS(matrix=np.eye(4))

    def addPhotos(self, paths):
        self.cameras = [NS(label=Path(p).stem, transform=None, photo=NS(path=p), meta={}) for p in paths]

    def analyzeImages(self, cameras):
        for c in cameras:
            c.meta['Image/Quality'] = '0.8'

    def matchPhotos(self, **kwargs):
        pass

    def alignCameras(self):
        time.sleep(float(os.environ.get('FAKE_METASHAPE_SLEEP', '0')))
        for i, c in enumerate(self.cameras):
            c.transform = _pose(i, len(self.cameras))
        self.tie_points = NS(points=[NS(valid=True, track_id=0)])

    def exportPointCloud(self, path, **kwargs):
        Path(path).write_text('ply\nend_header\n')

    def buildDepthMaps(self, **kwargs):
        pass

    def buildPointCloud(self):
        self.point_cloud = NS(point_count=100)

    def buildModel(self, **kwargs):
        self.model = NS(vertices=list(range(6)), faces=list(range(8)))

    def exportModel(self, path, **kwargs):
        v = ['1 0 0', '-1 0 0', '0 1 0', '0 -1 0', '0 0 1', '0 0 -1']
        f = ['1 3 5', '3 2 5', '2 4 5', '4 1 5', '3 1 6', '2 3 6', '4 2 6', '1 4 6']
        Path(path).write_text(''.join(f'v {x}\n' for x in v) + ''.join(f'f {x}\n' for x in f))


class Document:
    def __init__(self):
        self.chunks = []

    def addChunk(self):
        self.chunks.append(Chunk())
        return self.chunks[-1]

    def save(self, path):
        c = self.chunks[0]
        Path(path).write_text(json.dumps({'photos': [x.photo.path for x in c.cameras],
                                          'aligned': c.cameras[0].transform is not None}))
        files = Path(path).with_suffix('.files')
        files.mkdir(exist_ok=True)
        (files / 'doc.json').write_text('{}')

    def open(self, path):
        state = json.loads(Path(path).read_text())
        c = self.addChunk()
        c.addPhotos(state['photos'])
        if state['aligned']:
            c.alignCameras()
