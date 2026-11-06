from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from PIL import Image
import pytest
from bd3d_quality.manifests import create_manifest


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'images'
    root.mkdir()
    a = np.random.default_rng(5).integers(0, 256, (96, 128, 3), dtype=np.uint8)
    for i in range(3):
        Image.fromarray(np.roll(a, i * 2, axis=1)).save(root / f'{i}.png')
    return root, create_manifest(root, 'test-object')


@pytest.fixture
def fake_api():
    class Chunk:
        def __init__(self):
            self.cameras = []
            self.tie_points = None
            self.point_cloud = None
            self.model = None
        def addPhotos(self, paths):
            self.cameras = [NS(transform=None, photo=NS(path=p)) for p in paths]
        def matchPhotos(self, **kwargs):
            self.match_settings = kwargs
        def alignCameras(self):
            for c in self.cameras:
                c.transform = 'aligned'
            self.tie_points = NS(points=[NS(valid=True), NS(valid=False)])
        def exportPointCloud(self, path, **kwargs):
            Path(path).write_text('ply\nend_header\n')
        def buildDepthMaps(self, **kwargs):
            pass
        def buildPointCloud(self):
            self.point_cloud = NS(point_count=27)
        def buildModel(self, **kwargs):
            self.model = NS(vertices=[1, 2, 3], faces=[1])
        def exportModel(self, path, **kwargs):
            Path(path).write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
    chunks = []
    class Document:
        def __init__(self):
            self.chunks = []
        def addChunk(self):
            c = Chunk()
            self.chunks.append(c)
            chunks.append(c)
            return c
        def save(self, path):
            Path(path).write_text('mock checkpoint')
        def open(self, path):
            self.chunks = [chunks[-1]]
    api = NS(Document=Document, Chunk=Chunk, app=NS(version='2.2.1', activated=True, gpu_mask=3, cpu_enable=False))
    for name in ['TiePointsData', 'PointCloudData', 'PointCloudFormatPLY', 'DepthMapsData',
                 'Arbitrary', 'EnabledInterpolation', 'HighFaceCount', 'ModelFormatOBJ', 'MildFiltering']:
        setattr(api, name, name)
    return api


@pytest.fixture
def tetra(tmp_path):
    mesh = 'v 0 0 0\nv 1 0 0\nv 0 2 0\nv 0 0 3\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'
    p = tmp_path / 'tetra.obj'
    p.write_text(mesh)
    return p
