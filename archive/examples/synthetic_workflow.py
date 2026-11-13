"""Generate redistributable synthetic fixtures and a complete unlicensed smoke workflow.
Usage: PYTHONPATH=archive/src python archive/examples/synthetic_workflow.py results/synthetic-demo
"""
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from bd3d_quality.manifests import create_manifest, validate_manifest
from bd3d_quality.provenance import write_json
from bd3d_legacy.image_features import features_for_manifest
from bd3d_legacy.local_match_features import match_manifest
from bd3d_legacy.geometry_quality import compare_geometry
from bd3d_legacy.reporting import summarize, build_report


def main(output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    root = out / 'images'
    root.mkdir()
    rng = np.random.default_rng(71)
    pixels = rng.integers(0, 256, (192, 256, 3), dtype=np.uint8)
    for i in range(3):
        Image.fromarray(np.roll(pixels, i * 5, axis=1)).save(root / f'view-{i}.png')
    m = create_manifest(root, 'synthetic-texture-v1', description='CPU noise textures; not a physical reconstruction dataset')
    write_json(out / 'manifest.json', m)
    write_json(out / 'validation.json', validate_manifest(m, root))
    features = features_for_manifest(m, root)
    matches = match_manifest(m, root)
    write_json(out / 'features.json', features)
    write_json(out / 'matches.json', matches)
    mesh = 'v 0 0 0\nv 1 0 0\nv 0 2 0\nv 0 0 3\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'
    (out / 'reference.obj').write_text(mesh)
    (out / 'candidate.obj').write_text(mesh)
    write_json(out / 'geometry.json', compare_geometry(out / 'candidate.obj', out / 'reference.obj',
        reference_id='synthetic-tetrahedron', units='arbitrary_shared', sample_count=1000))
    # Geometry is standalone: these fixtures are not outputs of a reconstruction run.
    write_json(out / 'summary.json', summarize(m, features=features, matches=matches))
    build_report([out / 'summary.json'], out / 'report')
    print(out.resolve())


if __name__ == '__main__':
    main(sys.argv[1])
