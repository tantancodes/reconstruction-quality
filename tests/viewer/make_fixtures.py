"""Write tests/viewer/fixtures/coverage.json: what coverage.analyze computes for every tracked cameras.csv.
The Node tests (tests/viewer/*.test.mjs) check the viewer's JavaScript port against it.

  python tests/viewer/make_fixtures.py
"""
import json
from pathlib import Path

from bd3d_quality import coverage

REPO = Path(__file__).resolve().parents[2]


def main():
    out = []
    for path in sorted(REPO.glob('results/**/cameras.csv')):
        try:
            c = coverage.analyze(path)
        except ValueError as exc:      # too few aligned cameras: the viewer must refuse it too
            out.append({'path': str(path.relative_to(REPO)), 'error': str(exc)})
            continue
        labels, aligned, P, V = coverage.load_cameras(path)
        centre, B = coverage.camera_frame(P[aligned], V[aligned])
        out.append({'path': str(path.relative_to(REPO)), 'centre': centre.tolist(), 'B': B.tolist(),
                    'counts': c.counts.tolist(), 'el_edges': c.el_edges.tolist(), 'summary': c.summary,
                    'azimuth': c.azimuth.tolist(), 'elevation': c.elevation.tolist()})
    pairs = []      # every tracked run with a reference of the same capture
    for ref_name, folder in (('ref145b', 'results/pahma-5-613-v2'), ('ref50', 'results/gpu-pilot-d8'),
                             ('ref50', 'results/gpu-pilot-d8-ext'), ('_reference', 'results/hist16-old-alignment')):
        ref = (REPO / 'results/gpu-pilot-d8/ref50' if ref_name == 'ref50' else REPO / folder / ref_name) / 'cameras.csv'
        for run in sorted((REPO / folder).glob('*/cameras.csv')):
            if run.parent.name in (ref_name, '_reference') or not ref.exists():
                continue
            try:
                c = coverage.compare(run, ref)
            except ValueError as exc:
                pairs.append({'run': str(run.relative_to(REPO)), 'reference': str(ref.relative_to(REPO)), 'error': str(exc)})
                continue
            pairs.append({'run': str(run.relative_to(REPO)), 'reference': str(ref.relative_to(REPO)),
                          'counts': c.counts.tolist(), 'missing': c.missing.tolist(), 'never': c.never.tolist(),
                          'summary': c.summary, 'shared': c.fit[0], 'residual': c.fit[1],
                          'azimuth': c.azimuth.tolist(), 'elevation': c.elevation.tolist()})
    (Path(__file__).with_name('fixtures') / 'compare.json').write_text(json.dumps(pairs))
    print(f'{len(pairs)} run/reference pairs -> tests/viewer/fixtures/compare.json')
    dest = Path(__file__).with_name('fixtures') / 'coverage.json'
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out))
    print(f'{len(out)} cameras.csv files -> {dest.relative_to(REPO)}')


if __name__ == '__main__':
    main()
