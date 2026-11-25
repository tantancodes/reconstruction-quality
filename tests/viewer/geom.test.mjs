// node --test tests/viewer/   (fixtures: python tests/viewer/make_fixtures.py)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { analyze, eigh3, toDisplay, describeGaps } from '../../viewer/js/geom.js';
import { readCameras } from '../../viewer/js/csv.js';

const repo = fileURLToPath(new URL('../../', import.meta.url));
const fixtures = JSON.parse(readFileSync(new URL('./fixtures/coverage.json', import.meta.url)));

test('eigh3 reconstructs a symmetric matrix', () => {
  const M = [[4, 1, 0.5], [1, 3, 0.2], [0.5, 0.2, 1]];
  const { values, vectors } = eigh3(M);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    const r = [0, 1, 2].reduce((s, k) => s + values[k] * vectors[k][i] * vectors[k][j], 0);
    assert.ok(Math.abs(r - M[i][j]) < 1e-12);
  }
  assert.ok(values[0] <= values[1] && values[1] <= values[2]);
});

test(`coverage matches the Python on all ${fixtures.length} tracked cameras.csv`, () => {
  for (const f of fixtures) {
    const cams = readCameras(readFileSync(repo + f.path, 'utf8'), f.path);
    if (f.error) { assert.throws(() => analyze(cams)); continue; }
    const a = analyze(cams);
    const close = (x, y, tol, what) => assert.ok(Math.abs(x - y) <= tol, `${f.path}: ${what} ${x} vs ${y}`);
    a.centre.forEach((x, k) => close(x, f.centre[k], 1e-6 * (1 + Math.abs(f.centre[k])), 'centre'));
    a.B.forEach((row, r) => row.forEach((x, k) => close(x, f.B[r][k], 1e-6, `B[${r}]`)));
    a.azel.forEach(([az, el], i) => {
      const d = Math.abs(az - f.azimuth[i]);
      close(Math.min(d, 360 - d), 0, 1e-5, 'azimuth'); close(el, f.elevation[i], 1e-5, 'elevation');
    });
    // The first aligned camera defines azimuth 0, so it sits exactly on the 0/360 seam. Rounding puts it at
    // +1e-15 deg (first bin) in one implementation and 360 deg (last bin) in the other. Move such seam
    // cameras to the bin Python chose; every other count must match exactly.
    const counts = a.counts.map(r => r.slice()), last = counts[0].length - 1;
    a.azel.forEach(([az], i) => {
      const py = f.azimuth[i];
      if (Math.min(az, 360 - az) > 1e-9) return;
      const row = a.elEdges.findIndex((e, k) => k < a.elEdges.length - 1 && a.azel[i][1] <= a.elEdges[k + 1] && a.azel[i][1] >= e);
      const from = az > 180 ? last : 0, to = py > 180 ? last : 0;
      if (row >= 0 && from !== to) { counts[row][from]--; counts[row][to]++; }
    });
    assert.deepEqual(counts, f.counts, `${f.path}: counts`);
    a.elEdges.forEach((x, k) => close(x, f.el_edges[k], 1e-6, 'el_edges'));
    if (JSON.stringify(counts) === JSON.stringify(a.counts)) assert.equal(a.summary, f.summary, `${f.path}: summary`);
    else assert.ok(f.summary.includes(describeGaps(counts.map(r => r.map(c => c < 2)), a.azEdges) ?? 'No gaps'), `${f.path}: summary`);
  }
});

test('toDisplay maps up to +y and keeps the frame right-handed', () => {
  const B = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  assert.deepEqual(toDisplay([0, 0, 2], [0, 0, 0], B, 2), [0, 1, -0]);
  assert.deepEqual(toDisplay([0, 2, 0], [0, 0, 0], B, 2), [0, 0, -1]);
});
