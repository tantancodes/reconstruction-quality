// Parity of the browser's run-vs-reference coverage (geom.compare) with coverage.compare.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { compare, similarityFit } from '../../viewer/js/geom.js';
import { readCameras } from '../../viewer/js/csv.js';

const repo = fileURLToPath(new URL('../../', import.meta.url));
const pairs = JSON.parse(readFileSync(new URL('./fixtures/compare.json', import.meta.url)));
const load = p => readCameras(readFileSync(repo + p, 'utf8'), p);

test('similarityFit recovers a known similarity transform', () => {
  const src = [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [1, 1, 1]];
  const c = Math.cos(0.7), s = Math.sin(0.7), sc = 2.5, t = [1, -2, 0.5];
  const R = [[c, -s, 0], [s, c, 0], [0, 0, 1]];
  const tgt = src.map(p => R.map((r, i) => sc * (r[0] * p[0] + r[1] * p[1] + r[2] * p[2]) + t[i]));
  const T = similarityFit(src, tgt);
  T.sR.forEach((row, i) => row.forEach((x, j) => assert.ok(Math.abs(x - sc * R[i][j]) < 1e-9)));
  T.t.forEach((x, i) => assert.ok(Math.abs(x - t[i]) < 1e-9));
});

test(`run-vs-reference coverage matches the Python on all ${pairs.length} pairs`, () => {
  for (const f of pairs) {
    if (f.error) { assert.throws(() => compare(load(f.run), load(f.reference))); continue; }
    const c = compare(load(f.run), load(f.reference));
    assert.equal(c.fit.shared, f.shared, `${f.run}: shared`);
    assert.ok(Math.abs(c.fit.residual - f.residual) < 1e-8, `${f.run}: residual ${c.fit.residual} vs ${f.residual}`);
    c.azel.forEach(([az, el], i) => {
      const d = Math.abs(az - f.azimuth[i]);
      assert.ok(Math.min(d, 360 - d) < 1e-6 && Math.abs(el - f.elevation[i]) < 1e-6, `${f.run}: camera ${i}`);
    });
    assert.deepEqual(c.counts, f.counts, `${f.run}: counts`);
    assert.deepEqual(c.never, f.never, `${f.run}: never`);
    assert.deepEqual(c.missing, f.missing, `${f.run}: missing`);
    assert.equal(c.summary, f.summary, `${f.run}: summary`);
  }
});
