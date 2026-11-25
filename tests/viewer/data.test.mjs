// The OBJ reader and the camera scene built for the viewer.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { parseObjPositions, sceneFromCameras } from '../../viewer/js/data.js';
import { readCameras } from '../../viewer/js/csv.js';

const repo = fileURLToPath(new URL('../../', import.meta.url));
const entry = (path, extra = {}) => ({ name: path, source: 'local', files: [path], meta: {}, cameras: async () => readCameras(readFileSync(repo + path, 'utf8'), path), ...extra });

test('OBJ faces: 1-based, negative indices, quads split into two triangles', () => {
  const o = parseObjPositions('v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nf 1/1 2/1 3/1 4/1\nf -4 -3 -2\n');
  assert.deepEqual([...o.index], [0, 1, 2, 0, 2, 3, 0, 1, 2]);
  assert.equal(o.vertices.length, 12);
});

test('a build against its reference: every reference camera is shown once, with the right state', async () => {
  const run = 'results/pahma-5-613-v2/side40_s1/cameras.csv', ref = 'results/pahma-5-613-v2/ref145b/cameras.csv';
  const sc = await sceneFromCameras(entry(run), entry(ref));
  const refLabels = readCameras(readFileSync(repo + ref, 'utf8')).labels;
  assert.equal(new Set(sc.cameras.map(c => c.label)).size, sc.cameras.length);
  assert.equal(sc.cameras.filter(c => c.state === 'used').length, 122);
  assert.equal(sc.cameras.filter(c => c.state === 'removed').length, refLabels.length - 122);
  assert.ok(sc.cameras.every(c => c.p.every(Number.isFinite)));
  assert.ok(sc.fit.residual < 0.01);
});

test('the display frame puts up on +y: elevations agree with the y coordinate', async () => {
  const sc = await sceneFromCameras(entry('results/gpu-pilot-d8/side40/cameras.csv'));
  for (const c of sc.cameras) assert.ok(Math.abs(Math.asin(c.p[1] / Math.hypot(...c.p)) * 180 / Math.PI - c.el) < 1e-6);
});
