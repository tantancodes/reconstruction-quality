// Datasets the viewer can show, from four sources, all turned into one scene description:
//   public   viewer/public/datasets.json: tracked cameras.csv + measured labels (no restricted assets)
//   export   viewer/data/*/scene.json written by bd3d_quality.viewer_export (meshes, error maps); only
//            present when the viewer is served locally by `python -m bd3d_quality.viewer_export serve`
//   folder   a folder opened from disk: build folders (cameras.csv [+ raw-mesh.obj]) and export folders
//   file     one cameras.csv opened from disk
// Local files are read with the File API in this tab. Nothing is uploaded.
import { readCameras } from './csv.js';
import { analyze, compare, perCamera, azimuthGaps, displayToFrame, azimuthElevation, vec } from './geom.js';

const here = new URL('..', import.meta.url);           // viewer/
const fetchJson = async url => { const r = await fetch(url, { cache: 'no-store' }); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); };
const fetchText = async url => { const r = await fetch(url, { cache: 'no-store' }); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.text(); };

// ------------------------------------------------------------------ catalogue entries
// entry: { key, group, name, kind: 'cameras'|'export', role: 'build'|'reference', source, meta,
//          cameras(): Promise<parsed cameras.csv>, mesh: File|null, refKey, sizeNote }

export async function publicEntries() {
  let m;
  try { m = await fetchJson(new URL('public/datasets.json', here)); } catch { return { entries: [], manifest: null }; }
  const entries = m.datasets.map(d => ({
    key: `public:${d.id}`, group: `${d.object} · ${d.capture.split('/')[1]}`, name: d.run, kind: 'cameras', role: d.role,
    source: 'public', meta: d, refKey: d.reference ? `public:${d.reference}` : null,
    cameras: memo(async () => readCameras(await fetchText(new URL(d.cameras, here)), d.cameras_source)),
    files: [d.cameras_source],
  }));
  return { entries, manifest: m };
}

export async function exportEntries() {
  let list;
  try { list = await fetchJson(new URL('data/index.json', here)); } catch { return []; }
  const out = [];
  for (const o of list) {
    const base = new URL(`data/${encodeURIComponent(o.name)}/`, here);
    const scene = memo(() => fetchJson(new URL('scene.json', base)));
    const bin = file => fetch(new URL(file, base)).then(r => { if (!r.ok) throw new Error(`${file}: ${r.status}`); return r.arrayBuffer(); });
    out.push(...await exportRuns(o.title || o.name, scene, bin, 'export', `viewer/data/${o.name}/`));
  }
  return out;
}

async function exportRuns(title, scene, bin, source, where) {
  let s;
  try { s = await scene(); } catch { return []; }
  return s.runs.map(run => ({
    key: `${source}:${where}${run.id}`, group: `Export · ${title}`, name: run.id, kind: 'export', role: 'build', source,
    meta: { label: run.label, qc: run.qc, reference: s.reference }, scene, run, bin, files: [where + 'scene.json'],
  }));
}

// A folder opened from disk: every subfolder holding cameras.csv is a build, every one holding scene.json
// an export. A folder named ref* (or _reference) next to builds is offered as their reference.
export async function folderEntries(files) {
  const byDir = new Map();
  for (const f of files) {
    const path = f.webkitRelativePath || f.relPath || f.name, dir = path.includes('/') ? path.slice(0, path.lastIndexOf('/')) : '.';
    if (!byDir.has(dir)) byDir.set(dir, new Map());
    byDir.get(dir).set(path.slice(dir.length + 1) || path, f);
  }
  const entries = [];
  for (const [dir, fm] of byDir) {
    const name = dir.split('/').pop(), parent = dir.includes('/') ? dir.slice(0, dir.lastIndexOf('/')) : '';
    if (fm.has('scene.json')) {
      const scene = memo(async () => JSON.parse(await fm.get('scene.json').text()));
      const bin = async file => { const f = fm.get(file); if (!f) throw new Error(`${file} is not in ${dir}`); return f.arrayBuffer(); };
      entries.push(...await exportRuns(name, scene, bin, 'local', `${dir}/`));
    }
    if (fm.has('cameras.csv')) {
      const mesh = fm.get('raw-mesh.obj') || null;
      const json = async n => (fm.has(n) ? JSON.parse(await fm.get(n).text()) : null);
      entries.push({
        key: `local:${dir}`, group: `Local · ${parent || name}`, name, kind: 'cameras', parent,
        role: /^(ref|_reference)/.test(name) ? 'reference' : 'build', source: 'local', mesh,
        cameras: memo(async () => readCameras(await fm.get('cameras.csv').text(), `${dir}/cameras.csv`)),
        metrics: memo(async () => (await json('metrics-initial_alignment.json'))?.metrics ?? null),
        files: [...fm.keys()].map(k => `${dir}/${k}`),
        meta: {},
      });
    }
  }
  for (const e of entries) {   // default reference: a ref* folder beside it
    if (e.kind !== 'cameras' || e.role === 'reference') continue;
    const ref = entries.find(r => r.kind === 'cameras' && r.role === 'reference' && r.parent === e.parent);
    e.refKey = ref ? ref.key : null;
  }
  return entries;
}

export function fileEntry(file) {
  return { key: `local:${file.name}:${file.lastModified}`, group: 'Local · files', name: file.name, kind: 'cameras', role: 'build',
           source: 'local', mesh: null, cameras: memo(async () => readCameras(await file.text(), file.name)), files: [file.name], meta: {} };
}

// Files dropped on the page, folders included (DataTransferItem.webkitGetAsEntry).
export async function droppedFiles(items) {
  const out = [];
  const walk = (entry, path) => new Promise(resolve => {
    if (entry.isFile) entry.file(f => { f.relPath = path + f.name; out.push(f); resolve(); }, resolve);
    else {
      const reader = entry.createReader(), all = [];
      const more = () => reader.readEntries(async batch => {
        if (!batch.length) { for (const e of all) await walk(e, path + entry.name + '/'); resolve(); }
        else { all.push(...batch); more(); }
      }, resolve);
      more();
    }
  });
  for (const it of items) { const e = it.webkitGetAsEntry?.(); if (e) await walk(e, ''); }
  return out;
}

function memo(fn) { let p; return () => (p ??= fn()); }

// ------------------------------------------------------------------ scenes
// scene: { title, unit, frame, cameras: [{label, p, v, state, az, el, nnAngle, offAxis, distance, ...extra}],
//          failed, coverage {counts, missing, never, azEdges, elEdges, summary, relative}, gaps, fit,
//          mesh: null | () => Promise<{positions, index?, missing?, extra?}>, refMesh, metrics, notes, files }

const FRAME_NOTE = 'Frame from the cameras (coverage.camera_frame): origin = point nearest all optical axes; up = ring normal; '
  + 'azimuth 0° = first aligned camera. Metashape builds without markers have no true up or north.';

export async function sceneFromCameras(entry, refEntry) {
  const cams = await entry.cameras();
  const ref = refEntry ? await refEntry.cameras() : null;
  const c = ref ? compare(cams, ref) : analyze(cams);
  // Display scale: half the spread of the (reference) cameras, as viewer_export does for builds without a mesh.
  const basis = ref ?? cams, al = basis.P.filter((_, i) => basis.aligned[i]);
  const ptp = [0, 1, 2].map(k => Math.max(...al.map(p => p[k])) - Math.min(...al.map(p => p[k])));
  const diag = 0.5 * Math.hypot(...ptp);
  const disp = q => [q[0] / diag, q[2] / diag, -q[1] / diag];
  const stats = perCamera(c.Q, c.W);
  const cameras = c.index.map((i, k) => ({
    label: cams.labels[i], p: disp(c.Q[k]), v: [c.W[k][0], c.W[k][2], -c.W[k][1]], state: 'used',
    az: c.azel[k][0], el: c.azel[k][1], ...stats[k], distance: stats[k].distance / diag, ...cams.extra[i],
  }));
  if (ref) {   // reference cameras this build does not have, drawn where the reference has them
    const inRun = new Map(cams.labels.map((l, i) => [l, cams.aligned[i]]));
    const r = c.reference;
    r.labels.forEach((l, k) => {
      if (inRun.get(l)) return;
      cameras.push({ label: l, p: disp(r.Q[k]), v: [r.W[k][0], r.W[k][2], -r.W[k][1]], state: inRun.has(l) ? 'unaligned' : 'removed',
                     az: r.azel[k][0], el: r.azel[k][1], refOnly: true, ...ref.extra[ref.labels.indexOf(l)] });
    });
  }
  let mesh = null;
  if (entry.mesh) {
    const T = c.fit?.T, B = c.B, ctr = c.centre;
    mesh = async () => {
      const positions = parseObjPositions(await entry.mesh.text());
      for (let i = 0; i < positions.vertices.length; i += 3) {
        let p = [positions.vertices[i], positions.vertices[i + 1], positions.vertices[i + 2]];
        if (T) p = [0, 1, 2].map(r => vec.dot(T.sR[r], p) + T.t[r]);
        const d = vec.sub(p, ctr), q = [vec.dot(B[0], d), vec.dot(B[1], d), vec.dot(B[2], d)];
        positions.vertices.set(disp(q), i);
      }
      return { positions: positions.vertices, index: positions.index };
    };
  }
  const metrics = entry.metrics ? await entry.metrics() : null;
  const m = entry.meta || {};
  const notes = [];
  if (ref) notes.push(`Build cameras placed in the reference frame by a similarity fit on ${c.fit.shared} shared photos `
    + `(residual ${(100 * c.fit.residual).toFixed(2)}% of the reference camera spread). Same fit as evaluation.camera_transform; no ICP.`);
  if (entry.mesh && ref) notes.push('Mesh placed with the camera fit only. Surface error needs the ICP registration of the Python export (viewer_export compare).');
  return {
    title: entry.name, source: entry.source, entry, unit: ref ? 'half the reference camera spread' : 'half the camera spread', frame: FRAME_NOTE,
    cameras, failed: c.failed, total: cams.labels.length,
    coverage: { counts: c.counts, missing: c.missing, never: c.never, azEdges: c.azEdges, elEdges: c.elEdges, summary: c.summary, relative: !!ref },
    gaps: azimuthGaps(c.azel, c.elEdges), fit: c.fit ?? null, reference: refEntry?.name ?? null,
    mesh, refMesh: null, notes, files: entry.files,
    measured: m.label != null ? { label: m.label, qc: m.qc, completeness: m.completeness_at_1pct, accuracy: m.accuracy_at_1pct, source: m.label_source, reference: m.reference } : null,
    reprojection_px: m.reprojection_px ?? metrics?.reprojection_rmse_px ?? null,
    tie_points: metrics?.valid_tie_point_count ?? null,
  };
}

// An export from viewer_export: cameras, coverage and meshes are already in the display frame.
export async function sceneFromExport(entry) {
  const s = await entry.scene(), run = entry.run;
  const Q = s.camera_positions.map(displayToFrame), W = s.camera_views.map(displayToFrame);
  const azel = azimuthElevation(Q);
  const used = run.cameras.map(st => st === 'used');
  const usedIdx = used.map((u, i) => (u ? i : -1)).filter(i => i >= 0);
  const stats = perCamera(usedIdx.map(i => Q[i]), usedIdx.map(i => W[i]));
  const cameras = s.camera_labels.map((l, i) => {
    const k = usedIdx.indexOf(i);
    return { label: l, p: s.camera_positions[i], v: s.camera_views[i], state: run.cameras[i], az: azel[i][0], el: azel[i][1],
             refOnly: run.cameras[i] !== 'used', ...(k >= 0 ? stats[k] : {}), ...(s.camera_extra?.[i] ?? {}) };
  });
  const lo = s.bounds[0], span = s.bounds[1].map((h, a) => h - lo[a]);
  const decode = async (file, buffers) => {
    const buf = await entry.bin(file), view = b => new ({ uint16: Uint16Array, uint32: Uint32Array, uint8: Uint8Array }[b.type])(buf, b.offset, b.count);
    const q = view(buffers.positions), positions = new Float32Array(q.length);
    for (let i = 0; i < q.length; i++) positions[i] = lo[i % 3] + q[i] / 65535 * span[i % 3];
    return { positions, index: view(buffers.indices), missing: buffers.missing ? view(buffers.missing) : null, extra: buffers.extra ? view(buffers.extra) : null };
  };
  const relative = s.mode === 'compare';
  return {
    title: run.id, source: entry.source, entry, unit: s.unit, frame: FRAME_NOTE, cameras,
    failed: run.failed ?? [], total: s.reference_images,
    coverage: { counts: run.coverage.counts, missing: run.coverage.missing, never: run.coverage.never, azEdges: s.grid.az_edges,
                elEdges: s.grid.el_edges, summary: run.coverage.summary, relative },
    gaps: azimuthGaps(usedIdx.map(i => azel[i]), s.grid.el_edges), fit: null, reference: s.reference,
    mesh: run.file ? () => decode(run.file, run.buffers) : null,
    refMesh: s.reference_mesh ? () => decode(s.reference_mesh.file, s.reference_mesh.buffers) : null,
    cutoff: s.cutoff, notes: relative ? ['Meshes registered by the v3 scorer (camera-initialised similarity ICP), as for the labels.'] : [],
    files: entry.files,
    measured: run.label != null ? { label: run.label, qc: run.qc, completeness: run.completeness, accuracy: run.accuracy, reference: s.reference, source: entry.files[0] } : null,
    reprojection_px: run.reprojection_px, tie_points: run.tie_points, build_seconds: run.build_seconds,
  };
}

// Minimal OBJ reader: vertices and triangulated faces (fans), which is all raw-mesh.obj needs.
export function parseObjPositions(text) {
  const v = [], f = [];
  let i = 0;
  while (i < text.length) {
    let j = text.indexOf('\n', i); if (j < 0) j = text.length;
    const c0 = text.charCodeAt(i), c1 = text.charCodeAt(i + 1);
    if (c0 === 118 && c1 === 32) { const p = text.slice(i + 2, j).trim().split(/\s+/); v.push(+p[0], +p[1], +p[2]); }
    else if (c0 === 102 && c1 === 32) {
      const p = text.slice(i + 2, j).trim().split(/\s+/).map(t => parseInt(t, 10));
      const n = v.length / 3, idx = p.map(k => (k < 0 ? n + k : k - 1));
      for (let k = 1; k + 1 < idx.length; k++) f.push(idx[0], idx[k], idx[k + 1]);
    }
    i = j + 1;
  }
  return { vertices: new Float32Array(v), index: new Uint32Array(f) };
}
