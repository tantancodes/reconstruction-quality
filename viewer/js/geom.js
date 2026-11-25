// Camera geometry for the viewer. A port of src/bd3d_quality/coverage.py (camera_frame, analyze,
// describe_gaps) and viewer_export.to_display, so a folder opened in the browser is measured exactly
// as the pipeline measures it. tests/viewer/geom.test.mjs checks it against the Python on every
// tracked cameras.csv. No three.js here: plain arrays, runs in Node.

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const norm = a => Math.hypot(a[0], a[1], a[2]);
const unit = a => scale(a, 1 / norm(a));
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const vec = { dot, sub, scale, norm, unit, cross };

// Symmetric 3x3 eigen-decomposition (cyclic Jacobi). Returns eigenvalues ascending and the matching
// unit eigenvectors as rows, like numpy.linalg.eigh (columns there).
export function eigh3(M) {
  const A = M.map(r => r.slice()), V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 60; sweep++) {
    const off = A[0][1] ** 2 + A[0][2] ** 2 + A[1][2] ** 2;
    if (off < 1e-30) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(A[p][q]) < 1e-300) continue;
      const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) {            // A = J^T A J
        const akp = A[k][p], akq = A[k][q];
        A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = A[p][k], aqk = A[q][k];
        A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {            // V = V J (columns are eigenvectors)
        const vkp = V[k][p], vkq = V[k][q];
        V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
      }
    }
  }
  const order = [0, 1, 2].sort((i, j) => A[i][i] - A[j][j]);
  return { values: order.map(i => A[i][i]), vectors: order.map(i => [V[0][i], V[1][i], V[2][i]]) };
}

// Least-squares solve of a symmetric 3x3 system (pseudo-inverse through eigh, like np.linalg.lstsq).
function solveSym3(A, b) {
  const { values, vectors } = eigh3(A), tol = Math.max(...values.map(Math.abs)) * 1e-12;
  let x = [0, 0, 0];
  values.forEach((l, i) => { if (Math.abs(l) > tol) x = x.map((xi, k) => xi + vectors[i][k] * dot(vectors[i], b) / l); });
  return x;
}

function cov3(D, centred) {
  const n = D.length, m = [0, 0, 0];
  if (centred) for (const d of D) for (let k = 0; k < 3; k++) m[k] += d[k] / n;
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const d of D) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i][j] += (d[i] - m[i]) * (d[j] - m[j]);
  return C;
}

// coverage.camera_frame: centre nearest all viewing rays, and rows (e1, e2, up). See the Python
// docstring for why "up" is refined per elevation band.
export function cameraFrame(P, V) {
  if (P.length < 3) throw new Error(`need at least 3 aligned cameras, got ${P.length}`);
  const Vn = V.map(unit), A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], b = [0, 0, 0];
  P.forEach((p, i) => {
    const v = Vn[i];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
      const m = (r === c ? 1 : 0) - v[r] * v[c];
      A[r][c] += m; b[r] += m * p[c];
    }
  });
  const centre = solveSym3(A, b);
  const D = P.map(p => unit(sub(p, centre)));
  const signed = u => (D.reduce((s, d) => s + dot(d, u), 0) > 0 ? u : scale(u, -1));
  let up = signed(eigh3(cov3(D, true)).vectors[0]);
  for (let it = 0; it < 5; it++) {
    const h = D.map(d => dot(d, up)), lo = Math.min(...h), hi = Math.max(...h);
    const e = [lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3];
    const band = h.map(x => (x >= e[1] ? 2 : x >= e[0] ? 1 : 0));
    const R = D.map(d => d.slice());
    for (const bb of new Set(band)) {
      const idx = band.map((x, i) => (x === bb ? i : -1)).filter(i => i >= 0), m = [0, 0, 0];
      for (const i of idx) for (let k = 0; k < 3; k++) m[k] += D[i][k] / idx.length;
      for (const i of idx) R[i] = sub(R[i], m);
    }
    up = signed(eigh3(cov3(R, false)).vectors[0]);
  }
  let first = sub(D[0], scale(up, dot(D[0], up)));
  if (norm(first) < 1e-9) first = Math.abs(up[0]) < 0.9 ? cross(up, [1, 0, 0]) : cross(up, [0, 1, 0]);
  const e1 = unit(first);
  return { centre, B: [e1, cross(up, e1), up] };
}

const inFrame = (p, centre, B) => { const d = sub(p, centre); return [dot(B[0], d), dot(B[1], d), dot(B[2], d)]; };

export function azimuthElevation(Q) {
  return Q.map(q => {
    const raw = Math.atan2(q[1], q[0]) * 180 / Math.PI;
    const az = raw - 360 * Math.floor(raw / 360);   // Python/NumPy float modulo: -1e-15 -> 360.0, as in coverage.py
    const el = Math.asin(Math.max(-1, Math.min(1, q[2] / norm(q)))) * 180 / Math.PI;
    return [az, el];
  });
}

// numpy.histogram2d semantics: bins [a, b) except the last, which includes its right edge.
function binIndex(x, edges) {
  const n = edges.length - 1;
  if (x < edges[0] || x > edges[n]) return -1;
  if (x === edges[n]) return n - 1;
  for (let i = 0; i < n; i++) if (x < edges[i + 1]) return i;
  return -1;
}

export function coverageGrid(azel, nAz = 12, nEl = 3, elEdges = null) {
  let el = azel.map(a => a[1]);
  if (!elEdges) {
    let lo = Math.min(...el), hi = Math.max(...el);
    if (hi - lo < 1) { lo -= 0.5; hi += 0.5; }
    const step = (hi - lo) / nEl;            // numpy.linspace: i * step + start, endpoint set exactly
    elEdges = Array.from({ length: nEl + 1 }, (_, i) => (i === nEl ? hi : i * step + lo));
  } else {
    const lo = elEdges[0], hi = elEdges[elEdges.length - 1];
    el = el.map(x => (x < lo && x > lo - 1 ? lo : x > hi && x < hi + 1 ? hi : x));
  }
  const azEdges = Array.from({ length: nAz + 1 }, (_, i) => (i === nAz ? 360 : i * (360 / nAz)));
  const counts = Array.from({ length: elEdges.length - 1 }, () => new Array(nAz).fill(0));
  azel.forEach(([az], i) => {
    const r = binIndex(el[i], elEdges), c = binIndex(az, azEdges);
    if (r >= 0 && c >= 0) counts[r][c]++;
  });
  return { counts, azEdges, elEdges };
}

const BAND_NAMES = { 1: ['all'], 2: ['low', 'high'], 3: ['low', 'mid', 'high'] };

function circularRuns(empty) {
  const n = empty.length;
  if (empty.every(Boolean)) return [[0, n]];
  const start = empty.indexOf(false), runs = [];
  let i = 0;
  while (i < n) {
    const j = (start + i) % n;
    if (empty[j]) { let k = i; while (k < n && empty[(start + k) % n]) k++; runs.push([j, k - i]); i = k; } else i++;
  }
  return runs.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}

export function describeGaps(missing, azEdges) {
  const names = BAND_NAMES[missing.length] || missing.map((_, i) => `band ${i + 1}`);
  const step = azEdges[1] - azEdges[0], parts = [];
  missing.forEach((row, b) => {
    const runs = circularRuns(row);
    if (!runs.length) return;
    if (runs.length === 1 && runs[0][0] === 0 && runs[0][1] === row.length) { parts.push(`${names[b]} cameras at every azimuth`); return; }
    const spans = runs.map(([s, n]) => `${azEdges[s].toFixed(0)}° to ${((azEdges[s] + n * step) % 360 || 360).toFixed(0)}°`);
    parts.push(`${names[b]} cameras from ${spans.join(', ')}`);
  });
  return parts.length ? parts.join('; ') : null;
}

// coverage.analyze for one run in its own frame. cams: {labels, aligned, P, V}.
export function analyze(cams, { nAz = 12, nEl = 3, minCameras = 2 } = {}) {
  const idx = cams.aligned.map((a, i) => (a ? i : -1)).filter(i => i >= 0);
  const P = idx.map(i => cams.P[i]), V = idx.map(i => cams.V[i]);
  const { centre, B } = cameraFrame(P, V);
  const Q = P.map(p => inFrame(p, centre, B));
  const W = V.map(v => unit([dot(B[0], v), dot(B[1], v), dot(B[2], v)]));
  const azel = azimuthElevation(Q);
  const { counts, azEdges, elEdges } = coverageGrid(azel, nAz, nEl);
  const missing = counts.map(r => r.map(c => c < minCameras));
  const failed = cams.labels.filter((_, i) => !cams.aligned[i]);
  const gaps = describeGaps(missing, azEdges);
  let summary = gaps ? `Missing: ${gaps}` : `No gaps: every cell has ${minCameras}+ cameras.`;
  if (failed.length) summary += ` (${failed.length} of ${cams.labels.length} cameras failed to align)`;
  return { index: idx, labels: idx.map(i => cams.labels[i]), failed, centre, B, Q, W, azel, counts, azEdges, elEdges,
           minCameras, missing, never: missing.map(r => r.map(() => false)), summary };
}

// viewer_export.to_display: frame (e1, e2, up) -> display (x, -z, y) = (e1, up, -e2), scaled by diag.
export const toDisplay = (p, centre, B, diag) => { const q = inFrame(p, centre, B); return [q[0] / diag, q[2] / diag, -q[1] / diag]; };
export const dirToDisplay = (v, B) => { const w = unit([dot(B[0], v), dot(B[1], v), dot(B[2], v)]); return [w[0], w[2], -w[1]]; };
// Inverse for display-frame inputs (exports): display (x, y, z) -> frame (e1, e2, up) = (x, -z, y).
export const displayToFrame = d => [d[0], -d[2], d[1]];

// ------------------------------------------------------------------ measurements beyond coverage.py
// All in the run's own frame, about the ray-convergence centre. Pure descriptions of the cameras.

// Angle (deg) between the directions from the centre to two cameras.
export function angularSeparation(Q) {
  const D = Q.map(unit), n = D.length;
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) =>
    Math.acos(Math.max(-1, Math.min(1, dot(D[i], D[j])))) * 180 / Math.PI));
}

// Per camera: angle to its nearest neighbour (as seen from the centre), and how far its optical axis
// points away from the centre (0 deg = looking straight at the convergence point).
export function perCamera(Q, W) {
  const S = angularSeparation(Q);
  return Q.map((q, i) => {
    let nn = Infinity, nnIdx = -1;
    S[i].forEach((s, j) => { if (j !== i && s < nn) { nn = s; nnIdx = j; } });
    const toCentre = unit(scale(q, -1));
    const offAxis = Math.acos(Math.max(-1, Math.min(1, dot(unit(W[i]), toCentre)))) * 180 / Math.PI;
    return { nnAngle: nn, nnIndex: nnIdx, offAxis, distance: norm(q) };
  });
}

// Largest gap in azimuth (deg) among cameras in each elevation band (bands from the coverage grid),
// and over all cameras. A band with fewer than 2 cameras reports 360.
export function azimuthGaps(azel, elEdges) {
  const gap = az => {
    if (az.length < 2) return { gap: 360, from: null };
    const s = [...az].sort((a, b) => a - b);
    let best = s[0] + 360 - s[s.length - 1], from = s[s.length - 1];
    for (let i = 1; i < s.length; i++) if (s[i] - s[i - 1] > best) { best = s[i] - s[i - 1]; from = s[i - 1]; }
    return { gap: best, from };
  };
  const bands = [], L = elEdges[0], Hh = elEdges[elEdges.length - 1];
  // same band membership as coverageGrid: cameras up to 1 deg outside the outer edges count for the edge band
  const clamp = e => (e < L && e > L - 1 ? L : e > Hh && e < Hh + 1 ? Hh : e);
  for (let b = 0; b < elEdges.length - 1; b++) {
    const lo = elEdges[b], hi = elEdges[b + 1], last = b === elEdges.length - 2;
    const az = azel.filter(([, e0]) => { const e = clamp(e0); return e >= lo && (last ? e <= hi : e < hi); }).map(a => a[0]);
    bands.push({ lo, hi, n: az.length, ...gap(az) });
  }
  return { all: gap(azel.map(a => a[0])), bands };
}

export function quantiles(xs, qs) {
  const s = [...xs].sort((a, b) => a - b);
  return qs.map(q => { const t = (s.length - 1) * q, i = Math.floor(t); return s[i] + (s[Math.min(i + 1, s.length - 1)] - s[i]) * (t - i); });
}

// ------------------------------------------------------------------ comparison with a reference run

// 3x3 SVD through the eigen-decomposition of H^T H (H = U S V^T). Returns U, S, V with rows as vectors.
function svd3(H) {
  const HtH = [0, 1, 2].map(i => [0, 1, 2].map(j => H[0][i] * H[0][j] + H[1][i] * H[1][j] + H[2][i] * H[2][j]));
  const { values, vectors } = eigh3(HtH);
  const order = [2, 1, 0], V = order.map(i => vectors[i]), S = order.map(i => Math.sqrt(Math.max(values[i], 0)));
  const Hv = v => [dot(H[0], v), dot(H[1], v), dot(H[2], v)];
  const U = [unit(Hv(V[0])), null, null];
  U[1] = S[1] > 1e-12 * S[0] ? unit(Hv(V[1])) : unit(cross(U[0], Math.abs(U[0][0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]));
  U[2] = cross(U[0], U[1]);
  if (S[2] > 1e-12 * S[0] && dot(U[2], Hv(V[2])) < 0) U[2] = scale(U[2], -1);
  return { U, S, V };
}

const det3 = M => dot(M[0], cross(M[1], M[2]));

// evaluation.similarity_fit (Umeyama, with scale): returns {sR (3x3 rows), t} with target ≈ sR·source + t.
export function similarityFit(src, tgt) {
  const n = src.length, ms = [0, 0, 0], mt = [0, 0, 0];
  src.forEach((p, i) => { for (let k = 0; k < 3; k++) { ms[k] += p[k] / n; mt[k] += tgt[i][k] / n; } });
  const A = src.map(p => sub(p, ms)), Bm = tgt.map(p => sub(p, mt));
  const H = [0, 1, 2].map(i => [0, 1, 2].map(j => A.reduce((s, a, r) => s + a[i] * Bm[r][j], 0)));
  const { U, S, V } = svd3(H);                     // H = Σ u_k s_k v_k^T
  const VUt = [0, 1, 2].map(i => [0, 1, 2].map(j => V.reduce((s, v, k) => s + v[i] * U[k][j], 0)));
  const d = Math.sign(det3(VUt)) || 1, Dg = [1, 1, d];
  const R = [0, 1, 2].map(i => [0, 1, 2].map(j => V.reduce((s, v, k) => s + v[i] * Dg[k] * U[k][j], 0)));
  const s = S.reduce((acc, x, k) => acc + x * Dg[k], 0) / A.reduce((acc, a) => acc + dot(a, a), 0);
  const sR = R.map(r => scale(r, s));
  const t = sub(mt, [dot(sR[0], ms), dot(sR[1], ms), dot(sR[2], ms)]);
  return { sR, t };
}
const applyT = ({ sR, t }, p) => [dot(sR[0], p) + t[0], dot(sR[1], p) + t[1], dot(sR[2], p) + t[2]];

// coverage.compare: the run's cameras moved into the reference frame by the camera similarity fit on
// shared labels; a cell is missing where the reference has minCameras+ and the run has fewer.
export function compare(cams, ref, { nAz = 12, nEl = 3, minCameras = 2 } = {}) {
  const r = analyze(ref, { nAz, nEl, minCameras });
  const refPos = new Map(ref.labels.map((l, i) => [l, i]).filter(([, i]) => ref.aligned[i]).map(([l, i]) => [l, ref.P[i]]));
  const idx = cams.aligned.map((a, i) => (a ? i : -1)).filter(i => i >= 0);
  const shared = [...new Set(idx.map(i => cams.labels[i]))].filter(l => refPos.has(l)).sort();
  if (shared.length < 3) throw new Error('run and reference share fewer than 3 aligned camera labels; cannot align them');
  const pos = new Map(idx.map(i => [cams.labels[i], cams.P[i]]));
  const A = shared.map(l => pos.get(l)), Bt = shared.map(l => refPos.get(l));
  const T = similarityFit(A, Bt);
  const mb = [0, 1, 2].map(k => Bt.reduce((s, b) => s + b[k], 0) / Bt.length);
  const resid = Math.sqrt(A.reduce((s, a, i) => s + norm(sub(applyT(T, a), Bt[i])) ** 2, 0) / A.length)
              / Math.sqrt(Bt.reduce((s, b) => s + norm(sub(b, mb)) ** 2, 0) / Bt.length);
  const P = idx.map(i => applyT(T, cams.P[i]));
  const V = idx.map(i => { const v = cams.V[i]; return [dot(T.sR[0], v), dot(T.sR[1], v), dot(T.sR[2], v)]; });
  const Q = P.map(p => inFrame(p, r.centre, r.B)), W = V.map(v => unit([dot(r.B[0], v), dot(r.B[1], v), dot(r.B[2], v)]));
  const azel = azimuthElevation(Q);
  const { counts, azEdges, elEdges } = coverageGrid(azel, nAz, nEl, r.elEdges);
  const never = r.counts.map(row => row.map(c => c < minCameras));
  const missing = counts.map((row, b) => row.map((c, a) => !never[b][a] && c < minCameras));
  const failed = cams.labels.filter((_, i) => !cams.aligned[i]);
  const gaps = describeGaps(missing, azEdges);
  let summary = gaps ? `Missing vs reference: ${gaps} (reference angles)` : 'Covers every cell the reference covers.';
  if (failed.length) summary += ` (${failed.length} of ${cams.labels.length} cameras failed to align)`;
  return { index: idx, labels: idx.map(i => cams.labels[i]), failed, centre: r.centre, B: r.B, Q, W, azel, counts, azEdges, elEdges,
           minCameras, missing, never, summary, reference: r, fit: { shared: shared.length, residual: resid, T } };
}
