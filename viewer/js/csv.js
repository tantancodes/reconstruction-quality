// cameras.csv as written by the pipeline (src/bd3d_quality/metashape_runner.py). Same rules as
// coverage.load_cameras: required columns, aligned = "True" and a finite pose. Optional per-camera
// columns are kept when present.

export const REQUIRED = ['label', 'aligned', 'x', 'y', 'z', 'view_x', 'view_y', 'view_z'];
const OPTIONAL = { n_projections: 'projections', reprojection_rmse_px: 'reprojection_px', image_quality: 'image_quality' };

export function parseCsv(text) {
  const rows = [];
  let row = [], field = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; } else if (c === '"') q = false; else field += c;
    } else if (c === '"') q = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field); field = '';
      if (row.length > 1 || row[0] !== '') rows.push(row);
      row = [];
    } else field += c;
  }
  if (field !== '' || row.length) { row.push(field); rows.push(row); }
  const [head, ...body] = rows;
  return body.map(r => Object.fromEntries(head.map((h, i) => [h.trim(), r[i] ?? ''])));
}

export function readCameras(text, name = 'cameras.csv') {
  const rows = parseCsv(text);
  if (!rows.length) throw new Error(`${name} has no camera rows`);
  const missing = REQUIRED.filter(c => !(c in rows[0]));
  if (missing.length) throw new Error(`${name} is missing columns: ${missing.join(', ')} (expected the pipeline's cameras.csv)`);
  const num = v => (v === '' || v == null ? NaN : Number(v));
  const labels = [], aligned = [], P = [], V = [], extra = [];
  for (const r of rows) {
    const p = ['x', 'y', 'z'].map(k => num(r[k])), v = ['view_x', 'view_y', 'view_z'].map(k => num(r[k]));
    labels.push(r.label);
    aligned.push(r.aligned.trim() === 'True' && p.every(Number.isFinite) && v.every(Number.isFinite));
    P.push(p); V.push(v);
    const e = {};
    for (const [col, key] of Object.entries(OPTIONAL)) if (col in r && r[col] !== '') e[key] = num(r[col]);
    if (r.missing_reasons) e.missing_reasons = r.missing_reasons;
    extra.push(e);
  }
  return { labels, aligned, P, V, extra, run: rows[0].experiment_id || null };
}
