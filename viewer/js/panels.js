// Side panels and the analysis dock: dataset tree, inspector, camera table, coverage figures.
import { esc, frame, histogram, figure, toCsv, download, STATE } from './plot.js';
import { quantiles } from './geom.js';

const f = (x, d = 2) => (x == null || !Number.isFinite(x) ? '—' : x.toFixed(d));
const pctDiag = x => (x == null ? '—' : `${(100 * x).toFixed(2)}%`);
export const fmt = { f, pctDiag };

// Per-camera quantities: key -> [label, unit, digits, description]
export const QUANT = {
  az: ['Azimuth', '°', 1, 'about the estimated up axis, 0° = first aligned camera'],
  el: ['Elevation', '°', 1, 'above the plane through the centre, perpendicular to up'],
  nnAngle: ['Nearest-neighbour angle', '°', 1, 'angle at the centre to the closest other camera of this build'],
  offAxis: ['Off-axis angle', '°', 1, 'angle between the optical axis and the direction to the centre'],
  distance: ['Distance to centre', 'u', 3, 'in display units'],
  reprojection_px: ['Reprojection RMSE', 'px', 2, 'per camera, from Metashape (cameras.csv)'],
  projections: ['Projections', '', 0, 'tie-point projections in this image (cameras.csv)'],
  image_quality: ['Image quality', '', 3, 'Metashape image quality estimate (cameras.csv); blank when not computed'],
};
const STATE_NAME = { used: 'used', removed: 'not in this build', unaligned: 'failed to align' };

// ------------------------------------------------------------------ dataset tree
export function renderTree(el, entries, current, onSelect) {
  const groups = new Map();
  for (const e of entries) { if (!groups.has(e.group)) groups.set(e.group, []); groups.get(e.group).push(e); }
  el.innerHTML = '';
  if (!entries.length) { el.innerHTML = '<li class="grp">No datasets. Open a folder.</li>'; return; }
  for (const [g, list] of groups) {
    const li = document.createElement('li');
    li.innerHTML = `<div class="grp">${esc(g)}</div>`;
    list.sort((a, b) => (a.role === 'reference') - (b.role === 'reference') || (a.meta?.label ?? 1) - (b.meta?.label ?? 1) || a.name.localeCompare(b.name));
    for (const e of list) {
      const b = document.createElement('button');
      b.className = `ds${e.role === 'reference' ? ' ref' : ''}`;
      b.setAttribute('aria-current', String(e.key === current));
      b.title = `${e.name}${e.files ? '\n' + e.files[0] : ''}`;
      const v = e.role === 'reference' ? 'reference' : e.meta?.label != null ? pctDiag(e.meta.label) : e.source === 'local' ? 'local' : '';
      b.innerHTML = `<span class="nm">${esc(e.name)}</span><span class="v">${v}</span>`;
      b.onclick = () => onSelect(e);
      li.appendChild(b);
    }
    el.appendChild(li);
  }
}

// ------------------------------------------------------------------ inspector
export function renderInspector(el, sc, sel, meshInfo) {
  if (!sc) { el.innerHTML = ''; return; }
  const used = sc.cameras.filter(c => !c.refOnly), nn = used.map(c => c.nnAngle).filter(Number.isFinite);
  const [nnMed, nnMax] = nn.length ? [quantiles(nn, [0.5])[0], Math.max(...nn)] : [null, null];
  const src = { public: 'public (tracked in git)', local: 'local file (not uploaded)', export: 'local export (viewer/data)' }[sc.source] ?? sc.source;
  const rows = [
    ['Source', src], ['Reference', sc.reference ?? 'none'], ['Display unit', `1 u = ${sc.unit}`],
    ['Aligned / photos', `${used.length} / ${sc.total}`],
    ['Failed to align', sc.failed.length ? `<span class="bad">${sc.failed.length}</span>` : '0'],
    ['Reprojection RMSE', sc.reprojection_px != null ? `${f(sc.reprojection_px)} px` : '—'],
    ['Tie points', sc.tie_points != null ? Math.round(sc.tie_points).toLocaleString('en-US') : '—'],
    ['NN angle median / max', nn.length ? `${f(nnMed, 1)}° / ${f(nnMax, 1)}°` : '—'],
    ['Largest azimuth gap', sc.gaps ? `${f(sc.gaps.all.gap, 1)}°` : '—'],
  ];
  if (meshInfo?.faces) rows.push(['Mesh', `${Math.round(meshInfo.faces).toLocaleString('en-US')} faces`]);
  const m = sc.measured;
  let html = `<h3>Build</h3><div class="sec"><div style="font-weight:600;overflow-wrap:anywhere">${esc(sc.title)}</div>
    <dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl></div>`;
  html += `<h3>Coverage</h3><div class="sec"><p style="color:var(--ink)">${esc(sc.coverage.summary)}</p>
    <p>${sc.coverage.relative ? 'Red cells: the reference has ≥2 cameras, this build fewer. Grey: the reference has fewer than 2 too.' : 'Red cells: fewer than 2 cameras (12 azimuth × 3 elevation bands, coverage.analyze).'}</p></div>`;
  html += `<h3>Measured reconstruction error</h3><div class="sec">${m ? `<dl>
      <dt>v3 label</dt><dd>${pctDiag(m.label)} <span style="color:var(--dim)">of ref. diagonal</span></dd>
      <dt>QC</dt><dd class="${m.qc && m.qc !== 'pass' ? 'bad' : ''}">${esc(m.qc ?? '—')}</dd>
      <dt>Completeness @1%</dt><dd>${m.completeness != null ? (100 * m.completeness).toFixed(1) + '%' : '—'}</dd>
      <dt>Accuracy @1%</dt><dd>${m.accuracy != null ? (100 * m.accuracy).toFixed(1) + '%' : '—'}</dd></dl>
      <p>Symmetric surface RMSE against ${esc(m.reference?.split('/').pop() ?? 'the reference')} after camera-initialised similarity ICP (docs/METHOD.md). Source: ${esc(m.source ?? '')}</p>`
    : '<p>No label: measured error needs a mesh and a reference build, scored by scripts/score_runs_v3.py or viewer_export compare.</p>'}</div>`;
  if (sel) {
    const kv = [['Label', esc(sel.label)], ['State', STATE_NAME[sel.state]]];
    for (const [k, [name, unit, d]] of Object.entries(QUANT)) if (sel[k] != null) kv.push([name, `${f(sel[k], d)}${unit && unit !== 'u' ? ' ' + unit : unit === 'u' ? ' u' : ''}`]);
    html += `<h3>Selected camera</h3><div class="sec"><dl>${kv.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl></div>`;
  }
  if (sc.notes.length) html += `<h3>Notes</h3><div class="sec">${sc.notes.map(n => `<p>${esc(n)}</p>`).join('')}</div>`;
  html += `<h3>Frame</h3><div class="sec"><p>${esc(sc.frame)}</p><p>Frustum roll is not stored in cameras.csv (only the optical axis); the tick on each frustum marks an arbitrary image up.</p></div>`;
  html += `<h3>Files</h3><div class="sec"><p style="font-family:var(--mono);font-size:11px;overflow-wrap:anywhere">${(sc.files ?? []).map(esc).join('<br>')}</p></div>`;
  el.innerHTML = html;
}

// ------------------------------------------------------------------ camera table
export function renderCameraTable(pane, tools, sc, sel, onSelect, state) {
  state.sort ??= { key: 'az', dir: 1 };
  const cols = [['#', null], ['label', 'label'], ['state', 'state'], ...Object.entries(QUANT).map(([k, [n, u]]) => [`${n}${u ? ` (${u})` : ''}`, k])];
  const rows = sc.cameras.map((c, i) => ({ i, ...c }));
  const { key, dir } = state.sort;
  rows.sort((a, b) => { const x = a[key], y = b[key]; return (x == null) - (y == null) || (x < y ? -dir : x > y ? dir : 0); });
  const filt = (state.filter ?? '').toLowerCase();
  const shown = rows.filter(r => !filt || r.label.toLowerCase().includes(filt));
  pane.className = 'pane';
  pane.innerHTML = `<table class="t"><thead><tr>${cols.map(([n, k]) => `<th class="${k && k !== 'label' && k !== 'state' ? 'n' : ''}" data-k="${k ?? ''}" ${k === key ? `aria-sort="${dir > 0 ? 'ascending' : 'descending'}"` : ''}>${esc(n)}</th>`).join('')}</tr></thead>
    <tbody>${shown.map(r => `<tr data-i="${r.i}" aria-selected="${r.i === sel}"><td class="n">${r.i + 1}</td><td>${esc(r.label)}</td><td${r.state !== 'used' ? ' class="bad"' : ''}>${STATE_NAME[r.state]}</td>${Object.entries(QUANT).map(([k, [, , d]]) => `<td class="n">${f(r[k], d)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  pane.querySelectorAll('th[data-k]').forEach(th => th.onclick = () => {
    const k = th.dataset.k; if (!k) return;
    state.sort = { key: k, dir: state.sort.key === k ? -state.sort.dir : 1 };
    renderCameraTable(pane, tools, sc, sel, onSelect, state);
  });
  pane.querySelectorAll('tbody tr').forEach(tr => tr.onclick = () => onSelect(+tr.dataset.i));
  pane.querySelector('tr[aria-selected="true"]')?.scrollIntoView({ block: 'nearest' });
  tools.innerHTML = `<input type="search" placeholder="Filter labels" value="${esc(state.filter ?? '')}" style="width:140px"><span class="num" style="color:var(--dim)">${shown.length} / ${rows.length}</span><button>CSV</button>`;
  const inp = tools.querySelector('input');
  inp.oninput = () => { state.filter = inp.value; renderCameraTable(pane, tools, sc, sel, onSelect, state); tools.querySelector('input').focus(); };
  tools.querySelector('button').onclick = () => download(`${sc.title}-cameras.csv`, toCsv(rows.map(r => ({ label: r.label, state: r.state, ...Object.fromEntries(Object.keys(QUANT).map(k => [k, r[k]])), x_display: r.p[0], y_display: r.p[1], z_display: r.p[2] }))), 'text/csv');
}

// ------------------------------------------------------------------ coverage figures
export function renderCoverage(pane, tools, sc, sel, onSelect) {
  pane.className = 'pane flex'; pane.innerHTML = ''; tools.innerHTML = '';
  const cov = sc.coverage, id = sc.title;
  // 1. azimuth x elevation: cells shaded by count, cameras as points
  const elLo = Math.min(cov.elEdges[0], ...sc.cameras.map(c => c.el)) - 2, elHi = Math.max(cov.elEdges.at(-1), ...sc.cameras.map(c => c.el)) + 2;
  const F = frame({ w: 520, h: 210, x: [0, 360], y: [elLo, elHi], xticks: [0, 60, 120, 180, 240, 300, 360], xlabel: 'azimuth (°)', ylabel: 'elevation (°)' });
  let s = F.open + F.axes;
  for (let b = 0; b < cov.elEdges.length - 1; b++) for (let a = 0; a < cov.azEdges.length - 1; a++) {
    const n = cov.counts[b][a], miss = cov.missing[b][a], never = cov.never[b][a];
    const x = F.X(cov.azEdges[a]), w = F.X(cov.azEdges[a + 1]) - x, y = F.Y(cov.elEdges[b + 1]), h = F.Y(cov.elEdges[b]) - y;
    s += `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${miss ? '#d93a2b' : never ? '#9a9a94' : '#55554f'}" fill-opacity="${miss ? 0.18 : never ? 0.12 : Math.min(0.05 + n * 0.02, 0.2)}" stroke="#fff"/>`
       + `<text x="${x + w / 2}" y="${y + 10}" text-anchor="middle" fill="${miss ? '#a12a1f' : '#888'}" font-size="9">${n}</text>`;
  }
  sc.cameras.forEach((c, i) => {
    const color = i === sel ? STATE.selected : STATE[c.state];
    s += c.state === 'used' ? `<circle data-i="${i}" cx="${F.X(c.az)}" cy="${F.Y(c.el)}" r="${i === sel ? 4 : 2.6}" fill="${color}" style="cursor:pointer"/>`
      : `<circle data-i="${i}" cx="${F.X(c.az)}" cy="${F.Y(c.el)}" r="2.8" fill="${c.state === 'unaligned' ? color : 'none'}" stroke="${color}" stroke-dasharray="${c.state === 'removed' ? '1.5 1.2' : ''}" style="cursor:pointer"/>`;
  });
  s += '</svg>';
  const fig1 = figure(`${id}-coverage-az-el`, s,
    `Cameras by azimuth and elevation in the build's frame${cov.relative ? ' (reference frame and elevation bands)' : ''}. Cells: 12 × 3 coverage grid of coverage.py; number = cameras in the cell; red = missing${cov.relative ? ' relative to the reference' : ' (< 2 cameras)'}. Filled dots: used; dashed: reference photos not in this build; red: failed to align.`,
    sc.cameras.map(c => ({ label: c.label, state: c.state, azimuth_deg: c.az, elevation_deg: c.el })));
  fig1.querySelectorAll('circle[data-i]').forEach(c => c.onclick = () => onSelect(+c.dataset.i));
  pane.appendChild(fig1);

  // 2. largest azimuth gap per elevation band
  const g = sc.gaps, bands = g.bands;
  const G = frame({ w: 250, h: 210, x: [0, 360], y: [0, bands.length], xticks: [0, 90, 180, 270, 360], yticks: [], xlabel: 'largest azimuth gap (°)', ylabel: '' });
  let s2 = G.open + G.axes, names = { 1: ['all'], 2: ['low', 'high'], 3: ['low', 'mid', 'high'] }[bands.length] ?? bands.map((_, i) => `band ${i + 1}`);
  bands.forEach((b, i) => {
    const y = G.Y(i + 0.8), h = G.Y(i + 0.2) - y;
    s2 += `<rect x="${G.X(0)}" y="${y}" width="${G.X(b.gap) - G.X(0)}" height="${h}" fill="#55554f" fill-opacity="0.6"/>`
        + `<text x="${G.m.l - 4}" y="${y + h / 2 - 2}" text-anchor="end">${names[i]}</text><text x="${G.m.l - 4}" y="${y + h / 2 + 9}" text-anchor="end" fill="#888">n=${b.n}</text>`
        + `<text x="${G.X(b.gap) + 3}" y="${y + h / 2 + 3}" ${b.gap > 300 ? `text-anchor="end" dx="-6" fill="#fff"` : ''}>${b.gap.toFixed(1)}°</text>`;
  });
  s2 += '</svg>';
  pane.appendChild(figure(`${id}-azimuth-gaps`, s2,
    `Largest gap between azimuthally neighbouring cameras within each elevation band (band edges from the coverage grid). All cameras: ${g.all.gap.toFixed(1)}°. Same quantity as the synthetic benchmark's target, measured here on real cameras.`,
    bands.map((b, i) => ({ band: names[i], el_lo_deg: b.lo, el_hi_deg: b.hi, cameras: b.n, largest_gap_deg: b.gap, gap_starts_at_az_deg: b.from }))));

  // 3. nearest-neighbour angle
  const nn = sc.cameras.filter(c => !c.refOnly && Number.isFinite(c.nnAngle)).map(c => c.nnAngle);
  if (nn.length > 1) {
    const H = histogram(nn, { bins: 16, lo: 0 }), ymax = Math.max(...H.map(h => h.count));
    const P = frame({ w: 250, h: 210, x: [0, H.at(-1).hi], y: [0, ymax * 1.1], xlabel: 'nearest-neighbour angle (°)', ylabel: 'cameras' });
    let s3 = P.open + P.axes;
    H.forEach(h => { s3 += `<rect x="${P.X(h.lo) + 0.5}" y="${P.Y(h.count)}" width="${Math.max(P.X(h.hi) - P.X(h.lo) - 1, 0)}" height="${P.Y(0) - P.Y(h.count)}" fill="#55554f" fill-opacity="0.6"/>`; });
    if (sel >= 0 && Number.isFinite(sc.cameras[sel]?.nnAngle)) s3 += `<line x1="${P.X(sc.cameras[sel].nnAngle)}" x2="${P.X(sc.cameras[sel].nnAngle)}" y1="${P.m.t}" y2="${P.Y(0)}" stroke="${STATE.selected}"/>`;
    s3 += '</svg>';
    const [q50, q90] = quantiles(nn, [0.5, 0.9]);
    pane.appendChild(figure(`${id}-nn-angle`, s3,
      `Angle at the centre between each used camera and its closest neighbour (median ${q50.toFixed(1)}°, 90th percentile ${q90.toFixed(1)}°). Isolated cameras sit at the edge of a gap.`,
      H.map(h => ({ bin_lo_deg: h.lo, bin_hi_deg: h.hi, cameras: h.count }))));
  }
}
