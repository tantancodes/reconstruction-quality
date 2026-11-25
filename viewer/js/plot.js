// Small SVG plots for the analysis dock, with SVG and CSV export. No plotting library.

// viridis (perceptually uniform), polynomial fit by Matt Zucker (CC0), t in [0, 1] -> [r, g, b] in [0, 1]
export function viridis(t) {
  t = Math.max(0, Math.min(1, t));
  const c0 = [0.2777, 0.0054, 0.3341], c1 = [0.1051, 1.4046, 1.3846], c2 = [-0.3309, 0.2148, 0.0951],
    c3 = [-4.6342, -5.7991, -19.3324], c4 = [6.2283, 14.1799, 56.6906], c5 = [4.7764, -13.7451, -65.353], c6 = [-5.4355, 4.6459, 26.3124];
  return [0, 1, 2].map(k => Math.max(0, Math.min(1, c0[k] + t * (c1[k] + t * (c2[k] + t * (c3[k] + t * (c4[k] + t * (c5[k] + t * c6[k]))))))));
}
export const rgb = c => `rgb(${c.map(x => Math.round(x * 255)).join(',')})`;
export const STATE = { used: '#3a3a36', removed: '#d93a2b', unaligned: '#d93a2b', selected: '#0b5fd0' };

const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
export { esc };

export function ticks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
const fmt = v => (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(1) : +v.toPrecision(4) + '');

// frame: {w, h, x: [lo, hi], y: [lo, hi], xlabel, ylabel, xticks?, yticks?, title?} -> helpers + svg parts
export function frame(o) {
  const m = { l: 44, r: 10, t: o.title ? 18 : 8, b: 30 }, W = o.w - m.l - m.r, H = o.h - m.t - m.b;
  const X = v => m.l + (v - o.x[0]) / (o.x[1] - o.x[0]) * W, Y = v => m.t + H - (v - o.y[0]) / (o.y[1] - o.y[0]) * H;
  const xt = o.xticks ?? ticks(o.x[0], o.x[1]), yt = o.yticks ?? ticks(o.y[0], o.y[1], 4);
  let s = `<g class="grid">${xt.map(v => `<line x1="${X(v)}" x2="${X(v)}" y1="${m.t}" y2="${m.t + H}"/>`).join('')}`
        + `${yt.map(v => `<line x1="${m.l}" x2="${m.l + W}" y1="${Y(v)}" y2="${Y(v)}"/>`).join('')}</g>`;
  s += `<g class="axis"><path d="M${m.l},${m.t}V${m.t + H}H${m.l + W}" fill="none"/>`
     + xt.map(v => `<line x1="${X(v)}" x2="${X(v)}" y1="${m.t + H}" y2="${m.t + H + 3}"/><text x="${X(v)}" y="${m.t + H + 13}" text-anchor="middle">${fmt(v)}</text>`).join('')
     + yt.map(v => `<line x1="${m.l - 3}" x2="${m.l}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${m.l - 5}" y="${Y(v) + 3}" text-anchor="end">${fmt(v)}</text>`).join('')
     + `</g><text x="${m.l + W / 2}" y="${o.h - 3}" text-anchor="middle">${esc(o.xlabel ?? '')}</text>`
     + `<text transform="translate(10 ${m.t + H / 2}) rotate(-90)" text-anchor="middle">${esc(o.ylabel ?? '')}</text>`
     + (o.title ? `<text x="${m.l}" y="12" font-weight="600">${esc(o.title)}</text>` : '');
  return { X, Y, m, W, H, axes: s, open: `<svg class="plot" xmlns="http://www.w3.org/2000/svg" width="${o.w}" height="${o.h}" viewBox="0 0 ${o.w} ${o.h}">` };
}

export function histogram(values, { bins = 20, lo, hi } = {}) {
  lo ??= Math.min(...values); hi ??= Math.max(...values);
  if (hi <= lo) hi = lo + 1;
  const counts = new Array(bins).fill(0), w = (hi - lo) / bins;
  for (const v of values) counts[Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / w)))]++;
  return counts.map((c, i) => ({ lo: lo + i * w, hi: lo + (i + 1) * w, count: c }));
}

// A figure block: caption with its data source, and export buttons.
export function figure(id, svg, caption, csvRows) {
  const div = document.createElement('div');
  div.className = 'fig';
  div.innerHTML = `${svg}<div class="ft"><button data-x="svg" title="Save as SVG">SVG</button>${csvRows ? '<button data-x="csv" title="Save the plotted data as CSV">CSV</button>' : ''}</div><p class="cap">${caption}</p>`;
  div.querySelector('[data-x=svg]').onclick = () => download(`${id}.svg`, div.querySelector('svg').outerHTML, 'image/svg+xml');
  if (csvRows) div.querySelector('[data-x=csv]').onclick = () => download(`${id}.csv`, toCsv(csvRows), 'text/csv');
  return div;
}

export function toCsv(rows) {
  if (!rows.length) return '';
  const cols = Object.keys(rows[0]);
  const cell = v => (v == null ? '' : typeof v === 'number' ? String(v) : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
  return [cols.join(','), ...rows.map(r => cols.map(c => cell(r[c])).join(','))].join('\n') + '\n';
}

export function download(name, data, type) {
  const url = URL.createObjectURL(data instanceof Blob ? data : new Blob([data], { type }));
  const a = Object.assign(document.createElement('a'), { href: url, download: name });
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
