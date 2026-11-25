// Predicted vs measured reconstruction error, from the saved out-of-fold predictions
// (viewer/public/predictions.json, written by scripts/export_viewer_public.py from the results CSVs).
import { frame, figure, esc } from './plot.js';

const here = new URL('..', import.meta.url);
let data, loading;
const load = () => (loading ??= fetch(new URL('public/predictions.json', here)).then(r => (r.ok ? r.json() : null)).then(d => (data = d)).catch(() => null));

export const MODELS = {
  mean: 'training-fold mean label', reproj: 'linear fit on reprojection RMSE', ridge: 'Ridge on camera-set summaries',
  mlp: 'MLP on summaries', xgb: 'XGBoost on summaries', xgb_pair: 'XGBoost + pairwise-angle features',
  pointnet: 'PointNet', attention: 'self-attention set model', geo: 'geometry-aware attention',
  geo_inv: 'geometry-aware, rotation-invariant inputs', geo_shuf: 'control: geometry shuffled between pairs',
};
const GROUP = { 'arf-dog/hist16-cpu': '#4477aa', 'arf-dog/gpu-d8': '#66ccee', 'pahma-5-613/gpu-d8': '#aa3377' };
const pct = x => (100 * x).toFixed(2);
const ui = { source: 'pilot', scheme: 'LOAO', model: 'attention', grown: false };

export async function renderEvaluation(pane, tools, S, openDataset) {
  pane.className = 'pane flex'; tools.innerHTML = '';
  // the scatter needs ~380 px; make the dock that tall the first time (drag its top edge to change it)
  const main = document.querySelector('.main');
  if (!ui.grown && main && pane.parentElement.clientHeight < 380) main.style.setProperty('--dock', `${Math.min(400, main.clientHeight - 200)}px`);
  ui.grown = true;
  if (!data) { pane.innerHTML = '<p style="color:var(--dim)">Loading predictions…</p>'; await load(); if (S.tab !== 'evaluation') return; }
  if (!data) { pane.innerHTML = '<p>public/predictions.json not found: run python scripts/export_viewer_public.py.</p>'; return; }
  const src = data.sources.find(s => s.id === ui.source) ?? data.sources[0]; ui.source = src.id;
  const schemes = [...new Set(src.metrics.map(m => m.scheme))];
  if (!schemes.includes(ui.scheme)) ui.scheme = schemes.includes('LOAO') ? 'LOAO' : schemes[0];
  const models = src.metrics.filter(m => m.scheme === ui.scheme).map(m => m.model);
  if (!models.includes(ui.model)) ui.model = models.includes('attention') ? 'attention' : models[0];
  const sel = (id, opts, val, title) => `<select id="${id}" title="${title}">${opts.map(([v, t]) => `<option value="${v}" ${v === val ? 'selected' : ''}>${esc(t)}</option>`).join('')}</select>`;
  tools.innerHTML = sel('ev-src', data.sources.map(s => [s.id, `${s.id} (${s.produced_by})`]), ui.source, 'Which saved predictions')
    + sel('ev-split', schemes.map(s => [s, `${s}: ${data.splits[s].name}`]), ui.scheme, 'Cross-validation split')
    + sel('ev-model', models.map(m => [m, `${m}: ${MODELS[m] ?? ''}`]), ui.model, 'Model');
  const rerender = () => renderEvaluation(pane, tools, S, openDataset);
  tools.querySelector('#ev-src').onchange = e => { ui.source = e.target.value; rerender(); };
  tools.querySelector('#ev-split').onchange = e => { ui.scheme = e.target.value; rerender(); };
  tools.querySelector('#ev-model').onchange = e => { ui.model = e.target.value; rerender(); };

  const rows = src.builds.filter(b => b.scheme === ui.scheme && b.model === ui.model);
  const cur = S.entry?.source === 'public' ? S.entry.meta?.id : null;
  const hi = Math.max(...rows.map(b => Math.max(b.measured, b.predicted + b.predicted_sd))) * 1.08;
  const F = frame({ w: 330, h: 290, x: [0, 100 * hi], y: [0, 100 * hi], xlabel: 'measured v3 label (% of reference diagonal)', ylabel: 'predicted (% of reference diagonal)' });
  let s = F.open + F.axes + `<line x1="${F.X(0)}" y1="${F.Y(0)}" x2="${F.X(100 * hi)}" y2="${F.Y(100 * hi)}" stroke="#999" stroke-dasharray="3 3"/>`;
  for (const b of rows) {
    const x = F.X(100 * b.measured), y = F.Y(100 * b.predicted), c = GROUP[b.capture] ?? '#555';
    if (b.predicted_sd > 0) s += `<line x1="${x}" x2="${x}" y1="${F.Y(100 * (b.predicted - b.predicted_sd))}" y2="${F.Y(100 * (b.predicted + b.predicted_sd))}" stroke="${c}" stroke-opacity="0.6"/>`;
    s += `<circle data-d="${esc(b.dataset)}" cx="${x}" cy="${y}" r="${b.dataset === cur ? 5 : 3.2}" fill="${c}" ${b.dataset === cur ? 'stroke="#0b5fd0" stroke-width="2"' : ''} style="cursor:pointer"><title>${esc(b.run)}\nmeasured ${pct(b.measured)}%  predicted ${pct(b.predicted)}% ± ${pct(b.predicted_sd)} (${b.seeds} seeds)</title></circle>`;
  }
  let ly = F.m.t + 4;
  for (const [g, c] of Object.entries(GROUP)) { s += `<circle cx="${F.m.l + 8}" cy="${ly + 4}" r="3.2" fill="${c}"/><text x="${F.m.l + 15}" y="${ly + 7}">${g}</text>`; ly += 12; }
  s += '</svg>';
  const m = src.metrics.find(x => x.scheme === ui.scheme && x.model === ui.model);
  const fig = figure(`pred-vs-measured-${ui.source}-${ui.scheme}-${ui.model}`, s,
    `Out-of-fold predictions of <b>${esc(ui.model)}</b> (${esc(MODELS[ui.model] ?? '')}), split ${ui.scheme}: ${esc(data.splits[ui.scheme].name)}. `
    + `Points: one build, prediction averaged over ${m.seeds > 1 ? `${m.seeds} seeds (bar: ±1 SD)` : 'one deterministic fit'}. Dashed: perfect prediction. `
    + `Source: ${esc(src.predictions)}.`,
    rows.map(b => ({ dataset: b.dataset, held_out: b.held_out, measured: b.measured, predicted_mean: b.predicted, predicted_sd: b.predicted_sd, seeds: b.seeds })));
  fig.querySelectorAll('circle[data-d]').forEach(c => c.onclick = () => openDataset(`public:${c.dataset.d}`));
  pane.innerHTML = '';
  pane.appendChild(fig);

  // all models on this split: MAE over seeds and rank correlation inside each held-out fold
  const folds = Object.keys(m.within);
  const all = src.metrics.filter(x => x.scheme === ui.scheme).sort((a, b) => a.mae_mean - b.mae_mean);
  const tbl = document.createElement('div');
  tbl.className = 'fig';
  tbl.innerHTML = `<table class="t" style="width:auto"><thead><tr><th>model</th><th class="n">MAE (% diag)</th><th class="n">± SD</th><th class="n">seeds</th>
    ${folds.map(f => `<th class="n" title="Spearman rank correlation between predicted and measured inside this fold">ρ ${esc(f)} (n=${m.within[f].n})</th>`).join('')}</tr></thead><tbody>
    ${all.map(x => `<tr data-m="${x.model}" aria-selected="${x.model === ui.model}"><td title="${esc(MODELS[x.model] ?? '')}">${esc(x.model)}</td><td class="n">${pct(x.mae_mean)}</td><td class="n">${x.seeds > 1 ? pct(x.mae_sd) : '—'}</td><td class="n">${x.seeds}</td>
      ${folds.map(f => `<td class="n">${x.within[f]?.rho == null ? '—' : x.within[f].rho.toFixed(2)}</td>`).join('')}</tr>`).join('')}</tbody></table>
    <p class="cap">MAE: mean over seeds of the mean absolute error, identical to ${esc(src.summary)}. ρ: Spearman correlation of the seed-mean predictions with the labels inside each ${ui.scheme === 'LORO' ? 'object' : 'held-out fold'}, i.e. whether the model ranks builds of one capture correctly. A low MAE across objects can come from getting the level right without ranking builds.</p>`;
  tbl.querySelectorAll('tr[data-m]').forEach(tr => tr.onclick = () => { ui.model = tr.dataset.m; rerender(); });
  pane.appendChild(tbl);

  const note = document.createElement('div');
  note.className = 'fig'; note.style.maxWidth = '300px';
  const curRow = rows.find(b => b.dataset === cur);
  note.innerHTML = `${curRow ? `<p class="note info"><b>${esc(curRow.run)}</b>: measured ${pct(curRow.measured)}%, predicted ${pct(curRow.predicted)}% ± ${pct(curRow.predicted_sd)} (held out: ${esc(curRow.held_out)})</p>` : ''}
    <p class="note${ui.scheme === 'LORO' ? '' : ' info'}">${esc(data.splits[ui.scheme].note)}</p>
    <p class="note info">28 builds of 2 objects. Labels are relative to each capture's own reference build, so pooled errors mix three label scales. Every prediction here is out-of-fold. The capture app's models (models/quality-attention) were trained on all 28 builds, so their predictions on these builds would be in-sample and are not shown.</p>`;
  pane.appendChild(note);
}
