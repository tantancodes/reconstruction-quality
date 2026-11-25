// Viewer wiring: catalogue, selection, controls, dock tabs, URL state.
import * as THREE from 'three';
import { View3D } from './view3d.js';
import { publicEntries, exportEntries, folderEntries, fileEntry, droppedFiles, sceneFromCameras, sceneFromExport } from './data.js';
import { renderTree, renderInspector, renderCameraTable, renderCoverage, QUANT, fmt } from './panels.js';
import { viridis, rgb, download, esc } from './plot.js';
import { renderEvaluation } from './evaluation.js';

const $ = id => document.getElementById(id);
const S = { entries: [], entry: null, sc: null, sel: -1, tab: 'cameras', table: {}, colorBy: 'state', mesh: null, tabs: {} };
export const app = S;

const view = new View3D($('gl'), {
  onPick: i => select(i),
  onHover: (i, e) => {
    const tip = $('tip');
    if (i < 0 || !S.sc) { tip.style.display = 'none'; return; }
    const c = S.sc.cameras[i];
    tip.textContent = `${c.label}  ${c.state === 'used' ? '' : c.state + '  '}az ${c.az.toFixed(1)}° el ${c.el.toFixed(1)}°`;
    Object.assign(tip.style, { display: 'block', left: e.clientX + 12 + 'px', top: e.clientY + 12 + 'px' });
  },
});

// ------------------------------------------------------------------ catalogue
async function boot() {
  const [pub, exp] = await Promise.all([publicEntries(), exportEntries()]);
  S.manifest = pub.manifest;
  S.entries = [...exp, ...pub.entries];
  const q = new URLSearchParams(location.search);
  const want = q.get('dataset') ?? (q.get('object') ? S.entries.find(e => e.key.includes(`viewer/data/${q.get('object')}/`))?.key : null);
  const first = S.entries.find(e => e.key === want) ?? S.entries.find(e => e.kind === 'export') ?? S.entries.find(e => e.key.endsWith('/side40_s1')) ?? S.entries.find(e => e.role === 'build');
  if (first) await open(first, q.get('ref'));
  else { $('empty').textContent = 'No datasets found. Open a build folder (File > Open folder) or serve the viewer from the repository.'; drawTree(); }
}

function drawTree() { renderTree($('tree'), S.entries, S.entry?.key, e => open(e)); }

function refOptions(entry) {
  const sel = $('ref');
  sel.innerHTML = '';
  if (entry.kind === 'export') { sel.add(new Option(S.sc?.reference ? `${S.sc.reference} (export)` : 'none (export)', '')); sel.disabled = true; return; }
  sel.disabled = false;
  sel.add(new Option('none', ''));
  const same = S.entries.filter(e => e.kind === 'cameras' && e !== entry && (e.group === entry.group || e.role === 'reference'));
  for (const e of same) sel.add(new Option(`${e.name}${e.role === 'reference' ? ' (reference)' : ''} · ${e.group}`, e.key));
}

async function open(entry, refKey) {
  S.entry = entry; S.sel = -1;
  $('empty').hidden = false; $('empty').textContent = `Loading ${entry.name}…`;
  try {
    let ref = null;
    if (entry.kind === 'cameras') {
      const key = refKey === undefined || refKey === null ? entry.refKey : refKey;
      ref = key ? S.entries.find(e => e.key === key) ?? null : null;
      S.sc = await sceneFromCameras(entry, ref).catch(async err => {   // fall back to the build alone
        if (!ref) throw err; S.warn = `Reference not used: ${err.message}`; ref = null; return sceneFromCameras(entry, null);
      });
    } else S.sc = await sceneFromExport(entry);
    refOptions(entry);
    $('ref').value = entry.kind === 'cameras' && ref ? ref.key : '';
    view.setData(S.sc);
    applyColors();
    $('empty').hidden = true;
    S.mesh = null;
    if (S.sc.mesh) {
      status(`Reading mesh of ${entry.name}…`);
      view.setMeshes(S.sc).then(info => { S.mesh = info; refresh(); }).catch(err => { S.sc.notes.push(`Mesh not shown: ${err.message}`); refresh(); });
    }
    history.replaceState(null, '', entry.source === 'public' ? `?dataset=${encodeURIComponent(entry.key)}${ref && ref.key !== entry.refKey ? `&ref=${encodeURIComponent(ref.key)}` : !ref && entry.refKey ? '&ref=' : ''}` : location.pathname);
  } catch (err) {
    S.sc = null; $('empty').hidden = false; $('empty').textContent = `Cannot show ${entry.name}: ${err.message}`;
  }
  drawTree(); refresh();
}

// ------------------------------------------------------------------ colours and legend
function applyColors() {
  const sc = S.sc; if (!sc) return;
  const key = S.colorBy, lg = $('legend');
  if (key === 'state') {
    const C = { used: new THREE.Color('#3a3a36'), removed: new THREE.Color('#d93a2b'), unaligned: new THREE.Color('#d93a2b') };
    view.opts.colorOf = c => C[c.state];
    const has = s => sc.cameras.some(c => c.state === s);
    lg.innerHTML = `<div class="k"><span class="sw" style="background:#3a3a36"></span>used in this build</div>`
      + (has('removed') ? `<div class="k"><span class="sw" style="background:#fff;border:1px dashed #d93a2b"></span>reference photo not in this build</div>` : '')
      + (has('unaligned') ? `<div class="k"><span class="sw" style="background:#d93a2b"></span>in this build, failed to align</div>` : '')
      + `<div class="k"><span class="sw" style="background:#0b5fd0"></span>selected</div>`
      + (sc.coverage ? `<div class="k"><span class="sw" style="background:rgba(217,58,43,.2);border-color:#d93a2b"></span>coverage cell missing</div>` : '');
  } else {
    const vals = sc.cameras.map(c => c[key]).filter(Number.isFinite);
    const lo = vals.length ? Math.min(...vals) : 0, hi = vals.length ? Math.max(...vals) : 1;
    view.opts.colorOf = c => (Number.isFinite(c[key]) ? new THREE.Color(rgb(viridis((c[key] - lo) / (hi - lo || 1)))) : new THREE.Color('#b0b0aa'));
    const [name, unit, d] = QUANT[key];
    const stops = Array.from({ length: 11 }, (_, i) => rgb(viridis(i / 10))).join(',');
    lg.innerHTML = vals.length ? `<div>${name}${unit ? ` (${unit})` : ''}</div><div class="cb" style="background:linear-gradient(90deg,${stops})"></div>
      <div class="ax"><span>${lo.toFixed(d)}</span><span>${hi.toFixed(d)}</span></div>${vals.length < sc.cameras.length ? '<div class="k"><span class="sw" style="background:#b0b0aa"></span>no value</div>' : ''}`
      : `<div>${name}: not in this dataset's cameras.csv</div>`;
  }
  view.drawCameras();
}

// ------------------------------------------------------------------ selection, panels, status
function select(i) {
  S.sel = i; view.select(i);
  $('look').disabled = i < 0;
  refresh();
}

function refresh() {
  renderInspector($('insp'), S.sc, S.sc?.cameras[S.sel], S.mesh);
  if (S.warn) { $('insp').insertAdjacentHTML('afterbegin', `<p class="note" style="margin:6px">${esc(S.warn)}</p>`); S.warn = null; }
  renderTab();
  const sc = S.sc;
  $('status').innerHTML = sc ? `<span>${esc(sc.title)}</span><span>${sc.cameras.filter(c => !c.refOnly).length} cameras${sc.reference ? ` · vs ${esc(sc.reference)}` : ''}</span>
    <span>1 u = ${esc(sc.unit)}</span><span class="grow"></span>
    <span>${sc.source === 'public' ? 'Public camera data; meshes and photos are not published' : 'Local data: read in this tab, never uploaded'}</span>` : '<span class="grow"></span><span>No dataset</span>';
  $('vinfo').textContent = sc ? `up = estimated ring normal (solid) · az 0° = dashed` : '';
  const exp = sc?.refMesh != null;
  ['L-missing', 'L-error'].forEach(id => { $(id).disabled = !exp; $(id).parentElement.title = exp ? '' : 'Needs a Python export with a reference (viewer_export compare)'; });
  $('L-mesh').disabled = !sc?.mesh; $('thresh-row').hidden = !exp;
}

function status(text) { $('status').innerHTML = `<span>${esc(text)}</span>`; }

function renderTab() {
  const pane = $('pane'), tools = $('tab-tools');
  document.querySelectorAll('#tabs [role=tab]').forEach(b => b.setAttribute('aria-selected', String(b.dataset.tab === S.tab)));
  if (!S.sc && S.tab !== 'evaluation') { pane.innerHTML = ''; tools.innerHTML = ''; return; }
  if (S.tab === 'cameras') renderCameraTable(pane, tools, S.sc, S.sel, select, S.table);
  else if (S.tab === 'coverage') renderCoverage(pane, tools, S.sc, S.sel, select);
  else if (S.tabs[S.tab]) S.tabs[S.tab](pane, tools, S);
  else { pane.className = 'pane'; tools.innerHTML = ''; pane.innerHTML = '<p style="color:var(--dim)">Loading…</p>'; }
}
export function registerTab(name, render) { S.tabs[name] = render; if (S.tab === name) renderTab(); }
export { open, renderTab };

// ------------------------------------------------------------------ controls
document.querySelectorAll('#tabs [role=tab]').forEach(b => b.onclick = () => { S.tab = b.dataset.tab; renderTab(); });
document.querySelectorAll('[data-view]').forEach(b => b.onclick = () => view.setView(b.dataset.view));
$('ortho').onclick = e => { const on = e.target.getAttribute('aria-pressed') !== 'true'; e.target.setAttribute('aria-pressed', String(on)); view.setOrtho(on); };
$('look').onclick = () => S.sel >= 0 && view.lookThrough(S.sel);
const layers = { 'L-frusta': 'frusta', 'L-rays': 'rays', 'L-shell': 'shell', 'L-refcams': 'refcams', 'L-mesh': 'mesh', 'L-missing': 'missing', 'L-error': 'error', 'L-axes': 'axes' };
for (const [id, name] of Object.entries(layers)) { view.opts.layers[name] = $(id).checked; $(id).onchange = () => view.setLayer(name, $(id).checked); }
$('color-by').onchange = () => { S.colorBy = $('color-by').value; applyColors(); };
const fs = () => { $('fsize-v').textContent = (+$('fsize').value).toFixed(3); };
$('fsize').oninput = () => { view.opts.frustumSize = +$('fsize').value; fs(); view.drawCameras(); }; fs();
const th = () => { $('thresh-v').textContent = `${(+$('thresh').value).toFixed(1)}% diag`; view.setThreshold(+$('thresh').value / 100 / 0.05); };
$('thresh').oninput = th; th();
$('ref').onchange = () => S.entry && open(S.entry, $('ref').value || '');
$('shot').onclick = async () => { const blob = await view.screenshot(+$('shot-scale').value); download(`${S.sc?.title ?? 'view'}-${$('shot-scale').value}x.png`, blob); };

async function addLocal(entries) {
  if (!entries.length) { status('No cameras.csv or scene.json found in what was opened.'); return; }
  const keys = new Set(entries.map(e => e.key));
  S.entries = [...entries, ...S.entries.filter(e => !keys.has(e.key))];
  await open(entries.find(e => e.role === 'build') ?? entries[0]);
}
$('open-folder').onclick = () => $('pick-folder').click();
$('open-file').onclick = () => $('pick-file').click();
$('pick-folder').onchange = async e => { await addLocal(await folderEntries([...e.target.files])); e.target.value = ''; };
$('pick-file').onchange = async e => { if (e.target.files[0]) await addLocal([fileEntry(e.target.files[0])]); e.target.value = ''; };
const vp = $('vp');
vp.addEventListener('dragover', e => { e.preventDefault(); vp.classList.add('drop'); });
vp.addEventListener('dragleave', () => vp.classList.remove('drop'));
vp.addEventListener('drop', async e => {
  e.preventDefault(); vp.classList.remove('drop');
  const files = await droppedFiles([...e.dataTransfer.items]);
  const single = files.length === 1 && files[0].name.endsWith('.csv') && !files[0].relPath.includes('/');
  await addLocal(single ? [fileEntry(files[0])] : await folderEntries(files));
});
window.addEventListener('keydown', e => {
  if (e.target.matches('input, select, textarea') || !S.sc) return;
  if (e.key === 'f') view.setView('iso');
  else if (e.key === 'Escape') select(-1);
  else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); const n = S.sc.cameras.length; select(((S.sel < 0 ? -1 : S.sel) + (e.key === 'ArrowDown' ? 1 : -1) + n) % n); }
});

S.tabs.evaluation = (pane, tools) => renderEvaluation(pane, tools, S, key => { const e = S.entries.find(x => x.key === key); if (e) open(e); });

// drag the top edge of the dock to resize it
document.getElementById('grip').onpointerdown = e => {
  const main = document.querySelector('.main'), y0 = e.clientY, h0 = document.querySelector('.dock').getBoundingClientRect().height;
  const move = ev => main.style.setProperty('--dock', `${Math.max(120, Math.min(main.clientHeight - 160, h0 + y0 - ev.clientY))}px`);
  const up = () => { removeEventListener('pointermove', move); removeEventListener('pointerup', up); S.view?.resize?.(); };
  addEventListener('pointermove', move); addEventListener('pointerup', up);
};

boot();
