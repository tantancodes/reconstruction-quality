"""Figures for the README and the project page, drawn from tracked outputs only (nothing is trained).

    python scripts/make_readme_figures.py            # writes docs/figures/*.png

Inputs: results/pahma-5-613-v2/*/cameras.csv and labels-all.csv, docs/figures/data/real_runs.csv,
results/synthetic-coverage-gap/summary.csv, results/pilot-real-v3-all/summary.csv, and
experiments/geometry-aware/results/*/{summary,curves,predictions}.csv. The attention-map figures
are the saved outputs of experiments/geometry-aware/geometry_aware_attention_analysis.ipynb,
extracted unchanged.
"""
import base64
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from bd3d_quality import coverage  # noqa: E402

OUT = REPO / 'docs' / 'figures'
GA = REPO / 'experiments' / 'geometry-aware'
COFFIN = REPO / 'results' / 'pahma-5-613-v2'

INK, GREY, LIGHT, RED = '#222222', '#8a8a8a', '#d9d9d9', '#c0392b'
MODEL_C = {'mean': '#bdbdbd', 'reproj': '#bdbdbd', 'ridge': '#9e9e9e', 'mlp': '#9e9e9e', 'xgb': '#9e9e9e',
           'xgb_pair': '#7b5ea7', 'pointnet': '#6b6b6b', 'attention': '#2f4b7c', 'geo_shuf': '#7aa6c2',
           'geo': RED, 'geo_inv': '#e08a3c'}
GROUP_C = {'arf-dog (historical CPU)': '#4477aa', 'arf-dog (GPU d8)': '#66aacc', 'pahma-5-613 (GPU d8)': '#aa3377'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
    'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'legend.fontsize': 7, 'savefig.dpi': 220, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
    'axes.titleweight': 'normal', 'axes.titlelocation': 'left',
})


def read(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def panel(ax, letter):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight='bold', ha='right', va='bottom')


# ---------------------------------------------------------------- teaser
def coffin_cameras():
    labels, ok, P, V = coverage.load_cameras(COFFIN / 'ref145b' / 'cameras.csv')
    labels, P, V = np.array(labels)[ok], P[ok], V[ok]
    run = {r['label'] for r in read(COFFIN / 'side40_s1' / 'cameras.csv') if r['aligned'].strip() == 'True'}
    used = np.array([l in run for l in labels])
    centre, B = coverage.camera_frame(P, V)
    Q = (P - centre) @ B.T
    W = V @ B.T
    s = np.median(np.linalg.norm(Q, axis=1))
    return Q / s, W, used


def frustum(c, w, up_hint, size):
    """Corners of a small pyramid (apex at the camera centre, opening along the view w)."""
    w = w / np.linalg.norm(w)
    r = np.cross(w, up_hint)
    r /= np.linalg.norm(r)
    u = np.cross(r, w)
    base = c + w * size
    a = size * 0.45
    return c, [base + a * (sr * r + su * u * 0.7) for sr, su in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def oblique(X, az=-58, el=32):
    """Orthographic view: rotate about up (z) by az, tilt by el; returns screen x, y, depth."""
    a, e = np.radians(az), np.radians(el)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    Y = X @ Rz.T @ Rx.T
    return Y[..., 0], Y[..., 2], Y[..., 1]


def teaser():
    Q, W, used = coffin_cameras()
    fig = plt.figure(figsize=(8.2, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.2, 0.85], wspace=0.34)

    # (a) camera network
    ax = fig.add_subplot(gs[0])
    panel(ax, 'a')
    polys = []
    for c, w, u in zip(Q, W, used):
        apex, base = frustum(c, w, np.array([0, 0, 1.0]), 0.16)
        x, y, d = oblique(np.array([apex] + base))
        for k in range(4):
            polys.append((d.mean(), [(x[0], y[0]), (x[k + 1], y[k + 1]), (x[(k + 1) % 4 + 1], y[(k + 1) % 4 + 1])], u))
    for _, pts, u in sorted(polys, key=lambda t: -t[0]):
        ax.add_patch(Polygon(pts, closed=True, fc='white' if u else '#f6d5d1', ec=INK if u else RED, lw=0.35,
                             alpha=1 if u else 0.95))
    t = np.linspace(0, 2 * np.pi, 200)
    rx, ry, _ = oblique(np.stack([np.cos(t) * 0.55, np.sin(t) * 0.55, np.zeros_like(t)], 1))
    ax.plot(rx, ry, color=GREY, lw=0.5, ls=(0, (2, 2)))
    ux, uy, _ = oblique(np.array([[0, 0, 0], [0, 0, 0.9]]))
    ax.annotate('', (ux[1], uy[1]), (ux[0], uy[0]), arrowprops=dict(arrowstyle='-|>', lw=0.6, color=GREY, mutation_scale=6))
    ax.set_aspect('equal', adjustable='datalim')
    ax.autoscale_view()
    ax.axis('off')
    ax.set_title('Cameras, coffin (n = 145)')
    ax.text(0.0, 0.02, f'white: {used.sum()} kept in build side40_s1\nred: {(~used).sum()} wall-level photos removed',
            transform=ax.transAxes, fontsize=6.3, va='top')

    # (b) coverage in azimuth x elevation, reference frame
    ax = fig.add_subplot(gs[1])
    panel(ax, 'b')
    cov = coverage.compare(COFFIN / 'side40_s1' / 'cameras.csv', COFFIN / 'ref145b' / 'cameras.csv')
    ref = cov.reference
    az_e, el_e = cov.az_edges, cov.el_edges
    for i in range(len(el_e) - 1):
        for j in range(len(az_e) - 1):
            miss = cov.missing[i, j]
            ax.add_patch(Rectangle((az_e[j], el_e[i]), az_e[j + 1] - az_e[j], el_e[i + 1] - el_e[i],
                                   fc='#f6d5d1' if miss else 'white', ec=RED if miss else LIGHT, lw=0.8 if miss else 0.5,
                                   zorder=1 if not miss else 2))
            ax.text((az_e[j] + az_e[j + 1]) / 2, (el_e[i] + el_e[i + 1]) / 2, str(cov.counts[i, j]), ha='center',
                    va='center', fontsize=5.5, color=RED if miss else GREY, zorder=3)
    raz, rel_ = ref.azimuth, ref.elevation
    ref_used = np.isin(ref.labels, cov.labels)
    ax.scatter(raz[ref_used], rel_[ref_used], s=4, c=INK, lw=0, zorder=4, label='kept')
    ax.scatter(raz[~ref_used], rel_[~ref_used], s=7, facecolors='none', edgecolors=RED, lw=0.6, zorder=4, label='removed')
    ax.set_xlim(0, 360)
    ax.set_ylim(el_e[0] - 2, el_e[-1] + 2)
    ax.set_xticks(range(0, 361, 90))
    ax.set_xlabel('azimuth about ring normal (deg)')
    ax.set_ylabel('elevation (deg)')
    ax.set_title('Coverage cells (cameras per cell)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), handletextpad=0.2, borderaxespad=0.1, fontsize=6.5)

    # (c) the quick check after alignment does not predict geometry
    ax = fig.add_subplot(gs[2])
    panel(ax, 'c')
    rows = [r for r in read(REPO / 'docs/figures/data/real_runs.csv') if r['reprojection_rmse_px'] and r['label_v3']]
    stats = json.loads((REPO / 'docs/figures/data/stats.json').read_text())['reprojection_vs_label']
    for g, c in GROUP_C.items():
        R = [r for r in rows if r['group'] == g]
        if not R or g.startswith('arf-dog (hist'):
            continue
        x = np.array([float(r['reprojection_rmse_px']) for r in R])
        y = np.array([float(r['label_v3']) for r in R]) * 100
        o = np.argsort(x)
        ax.plot(x[o], y[o], color=c, lw=0.5, alpha=0.5)
        ax.scatter(x, y, s=12, color=c, lw=0, zorder=3,
                   label=f"{'coffin' if g.startswith('pahma') else 'dog'}, ρ = {stats[g]['spearman_rho']:+.2f}")
    ax.set_xlabel('reprojection RMSE (px)')
    ax.set_ylabel('surface error (% of diagonal)')
    ax.set_title('Reprojection vs. surface error')
    ax.legend(loc='upper right', handletextpad=0.1, fontsize=6.5)
    fig.savefig(OUT / 'teaser.png')
    plt.close(fig)


# ---------------------------------------------------------------- architecture
def box(ax, x, y, w, h, text, fc='white', ec=INK, fs=6.8, lw=0.7, color=INK, style='square,pad=0'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=color, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color=INK):
    ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', lw=0.6, color=color, mutation_scale=7,
                                                         shrinkA=0, shrinkB=0))


def architecture():
    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis('off')
    X = dict(inp=(0, 11), phi=(13.5, 13), att=(29.5, 25), ff=(57, 9), pool=(68.5, 10), rho=(81, 8))
    rows = [(36, 'PointNet', 'each camera is encoded alone'), (22, 'Self-attention', 'cameras are compared with each other before pooling'),
            (8, 'Geometry-aware attention', 'attention logits receive a learned bias from pairwise camera geometry')]
    for y, name, sub in rows:
        geo = name.startswith('Geometry')
        ax.text(0, y + 5.4, name, fontsize=7.5, fontweight='bold', va='bottom')
        ax.text(len(name) * 1.25 + 1.5, y + 5.4, sub, fontsize=6.6, color=GREY, va='bottom')
        mid = y + 2.3
        box(ax, *X['inp'][:1], y, X['inp'][1], 4.6, 'camera set\nN × 6', fs=6.4)
        box(ax, X['phi'][0], y, X['phi'][1], 4.6, 'per-camera MLP\nφ: 6 → h', fs=6.4)
        arrow(ax, 11, mid, 13.5, mid)
        if name == 'PointNet':
            arrow(ax, 26.5, mid, 68.5, mid)
        else:
            arrow(ax, 26.5, mid, 29.5, mid)
            box(ax, X['att'][0], y, X['att'][1], 4.6,
                'multi-head self-attention, H = 4\nsoftmax(QKᵀ/√d' + (' + g(rᵢⱼ)' if geo else '') + ') V', fs=6.4,
                fc='#fbeceb' if geo else 'white', ec=RED if geo else INK)
            arrow(ax, 54.5, mid, 57, mid)
            box(ax, X['ff'][0], y, X['ff'][1], 4.6, 'residual\n+ FF', fs=6.4)
            arrow(ax, 66, mid, 68.5, mid)
        box(ax, X['pool'][0], y, X['pool'][1], 4.6, 'max-pool\nover N', fs=6.4)
        arrow(ax, 78.5, mid, 81, mid)
        box(ax, X['rho'][0], y, X['rho'][1], 4.6, 'MLP ρ', fs=6.4)
        arrow(ax, 89, mid, 91.5, mid)
        ax.text(92.2, mid, 'ŷ', fontsize=8.5, va='center')
    box(ax, 28.5, 0.4, 27, 5.6, 'rᵢⱼ = ( ‖pᵢ − pⱼ‖,  r̂ᵢ·r̂ⱼ,  vᵢ·vⱼ )\ng: MLP 3 → 16 → H, shared over pairs', fs=6.2,
        ec=RED)
    arrow(ax, 42, 6.0, 42, 8, color=RED)
    ax.text(57, 3.2, 'rotation-invariant by construction;\n~100 extra parameters', fontsize=6.2, color=GREY, va='center')
    fig.savefig(OUT / 'architecture.png')
    plt.close(fig)


# ---------------------------------------------------------------- results
def dots(ax, names, means, sds, highlight=(), xlabel='', xlim=None, fmt='{:.1f}'):
    y = np.arange(len(names))[::-1]
    for yi, n, m, s in zip(y, names, means, sds):
        c = MODEL_C.get(n, GREY)
        ax.plot([m - s, m + s], [yi, yi], color=c, lw=1.1, solid_capstyle='butt')
        ax.scatter([m], [yi], s=16 if n in highlight else 11, color=c, zorder=3, lw=0)
        ax.text(m + s + (xlim[1] - xlim[0]) * 0.015 if xlim else m + s, yi, fmt.format(m), va='center', fontsize=6, color=GREY)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(xlabel)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(axis='x', color='#eeeeee', lw=0.5)
    ax.set_axisbelow(True)


def results_synthetic():
    orig = read(REPO / 'results/synthetic-coverage-gap/summary.csv')
    loc = read(GA / 'results/synthetic-localized/summary.csv')
    cs = read(GA / 'results/synthetic-count-shift/summary.csv')
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.5), gridspec_kw=dict(wspace=0.75, width_ratios=[1, 1.1, 1.1]))
    order = ['mean', 'ridge', 'pointnet', 'attention']
    ax = axs[0]
    for k, (cond, off, fill) in enumerate((('global', 0.18, False), ('localized', -0.18, True))):
        R = {r['model']: r for r in orig if r['condition'] == cond}
        for yi, m in zip(np.arange(len(order))[::-1], order):
            mu, sd, c = float(R[m]['mae_mean']), float(R[m]['mae_std']), MODEL_C[m]
            ax.plot([mu - sd, mu + sd], [yi + off] * 2, color=c, lw=1.1)
            ax.scatter([mu], [yi + off], s=13, color=c if fill else 'white', edgecolors=c, lw=0.8, zorder=3,
                       label=cond if m == 'mean' else None)
            ax.text(mu + sd + 0.8, yi + off, f'{mu:.1f}', va='center', fontsize=6, color=GREY)
    ax.set_yticks(np.arange(len(order))[::-1])
    ax.set_yticklabels(order)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 42)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel('test MAE (deg)')
    ax.grid(axis='x', color='#eeeeee', lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='center right', fontsize=6.3, handletextpad=0.1)
    axs[0].set_title('Original, 400 ep., 5 seeds')
    panel(axs[0], 'a')
    for ax, rows, title, letter in ((axs[1], loc, 'Localized gap, 150 ep., 3 seeds', 'b'),
                                    (axs[2], cs, 'Count shift: train 20–35, test 45–60', 'c')):
        R = {r['model']: r for r in rows}
        order = [m for m in ['mean', 'ridge', 'mlp', 'xgb', 'xgb_pair', 'pointnet', 'attention', 'geo_shuf', 'geo', 'geo_inv']
                 if m in R]
        dots(ax, order, [float(R[m]['mae_mean']) for m in order], [float(R[m]['mae_std']) for m in order],
             ('geo',), 'test MAE (deg)', (0, 42))
        ax.set_title(title)
        panel(ax, letter)
    fig.savefig(OUT / 'results_synthetic.png')
    plt.close(fig)


def curves():
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.3), gridspec_kw=dict(wspace=0.32))
    for ax, name, title, letter in ((axs[0], 'synthetic-localized', 'Localized gap (3 seeds)', 'a'),
                                    (axs[1], 'synthetic-count-shift', 'Count shift (3 seeds)', 'b')):
        rows = read(GA / 'results' / name / 'curves.csv')
        by = defaultdict(lambda: defaultdict(list))
        for r in rows:
            by[r['model']][int(r['epoch'])].append(float(r['val_mae_deg']))
        for m in ['pointnet', 'attention', 'geo_shuf', 'geo', 'geo_inv']:
            if m not in by:
                continue
            e = np.array(sorted(by[m]))
            v = np.array([by[m][k] for k in e])
            mu, sd = v.mean(1), v.std(1, ddof=1)
            ax.fill_between(e, mu - sd, mu + sd, color=MODEL_C[m], alpha=0.15, lw=0)
            ax.plot(e, mu, color=MODEL_C[m], lw=1.0, label=m)
        ax.set_xlabel('epoch')
        ax.set_title(title)
        panel(ax, letter)
    axs[0].set_ylabel('validation MAE (deg)')
    axs[0].legend(loc='upper right', ncol=1, fontsize=6.3)
    ax = axs[2]
    rows = read(GA / 'results/synthetic-localized-long/curves.csv')
    for m in ['attention', 'geo']:
        R = [r for r in rows if r['model'] == m]
        ax.plot([int(r['epoch']) for r in R], [float(r['val_mae_deg']) for r in R], color=MODEL_C[m], lw=1.0, label=m)
    ax.axhline(13.09, color=MODEL_C['attention'], lw=0.7, ls=(0, (3, 2)))
    ax.text(395, 13.09 - 0.5, 'original attention (Xavier init.), test 13.09°', fontsize=5.8, ha='right', va='top',
            color=MODEL_C['attention'])
    ax.set_ylim(11, 36)
    ax.set_xlabel('epoch')
    ax.set_title('Localized, 400 epochs (1 seed)')
    ax.legend(loc='upper right', fontsize=6.3)
    panel(ax, 'c')
    fig.savefig(OUT / 'curves.png')
    plt.close(fig)


def ranks(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind='mergesort')
    r = np.empty(len(x))
    r[o] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        r[x == v] = r[x == v].mean()
    return r


def spearman(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1]) if np.ptp(a) > 0 and np.ptp(b) > 0 else np.nan


def results_real():
    summ = read(GA / 'results/real/summary.csv')
    preds = read(GA / 'results/real/predictions.csv')
    order = ['mean', 'reproj', 'ridge', 'xgb', 'xgb_pair', 'pointnet', 'attention', 'geo_shuf', 'geo', 'geo_inv']
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.6), gridspec_kw=dict(wspace=0.75, width_ratios=[1, 1, 1.25]))
    for ax, scheme, title, letter in ((axs[0], 'LOGO', 'Capture group held out (3 folds)', 'a'),
                                      (axs[1], 'LOAO', 'Object held out (2 folds)', 'b')):
        R = {r['model']: r for r in summ if r['scheme'] == scheme}
        dots(ax, order, [100 * float(R[m]['mae_mean']) for m in order], [100 * float(R[m]['mae_std']) for m in order],
             ('geo',), 'MAE (% of diagonal)', (0, 4.6), fmt='{:.2f}')
        ax.set_title(title)
        panel(ax, letter)
    # (c) within-fold Spearman rho, LOAO and LOGO, seed-mean predictions
    agg = defaultdict(list)
    for r in preds:
        agg[(r['scheme'], r['model'], r['held_out'], r['run'])].append((float(r['predicted']), float(r['actual'])))
    cols = [('LOAO', 'arf-dog', 'dog\n(n=23)'), ('LOAO', 'pahma-5-613', 'coffin\n(n=5)'),
            ('LOGO', 'arf-dog/hist16-cpu', 'CPU dog\n(n=16)'), ('LOGO', 'arf-dog/gpu-d8', 'GPU dog\n(n=7)')]
    models = ['reproj', 'ridge', 'xgb', 'pointnet', 'attention', 'geo_shuf', 'geo']
    M = np.full((len(models), len(cols)), np.nan)
    for i, m in enumerate(models):
        for j, (s, h, _) in enumerate(cols):
            pts = [(np.mean([p for p, _ in v]), v[0][1]) for (s_, m_, h_, _), v in agg.items() if s_ == s and m_ == m and h_ == h]
            if pts:
                M[i, j] = spearman([p for p, _ in pts], [a for _, a in pts])
    ax = axs[2]
    im = ax.imshow(M, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    for i in range(len(models)):
        for j in range(len(cols)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f'{M[i, j]:+.2f}', ha='center', va='center', fontsize=6,
                        color='white' if abs(M[i, j]) > 0.6 else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[2] for c in cols], fontsize=6)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axvline(1.5, color='white', lw=2)
    ax.text(0.5, len(models) + 0.75, 'object held out', ha='center', va='top', fontsize=6.3, color=GREY)
    ax.text(2.5, len(models) + 0.75, 'group held out', ha='center', va='top', fontsize=6.3, color=GREY)
    ax.set_title('Ranking within held-out fold (Spearman ρ)')
    panel(ax, 'c')
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=2)
    fig.savefig(OUT / 'results_real.png')
    plt.close(fig)
    return M, models, cols


def notebook_figures():
    """The attention maps are saved outputs of the analysis notebook; copy them out unchanged."""
    nb = json.loads((GA / 'geometry_aware_attention_analysis.ipynb').read_text())
    wanted = {'## 8. Attention matrices': ['attention_maps.png', 'attention_vs_angle.png'],
              '## 7. Geometry-aware attention': ['learned_bias.png'],
              '## 3. Camera positions': [None, 'pairwise_relations.png']}
    section, out = None, []
    for c in nb['cells']:
        if c['cell_type'] == 'markdown':
            head = ''.join(c['source']).split('\n')[0]
            section = next((k for k in wanted if head.startswith(k)), None if head.startswith('## ') else section)
            continue
        for o in c.get('outputs', []):
            png = o.get('data', {}).get('image/png')
            if png and section and wanted[section]:
                name = wanted[section].pop(0)
                if name is None:
                    continue
                (OUT / name).write_bytes(base64.b64decode(png))
                out.append(name)
    # the notebook's in-figure title duplicates the caption here; trim it (pixels only, no data change)
    from PIL import Image
    im = Image.open(OUT / 'attention_vs_angle.png')
    im.crop((0, 38, im.width, im.height)).save(OUT / 'attention_vs_angle.png')
    return out


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    teaser()
    architecture()
    results_synthetic()
    curves()
    M, models, cols = results_real()
    print('notebook figures:', notebook_figures())
    print('rho:', {m: [None if np.isnan(v) else round(v, 2) for v in row] for m, row in zip(models, M)})
