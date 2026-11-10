"""Generate every README figure from saved results (no training, no Metashape).

  python3 scripts/make_figures.py            # writes docs/figures/*.png and docs/figures/data/*

Inputs are saved result files. Two inputs are local-only (not in git): TensorBoard logs in
runs/synthetic and the per-run camera tables in ignored run folders. The small derived tables
those figures need are cached in docs/figures/data/; when the local inputs are absent (for
example in a fresh clone) the cache is used instead, so every figure regenerates from git.

Figures
  hero.png                     coffin camera layout, largest gap per elevation band shaded
  camera_layouts.png           dog and coffin example subsets, same encoding
  evaluation_audit.png         v1 vs v3 labels against the initial frame rotation
  synthetic_mae.png            synthetic benchmark test MAE, mean +/- std over seeds
  learning_curves_synthetic.png  train loss and validation MAE per model (TensorBoard export)
  learning_curves_pilot.png    real-data pilot training loss
  label_vs_images.png          v3 label vs image count per artifact, by removal type
  reprojection_vs_label.png    run reprojection RMSE vs v3 label
  pilot_loao.png               pilot predicted vs actual, leave-one-artifact-out
Derived numbers quoted in the README are written to docs/figures/data/stats.json.
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
R = REPO / 'results'
OUT = REPO / 'docs' / 'figures'
DATA = OUT / 'data'

# Okabe-Ito palette (colorblind-safe)
C = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'red': '#D55E00', 'sky': '#56B4E9',
     'purple': '#CC79A7', 'yellow': '#F0E442', 'grey': '#7F7F7F', 'black': '#000000'}
MODEL_COLOR = {'mean': C['grey'], 'reproj': C['purple'], 'ridge': C['blue'], 'pointnet': C['orange'],
               'attention': C['green']}
MODEL_NAME = {'mean': 'mean', 'reproj': 'reprojection RMSE', 'ridge': 'Ridge', 'pointnet': 'PointNet',
              'attention': 'attention'}
REMOVAL_COLOR = {'random': C['blue'], 'contiguous side': C['red'], 'elevation band': C['green'],
                 'structured thinning': C['purple']}
REMOVAL_MARKER = {'random': 'o', 'contiguous side': 's', 'elevation band': '^', 'structured thinning': 'D'}
ARTIFACT_COLOR = {'arf-dog (historical CPU)': C['sky'], 'arf-dog (GPU d8)': C['blue'],
                  'pahma-5-613 (GPU d8)': C['orange']}

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'font.size': 10, 'axes.titlesize': 10.5, 'axes.labelsize': 10,
    'legend.fontsize': 8.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white'})

STATS = {}


def log_ticks(axis, ticks):
    """Readable fixed ticks on a log axis (no minor-tick labels)."""
    axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    axis.set_major_formatter(matplotlib.ticker.FixedFormatter([f'{t:g}' for t in ticks]))
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print('wrote', (OUT / name).relative_to(REPO))


# ------------------------------------------------------------------ camera layouts

def kmeans_1d(x, k, iters=100):
    c = np.quantile(x, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        lab = np.argmin(np.abs(x[:, None] - c[None]), 1)
        new = np.array([x[lab == j].mean() if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return np.argsort(np.argsort(c))[lab]


def reference_frame(P, V):
    """Azimuth/elevation (deg) about the least-squares viewing-ray centre; up = least-spread axis."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for p, v in zip(P, V):
        M = np.eye(3) - np.outer(v, v)
        A += M
        b += M @ p
    D = P - np.linalg.solve(A, b)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    up = np.linalg.eigh(np.cov(D.T))[1][:, 0]
    up = up if (D @ up).mean() > 0 else -up
    e1 = np.linalg.svd(np.eye(3) - np.outer(up, up))[0][:, 0]
    e2 = np.cross(up, e1)
    return np.degrees(np.arctan2(D @ e2, D @ e1)) % 360, np.degrees(np.arcsin(np.clip(D @ up, -1, 1)))


LAYOUTS = [  # (example id, artifact, reference run dir, candidate run dir, elevation bands)
    ('dog_side40', 'ARF dog', R / 'gpu-pilot-d8/ref50', R / 'gpu-pilot-d8/side40', 2),
    ('dog_random10', 'ARF dog', R / 'gpu-pilot-d8/ref50', R / 'gpu-pilot-d8/random10', 2),
    ('coffin_side40_s1', 'PAHMA 5-613 coffin', R / 'pahma-5-613-v2/ref145b', R / 'pahma-5-613-v2/side40_s1', 3),
    ('coffin_random30_s20261111', 'PAHMA 5-613 coffin', R / 'pahma-5-613-v2/ref145b',
     R / 'pahma-5-613-v2/random30_s20261111', 3),
]


def aligned(path):
    rows = [r for r in csv.DictReader(open(path, newline='')) if r.get('aligned', 'True') == 'True' and r['x']]
    return (rows, np.array([[float(r[k]) for k in 'xyz'] for r in rows]),
            np.array([[float(r['view_' + k]) for k in 'xyz'] for r in rows]))


def layout_table():
    cache = DATA / 'camera_layouts.csv'
    if all((ref / 'cameras.csv').exists() and (cand / 'cameras.csv').exists() for _, _, ref, cand, _ in LAYOUTS):
        out = []
        for ex, art, ref, cand, k in LAYOUTS:
            rows, P, V = aligned(ref / 'cameras.csv')
            az, el = reference_frame(P, V)
            band = kmeans_1d(el, k)
            used = {r['label'] for r in aligned(cand / 'cameras.csv')[0]}
            for r, a_, e_, b_ in zip(rows, az, el, band):
                out.append(dict(example=ex, artifact=art, reference=ref.name, candidate=cand.name, label=r['label'],
                                azimuth_deg=round(float(a_), 2), elevation_deg=round(float(e_), 2), band=int(b_),
                                used=r['label'] in used))
        pd.DataFrame(out).to_csv(cache, index=False)
    return pd.read_csv(cache)


def largest_gap(az):
    s = np.sort(np.asarray(az) % 360)
    if len(s) == 0:
        return 0.0, 360.0
    d = np.diff(np.r_[s, s[0] + 360])
    i = int(d.argmax())
    return float(s[i]), float(d[i])


def draw_layout(ax, g, title, legend=True, shade_bands=None):
    """Polar top-down view: angle = azimuth, radius = 90 - elevation (top-down cameras at the centre)."""
    gaps = []
    for b, gb in g.groupby('band'):
        r_lo, r_hi = 90 - gb.elevation_deg.max() - 2.5, 90 - gb.elevation_deg.min() + 2.5
        start, width = largest_gap(gb.azimuth_deg[gb.used])
        gaps.append(width)
        if shade_bands is not None and b not in shade_bands:
            continue
        ax.bar(np.radians(start + width / 2), r_hi - r_lo, width=np.radians(width), bottom=r_lo,
               color=C['red'], alpha=0.16, linewidth=0, label='largest gap in band' if b == 0 else None)
    nu = g[~g.used]
    ax.scatter(np.radians(nu.azimuth_deg), 90 - nu.elevation_deg, s=16, facecolors='none', edgecolors=C['grey'],
               linewidths=0.8, label='not used')
    u = g[g.used]
    ax.scatter(np.radians(u.azimuth_deg), 90 - u.elevation_deg, s=16, color=C['blue'], label='used')
    ax.set_ylim(0, 90)
    ax.set_yticks([30, 60, 90], ['elev. 60°', '30°', '0°'], fontsize=7, color=C['grey'])
    ax.set_rlabel_position(100)
    ax.set_xticks(np.radians([0, 90, 180, 270]), ['0°', '90°', '180°', '270°'], fontsize=8)
    ax.grid(color='#dddddd', linewidth=0.6)
    ax.set_title(title, pad=12)
    if legend:
        ax.legend(loc='lower left', bbox_to_anchor=(-0.18, -0.2), fontsize=7.5, frameon=False)
    return gaps


def fig_layouts():
    t = layout_table()
    STATS['camera_layouts'] = {}
    titles = {'dog_side40': 'ARF dog: side40', 'dog_random10': 'ARF dog: random10',
              'coffin_side40_s1': 'Coffin: c, side40_s1', 'coffin_random30_s20261111': 'Coffin: d, random30 (seed 2)'}
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6), subplot_kw={'projection': 'polar'})
    for ax, (ex, g) in zip(axes, t.groupby('example', sort=False)):
        gaps = draw_layout(ax, g, '', legend=ex == 'dog_side40')
        ax.set_title(f"{titles[ex]}\n{int(g.used.sum())} of {len(g)} cameras used\nlargest gap per band: "
                     + ', '.join(f'{x:.0f}°' for x in gaps), pad=14, fontsize=9.5)
        STATS['camera_layouts'][ex] = {'used': int(g.used.sum()), 'reference_cameras': len(g),
                                       'largest_gap_per_band_deg': [round(x, 1) for x in gaps]}
    fig.text(0.5, -0.04, 'Top-down view in the reference frame: angle = azimuth, radius = 90° − elevation '
             '(near-top-down cameras at the centre). Shaded: largest azimuth gap among used cameras, per elevation band.',
             ha='center', fontsize=8.5, color='#444444')
    fig.tight_layout(w_pad=3)
    save(fig, 'camera_layouts.png')

    g = t[t.example == 'coffin_side40_s1']
    fig, ax = plt.subplots(figsize=(5.2, 5.2), subplot_kw={'projection': 'polar'})
    gaps = draw_layout(ax, g, '', legend=False, shade_bands={0})
    ax.set_title(f'PAHMA 5-613 coffin: {int(g.used.sum())} of {len(g)} reference cameras used\n'
                 f'({len(g) - int(g.used.sum())} wall-level cameras removed; largest gap at wall level '
                 f'{gaps[0]:.0f}°)', pad=14)
    ax.legend(loc='upper left', bbox_to_anchor=(0.86, 1.05), fontsize=8, frameon=False)
    save(fig, 'hero.png')


# ------------------------------------------------------------------ evaluation audit

def fig_audit():
    rows = []
    for f in sorted((R / 'geometry-labels-v3' / 'reports').glob('*.json')):
        d = json.loads(f.read_text())
        M = np.array(d['camera_init_transform'])[:3, :3]
        Rm = M / np.cbrt(np.linalg.det(M))
        ang = float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))))
        r = d['row']
        rows.append(dict(experiment_id=r['experiment_id'], run_id=r['run_id'], images=int(r['images_used_count']),
                         initial_rotation_deg=round(ang, 1), label_v1=r['label_v1'], qc_v1=r['qc_v1'],
                         label_v3=r['label_v3'], qc_v3=r['qc_v3']))
    t = pd.DataFrame(rows).sort_values('initial_rotation_deg')
    t.to_csv(DATA / 'evaluation_audit.csv', index=False)
    big = t[t.initial_rotation_deg > 90]
    small = t[t.initial_rotation_deg <= 90]
    STATS['evaluation_audit'] = {
        'runs': len(t), 'large_rotation_runs': len(big),
        'large_rotation_deg_range': [big.initial_rotation_deg.min(), big.initial_rotation_deg.max()],
        'small_rotation_deg_range': [small.initial_rotation_deg.min(), small.initial_rotation_deg.max()],
        'large_rotation_qc_v1': sorted(big.qc_v1.unique()), 'small_rotation_qc_v1': sorted(small.qc_v1.unique()),
        'label_v1_range_large_rotation': [round(big.label_v1.min(), 4), round(big.label_v1.max(), 4)],
        'label_v3_range_all': [round(t.label_v3.min(), 4), round(t.label_v3.max(), 4)],
        'label_v3_range_large_rotation': [round(big.label_v3.min(), 4), round(big.label_v3.max(), 4)]}

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={'width_ratios': [1.35, 1]})
    for _, r in t.iterrows():
        a.plot([r.initial_rotation_deg] * 2, [r.label_v1, r.label_v3], color='#bbbbbb', lw=0.8, zorder=1)
    a.scatter(t.initial_rotation_deg, t.label_v1, s=30, color=C['red'], label='v1 (rigid ICP, identity start)', zorder=2)
    a.scatter(t.initial_rotation_deg, t.label_v3, s=30, facecolors='white', edgecolors=C['blue'], linewidths=1.3,
              label='v3 (camera-initialized similarity ICP)', zorder=3)
    a.set_yscale('log')
    a.set_xlabel('rotation between run frame and reference frame (degrees)')
    a.set_ylabel('label: symmetric RMSE / bbox diagonal')
    a.set_title('ARF dog, 16 historical runs: v1 error tracks frame rotation, not geometry')
    a.legend(frameon=False, loc='center', bbox_to_anchor=(0.5, 0.62))
    lim = [0.0025, 0.06]
    b.plot(lim, lim, color=C['grey'], lw=0.8)
    for qc, mk, col in (('pass', 'o', C['blue']), ('caution', 's', C['red'])):
        s = t[t.qc_v1 == qc]
        b.scatter(s.label_v1, s.label_v3, marker=mk, s=30, color=col, label=f'v1 QC "{qc}" (n={len(s)})')
    b.set_xscale('log')
    b.set_yscale('log')
    b.set_xlim(lim)
    b.set_ylim(lim)
    for axis in (a.yaxis, b.xaxis, b.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator([0.003, 0.01, 0.03]))
        axis.set_major_formatter(matplotlib.ticker.FixedFormatter(['0.003', '0.01', '0.03']))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    b.set_xlabel('v1 label')
    b.set_ylabel('v3 label')
    b.set_title('Same runs, v1 vs v3 (line: equal)')
    b.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    save(fig, 'evaluation_audit.png')


# ------------------------------------------------------------------ synthetic benchmark

def fig_synthetic():
    syn = pd.read_csv(R / 'synthetic-coverage-gap' / 'summary.csv')
    conds = ['global', 'localized']
    models = ['mean', 'ridge', 'pointnet', 'attention']
    fig, ax = plt.subplots(figsize=(8, 3.9))
    w = 0.19
    for j, m in enumerate(models):
        s = syn[syn.model == m].set_index('condition').loc[conds]
        bars = ax.bar(np.arange(2) + (j - 1.5) * w, s.mae_mean, w, yerr=s.mae_std, capsize=3,
                      color=MODEL_COLOR[m], label=MODEL_NAME[m], error_kw={'lw': 1})
        ax.bar_label(bars, labels=[f'{v:.1f}' for v in s.mae_mean], padding=2, fontsize=8)
    ax.set_xticks([0, 1], ['global gap\n(one elevation band)', 'localized gap\n(upper ring only, full lower ring)'])
    ax.set_ylabel('test MAE of largest azimuth gap (degrees)')
    ax.set_title(f'Synthetic camera sets: predicting the largest coverage gap '
                 f'(mean ± std over {int(syn.n_seeds.iloc[0])} seeds)')
    ax.legend(frameon=False, ncols=4, loc='upper left')
    ax.set_ylim(0, (syn.mae_mean + syn.mae_std).max() * 1.22)
    ax.grid(axis='x', visible=False)
    save(fig, 'synthetic_mae.png')


def tb_table():
    cache = DATA / 'learning_curves_synthetic.csv'
    runs = REPO / 'runs' / 'synthetic'
    if runs.exists():
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        out = []
        for cond in ('global', 'localized'):
            for m in ('pointnet', 'attention'):
                for d in sorted((runs / cond / m).iterdir(), key=lambda p: int(p.name)):
                    e = EventAccumulator(str(d))
                    e.Reload()
                    for tag in ('loss/train_mse_scaled', 'mae_deg/val'):
                        out += [dict(condition=cond, model=m, seed=int(d.name), tag=tag, epoch=s.step, value=s.value)
                                for s in e.Scalars(tag)]
        pd.DataFrame(out).to_csv(cache, index=False)
    return pd.read_csv(cache)


def band(ax, g, color, label):
    piv = g.pivot(index='epoch', columns='seed', values='value')
    m, s = piv.mean(axis=1), piv.std(axis=1)
    ax.plot(piv.index, m, color=color, lw=1.4, label=label)
    ax.fill_between(piv.index, m - s, m + s, color=color, alpha=0.2, lw=0)


def fig_learning_curves():
    t = tb_table()
    syn = pd.read_csv(R / 'synthetic-coverage-gap' / 'summary.csv')
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
    for j, cond in enumerate(('global', 'localized')):
        for m in ('pointnet', 'attention'):
            g = t[(t.condition == cond) & (t.model == m)]
            band(axes[0, j], g[g.tag == 'loss/train_mse_scaled'], MODEL_COLOR[m], MODEL_NAME[m])
            band(axes[1, j], g[g.tag == 'mae_deg/val'], MODEL_COLOR[m], MODEL_NAME[m])
        for m in ('mean', 'ridge'):
            v = syn[(syn.condition == cond) & (syn.model == m)].mae_mean.iloc[0]
            axes[1, j].axhline(v, color=MODEL_COLOR[m], ls='--', lw=1, label=f'{MODEL_NAME[m]} (test, no training)')
        axes[0, j].set_yscale('log')
        log_ticks(axes[0, j].yaxis, [0.01, 0.03, 0.1, 0.3])
        axes[0, j].set_title(f'{cond} gap: training loss')
        axes[1, j].set_title(f'{cond} gap: validation MAE')
        axes[1, j].set_xlabel('epoch')
    axes[0, 0].set_ylabel('train MSE (target = degrees / 100)')
    axes[1, 0].set_ylabel('validation MAE (degrees)')
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.suptitle('Synthetic benchmark learning curves (mean ± std over 5 seeds; exported from TensorBoard)', y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(h, l, loc='lower center', ncols=4, frameon=False)
    save(fig, 'learning_curves_synthetic.png')

    lc = pd.read_csv(R / 'pilot-real-v3-all' / 'learning_curves.csv')
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for m in ('pointnet', 'attention'):
        band(ax, lc[lc.model == m].rename(columns={'train_mse_scaled': 'value'}), MODEL_COLOR[m], MODEL_NAME[m])
    ax.set_yscale('log')
    log_ticks(ax.yaxis, [0.1, 0.3, 1, 3])
    ax.set_xlabel('epoch (full batch)')
    ax.set_ylabel('train MSE (target / fold mean label)')
    ax.set_title('Real-data pilot: training loss, first leave-one-run-out fold\n(mean ± std over 5 seeds; '
                 'no validation curve, one held-out run per fold)')
    ax.legend(frameon=False)
    save(fig, 'learning_curves_pilot.png')


# ------------------------------------------------------------------ real runs

def removal_type(name):
    n = name.lower()
    if 'random' in n or 'image-count' in n:
        return 'random'
    if 'side' in n or 'central-gap' in n:
        return 'contiguous side'
    if n.startswith('drop_s'):
        return 'elevation band'
    return 'structured thinning'


def real_runs():
    """All real candidate runs with a v3 label, plus the failed coffin alignment (no label)."""
    rows = []
    for r in pd.read_csv(R / 'geometry-labels-v3' / 'labels_v3.csv').itertuples():
        rows.append(dict(group='arf-dog (historical CPU)', run=r.experiment_id, images=r.images_used_count,
                         label_v3=r.label_v3, status='scored'))
    for r in pd.read_csv(R / 'gpu-pilot-d8-ext' / 'labels.csv').itertuples():
        rows.append(dict(group='arf-dog (GPU d8)', run=r.run, images=r.images, label_v3=r.label_v3, status='scored',
                         reprojection_rmse_px=r.run_reprojection_rmse_px, missing_rmse=r.missing_surface_rmse_norm,
                         extra_rmse=r.extra_surface_rmse_norm))
    for r in pd.read_csv(R / 'pahma-5-613-v2' / 'labels-all.csv').itertuples():
        rows.append(dict(group='pahma-5-613 (GPU d8)', run=r.run, images=r.images, label_v3=r.label_v3, status='scored',
                         reprojection_rmse_px=r.run_reprojection_rmse_px, missing_rmse=r.missing_surface_rmse_norm,
                         extra_rmse=r.extra_surface_rmse_norm))
    failed = R / 'pahma-5-613-v2' / 'random30_s20261110' / 'metrics-initial_alignment.json'
    cache = DATA / 'real_runs.csv'
    if failed.exists():
        m = json.loads(failed.read_text())['metrics']
        rows.append(dict(group='pahma-5-613 (GPU d8)', run='random30_s20261110', images=m['input_image_count'],
                         label_v3=np.nan, status=f"alignment failed ({m['aligned_camera_count']}/{m['camera_count']} aligned)",
                         reprojection_rmse_px=m['reprojection_rmse_px']))
        t = pd.DataFrame(rows)
        t['removal'] = t.run.map(removal_type)
        t.to_csv(cache, index=False)
    return pd.read_csv(cache)


def fig_label_vs_images():
    t = real_runs()
    groups = ['arf-dog (historical CPU)', 'arf-dog (GPU d8)', 'pahma-5-613 (GPU d8)']
    refs = {'arf-dog (historical CPU)': 'vs 50-image CPU reference', 'arf-dog (GPU d8)': 'vs 50-image GPU d8 reference',
            'pahma-5-613 (GPU d8)': 'vs 145-image GPU d8 reference'}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=True)
    for ax, gname in zip(axes, groups):
        g = t[t.group == gname]
        for rem in REMOVAL_COLOR:
            s = g[(g.removal == rem) & (g.status == 'scored')]
            if len(s):
                ax.scatter(s.images, s.label_v3, s=34, marker=REMOVAL_MARKER[rem], color=REMOVAL_COLOR[rem], label=rem)
        f = g[g.status != 'scored']
        for r in f.itertuples():
            ax.scatter([r.images], [0.06], marker='x', s=40, color=C['black'], clip_on=False)
            ax.annotate(f'{r.status}\n(no mesh, no label)', (r.images, 0.06), xytext=(6, -4),
                        textcoords='offset points', fontsize=7.5, va='top')
        ax.set_title(f'{gname}\n{refs[gname]}')
        ax.set_xlabel('images used')
        ax.set_yscale('log')
        log_ticks(ax.yaxis, [0.003, 0.01, 0.03])
        ax.legend(frameon=False, loc='upper right', fontsize=7.5)
    axes[0].set_ylabel('v3 label (symmetric RMSE / bbox diagonal)')
    axes[0].set_ylim(0.0025, 0.07)
    fig.tight_layout()
    save(fig, 'label_vs_images.png')


def fig_reprojection():
    t = pd.read_csv(R / 'pilot-real-v3-all' / 'runs.csv')
    t['group'] = np.where(t.artifact == 'pahma-5-613', 'pahma-5-613 (GPU d8)',
                          np.where(t.source == 'hist16-cpu', 'arf-dog (historical CPU)', 'arf-dog (GPU d8)'))
    STATS['reprojection_vs_label'] = {}
    fig, ax = plt.subplots(figsize=(7, 4))
    for gname, g in t.groupby('group', sort=False):
        rho = float(spearmanr(g.reproj, g.label).statistic)
        STATS['reprojection_vs_label'][gname] = {'n': len(g), 'spearman_rho': round(rho, 3)}
        ax.scatter(g.reproj, g.label, s=30, color=ARTIFACT_COLOR[gname], label=f'{gname}: n={len(g)}, ρ={rho:+.2f}')
    rho = float(spearmanr(t.reproj, t.label).statistic)
    STATS['reprojection_vs_label']['all'] = {'n': len(t), 'spearman_rho': round(rho, 3)}
    ax.set_yscale('log')
    log_ticks(ax.yaxis, [0.003, 0.01, 0.03])
    ax.set_xlabel('run reprojection RMSE after alignment (pixels)')
    ax.set_ylabel('v3 label (symmetric RMSE / bbox diagonal)')
    ax.set_title(f'Reprojection RMSE vs geometric error (pooled n={len(t)}, ρ={rho:+.2f}; within each\n'
                 'GPU artifact, worse geometry comes with lower reprojection RMSE)')
    ax.legend(frameon=False, loc='lower right')
    save(fig, 'reprojection_vs_label.png')


def fig_pilot():
    p = pd.read_csv(R / 'pilot-real-v3-all' / 'predictions.csv')
    p = p[p.scheme == 'LOAO'].groupby(['model', 'run', 'artifact'], as_index=False)[['actual', 'predicted']].mean()
    summ = pd.read_csv(R / 'pilot-real-v3-all' / 'summary.csv')
    models = ['mean', 'reproj', 'ridge', 'pointnet', 'attention']
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.3), sharex=True, sharey=True)
    lim = [0.002, 0.2]
    for ax, m in zip(axes, models):
        ax.plot(lim, lim, color=C['grey'], lw=0.8)
        for art, col, lab in (('arf-dog', C['blue'], 'ARF dog (trained on coffin)'),
                              ('pahma-5-613', C['orange'], 'coffin (trained on dog)')):
            g = p[(p.model == m) & (p.artifact == art)]
            ax.scatter(g.actual, g.predicted.clip(lower=lim[0]), s=18, color=col, label=lab)
        s = summ[(summ.scheme == 'LOAO') & (summ.model == m)].iloc[0]
        ax.set_title(f'{MODEL_NAME[m]}\nMAE {s.mae_mean:.4f}' + (f' ± {s.mae_std:.4f}' if s.n_seeds > 1 else ''))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        log_ticks(ax.xaxis, [0.003, 0.01, 0.03, 0.1])
        log_ticks(ax.yaxis, [0.003, 0.01, 0.03, 0.1])
        ax.set_xlabel('actual v3 label')
    axes[0].set_ylabel('predicted (mean over seeds)')
    axes[0].legend(frameon=False, fontsize=7, loc='upper left')
    fig.suptitle('PILOT (28 runs, 2 artifacts): leave-one-artifact-out predictions. Line: perfect prediction.', y=1.04)
    fig.tight_layout()
    save(fig, 'pilot_loao.png')


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    fig_layouts()
    fig_audit()
    fig_synthetic()
    fig_learning_curves()
    fig_label_vs_images()
    fig_reprojection()
    fig_pilot()
    (DATA / 'stats.json').write_text(json.dumps(STATS, indent=2, default=float) + '\n')
    print('wrote', (DATA / 'stats.json').relative_to(REPO))


if __name__ == '__main__':
    main()
