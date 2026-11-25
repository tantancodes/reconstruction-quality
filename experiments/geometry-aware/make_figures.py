"""Figures for experiments/geometry-aware/results (run after run_all.sh)."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

R = Path(__file__).resolve().parent / 'results'
ORDER = ['mean', 'reproj', 'ridge', 'mlp', 'xgb', 'xgb_pair', 'pointnet', 'attention', 'geo_shuf', 'geo', 'geo_inv']
COLOR = {'geo': '#c0392b', 'geo_inv': '#e67e22', 'geo_shuf': '#95a5a6', 'attention': '#2c3e50', 'pointnet': '#7f8c8d'}


def bars(ax, s, title, unit):
    s = s.set_index('model').reindex([m for m in ORDER if m in set(s['model'])]).reset_index()
    ax.barh(s['model'], s['mae_mean'], xerr=s['mae_std'], color=[COLOR.get(m, '#bdc3c7') for m in s['model']])
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f'test MAE ({unit}), mean ± std over seeds')


def main():
    panels = []
    for tag, cond, title in (('synthetic-localized', 'localized', 'Synthetic, localized gap'),
                             ('synthetic-count-shift', 'count_shift', 'Synthetic, train 20-35 / test 45-60 cameras')):
        f = R / tag / 'summary.csv'
        if f.exists():
            panels.append((pd.read_csv(f).query('condition == @cond'), title, 'degrees'))
    f = R / 'real' / 'summary.csv'
    if f.exists():
        s = pd.read_csv(f)
        for scheme, title in (('LOGO', 'Real, leave one capture group out (3 folds)'),
                              ('LOAO', 'Real, leave one object out (2 folds)')):
            panels.append((s[s.scheme == scheme], title, 'label units'))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.8))
    for ax, (s, title, unit) in zip(axes, panels):
        bars(ax, s, title, unit)
    fig.tight_layout()
    fig.savefig(R / 'comparison.png', dpi=160)

    f = R / 'synthetic-localized' / 'curves.csv'
    if f.exists():
        c = pd.read_csv(f)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for m, g in c.groupby('model'):
            g = g.groupby('epoch')['val_mae_deg'].agg(['mean', 'std']).reset_index()
            ax.plot(g['epoch'], g['mean'], label=m, color=COLOR.get(m))
            ax.fill_between(g['epoch'], g['mean'] - g['std'], g['mean'] + g['std'], color=COLOR.get(m), alpha=0.15)
        f2 = R / 'synthetic-localized-long' / 'curves.csv'
        if f2.exists():
            for m, g in pd.read_csv(f2).groupby('model'):
                ax.plot(g['epoch'], g['val_mae_deg'], ls='--', color=COLOR.get(m), label=f'{m} (400 ep, n=3000)')
        ax.set_xlabel('epoch')
        ax.set_ylabel('validation MAE (degrees)')
        ax.set_title('Synthetic localized: validation curves', fontsize=10)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(R / 'curves_localized.png', dpi=160)


if __name__ == '__main__':
    main()
