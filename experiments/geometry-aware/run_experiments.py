"""Geometry-aware attention vs. the existing models, on the synthetic benchmark and the 28 real builds.

Reuses, unchanged: the synthetic generator (scripts/synthetic_coverage_gap.py) and the real-data
loader (scripts/pilot_set_models.py). Writes only under the --out folder.

  python experiments/geometry-aware/run_experiments.py synthetic --out experiments/geometry-aware/results
  python experiments/geometry-aware/run_experiments.py real      --out experiments/geometry-aware/results
  add --pilot for a fast timing run (1 seed, few epochs)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / 'scripts')]
import geo_models as gm                                   # noqa: E402
import synthetic_coverage_gap as syn                      # noqa: E402
import pilot_set_models as pilot                          # noqa: E402

NETS = ('pointnet', 'attention', 'geo', 'geo_inv', 'geo_shuf')


# ------------------------------------------------------------------ hand-built features

def pair_features(P, V=None):
    """Rotation-invariant features built from the same pairwise geometry the geo models get:
    histogram of angular separations between cameras (seen from the centroid) and statistics of each
    camera's nearest-neighbour angle. Lets a tree model use pairwise geometry without attention."""
    r = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
    ang = np.degrees(np.arccos(np.clip(r @ r.T, -1, 1)))
    iu = np.triu_indices(len(P), 1)
    hist = np.histogram(ang[iu], bins=12, range=(0, 180))[0] / max(len(iu[0]), 1)
    np.fill_diagonal(ang, np.inf)
    nn_ang = ang.min(1)
    out = [*hist, nn_ang.mean(), nn_ang.std(), nn_ang.max(), np.percentile(nn_ang, 90)]
    if V is not None:
        out.append(float(np.abs((V * r).sum(1)).mean()))
    return out


def fit_tabular(name, Str, ytr, Sva=None, yva=None, seed=0):
    """Ridge / MLP / XGBoost on summary features. With a validation set, pick hyperparameters on it;
    without one (real data, n=28) use one fixed setting chosen before seeing results."""
    if name == 'ridge':
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(Str, ytr)
    if name == 'mlp':
        grid = [1e-4, 1e-2, 1.0] if Sva is not None else [1e-2]
        # the target is standardised too: real labels are ~0.005, which an unscaled MLP cannot fit
        mk = lambda a: TransformedTargetRegressor(
            make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), alpha=a, max_iter=2000,
                                                         random_state=seed)), transformer=StandardScaler())
    elif name.startswith('xgb'):
        grid = [(3, 200), (3, 600), (5, 300)] if Sva is not None else [(2, 200)]
        mk = lambda g: XGBRegressor(max_depth=g[0], n_estimators=g[1], learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.8, random_state=seed, n_jobs=2)
    else:
        raise ValueError(name)
    if Sva is None:
        return mk(grid[0]).fit(Str, ytr)
    best = min(grid, key=lambda g: np.abs(mk(g).fit(Str, ytr).predict(Sva) - yva).mean())
    return mk(best).fit(np.r_[Str], ytr)


# ------------------------------------------------------------------ network training

def net_inputs(name, X, has_view):
    """Token features each net sees. Frame-dependent nets get raw coordinates; geo_inv gets invariants."""
    P = X[..., :3]
    V = X[..., 3:6] if has_view else None
    return gm.invariant_tokens(P, V) if name == 'geo_inv' else X


def rotate(X, R, has_view):
    out = X[..., :3] @ R.transpose(1, 2)
    return torch.cat([out, X[..., 3:6] @ R.transpose(1, 2)], -1) if has_view else out


def train_net(name, Xtr, Mtr, Rtr, ytr, seed, *, epochs, h, batch, lr, wd, has_view, Xva=None, Mva=None, Rva=None,
              yva=None, eval_every=10, log=None, cosine=False):
    """Adam on MSE of the scaled target. Fresh random rotation of the coordinates per example per step
    (the relations are rotation-invariant, so they are computed once and reused)."""
    torch.manual_seed(seed)
    aug = np.random.default_rng(seed + 10_000)
    gen = torch.Generator().manual_seed(seed)
    model = gm.make_model(name, Xtr.shape[-1], h, Rtr.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs) if cosine else None
    rel_dim_used = name in ('geo', 'geo_inv', 'geo_shuf')
    for epoch in range(epochs):
        perm = torch.randperm(len(ytr), generator=gen)
        tot = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            x = Xtr[b]
            if name != 'geo_inv':
                x = rotate(x, torch.tensor(np.stack([syn.random_rotation(aug) for _ in b]), dtype=torch.float32),
                           has_view)
            x = net_inputs(name, x, has_view)
            rel = Rtr[b] if rel_dim_used else None
            if name == 'geo_shuf':
                rel = gm.shuffle_pairs(rel, Mtr[b], gen)
            loss = nn.functional.mse_loss(model(x, Mtr[b], rel), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if sched is not None:
            sched.step()
        if log is not None and (epoch % eval_every == 0 or epoch == epochs - 1):
            row = dict(epoch=epoch, train_mse=tot / len(ytr))
            if Xva is not None:
                row['val_mae'] = float((predict(model, name, Xva, Mva, Rva, has_view, seed) - yva).abs().mean())
            log.append(row)
    model.eval()
    return model


@torch.no_grad()
def predict(model, name, X, M, Rel, has_view, seed, rotations=1, chunk=256):
    """Prediction; rotations > 1 averages over random frames (used on the real data, as the pilot does)."""
    model.eval()
    rng = np.random.default_rng(seed + 20_000)
    outs = []
    for _ in range(rotations):
        preds = []
        for i in range(0, len(X), chunk):
            x = X[i:i + chunk]
            if rotations > 1 and name != 'geo_inv':
                x = rotate(x, torch.tensor(np.stack([syn.random_rotation(rng) for _ in range(len(x))]),
                                           dtype=torch.float32), has_view)
            rel = Rel[i:i + chunk] if name in ('geo', 'geo_inv', 'geo_shuf') else None
            if name == 'geo_shuf':
                rel = gm.shuffle_pairs(rel, M[i:i + chunk], torch.Generator().manual_seed(seed))
            preds.append(model(net_inputs(name, x, has_view), M[i:i + chunk], rel))
        outs.append(torch.cat(preds))
    model.train()
    return torch.stack(outs).mean(0)


# ------------------------------------------------------------------ synthetic

def make_set_n(rng, n_lo, n_hi):
    """syn.make_set('global') with the camera count drawn from [n_lo, n_hi]."""
    hole, start = rng.uniform(0, 150), rng.uniform(0, 360)
    p, az = syn.ring(rng, int(rng.integers(n_lo, n_hi + 1)), 10, 50, hole, start)
    target = syn.largest_gap(az)
    p = (p @ syn.random_rotation(rng).T) * rng.uniform(0.2, 5) + rng.normal(0, 10, 3)
    p = p - p.mean(0)
    p = p / np.median(np.linalg.norm(p, axis=1))
    rng.shuffle(p)
    return p, target


def build_n(rng, m, n_lo, n_hi):
    X = np.zeros((m, syn.NMAX, 3), np.float32)
    M = np.zeros((m, syn.NMAX), bool)
    y = np.zeros(m, np.float32)
    for i in range(m):
        p, y[i] = make_set_n(rng, n_lo, n_hi)
        X[i, :len(p)], M[i, :len(p)] = p, True
    return torch.tensor(X), torch.tensor(M), torch.tensor(y)


def synthetic_data(condition, seed, n_train, n_val, n_test):
    if condition in syn.CONDITIONS:            # identical draws to scripts/synthetic_coverage_gap.py
        rng = np.random.default_rng([seed, syn.CONDITIONS.index(condition)])
        return [syn.build(rng, m, condition) for m in (n_train, n_val, n_test)]
    if condition == 'count_shift':             # train/val on 20-35 cameras, test on 45-60
        rng = np.random.default_rng([seed, 7])
        return [build_n(rng, n_train, 20, 35), build_n(rng, n_val, 20, 35), build_n(rng, n_test, 45, 60)]
    raise ValueError(condition)


def run_synthetic(a):
    out = a.out / a.tag
    out.mkdir(parents=True, exist_ok=True)
    rows, curves = [], []
    for condition in a.conditions:
        for seed in a.seeds:
            (Xtr, Mtr, ytr), (Xva, Mva, yva), (Xte, Mte, yte) = synthetic_data(condition, seed, a.n_train, a.n_val, a.n_test)
            R = {k: gm.relations(X, None, M) for k, (X, M) in dict(tr=(Xtr, Mtr), va=(Xva, Mva), te=(Xte, Mte)).items()}
            S = {k: (syn.summary(X, M), np.array([pair_features(x[m].numpy()) for x, m in zip(X, M)]))
                 for k, (X, M) in dict(tr=(Xtr, Mtr), va=(Xva, Mva), te=(Xte, Mte)).items()}

            def record(model, pred, secs, params):
                err = np.asarray(pred) - yte.numpy()
                rows.append(dict(condition=condition, seed=seed, model=model, mae=float(np.abs(err).mean()),
                                 rmse=float(np.sqrt((err ** 2).mean())), seconds=round(secs, 1), params=params))
                print(f'{condition:11s} s{seed} {model:9s} MAE {np.abs(err).mean():6.2f}  ({secs:.0f}s)', flush=True)

            record('mean', np.full(len(yte), float(ytr.mean())), 0, 1)
            for name in ('ridge', 'mlp', 'xgb', 'xgb_pair'):
                t = time.time()
                sel = (lambda k: S[k][0]) if name != 'xgb_pair' else (lambda k: np.c_[S[k][0], S[k][1]])
                model = fit_tabular(name, sel('tr'), ytr.numpy(), sel('va'), yva.numpy(), seed)
                record(name, model.predict(sel('te')), time.time() - t, None)
            for name in a.nets:
                t = time.time()
                log = []
                model = train_net(name, Xtr, Mtr, R['tr'], ytr / syn.SCALE, seed, epochs=a.epochs, h=64, batch=128,
                                  lr=a.lr, wd=0.0, cosine=a.cosine, has_view=False, Xva=Xva, Mva=Mva, Rva=R['va'], yva=yva / syn.SCALE,
                                  log=log)
                pred = predict(model, name, Xte, Mte, R['te'], False, seed) * syn.SCALE
                record(name, pred.numpy(), time.time() - t, gm.n_params(model))
                curves += [dict(condition=condition, seed=seed, model=name, **r, val_mae_deg=r['val_mae'] * syn.SCALE)
                           for r in log]
            write(out / 'results.csv', rows)
            write(out / 'curves.csv', curves)
    write(out / 'summary.csv', summarize(rows, ('condition',)))
    (out / 'config.json').write_text(json.dumps({**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                                                  'torch': torch.__version__}, indent=2))


# ------------------------------------------------------------------ real

def run_real(a):
    out = a.out / 'real'
    out.mkdir(parents=True, exist_ok=True)
    runs = pilot.load_runs(REPO / 'results/pahma-5-613-v2/labels-all.csv')   # the 28 builds of results/pilot-real-v3-all
    for r in runs:
        r['group'] = r['artifact'] + '/' + r['source']
    X, M = pilot.pad(runs)
    Rel = gm.relations(X[..., :3], X[..., 3:6], M)
    y = np.array([r['label'] for r in runs], np.float32)
    S = np.array([pilot.summary(r['X']) for r in runs])
    SP = np.array([pilot.summary(r['X']) + pair_features(r['X'][:, :3], r['X'][:, 3:]) for r in runs])
    rp = np.array([[r['reproj']] for r in runs])
    rows = []
    for scheme in a.schemes:
        key = dict(LORO='run', LOAO='artifact', LOGO='group')[scheme]
        groups = [r[key] for r in runs]
        for g in dict.fromkeys(groups):
            te = np.array([x == g for x in groups])
            tr = ~te
            base = lambda i: dict(scheme=scheme, held_out=g, run=runs[i]['run'], group=runs[i]['group'], actual=float(y[i]))
            for i in np.where(te)[0]:
                rows.append({**base(i), 'model': 'mean', 'seed': -1, 'predicted': float(y[tr].mean())})
                rows.append({**base(i), 'model': 'reproj', 'seed': -1,
                             'predicted': float(LinearRegression().fit(rp[tr], y[tr]).predict(rp[i:i + 1])[0])})
            for name, feats in (('ridge', S), ('mlp', S), ('xgb', S), ('xgb_pair', SP)):
                for seed in (a.seeds if name in ('mlp', 'xgb', 'xgb_pair') else [-1]):
                    m = fit_tabular(name, feats[tr], y[tr], seed=max(seed, 0))
                    p = m.predict(feats[te])
                    rows += [{**base(i), 'model': name, 'seed': seed, 'predicted': float(v)} for i, v in zip(np.where(te)[0], p)]
            ytr = torch.tensor(y[tr])
            scale = float(ytr.mean())
            for name in a.nets:
                for seed in a.seeds:
                    model = train_net(name, X[tr], M[tr], Rel[tr], ytr / scale, seed, epochs=a.epochs, h=32,
                                      batch=len(ytr), lr=1e-3, wd=1e-4, has_view=True)
                    p = predict(model, name, X[te], M[te], Rel[te], True, seed, rotations=8) * scale
                    rows += [{**base(i), 'model': name, 'seed': seed, 'predicted': float(v)}
                             for i, v in zip(np.where(te)[0], p.numpy())]
            print(f'{scheme} held out {g}: done', flush=True)
            write(out / 'predictions.csv', rows)
    summ = real_summary(rows, a.schemes)
    write(out / 'summary.csv', summ)
    (out / 'config.json').write_text(json.dumps({**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                                                  'n_runs': len(runs), 'groups': sorted({r['group'] for r in runs}),
                                                  'torch': torch.__version__}, indent=2))


# ------------------------------------------------------------------ util

def real_summary(rows, schemes):
    """MAE / RMSE per (scheme, model): mean and std over seeds of the fold-pooled errors."""
    summ = []
    for scheme in schemes:
        P = [r for r in rows if r['scheme'] == scheme]
        for model in dict.fromkeys(r['model'] for r in P):
            per_seed = {}
            for r in P:
                if r['model'] == model:
                    per_seed.setdefault(r['seed'], []).append(float(r['predicted']) - float(r['actual']))
            mae = np.array([np.abs(e).mean() for e in per_seed.values()])
            rmse = np.array([np.sqrt(np.square(e).mean()) for e in per_seed.values()])
            summ.append(dict(scheme=scheme, model=model, n_seeds=len(mae), mae_mean=mae.mean(),
                             mae_std=mae.std(ddof=1) if len(mae) > 1 else 0.0, rmse_mean=rmse.mean(),
                             rmse_std=rmse.std(ddof=1) if len(rmse) > 1 else 0.0))
    return summ


def summarize(rows, keys):
    out = []
    for k in dict.fromkeys(tuple(r[x] for x in keys) for r in rows):
        for model in dict.fromkeys(r['model'] for r in rows):
            R = [r for r in rows if tuple(r[x] for x in keys) == k and r['model'] == model]
            if not R:
                continue
            mae, rmse = np.array([r['mae'] for r in R]), np.array([r['rmse'] for r in R])
            out.append({**dict(zip(keys, k)), 'model': model, 'n_seeds': len(R), 'mae_mean': mae.mean(),
                        'mae_std': mae.std(ddof=1) if len(R) > 1 else 0.0, 'rmse_mean': rmse.mean(),
                        'rmse_std': rmse.std(ddof=1) if len(R) > 1 else 0.0, 'params': R[0]['params']})
    return out


def write(path, rows):
    if rows:
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('synthetic', 'real'))
    ap.add_argument('--out', type=Path, default=HERE / 'results')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--nets', nargs='+', default=list(NETS))
    ap.add_argument('--conditions', nargs='+', default=['global', 'localized', 'count_shift'])
    ap.add_argument('--schemes', nargs='+', default=['LOGO', 'LOAO'])
    ap.add_argument('--n-train', type=int, default=3000)
    ap.add_argument('--n-val', type=int, default=600)
    ap.add_argument('--n-test', type=int, default=600)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--tag', default='synthetic')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--cosine', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.epochs is None:
        a.epochs = 400
    t = time.time()
    (run_synthetic if a.mode == 'synthetic' else run_real)(a)
    print(f'done in {time.time() - t:.0f}s')


if __name__ == '__main__':
    main()
