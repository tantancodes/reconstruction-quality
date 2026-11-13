"""PILOT (tiny n): predict the v3 geometric label of a real run from its aligned camera set.

Data: every real run with a v3 label and a cameras.csv
  - arf-dog historical 16 (results/geometry-labels-v3 + results/hist16-old-alignment)
  - arf-dog GPU d8 runs (results/gpu-pilot-d8-ext/labels.csv, else gpu-pilot-d8/geometry-v3)
  - pahma-5-613 coffin runs (results/pahma-5-613-v2/labels.csv), if any
Labels are reference-relative (each artifact/pipeline against its own reference), so pooled
values are only roughly comparable. With n of a few dozen this is a feasibility check of the
pipeline, not evidence that any model works.

Input per run: aligned cameras only; position and viewing direction. Per-run normalization:
subtract the camera-centre centroid, divide positions by the median camera distance from it.
Orientation is unknown (unreferenced frames) -> random rotation augmentation in training.

Models
  mean        training-fold mean label
  reproj      linear fit of label on run-level reprojection RMSE (px), the usual QC number
  ridge       RidgeCV on rotation-invariant global summaries (n, covariance eigenvalues,
              radial mean/std, mean |view . radial|)
  pointnet    per-camera MLP + max-pool, h=32
  attention   one 4-head self-attention layer before the max-pool, h=32
Evaluation: leave-one-run-out (LORO) and, if >= 2 artifacts, leave-one-artifact-out (LOAO).
5 seeds for the networks (init + augmentation); deterministic baselines have std 0.
MAE in label units, also reported as skill = 1 - MAE / MAE(mean) per seed.

  python3 scripts/pilot_set_models.py --output results/pilot-real-v3
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 400
H = 32
MODELS = ('mean', 'reproj', 'ridge', 'pointnet', 'attention')


def rel(path):
    """Path as written to outputs: repo-relative when inside the repo."""
    p = Path(path).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(path)


# ------------------------------------------------------------------ data

def read_cameras(path):
    rows = [r for r in csv.DictReader(open(path, newline=''))
            if r.get('aligned', 'True') == 'True' and r['x'] not in ('', None)]
    P = np.array([[float(r[k]) for k in 'xyz'] for r in rows])
    V = np.array([[float(r['view_' + k]) for k in 'xyz'] for r in rows])
    return P, V


def load_runs(coffin=REPO / 'results/pahma-5-613-v2/labels.csv'):
    runs = []
    hist = {r['experiment_id']: r for r in csv.DictReader(open(REPO / 'results/geometry-labels-v3/labels_v3.csv'))}
    for eid, r in sorted(hist.items()):
        cam = REPO / 'results/hist16-old-alignment' / eid / 'cameras.csv'
        rr = next(csv.DictReader(open(cam)))['run_reprojection_rmse_px']
        runs.append(dict(artifact='arf-dog', source='hist16-cpu', run=eid, label=float(r['label_v3']),
                         reproj=float(rr), cameras=cam))
    gpu = REPO / 'results/gpu-pilot-d8-ext/labels.csv'
    gpu_rows = list(csv.DictReader(open(gpu))) if gpu.exists() else \
        list(csv.DictReader(open(REPO / 'results/gpu-pilot-d8/geometry-v3/labels_v3.csv')))
    for r in gpu_rows:
        d = next(p / r['run'] for p in (REPO / 'results/gpu-pilot-d8', REPO / 'results/gpu-pilot-d8-ext')
                 if (p / r['run'] / 'cameras.csv').exists())
        rr = json.loads((d / 'metrics-initial_alignment.json').read_text())['metrics']['reprojection_rmse_px']
        runs.append(dict(artifact='arf-dog', source='gpu-d8', run=r['run'], label=float(r['label_v3']),
                         reproj=float(rr), cameras=d / 'cameras.csv'))
    if coffin.exists():
        for r in csv.DictReader(open(coffin)):
            d = REPO / 'results/pahma-5-613-v2' / r['run']
            runs.append(dict(artifact='pahma-5-613', source='gpu-d8', run=r['run'], label=float(r['label_v3']),
                             reproj=float(r['run_reprojection_rmse_px']), cameras=d / 'cameras.csv'))
    for r in runs:
        P, V = read_cameras(r['cameras'])
        c = P.mean(0)
        P = P - c
        P = P / np.median(np.linalg.norm(P, axis=1))
        r['X'] = np.c_[P, V].astype(np.float32)
        r['n'] = len(P)
    return runs


def summary(x):
    p, v = x[:, :3], x[:, 3:]
    d = np.linalg.norm(p, axis=1)
    radial = p / np.maximum(d[:, None], 1e-9)
    return [len(p), *np.linalg.eigvalsh(np.cov(p.T)), d.mean(), d.std(), np.abs((v * radial).sum(1)).mean()]


def pad(runs):
    nmax = max(r['n'] for r in runs)
    X = np.zeros((len(runs), nmax, 6), np.float32)
    M = np.zeros((len(runs), nmax), bool)
    for i, r in enumerate(runs):
        X[i, :r['n']] = r['X']
        M[i, :r['n']] = True
    return torch.tensor(X), torch.tensor(M)


# ------------------------------------------------------------------ models

class PointNetSet(nn.Module):
    def __init__(self, h=H):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(6, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x, mask):
        return self.rho(self.phi(x).masked_fill(~mask[..., None], -1e9).max(1).values).squeeze(-1)


class AttnSet(nn.Module):
    def __init__(self, h=H):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(6, h), nn.ReLU(), nn.Linear(h, h))
        self.attn = nn.MultiheadAttention(h, num_heads=4, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x, mask):
        f = self.embed(x)
        a, _ = self.attn(f, f, f, key_padding_mask=~mask)
        return self.rho(self.ff(f + a).masked_fill(~mask[..., None], -1e9).max(1).values).squeeze(-1)


def random_rotations(rng, k):
    out = []
    for _ in range(k):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q * np.where(np.diag(r) < 0, -1.0, 1.0)
        out.append(q if np.linalg.det(q) > 0 else -q)
    return torch.tensor(np.stack(out), dtype=torch.float32)


def rotate(X, R):
    return torch.cat([X[..., :3] @ R.transpose(1, 2), X[..., 3:] @ R.transpose(1, 2)], -1)


def train_net(name, X, M, y, seed, curve):
    """Full-batch Adam on the training fold; target scaled by the fold's mean label."""
    torch.manual_seed(seed)
    aug = np.random.default_rng(seed + 10_000)
    model = PointNetSet() if name == 'pointnet' else AttnSet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scale = float(y.mean())
    for epoch in range(EPOCHS):
        loss = nn.functional.mse_loss(model(rotate(X, random_rotations(aug, len(y))), M), y / scale)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if curve is not None:
            curve.append(loss.item())
    model.eval()
    return model, scale


@torch.no_grad()
def predict_net(model, scale, X, M, seed, k=8):
    """Average over k random rotations (the frame is arbitrary at test time too)."""
    R = np.random.default_rng(seed + 20_000)
    return float(np.mean([(model(rotate(X, random_rotations(R, len(X))), M) * scale).item() for _ in range(k)]))


# ------------------------------------------------------------------ evaluation

def evaluate(runs, scheme):
    groups = [r['run'] for r in runs] if scheme == 'LORO' else [r['artifact'] for r in runs]
    X, M = pad(runs)
    y = np.array([r['label'] for r in runs], np.float32)
    S = np.array([summary(r['X']) for r in runs])
    rp = np.array([[r['reproj']] for r in runs])
    preds, curves = [], {}
    for g in dict.fromkeys(groups):
        te = np.array([x == g for x in groups])
        tr = ~te
        for i in np.where(te)[0]:
            base = dict(scheme=scheme, held_out=g, run=runs[i]['run'], artifact=runs[i]['artifact'],
                        source=runs[i]['source'], images=runs[i]['n'], actual=float(y[i]))
            preds.append({**base, 'model': 'mean', 'seed': -1, 'predicted': float(y[tr].mean())})
            lin = LinearRegression().fit(rp[tr], y[tr])
            preds.append({**base, 'model': 'reproj', 'seed': -1, 'predicted': float(lin.predict(rp[i:i + 1])[0])})
            ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(S[tr], y[tr])
            preds.append({**base, 'model': 'ridge', 'seed': -1, 'predicted': float(ridge.predict(S[i:i + 1])[0])})
        Xtr, Mtr, ytr = X[tr], M[tr], torch.tensor(y[tr])
        for name in ('pointnet', 'attention'):
            for seed in SEEDS:
                curve = curves.setdefault((name, seed), []) if not curves.get((name, seed)) else None
                model, scale = train_net(name, Xtr, Mtr, ytr, seed, curve)
                for i in np.where(te)[0]:
                    base = dict(scheme=scheme, held_out=g, run=runs[i]['run'], artifact=runs[i]['artifact'],
                                source=runs[i]['source'], images=runs[i]['n'], actual=float(y[i]))
                    p = predict_net(model, scale, X[i:i + 1], M[i:i + 1], seed)
                    preds.append({**base, 'model': name, 'seed': seed, 'predicted': p})
        print(f'  {scheme} held out {g}: done', flush=True)
    return preds, curves


def aggregate(preds):
    out = []
    for scheme in dict.fromkeys(p['scheme'] for p in preds):
        P = [p for p in preds if p['scheme'] == scheme]
        mean_mae = np.mean([abs(p['predicted'] - p['actual']) for p in P if p['model'] == 'mean'])
        for m in MODELS:
            seeds = sorted({p['seed'] for p in P if p['model'] == m})
            maes = np.array([np.mean([abs(p['predicted'] - p['actual']) for p in P
                                      if p['model'] == m and p['seed'] == s]) for s in seeds])
            skill = 1 - maes / mean_mae
            out.append(dict(scheme=scheme, model=m, n_runs=len({p['run'] for p in P}), n_seeds=len(seeds),
                            mae_mean=maes.mean(), mae_std=maes.std(ddof=1) if len(maes) > 1 else 0.0,
                            skill_vs_mean=skill.mean(), skill_std=skill.std(ddof=1) if len(maes) > 1 else 0.0))
    return out


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--coffin-labels', type=Path, default=REPO / 'results/pahma-5-613-v2/labels.csv')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    runs = load_runs(a.coffin_labels)
    artifacts = sorted({r['artifact'] for r in runs})
    print(f'{len(runs)} runs: ' + ', '.join(f'{x} {sum(r["artifact"] == x for r in runs)}' for x in artifacts), flush=True)
    a.output.mkdir(parents=True, exist_ok=False)
    write_csv(a.output / 'runs.csv', [{k: rel(v) if k == 'cameras' else v for k, v in r.items() if k != 'X'} for r in runs])
    preds, curves = evaluate(runs, 'LORO')
    if len(artifacts) >= 2:
        p2, _ = evaluate(runs, 'LOAO')
        preds += p2
    write_csv(a.output / 'predictions.csv', preds)
    summ = aggregate(preds)
    write_csv(a.output / 'summary.csv', summ)
    write_csv(a.output / 'learning_curves.csv',
              [dict(model=m, seed=s, epoch=e, train_mse_scaled=v) for (m, s), c in curves.items() for e, v in enumerate(c)])
    (a.output / 'config.json').write_text(json.dumps({
        'status': 'PILOT - tiny n, not evidence of model skill', 'n_runs': len(runs), 'artifacts': artifacts,
        'seeds': SEEDS, 'epochs': EPOCHS, 'hidden': H, 'optimizer': 'Adam lr 1e-3 wd 1e-4, full batch',
        'input': 'aligned cameras: position (centroid-centred, / median distance) + view direction',
        'augmentation': 'fresh random rotation per run per epoch; test = mean over 8 random rotations',
        'learning_curves': 'training loss of the first LORO fold per (model, seed)',
        'label': 'v3 symmetric normalized RMSE vs each artifact/pipeline reference',
        'coffin_labels': rel(a.coffin_labels), 'seconds': round(time.time() - t0, 1)}, indent=2))
    for r in summ:
        print(f"{r['scheme']:5s} {r['model']:9s} MAE {r['mae_mean']:.5f} ± {r['mae_std']:.5f}  "
              f"skill {r['skill_vs_mean']:+.2f} ± {r['skill_std']:.2f}", flush=True)


if __name__ == '__main__':
    main()
