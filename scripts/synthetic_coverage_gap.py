"""Synthetic coverage-gap benchmark: can a set model read camera layout?

Each example is a set of camera centres around an object, expressed in an
arbitrary similarity frame (random rotation, scale, offset), like an
unreferenced Metashape alignment. The target is the largest azimuthal gap in
coverage (degrees). Four predictors are compared:

  mean       predict the training-set mean (floor)
  ridge      RidgeCV on rotation-invariant global summaries
             (n, covariance eigenvalues, radial distance mean/std)
  pointnet   per-camera MLP + max-pool (cameras never see each other)
  attention  one self-attention layer before the max-pool (cameras compare)

Conditions:
  global     one elevation band (10-50 deg); gap cut from all cameras.
             This is the original notebook experiment.
  localized  two elevation rings; full 360 deg lower ring, gap cut from the
             upper ring only; target = largest gap in the upper ring.

Synthetic data only. Nothing here says anything about Metashape quality yet.

Usage:
  python3 scripts/synthetic_coverage_gap.py                    # full run
  python3 scripts/synthetic_coverage_gap.py --epochs 5 --seeds 0 --out /tmp/x  # smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter

NMAX = 60
SCALE = 100.0  # networks predict degrees / 100 so targets are ~1
MODELS = ("mean", "ridge", "pointnet", "attention")
CONDITIONS = ("global", "localized")


# ---------------------------------------------------------------- data

def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.where(np.diag(r) < 0, -1.0, 1.0)  # make QR unique -> Haar-uniform rotation
    return q if np.linalg.det(q) > 0 else -q


def largest_gap(az_deg: np.ndarray) -> float:
    s = np.sort(az_deg % 360)
    return float(np.diff(np.r_[s, s[0] + 360]).max())


def ring(rng, n, el_lo, el_hi, hole, start):
    """n cameras with azimuth avoiding [start, start+hole), elevation in band."""
    az = (start + hole + rng.uniform(0, 360 - hole, n)) % 360
    el = np.radians(rng.uniform(el_lo, el_hi, n))
    r = 5 + rng.normal(0, 0.3, n)
    a = np.radians(az)
    p = np.c_[r * np.cos(el) * np.cos(a), r * np.sin(el), r * np.cos(el) * np.sin(a)]
    return p, az


def make_set(rng, condition):
    hole = rng.uniform(0, 150)
    start = rng.uniform(0, 360)
    if condition == "global":
        n = int(rng.integers(20, NMAX + 1))
        p, az = ring(rng, n, 10, 50, hole, start)
        target = largest_gap(az)
    elif condition == "localized":
        n_lo = int(rng.integers(10, NMAX // 2 + 1))
        n_up = int(rng.integers(10, NMAX // 2 + 1))
        p_lo, _ = ring(rng, n_lo, 0, 15, 0.0, 0.0)          # full lower ring
        p_up, az_up = ring(rng, n_up, 40, 55, hole, start)  # gap only up here
        p = np.r_[p_lo, p_up]
        target = largest_gap(az_up)
    else:
        raise ValueError(condition)
    # Arbitrary frame (as in an unreferenced alignment), then normalise what
    # we can recover without a reference: centroid and scale. Orientation is
    # left unknown on purpose.
    p = (p @ random_rotation(rng).T) * rng.uniform(0.2, 5) + rng.normal(0, 10, 3)
    p = p - p.mean(0)
    p = p / np.median(np.linalg.norm(p, axis=1))
    rng.shuffle(p)  # order carries no information
    return p, target


def build(rng, m, condition):
    X = np.zeros((m, NMAX, 3), np.float32)
    M = np.zeros((m, NMAX), bool)
    y = np.zeros(m, np.float32)
    for i in range(m):
        p, y[i] = make_set(rng, condition)
        X[i, : len(p)] = p
        M[i, : len(p)] = True
    return torch.tensor(X), torch.tensor(M), torch.tensor(y)


def summary(X, M):
    rows = []
    for x, m in zip(X.numpy(), M.numpy()):
        p = x[m]
        d = np.linalg.norm(p, axis=1)
        rows.append([len(p), *np.linalg.eigvalsh(np.cov(p.T)), d.mean(), d.std()])
    return np.array(rows)


# ---------------------------------------------------------------- models

class PointNetSet(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(3, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x, mask):
        f = self.phi(x).masked_fill(~mask[..., None], -1e9)
        return self.rho(f.max(1).values).squeeze(-1)


class AttnSet(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(3, h), nn.ReLU(), nn.Linear(h, h))
        self.attn = nn.MultiheadAttention(h, num_heads=4, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x, mask):
        f = self.embed(x)
        a, _ = self.attn(f, f, f, key_padding_mask=~mask)
        f = self.ff(f + a).masked_fill(~mask[..., None], -1e9)
        return self.rho(f.max(1).values).squeeze(-1)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def mae_deg(model, X, M, y):
    model.eval()
    out = (model(X, M) * SCALE - y).abs().mean().item()
    model.train()
    return out


def train_net(name, data, seed, epochs, writer, eval_every):
    (Xtr, Mtr, ytr), (Xva, Mva, yva) = data["train"], data["val"]
    torch.manual_seed(seed)
    model = PointNetSet() if name == "pointnet" else AttnSet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    aug = np.random.default_rng(seed + 10_000)  # augmentation stream, separate from data
    gen = torch.Generator().manual_seed(seed)
    step = 0
    for epoch in range(epochs):
        perm = torch.randperm(len(ytr), generator=gen)
        tot, cnt = 0.0, 0
        for i in range(0, len(perm), 128):
            b = perm[i : i + 128]
            R = torch.tensor(np.stack([random_rotation(aug) for _ in b]), dtype=torch.float32)
            xb = Xtr[b] @ R.transpose(1, 2)  # fresh rotation per example per step
            loss = nn.functional.mse_loss(model(xb, Mtr[b]), ytr[b] / SCALE)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            cnt += len(b)
            step += 1
        writer.add_scalar("loss/train_mse_scaled", tot / cnt, epoch)
        if epoch % eval_every == 0 or epoch == epochs - 1:
            writer.add_scalar("mae_deg/val", mae_deg(model, Xva, Mva, yva), epoch)
    return model


# ---------------------------------------------------------------- driver

def run(condition, seed, args, runs_root):
    rng = np.random.default_rng([seed, CONDITIONS.index(condition)])
    data = {
        "train": build(rng, args.n_train, condition),
        "val": build(rng, args.n_val, condition),
        "test": build(rng, args.n_test, condition),
    }
    Xtr, Mtr, ytr = data["train"]
    Xte, Mte, yte = data["test"]
    rows = []

    def record(model_name, mae, secs, params):
        rows.append(dict(condition=condition, seed=seed, model=model_name,
                         test_mae_deg=round(mae, 4), train_seconds=round(secs, 1),
                         n_params=params, target_mean_deg=round(float(yte.mean()), 3),
                         target_std_deg=round(float(yte.std()), 3)))
        print(f"  {condition:9s} seed={seed} {model_name:9s} MAE {mae:6.2f} deg  ({secs:.0f}s)", flush=True)

    def writer_for(model_name):
        return SummaryWriter(str(runs_root / condition / model_name / str(seed)))

    t = time.time()
    mean_mae = (ytr.mean() - yte).abs().mean().item()
    record("mean", mean_mae, time.time() - t, 1)

    t = time.time()
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    ridge.fit(summary(Xtr, Mtr), ytr.numpy())
    ridge_mae = float(np.abs(ridge.predict(summary(Xte, Mte)) - yte.numpy()).mean())
    record("ridge", ridge_mae, time.time() - t, 7)

    # Flat reference lines so mean/ridge show up next to the curves in TensorBoard.
    for name, mae in (("mean", mean_mae), ("ridge", ridge_mae)):
        w = writer_for(name)
        for e in (0, args.epochs - 1):
            w.add_scalar("mae_deg/test", mae, e)
        w.close()

    for name in ("pointnet", "attention"):
        t = time.time()
        w = writer_for(name)
        model = train_net(name, data, seed, args.epochs, w, args.eval_every)
        mae = mae_deg(model, Xte, Mte, yte)
        w.add_scalar("mae_deg/test", mae, args.epochs - 1)
        w.close()
        record(name, mae, time.time() - t, n_params(model))
    return rows


def aggregate(rows):
    out = []
    for c in CONDITIONS:
        base = {r["seed"]: r["test_mae_deg"] for r in rows if r["condition"] == c and r["model"] == "mean"}
        for m in MODELS:
            sel = [r for r in rows if r["condition"] == c and r["model"] == m]
            if not sel:
                continue
            v = np.array([r["test_mae_deg"] for r in sel])
            skill = np.array([1 - r["test_mae_deg"] / base[r["seed"]] for r in sel])
            out.append(dict(condition=c, model=m, n_seeds=len(v),
                            mae_mean=round(v.mean(), 3), mae_std=round(v.std(ddof=1) if len(v) > 1 else 0.0, 3),
                            skill_vs_mean=round(skill.mean(), 3),
                            skill_std=round(skill.std(ddof=1) if len(v) > 1 else 0.0, 3)))
    return out


def plot(summary_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"mean": "#9e9e9e", "ridge": "#4c72b0", "pointnet": "#dd8452", "attention": "#55a868"}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.2
    for j, m in enumerate(MODELS):
        xs, ms, ss = [], [], []
        for i, c in enumerate(CONDITIONS):
            r = next((r for r in summary_rows if r["condition"] == c and r["model"] == m), None)
            if r:
                xs.append(i + (j - 1.5) * width); ms.append(r["mae_mean"]); ss.append(r["mae_std"])
        bars = ax.bar(xs, ms, width, yerr=ss, capsize=3, label=m, color=colors[m])
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in ms], padding=2, fontsize=8)
    ax.set_xticks(range(len(CONDITIONS)), ["global gap\n(one band)", "localized gap\n(upper ring only)"])
    ax.set_ylabel("test MAE, largest gap (degrees)")
    n = summary_rows[0]["n_seeds"]
    ax.set_title(f"Synthetic coverage-gap prediction (mean ± std over {n} seeds)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=4, loc="upper left")
    ax.set_ylim(0, max(r["mae_mean"] + r["mae_std"] for r in summary_rows) * 1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=repo / "results" / "synthetic-coverage-gap")
    ap.add_argument("--runs", type=Path, default=repo / "runs" / "synthetic")
    ap.add_argument("--resume", action="store_true",
                    help="keep finished (condition, seed) pairs from results.csv; rerun the rest")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.resume and (args.out / "results.csv").exists():
        old = list(csv.DictReader((args.out / "results.csv").open(newline="")))
        for r in old:
            r["seed"] = int(r["seed"]); r["test_mae_deg"] = float(r["test_mae_deg"])
        done = {(c, s) for c in CONDITIONS for s in args.seeds
                if {r["model"] for r in old if r["condition"] == c and r["seed"] == s} == set(MODELS)}
        rows = [r for r in old if (r["condition"], r["seed"]) in done]
        print(f"resuming: keeping {sorted(done)}")
    else:
        done = set()
    t0 = time.time()
    for c in args.conditions:
        for s in args.seeds:
            if (c, s) in done:
                continue
            for m in MODELS:  # drop partial TensorBoard logs from an interrupted seed
                d = args.runs / c / m / str(s)
                if d.exists():
                    for f in d.glob("events.out.tfevents.*"):
                        f.unlink()
            rows += run(c, s, args, args.runs)
            # write as we go so an interrupted run keeps finished seeds
            with open(args.out / "results.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)

    summ = aggregate(rows)
    with open(args.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader(); w.writerows(summ)
    plot(summ, args.out / "comparison.png")
    (args.out / "config.json").write_text(json.dumps(dict(
        vars(args) | {"out": str(args.out), "runs": str(args.runs)},
        nmax=NMAX, torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__,
        python=platform.python_version(), wall_seconds=round(time.time() - t0, 1)), indent=2))

    print("\ncondition  model      MAE (deg)        skill vs mean")
    for r in summ:
        print(f"{r['condition']:10s} {r['model']:9s}  {r['mae_mean']:6.2f} ± {r['mae_std']:5.2f}   "
              f"{r['skill_vs_mean']:.3f} ± {r['skill_std']:.3f}")


if __name__ == "__main__":
    main()
