"""EXPERIMENTAL predicted quality from an aligned camera set (no UI code).

The pilot attention set model (scripts/pilot_set_models.py, same settings) trained on every
labelled run, one model per seed; scripts/build_quality_models.py rebuilds them into
models/quality-attention/ (gitignored). The models predict the v3 label (symmetric surface RMSE /
reference bounding-box diagonal; lower is better) from camera positions and view directions only.
They were trained on 2 objects and a few dozen runs: a prediction is a rough hint, not a
measurement, and is not reliable on objects unlike the training ones.

PyTorch is imported only when models are loaded, so the rest of the package works without it.
"""
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

H = 32              # hidden width (pilot_set_models.H)
ROTATIONS = 8       # test-time random rotations per prediction (pilot_set_models.predict_net)
MIN_CAMERAS = 3
META = 'model.json'


class ModelsMissing(Exception):
    """No trained models in the folder (or PyTorch is not installed)."""


# ---------------------------------------------------------------- input

def read_cameras(path):
    """(positions, view directions) of the aligned cameras in a cameras.csv."""
    with open(path, newline='') as f:
        rows = [r for r in csv.DictReader(f)
                if r.get('aligned', 'True') == 'True' and r.get('x') not in ('', None)]
    P = np.array([[float(r[k]) for k in 'xyz'] for r in rows], dtype=float).reshape(-1, 3)
    V = np.array([[float(r['view_' + k]) for k in 'xyz'] for r in rows], dtype=float).reshape(-1, 3)
    return P, V


def camera_features(path):
    """(n, 6) float32 model input, as in pilot_set_models.load_runs: positions minus their centroid,
    divided by the median distance from it, then view directions."""
    P, V = read_cameras(path)
    if len(P) < MIN_CAMERAS:
        raise ValueError(f'only {len(P)} aligned cameras (at least {MIN_CAMERAS} needed)')
    P = P - P.mean(0)
    scale = np.median(np.linalg.norm(P, axis=1))
    if not scale > 0:
        raise ValueError('all aligned cameras are at the same position')
    return np.c_[P / scale, V].astype(np.float32)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- model (torch)

def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ModelsMissing('PyTorch is not installed (pip install -e ".[ml]")') from exc
    return torch


def attention_model(h=H):
    """The pilot's AttnSet; parameter names match pilot_set_models.AttnSet so state dicts load."""
    torch = _torch()
    nn = torch.nn

    class AttnSet(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Sequential(nn.Linear(6, h), nn.ReLU(), nn.Linear(h, h))
            self.attn = nn.MultiheadAttention(h, num_heads=4, batch_first=True)
            self.ff = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
            self.rho = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

        def forward(self, x, mask):
            f = self.embed(x)
            a, _ = self.attn(f, f, f, key_padding_mask=~mask)
            return self.rho(self.ff(f + a).masked_fill(~mask[..., None], -1e9).max(1).values).squeeze(-1)

    return AttnSet()


def random_rotations(rng, k):
    """k random proper rotations (pilot_set_models.random_rotations)."""
    out = []
    for _ in range(k):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q * np.where(np.diag(r) < 0, -1.0, 1.0)
        out.append(q if np.linalg.det(q) > 0 else -q)
    return np.stack(out)


def predict_one(model, scale, seed, x, rotations=ROTATIONS):
    """Prediction for one (n, 6) camera set: mean over random rotations drawn from seed + 20000,
    exactly as pilot_set_models.predict_net."""
    torch = _torch()
    X = torch.tensor(x[None])
    M = torch.ones(X.shape[:2], dtype=torch.bool)
    rng = np.random.default_rng(seed + 20_000)
    out = []
    with torch.no_grad():
        for _ in range(rotations):
            R = torch.tensor(random_rotations(rng, 1), dtype=torch.float32).transpose(1, 2)
            Xr = torch.cat([X[..., :3] @ R, X[..., 3:] @ R], -1)
            out.append((model(Xr, M) * scale).item())
    return float(np.mean(out))


def save_model(folder, model, scale, seed, hidden=H):
    """Write one trained model as seed<k>.pt (weights, label scale, seed)."""
    torch = _torch()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'seed{seed}.pt'
    torch.save({'state_dict': model.state_dict(), 'scale': float(scale), 'seed': int(seed), 'hidden': int(hidden)},
               path)
    return path


def write_meta(folder, meta):
    (Path(folder) / META).write_text(json.dumps(meta, indent=2) + '\n')


_cache = {}


def load_models(folder):
    """(meta dict, [(model, scale, seed)]) from a folder made by scripts/build_quality_models.py.
    Raises ModelsMissing when the folder has no models or PyTorch is missing."""
    folder = Path(folder)
    files = sorted(folder.glob('seed*.pt')) if folder.is_dir() else []
    if not files or not (folder / META).is_file():
        raise ModelsMissing('the experimental models have not been built (python scripts/build_quality_models.py)')
    key = (str(folder.resolve()), tuple((f.name, f.stat().st_mtime_ns) for f in files),
           (folder / META).stat().st_mtime_ns)
    if key in _cache:
        return _cache[key]
    torch = _torch()
    meta = json.loads((folder / META).read_text())
    models = []
    for f in files:
        d = torch.load(f, map_location='cpu', weights_only=True)
        model = attention_model(d.get('hidden', H))
        model.load_state_dict(d['state_dict'])
        model.eval()
        models.append((model, float(d['scale']), int(d['seed'])))
    _cache.clear()
    _cache[key] = (meta, models)
    return meta, models


# ---------------------------------------------------------------- prediction

@dataclass
class Prediction:
    mean: float
    low: float                  # min across seeds
    high: float                 # max across seeds
    per_seed: dict              # {seed: prediction}
    cameras: int                # aligned cameras used
    n_runs: int                 # training runs
    n_objects: int              # training objects
    in_training: str = None     # training run with the same cameras.csv, if any

    @property
    def caption(self):
        objects = f'{self.n_objects} object{"s" * (self.n_objects != 1)}'
        return f'Experimental: trained on {objects} ({self.n_runs} runs); not reliable on new objects'

    @property
    def text(self):
        """One plain line. Deliberately not a score or a verdict: a number and its spread."""
        k = len(self.per_seed)
        return (f'Predicted quality: error about {self.mean:.4f} '
                f'({k} model{"s" * (k != 1)} range {self.low:.4f}–{self.high:.4f}; lower is better)')


def predict_quality(cameras_csv, folder):
    """Prediction for the aligned cameras in cameras_csv with every model in folder.
    Raises ModelsMissing (no models / no PyTorch) or ValueError (too few aligned cameras)."""
    meta, models = load_models(folder)
    x = camera_features(cameras_csv)
    per_seed = {seed: predict_one(model, scale, seed, x) for model, scale, seed in models}
    values = list(per_seed.values())
    sha = file_sha256(cameras_csv)
    seen = next((r['run'] for r in meta.get('runs', []) if r.get('cameras_sha256') == sha), None)
    return Prediction(float(np.mean(values)), min(values), max(values), per_seed, len(x),
                      int(meta.get('n_runs', 0)), len(meta.get('artifacts', [])), seen)
