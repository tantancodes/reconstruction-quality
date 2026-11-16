"""Rebuild the EXPERIMENTAL quality models used by the capture app (models/ is gitignored).

Trains the pilot attention set model with the settings of scripts/pilot_set_models.py (its own
load_runs, pad and train_net: h=32, 400 epochs, Adam lr 1e-3 wd 1e-4, full batch, random rotation
augmentation) on ALL labelled runs at once, one model per seed (0-4), and writes
  models/quality-attention/seed<k>.pt   weights + label scale
  models/quality-attention/model.json   training runs, settings, versions
There is no held-out data here: the pilot's leave-one-out numbers (results/pilot-real-v3-all)
are the only estimate of error, and they come from 2 objects.

  python scripts/build_quality_models.py            # about 10 seconds on a laptop CPU
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'src'))
import pilot_set_models as pilot  # noqa: E402
from bd3d_quality import quality_model as qm  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, default=REPO / 'models/quality-attention')
    ap.add_argument('--coffin-labels', type=Path, default=REPO / 'results/pahma-5-613-v2/labels-all.csv')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    runs = pilot.load_runs(a.coffin_labels)
    artifacts = sorted({r['artifact'] for r in runs})
    print(f'{len(runs)} runs: ' + ', '.join(f'{x} {sum(r["artifact"] == x for r in runs)}' for x in artifacts))
    X, M = pilot.pad(runs)
    y = torch.tensor(np.array([r['label'] for r in runs], np.float32))
    a.output.mkdir(parents=True, exist_ok=True)
    for old in a.output.glob('seed*.pt'):
        old.unlink()
    fits = []
    for seed in pilot.SEEDS:
        model, scale = pilot.train_net('attention', X, M, y, seed, None)
        qm.save_model(a.output, model, scale, seed, pilot.H)
        fit = [pilot.predict_net(model, scale, X[i:i + 1], M[i:i + 1], seed) for i in range(len(runs))]
        fits.append(np.mean(np.abs(np.array(fit) - y.numpy())))
        print(f'  seed {seed}: trained, training-set MAE {fits[-1]:.5f} (in-sample, optimistic)', flush=True)
    qm.write_meta(a.output, {
        'status': 'EXPERIMENTAL - trained on a few dozen runs of 2 objects; not reliable on new objects',
        'model': 'attention (scripts/pilot_set_models.py AttnSet)', 'n_runs': len(runs), 'artifacts': artifacts,
        'seeds': list(pilot.SEEDS), 'epochs': pilot.EPOCHS, 'hidden': pilot.H,
        'optimizer': 'Adam lr 1e-3 wd 1e-4, full batch', 'test_rotations': qm.ROTATIONS,
        'label': 'v3 symmetric normalized RMSE vs each artifact/pipeline reference (lower is better)',
        'held_out_error': 'see results/pilot-real-v3-all/summary.csv (LORO / LOAO); none measured here',
        'training_set_mae': [round(float(f), 6) for f in fits],
        'coffin_labels': pilot.rel(a.coffin_labels),
        'runs': [{'artifact': r['artifact'], 'run': r['run'], 'label': r['label'], 'cameras': pilot.rel(r['cameras']),
                  'cameras_sha256': qm.file_sha256(r['cameras'])} for r in runs],
        'torch': torch.__version__, 'built_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'seconds': round(time.time() - t0, 1)})
    print(f'wrote {len(pilot.SEEDS)} models to {pilot.rel(a.output)} in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
