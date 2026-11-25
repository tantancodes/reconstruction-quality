"""Train and save the geo model used for the attention maps in geometry_aware_attention_analysis.ipynb.

Same data and settings as runs/attention-maps/attention_localized_seed0.pt (notebook 03) and as the
long check in run_all.sh: localized condition, seed 0, 3000 training sets, 400 epochs, Adam 1e-3, batch 128.
About 11 min on 2 CPU threads. Writes experiments/geometry-aware/results/models/geo_localized_seed0.pt
and its validation curve.

  python experiments/geometry-aware/train_cached_models.py
"""
import csv
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiments as E   # noqa: E402
import geo_models as gm       # noqa: E402


def main():
    torch.set_num_threads(2)
    out = HERE / 'results' / 'models'
    out.mkdir(parents=True, exist_ok=True)
    (Xtr, Mtr, ytr), (Xva, Mva, yva), (Xte, Mte, yte) = E.synthetic_data('localized', 0, 3000, 600, 600)
    R = {k: gm.relations(X, None, M) for k, (X, M) in dict(tr=(Xtr, Mtr), va=(Xva, Mva), te=(Xte, Mte)).items()}
    t, log = time.time(), []
    model = E.train_net('geo', Xtr, Mtr, R['tr'], ytr / E.syn.SCALE, 0, epochs=400, h=64, batch=128, lr=1e-3,
                        wd=0.0, has_view=False, Xva=Xva, Mva=Mva, Rva=R['va'], yva=yva / E.syn.SCALE, log=log)
    mae = float((E.predict(model, 'geo', Xte, Mte, R['te'], False, 0) * E.syn.SCALE - yte).abs().mean())
    torch.save({'state_dict': model.state_dict(), 'name': 'geo', 'coord_dim': 3, 'h': 64, 'rel_dim': 2,
                'condition': 'localized', 'seed': 0, 'n_train': 3000, 'epochs': 400, 'lr': 1e-3,
                'test_mae_deg': mae, 'seconds': round(time.time() - t)}, out / 'geo_localized_seed0.pt')
    with open(out / 'geo_localized_seed0_curve.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'train_mse', 'val_mae_deg'])
        w.writeheader()
        w.writerows(dict(epoch=r['epoch'], train_mse=r['train_mse'], val_mae_deg=r['val_mae'] * E.syn.SCALE) for r in log)
    print(f'geo test MAE {mae:.2f} deg in {time.time() - t:.0f}s')


if __name__ == '__main__':
    main()
