#!/bin/sh
# Every result in experiments/geometry-aware/results, in the order it was produced (about 80 min on 2 CPU threads).
set -e
cd "$(dirname "$0")/../.."
S=experiments/geometry-aware/run_experiments.py
O=experiments/geometry-aware/results
python3 $S real --out $O --schemes LOGO LOAO --seeds 0 1 2 3 4 --epochs 400
python3 $S synthetic --out $O --tag synthetic-localized --conditions localized --seeds 0 1 2 --n-train 2000 --epochs 150 --lr 3e-3 --cosine
python3 $S synthetic --out $O --tag synthetic-count-shift --conditions count_shift --seeds 0 1 2 --n-train 2000 --epochs 150 --lr 3e-3 --cosine --nets pointnet attention geo geo_inv
python3 $S synthetic --out $O --tag synthetic-localized-long --conditions localized --seeds 0 --n-train 3000 --epochs 400 --lr 1e-3 --nets attention geo
