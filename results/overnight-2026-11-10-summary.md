# Overnight run, 2026-11-10: summary

**Result.** Every Metashape job ran except one: random30 seed 20261110 failed alignment (7 of 30 cameras aligned). That covers the coffin reference, all coffin candidates (a, b, c, d ×2, e) and all 5 new dog runs. Every completed run is scored with v3 and passes QC, with camera and ICP scale agreeing. The pilot model comparison and the results notebook are built.

There were no license problems: the check passed before every job. There were no watchdog or resource stops, and no Metashape or watchdog process was left running at the end. Nothing was deleted and nothing was committed.

## Timeline

| time | what |
|---|---|
| 02:52–03:50 | coffin reference `ref145b`, 145 images: complete, 57.5 min. It ran `analyzeImages` once for all images. |
| 03:50 | image-quality cache (145 rows), pose-based series split, candidate manifests |
| 03:50–04:51 | a `drop_s3` (95 images, 27.8 min) and b `drop_s2` (107 images, 31.0 min): complete. d `random30_s20261110` **failed**, see below. |
| 04:51–05:22 | dog runs (Part 2): `random15` ×2, `random20` ×2, `side25`, all complete, 4–12 min each |
| 05:22–05:26 | v3 scoring, labels, pilot models and notebook on what had finished so far |
| 05:27–06:24 | continuation: d `random30_s20261111` (6.7 min), c `side40_s1` (35.8 min), e `random60_s20261112` (14.8 min), all complete |
| 06:24–06:29 | rescoring, pilot models and notebook on everything |

Every coffin subset reused the cached image quality instead of running `analyzeImages`. Each run's `run.json` records this, with the cache path and its SHA-256, plus the source run and experiment. The dog runs ran `analyzeImages` themselves, as in the earlier dog pilot.

## What failed and why

1. **Coffin d, `random30_s20261110`: alignment failure, a result.** Only 7 of the 30 cameras aligned (23%), so the run failed the alignment criteria and no mesh or label exists. Thirty random photos aren't enough coverage for this object. The other seed aligned 29 of 30 cameras and has the worst coffin label (0.035).
2. **The driver stopped Part 1 too early.** `run_watched.py` treats any failure as a stop, so after this alignment failure the chain skipped random30 seed 2, c and e and moved on to the dog runs. Your stop rule covers watchdog and resource stops only. At 05:27 I ran those three approved jobs with `scripts/overnight_continue.py`: one job at a time, so an alignment failure no longer blocks the next one, while watchdog or license problems still stop everything. Their scores went to new versioned files; the 05:22 outputs are kept as they were.
3. **The series split disagreed with the capture-time split; the pose split was used, as your rule says.**
   - The capture-time column in `series.csv` is wrong. It takes the two largest EXIF gaps, and one of them is a 110 s pause after the first photo (`_MG_4748`), so it gives 1/106/38. Against the known split (gaps after `_MG_4804` and `_MG_4854`, 57/50/38), 119 of 145 photos match the pose split.
   - The walls circuit matches exactly: 57 of 57, elevation 2–12°.
   - The 45° circuit (21–43°) and the near-top-down series (31–53°) overlap in elevation. Pose series 3 (the 50 highest cameras) is 31 top-down plus 19 from the 45° circuit. Pose series 2 (38 cameras) is 31 from the 45° circuit plus 7 top-down.
   - So **a drops the 50 highest cameras and b drops the 38 middle ones**. Neither removes exactly one shooting series. If you want the true shooting series, a and b can be rebuilt from the 57/50/38 split.

## Labels per artifact (v3, normalized by the reference bounding-box diagonal)

**Missing** surface is reference surface the candidate doesn't reconstruct (reference → candidate). **Extra** surface is candidate surface with no counterpart in the reference (candidate → reference). "Beyond cutoff" is the fraction of points more than 5% of the diagonal away.

### PAHMA 5-613 coffin, against `ref145b`: `results/pahma-5-613-v2/labels-all.csv`

| run | images | v3 label | missing RMSE | extra RMSE | missing beyond cutoff | extra beyond cutoff | dominant | reproj. RMSE (px) | QC / scale |
|---|---|---|---|---|---|---|---|---|---|
| c `side40_s1` (23 of 57 wall cameras removed) | 122 | 0.0080 | 0.0112 | 0.0015 | 1.7% | 0.0% | missing (98%) | 1.16 | pass / agree |
| b `drop_s2` (middle elevation cluster) | 107 | 0.0102 | 0.0080 | 0.0120 | 0.3% | 1.9% | **extra** (69%) | 0.98 | pass / agree |
| a `drop_s3` (highest elevation cluster) | 95 | 0.0106 | 0.0139 | 0.0055 | 2.1% | 0.2% | missing (86%) | 0.90 | pass / agree |
| e `random60_s20261112` | 60 | 0.0182 | 0.0257 | 0.0022 | 8.5% | 0.0% | missing (99%) | 0.93 | pass / agree |
| d `random30_s20261111` (29 of 30 aligned) | 29 | 0.0354 | 0.0499 | 0.0039 | 22.4% | 0.0% | missing (99%) | 0.67 | pass / agree |
| d `random30_s20261110` | 30 | — | — | — | — | — | alignment failed (7 of 30) | 0.33 | no mesh |

`labels.csv` in the same folder is the 05:22 version, with a and b only.

### ARF dog, GPU depth-downscale 8, against `gpu-pilot-d8/ref50`: `results/gpu-pilot-d8-ext/labels.csv`

| run | images | v3 label | missing RMSE | extra RMSE | missing beyond cutoff | extra beyond cutoff | dominant | reproj. RMSE (px) | QC / scale |
|---|---|---|---|---|---|---|---|---|---|
| random10 (earlier pilot) | 10 | 0.0225 | 0.0304 | 0.0094 | 10.9% | 0.8% | missing (91%) | 0.25 | pass / agree |
| random15_s20261111 | 15 | 0.0091 | 0.0119 | 0.0050 | 1.2% | 0.0% | missing (85%) | 0.33 | pass / agree |
| side40 (earlier pilot) | 30 | 0.0085 | 0.0101 | 0.0065 | 0.9% | 0.0% | missing (71%) | 0.43 | pass / agree |
| random15_s20261110 | 15 | 0.0085 | 0.0105 | 0.0059 | 0.7% | 0.0% | missing (76%) | 0.32 | pass / agree |
| side25 (new) | 38 | 0.0078 | 0.0073 | 0.0083 | 0.3% | 0.7% | **extra** (56%) | 0.45 | pass / agree |
| random20_s20261110 | 20 | 0.0072 | 0.0091 | 0.0044 | 0.4% | 0.0% | missing (82%) | 0.34 | pass / agree |
| random20_s20261111 | 20 | 0.0062 | 0.0073 | 0.0050 | 0.3% | 0.0% | missing (68%) | 0.36 | pass / agree |

Both artifacts follow the same pattern:
- **Random subsets fail mainly through missing surface,** and it grows as images are removed. On the coffin, missing surface accounts for 99% of the squared error at 60 and 30 images.
- **Removing a whole side or band costs much less than removing the same number of images at random.** Coffin c, with 23 images removed, scores 0.008 against 0.018 for random60. Dog side40 (30 images) scores 0.0085, about the same as random15.
- **Two runs are dominated by extra surface:** coffin b (middle band dropped) and dog side25. The candidate there adds surface the reference doesn't have, likely noisier geometry where coverage is thinner.
- **Reprojection RMSE doesn't track the label.** The worst coffin run has the lowest reprojection RMSE (0.67 px), and the best has the highest (1.16 px).

## Pilot model comparison: tiny n, not evidence

Data: 28 runs with a v3 label and `cameras.csv`. That's 23 dog runs (16 historical CPU runs plus 7 GPU d8 runs) and 5 coffin runs. Mean ± std over 5 seeds; the mean, reprojection and Ridge baselines are deterministic, so their std is 0. MAE is in label units; skill = 1 − MAE / MAE of the mean baseline. Output: `results/pilot-real-v3-all/`.

| model | leave-one-run-out MAE | skill | leave-one-artifact-out MAE | skill |
|---|---|---|---|---|
| mean | 0.00461 | 0 | 0.01082 | 0 |
| reprojection-RMSE linear fit | 0.00422 | +0.08 | 0.03945 | −2.65 |
| Ridge on global summaries | **0.00303** | **+0.34** | 0.02365 | −1.19 |
| PointNet | 0.00365 ± 0.00003 | +0.21 ± 0.01 | 0.00409 ± 0.00044 | +0.62 ± 0.04 |
| attention | 0.00343 ± 0.00013 | +0.26 ± 0.03 | **0.00351 ± 0.00053** | **+0.68 ± 0.05** |

How to read it:
- **Leave one run out:** Ridge is best. All three camera-layout models beat the mean, and reprojection RMSE barely does. Runs from the same capture share images, so this scheme is optimistic.
- **Leave one artifact out (two folds):**
  - Ridge and the reprojection fit break down. Their features shift between objects: coffin reprojection RMSE is about 0.9 px against about 0.35 px for the dog. Ridge predicts 0.03–0.13 for coffin runs whose labels are 0.008–0.035.
  - The set networks stay in range. Trained on the dog only, attention comes closest on the coffin (MAE 0.0067 against 0.0104 for the mean), followed by PointNet (0.0074). Both underpredict the worst coffin run (0.035 → about 0.013–0.019) and overpredict side40_s1.
  - The leave-one-artifact-out mean baseline is weak by design: it predicts the other object's mean. The large positive skill numbers mostly reflect that.
- **Bottom line:** the pipeline runs end to end on real data. The set models transfer across objects more sensibly than the summary features do. With 28 correlated runs from 2 objects, none of these differences can be called reliable.

## Results notebook

The notebook is `~/bd3dtorch/analysis/pytorch_reconstruction_results-2.ipynb`; this is the final one, built from all runs. `pytorch_reconstruction_results.ipynb` is the 05:26 version, built before the coffin continuation, and the original `pytorch_reconstruction.ipynb` is untouched.

It only loads saved CSVs, TensorBoard logs and camera tables, has no hand-typed numbers, and ran without errors. Contents:
- markdown sections: question, setup, results, limitations
- example camera sets with the largest azimuth gap shaded (dog and coffin)
- synthetic MAE bar chart with seed error bars
- learning curves: synthetic validation MAE from TensorBoard, plus the pilot's training loss
- label tables per artifact
- pilot table and predicted-vs-actual plots

To rebuild it: `python3 scripts/build_results_notebook.py --pilot results/pilot-real-v3-all --coffin-labels results/pahma-5-613-v2/labels-all.csv`.

## Files and code changes (uncommitted)

- **New scripts:**
  - `scripts/overnight_chain.py`, `scripts/overnight_continue.py` (now in `archive/scripts/`)
  - `scripts/make_dog_failure_manifests.py`, `scripts/combine_labels.py`
  - `scripts/pilot_set_models.py`, `scripts/build_results_notebook.py`
- **Changed scripts:**
  - `scripts/run_watched.py`: license check before each job.
  - `scripts/make_pahma_manifests.py`: adds c `side40_s1` and e `random60`.
- **Manifests:** `experiments/pahma-5-613/candidates/`, `experiments/gpu-pilot-d8-ext/`, `experiments/pahma-5-613/ref145b.json`.
- **Logs:** `results/overnight-2026-11-10/`, with `chain.log`, `continue.log`, `chain-status.json`, `continue-status.json` and per-step logs. Per-run watchdog, resource and status files sit next to each run.

## Suggested next steps

1. Fix the capture-time gap detection in `series_split_from_poses.py`, for example by ignoring a gap after the first or last photo. Then decide whether to rerun a and b on the true 57/50/38 shooting series.
2. Change `run_watched.py` so an alignment failure is recorded as a result, not as a stop. `overnight_continue.py` already behaves this way.
3. Before reading anything into model skill: more objects, or at least more coffin runs at mid-range image counts.
