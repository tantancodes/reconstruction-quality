# Implementation validation — 2026-11-06

## Automated and package checks

Final local automated suite: **63 passed, 0 failed**, Python 3.14.7. Tests cover
portable identities, file hashing, paths/symlinks, missing/unreadable/changed inputs,
configuration validation, exclusive evidence writes, image formulas, CPU pair
matches, metric parsing/residual calculation, Metashape mocks, failure ledgers,
raw exports, dense/mesh call flow, resume mismatches, process locks, geometry
self/translation/scale/registration failures, reporting compatibility/duplicates,
CLI failures, and a standalone synthetic smoke workflow. An early run had four
failures due to a missing hash import in adapted geometry code; fixed before the
passing suite. The real resume test additionally found vendor transient lock
handling absent from mocks; implementation and regression tests were added.

Editable installation, pip dependency consistency, compilation and command help
checks passed. Wheel packaging was exercised; the final wheel is locally generated
under ignored results. requirements-lock.txt records exact non-ML development
dependencies. No global package installation or ML package installation occurred.

## Synthetic workflow

`results/synthetic-demo/` contains three generated texture PNGs, source manifest,
file validation, image features, all three image pairs, OBJ tetrahedron fixtures,
self-comparison geometry JSON, source-preserving summary, CSV/JSON reports,
status/validation report and generated feature dictionary. Geometry self-distance
is approximately zero. These are software checks, not photographs of a physical
object and not evidence of successful photogrammetry.

An invocation against the actual Metashape API on these textures failed with
“Can't align cameras,” as expected for this nonphysical scene. Its failure ledger
is retained at `results/metashape-license-check/run.json`; the directory name
reflects its initial purpose, but failure was alignment, not licensing.

## Real archaeological image integration

All 50 historical baseline hashes were independently verified against the original
images. No image was copied into the repository or changed. New manifest:
`results/arf-dog-50-manifest.json`; CPU image features and all 1,225 pair measurements
are stored as `results/arf-dog-50-features.json` and `results/arf-dog-50-matches.json`.

Metashape 2.2.1 executed through the existing Python 3.12.14 interpreter in read-only
mode for source/environment files. An initial probe denied licensing; later calls
obtained a license. Fresh projects and outputs were written only into this new
repository. No original project was opened for modification.

First run: `results/arf-dog-50-alignment/` completed matching/alignment and raw PLY
export. Measurements: 50/50 cameras aligned (1.0), 23,447 valid tie points,
157,619 eligible reprojection observations, RMSE 0.4772728355651178 pixels;
matching/alignment runtime 103.9912 seconds on CPU. The first resume check correctly
found a hash mismatch, but the mismatched entry was a transient vendor lock, not
scientific project state. That run was left intact. The runner now excludes only
`project.files/lock` from fingerprints and explicitly rejects a live vendor lock.
The fresh second run `results/arf-dog-50-alignment-v2/` completed successfully:
50/50 aligned cameras, 23,551 valid tie points, 157,633 residual observations,
0.4884375710632685 pixel reprojection RMSE, and 117.1599 seconds matching/alignment.
A separate process then successfully verified this completed checkpoint with
`--resume`. All project data and artifact hashes matched after process closure.
The second run is the explicitly selected run in `results/arf-dog-50-summary.json`
and `results/arf-dog-50-table/experiments.csv`; the first run has its own summary
and table. Both conditions share an ID but have distinct UUIDs, and are never
silently combined in one condition-level table. Their numerical variation
illustrates why condition identity does not imply bitwise deterministic outcomes.

Real dense/mesh reconstruction and external-ground-truth validation have not been
performed by this new implementation. Dense/mesh stage flow is mock-tested only.
There is no physical scale calibration. These settings also differ from historical
Aphoma defaults/refinement: new counts and residuals must not be interpreted as a
repeat of the historical full-reference condition.

## Reproduction

```bash
cd /Users/tristan/Documents/bd3d-reconstruction-quality
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[local,test]'
python -m pytest -q
python examples/synthetic_workflow.py results/my-fresh-demo
```

The local real-data integration used the existing compatible interpreter without
installing into it or writing bytecode in the old repository:

```bash
PYTHONPATH=/Users/tristan/Documents/bd3d-reconstruction-quality/src \
/Users/tristan/Documents/ChatGPT/bd3d/venv312/bin/python -B -m bd3d_quality.cli \
reconstruct results/arf-dog-50-manifest.json \
--root /Users/tristan/Downloads/Jpg-processed --output results/my-fresh-real-run
```

The real manifest is intentionally ignored by Git. On another machine, obtain the
authorized dataset and install this package plus Metashape in a separately licensed,
vendor-compatible environment. Do not infer that license availability here
establishes availability elsewhere. Source data and generated results stay local.
