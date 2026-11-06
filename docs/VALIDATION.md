# Implementation validation — 2026-11-06

## Automated and package checks

Initial infrastructure suite: **63 passed, 0 failed**, Python 3.14.7. Tests cover
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

## Resource-safe pre-ML continuation — 2026-11-06

Latest automated suite: **74 passed, 0 failed**. Added protocol-identity,
resource-gate, checkpoint-copy/lineage, deterministic-subset and watchdog tests.
All original baseline images and the original alignment-v2 checkpoint were
validated before execution. No original images/projects were altered.

**New end-to-end dense/mesh validation is resource-blocked, not complete.**
The bounded new pilot was not launched because its prerequisite (a successful
monitored baseline) did not pass. Do not describe archival meshes as validation
of the new runner or substitute them for missing newly generated dense points.

| Monitored attempt | Wall seconds | Peak sampled RSS MiB | Maximum pressure | Result |
| --- | ---: | ---: | ---: | --- |
| Original 50 images, copied alignment, Lowest depth/four neighbors | 4.08 | 1404.5 | 2 (warning) | Swap grew 450.3 MiB; terminated |
| New 1296-pixel derivative alignment | 22.67 | 121.8 | 2 | Swap grew 277.2 MiB; terminated |
| New 648-pixel derivative alignment/depth | 300.75 | 654.6 | 2 | Five-minute wall-time gate; terminated near depth-map completion |
| Verified copy of 648-pixel alignment checkpoint, same condition continuation | 6.30 | 786.9 | 2 | Swap grew 650.3 MiB; terminated |

These are sampled system-wide swap changes, not proof of a Metashape-specific
memory leak. RSS omits compressed/swapped process memory. No observed pressure
sample reached critical level 4. The system already used approximately 25–26 GiB
of swap on a 16 GiB machine. Safety gates were retained rather than weakened.
All process groups exited; saved files, failed-attempt ledgers, console logs,
derivative rasters and original checkpoints remain. There are no newly completed
dense-cloud counts, mesh counts or raw dense/mesh exports to report.

The 648-pixel alignment succeeded with 50/50 cameras, 2,572 valid tie points,
8,281 reprojection observations and 0.3685228 px RMSE in 112.8994 wall seconds
under the duty-cycle controller. Residuals are in derivative pixels and are not
directly interchangeable with original-resolution residuals. Its checkpoint was
hash-verified before the continuation copy. Runtime gates apply per slice;
all four monitored reconstruction attempts together consumed approximately 334
wall seconds, excluding derivative preparation and separate read-only analysis.

New saved evidence (ignored by Git): `pre-ml-validation-status.json`, four
`*-resources.json` traces and associated console logs; `arf-dog-derived-1296/`
and `arf-dog-derived-648/` source-hash derivation ledgers; and
`derived-648-experiment-design.json`. Total generated results were approximately
105 MiB at audit, including approximately 82 MiB of derivative rasters. The pilot
remains limited to three additional conditions / 900 seconds / 500 MB, pending
successful validation. The full 14-execution matrix is not authorized for automatic
batch launch and could consume up to 70 minutes at its five-minute per-run ceiling.
RAM and storage cannot be reliably projected from a failed end-to-end baseline;
premeasurement estimates remain provisional.

### Actual historical geometry measurements (separate archival stratum)

A read-only Open3D 0.20.0 audit loaded three existing FAST_DENSE mesh exports.
Historical manifest file hashes match the recorded report hashes. Every declared
image hash matches the current verified 50-image source pool. This verifies the
current inputs and intended historical manifests, not historical image bytes at
execution. No historical project or mesh was rewritten.

| Historical condition | Open3D vertices | Open3D faces | Surface area, arbitrary units² |
| --- | ---: | ---: | ---: |
| Baseline 50, seed 20260822 | 33,570 | 67,136 | 14.5496 |
| Random 25, seed 20260823 | 31,991 | 63,978 | 15.4438 |
| Random 25, seed 20260824 | 31,138 | 62,272 | 15.5101 |

All three have finite coordinates, permit boundary edges under the manifold check,
and are not watertight. No separate dense-cloud export was found for these audited
legacy runs, so dense-point counts remain missing. Legacy refinement/orientation
and High face-count settings differ from the new raw protocol. At least one legacy
report has inconsistent total vs summed stage durations; source values are retained
and explicitly flagged, not repaired or used as new-run timing benchmarks.

The initial 3,000-sample / 60-iteration rigid ICP audit rejected both subsets at the
iteration limit. A separate, uniform sensitivity check kept the same hashes,
seed, sampling and QC thresholds but raised the iteration cap to 200. The two
comparisons converged in 97 and 140 iterations, with fitness 0.8267 and 0.8260,
scale-radius ratios 1.0179 and 1.0342, and squared Chamfer-style agreement 0.0320462
and 0.0286345 in historical export units². These are marginal-fitness local
registration results, requiring visual review; they are neither physical accuracy
nor independently validated training labels. The 60-iteration failures remain
preserved. Baseline self-distance is only an evaluator check and must not be
counted as an independent quality target.

`archival-pilot-audit-v2/` holds source reports, hash evidence, Open3D diagnostics
and initial QC; `archival-registration-200/` holds the sensitivity results;
`archival-pre-ml-observations/observations.csv` and JSON explicitly join current
image/pair measurements to the three archival runs by filename/content hash.
These are **three archival observations, zero newly completed pilot executions**.
All share one physical object/session and overlapping input images. The historical
38-image subset is still unresolved. No ML frameworks were imported by the
Open3D audit, and no neural-network/model work was performed.

### Remaining prerequisites

Reduce system-wide memory pressure outside this pipeline before attempting another
monitored reconstruction. Recheck pressure and swap behavior; do not bypass gates.
Then use a new output directory or a verified checkpoint copy, finish/export actual
dense and raw mesh assets, validate them, and only then launch the bounded pilot.
The original pause/stop requests were honored; no background reconstruction is
left running. Existing independently verified historical meshes do not remove
these remaining validation and data-collection requirements.

Final lightweight checks passed: 74 tests in 1.39 seconds, source/example
compilation, dependency consistency and unchanged original alignment-v2 project
fingerprints. Aphoma remained clean. At the final check, no reconstruction or
watchdog child was active; the machine still reported warning-level pressure,
27,103 MiB system swap in use and 229 GiB available disk. Repository size was
approximately 402 MiB including the 296 MiB environment and 105 MiB local results.
These final system conditions reinforce that the resource prerequisite remains
unresolved outside this pipeline. No public or private push was performed.
