# BD3D reconstruction quality

A standalone, CPU-based research pipeline for archaeological photogrammetry inputs,
controlled Metashape runs, stage metrics, reference-relative geometry agreement, and
analysis tables. No training, prediction, neural networks, or ML frameworks.

## Setup on macOS

From this repository, using Python 3.11 or later (development tested on 3.14.7):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[local,test]'
python -m pytest -q
bd3d-quality --help
python examples/synthetic_workflow.py results/my-demo
```

`numpy`, `Pillow`, and `scipy` are the core dependencies. CPU OpenCV is optional
(`local` extra) for ORB descriptors; pytest is a development extra. No Open3D is
required: Aphoma's research-branch geometry implementation already used NumPy and
SciPy, allowing deterministic mesh sampling without Open3D's larger dependencies.
`requirements-lock.txt` records the actual Python 3.14 development environment:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

This freeze is platform/interpreter-specific, not a cross-platform lock. For another
interpreter use the version ranges in `pyproject.toml` and record a fresh freeze.
Metashape is separately installed and licensed. Do not rename wheels to bypass
Python compatibility checks. See [Metashape setup](docs/METASHAPE.md).

## Workflow

All output commands refuse to overwrite existing evidence. Choose a new output
filename/directory for each analysis or reconstruction invocation. Commands return
0 for success and 2 for validation, dependency, or scientific-QC failures.

```bash
bd3d-quality manifest create --root /path/to/photos --dataset-id object-001 \
  --config config/alignment.json --description 'Capture condition and object' \
  --output experiments/object-001.json
bd3d-quality manifest inspect experiments/object-001.json
bd3d-quality manifest validate experiments/object-001.json --root /path/to/photos \
  --output results/validation.json
bd3d-quality features extract experiments/object-001.json --root /path/to/photos \
  --output results/image-features.json
bd3d-quality features match experiments/object-001.json --root /path/to/photos \
  --output results/pair-matches.json
bd3d-quality reconstruct experiments/object-001.json --root /path/to/photos \
  --output results/run-001
```

Manifest creation recursively selects `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, in
sorted relative-path order. It does not silently omit unreadable supported files.
For subsets, create a separate directory of selected images (or carefully construct
an ordered manifest using the Python API), then validate. Never alter originals.
Additional files appearing after manifest creation are excluded: the manifest is
the explicit input set, rather than a live directory listing.

A manifest stores dataset ID, description, ordered portable paths, SHA-256 hashes,
byte counts, total count, effective configuration, schema version, creation time,
software versions, initial status, and empty artifact/failure containers. Its stable
condition ID hashes schema, dataset ID, ordered image records, count, and all
configuration settings. It excludes absolute roots, descriptions, timestamps, and
software discovery. Runtime software belongs to the run record and report
compatibility checks. Order is significant. Experiment ID denotes a condition;
each execution has a separate UUID run ID. Edit settings by creating a new manifest,
not by reusing a project. The source manifest stays immutable; `run.json` records
execution status, timestamps, settings, software, stage measurements and artifacts.

### Geometry

```bash
bd3d-quality geometry compare results/run-001/raw-mesh.obj /path/to/reference.obj \
  --reference-id scanner-or-baseline-ID --units mm --sample-count 10000 \
  --output results/geometry.json
```

OBJ geometry only, including negative indices and fan triangulation. Use triangulated
meshes; concave polygons may not be faithfully fan-triangulated. Input geometry is
never edited. Specify shared units explicitly: reconstruction coordinates are not
millimeters without an independent scale calibration. The runner does not establish
physical scale or detect scale bars. Registering arbitrary-scale reconstructions
against physical scanner data without prior calibration will fail or mislead.

Rigid centroid-initialized point-to-point ICP records the transformation, fitness,
inlier RMSE, correspondence count, settings, sample seed and hashes. It never
rescales geometry. Major RMS-radius disagreement, failed registration, or reaching
the iteration limit invalidates the comparison and suppresses distance metrics.
Outputs include both directional nearest-sample statistics, symmetric squared
Chamfer-style distance, and thresholded candidate coverage/reference completeness.
These are **reference agreement**, not automatically physical accuracy. A reference
reconstruction shares its own errors. See [methodology](docs/METHODOLOGY.md).

### Reports

```bash
bd3d-quality report summarize experiments/object-001.json \
  --features results/image-features.json --matches results/pair-matches.json \
  --run results/run-001/run.json --geometry results/geometry.json \
  --output results/summary-001.json
bd3d-quality report build results/summary-001.json --output results/table-001
```

All summary attachments are optional. Geometry can be attached only when the
candidate hash matches a raw OBJ recorded in the selected run. Standalone geometry
fixtures remain separate from experiment summaries. Summaries embed all supplied
records and source file hashes; every reported stage metric remains traceable.
The output directory contains `experiments.csv`, `experiments.json`,
`validation.json` (including status counts) and `feature_dictionary.json`.

One row = one input/configuration condition with one explicitly selected run.
Duplicate conditions are rejected, including repeat executions; select one run per
condition deliberately. Preserve repeated runs separately for repeatability analysis.
Different effective configurations, feature settings, recorded software signatures,
references or geometry units are rejected from a single table. Build separate tables
for incompatible strata. Missing attachments remain missing; JSON null and CSV
empty cells never mean zero. Stage names prefix metric columns with `__`.

## Architecture

- `manifests.py`, `config.py`, `provenance.py`, `validation.py`: identity, image verification, typed settings, atomic output.
- `image_features.py`, `local_match_features.py`: measurable image and pair descriptors.
- `metashape_runner.py`, `reconstruction_metrics.py`: isolated licensed API, fresh projects, verified checkpoints, stage snapshots, raw exports.
- `geometry_quality.py`: deterministic area-weighted sampling, rigid ICP and validity checks.
- `reporting.py`, `cli.py`: explicit record joins, portable reports and commands.
- `tests/`: provenance, corruption, reconstruction mocks, geometry, features, reporting, CLI and synthetic smoke coverage.
- `config/`, `examples/`: fully specified profiles and redistributable synthetic workflow.
- `experiments/`, `results/`: ignored local research records and artifacts.

## Safeguards and limitations

Fresh run directories are mandatory except explicit `--resume`. Resume accepts only
clean, hashed checkpoints with matching identity/configuration and camera paths.
It verifies project XML/data files and recorded artifact hashes; changes cause
rejection. Failed or interrupted runs need a new directory. A `.run.lock` prevents
concurrent writers. After a crash inspect the run and process before manually
removing a stale lock; this does not make a failed project resumable. Hashes detect
accidental changes, not malicious alteration of an entire ledger.

No refinement, cleanup, texture creation, marker detection or physical scaling is
implemented. Dense cloud and raw mesh are optional profiles; raw data are never
cleaned or overwritten. Absent metrics have null values with reasons. Reprojection
RMSE is calculated only from valid tie-point observations of aligned cameras when
the installed API exposes the necessary methods. Other uncertainty statistics
are not inferred. Metashape settings explicitly exposed here are versioned; API
internal defaults remain dependent on the recorded 2.2.x version. Matching may
still vary across hardware or repeated runs; identity does not imply deterministic
numerical reconstruction.

Read [audit](docs/AUDIT.md), [validation record](docs/VALIDATION.md),
[feature definitions](docs/FEATURE_DICTIONARY.md), [attribution](docs/ATTRIBUTION.md),
and [future statistical/ML work](FUTURE_ML.md). Optional plots and mesh
postprocessing are deliberately outside the initial implementation. Never commit
private photos, archaeological geometry, credentials, environments or generated
research results. Publication instructions are in [GitHub setup](docs/PUBLISHING.md).

## Resource-safe reconstruction and controlled pilot

For a 16 GB laptop, use the watchdog in `bd3d_quality.resource_monitor` rather than
launching the original-raster mesh profile unattended. Its macOS pressure/swap,
RSS, runtime, output-size and CPU-activity gates stop its own child process without
deleting evidence. Metashape may override OpenMP worker requests; the watchdog
also uses a 20% process run duty cycle. These are sampled safeguards, not hard
kernel memory or storage limits. Stopped runs may retain stale lock files; preserve
them and use a fresh output directory.

A versioned `execution_protocol` is included in condition identity when present.
It specifies four depth neighbors, small work groups, no mesh trimming,
CustomFaceCount=0 (no automatic decimation), and no preview textures. Original
manifests without this protocol retain their existing identity and behavior.
`reconstruct --seed-checkpoint PATH` verifies an alignment-only parent's manifest,
project and artifact hashes, then copies its project into a new output directory.
Changing raster resolution requires an independently aligned derivative condition;
it is never treated as the original checkpoint's condition.

`examples/prepare_low_resolution.py` creates lossless, EXIF-oriented derivatives
and a source-hash/derivation ledger. `experiment_design.py` defines deterministic
subsets with explicit overlap/dependence metadata. `examples/run_bounded_pilot.py`
requires a successful monitored baseline, records a measured-cost budget before
execution, and runs at most three conditions sequentially. Geometry export checks
use `examples/validate_exports.py` in an interpreter with Open3D installed; Open3D
is not a core package dependency and no ML extras are needed.

See [controlled experiment protocol](docs/CONTROLLED_EXPERIMENTS.md) for settings,
selection strategies, cancellation rules, pre-execution budgets and scientific
limitations. The original 38-image historical subset remains unresolved; newly
sampled 38-image conditions are explicitly distinct.

Current continuation outcome: **resource-blocked; no newly completed dense/mesh
validation or new pilot executions**. The safety gates stopped original-raster and
derivative attempts on swap growth or timeout. A separately labeled archival
stratum now contains three real mesh observations with Open3D checks, image-hash
joins and reference-relative geometry measurements. These do not establish new-run
success or ground-truth accuracy. Latest automated suite: **74 passed**. See the
[actual measurements and remaining prerequisites](docs/VALIDATION.md#resource-safe-pre-ml-continuation--2026-11-06).

## Lightweight archival table (no reconstruction)

The reviewed array-ready dataset is `results/pytorch-ready-archival-v2/`: **3 archival
observations, 2 provisional non-self reference labels, 1 object group**. It includes
a CSV/JSON table, plain float32 NumPy arrays with boolean masks, explicit input-only
and post-refinement feature lists, target definitions, file/row provenance hashes,
and a leakage/dependence audit. Baseline self-comparison targets are masked; physical
accuracy and dense-point counts are missing. No train/test split or model is fitted.
It is format-ready, not statistically training-ready. See
[dataset columns, verification, and remaining blockers](docs/TABULAR_DATASET.md).

The independently measured dimensional/surface pathway is preserved as a future
extension: see [its protocol, schemas and CLI](docs/DIMENSIONAL_ACCURACY.md).
It does not change the archival tables or convert reprojection error into physical
accuracy. Current research focus: historical alignment-consistency prediction from
photographs before alignment; no new reconstruction or model training is required
for dataset inventory and metadata preparation.

## Current study: historical alignment-consistency prediction

The active objective is predicting historical Metashape **pixel reprojection RMS
from photographs before alignment**. It is not physical mesh-accuracy prediction.
The current local dataset is `results/historical-alignment-recovered-v2/`:
33 verified historical RMSE labels (1 priority Drive PDF + 32 local archival
reports), including 29 additional labels recovered from ARF Dog. There are 35
manifest/cache-linked feature rows, 29 with numeric labels, all from one artifact.
The 145 candidate PAHMA 5-613 JPEGs still lack exact report/input association.
Small PAHMA 5-1405 metadata links 113 camera labels to candidate TIFF stems but
contains no verified pixel RMSE or photo paths. The 61.6 MB PDF was not downloaded.
No row passes the strict readiness gate; session/settings gaps and dependent
observations prevent meaningful grouped evaluation. No models were trained.
See [recovery findings and exact missing files](docs/HISTORICAL_ALIGNMENT_RECOVERY.md)
and the [preserved initial inventory and CLI](docs/HISTORICAL_ALIGNMENT_DATASET.md).
The dimensional/surface extension and all previous datasets remain available.

The [2026-11-07 provenance/readiness audit](docs/PROVENANCE_READINESS.md) clarifies
the five disconnected 5-1405 alignment components and provides a
[deterministic report-to-photo protocol](docs/PHOTO_PROVENANCE_PROTOCOL.md).
Counts remain 33 verified historical labels, 29 label-and-feature pairs and zero
strict-ready observations or eligible held-out groups. No Dog-derived examples
were added. The 6-19929 collection is a prospective acquisition lead; the separate
shared-drive archive link currently returns 404 through the connector.
