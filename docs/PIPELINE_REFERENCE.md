# Pipeline reference: CLI, provenance and safeguards

Operational reference for the `bd3d-quality` package (moved here from the project README on
2026-11-10). For the research question, findings and figures see the [README](../README.md).

## Setup on macOS

From this repository, with Python 3.14.7 (the version that produced the results):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,ml,local]' -c requirements-lock.txt
python -m pytest -q
bd3d-quality --help
```

`numpy`, `Pillow`, and `scipy` are the core dependencies; the `ml` extra adds torch,
scikit-learn, pandas, matplotlib, TensorBoard and the notebook tools used by the models and
figures; `local` adds CPU OpenCV (only for the archived image-feature code); `test` adds
pytest. `requirements-lock.txt` is a full freeze of that environment, so tests, models and
figures all run with the same versions. It is interpreter-specific, not a cross-platform
lock: for another interpreter use the version ranges in `pyproject.toml` and record a fresh
freeze. Metashape is separately installed and licensed. Do not rename wheels to bypass
Python compatibility checks. See [Metashape setup](METASHAPE.md).

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

### Geometry labels (evaluator v3)

```bash
python scripts/score_runs_v3.py --reference results/run-ref --candidates results/run-001 \
  --output results/geometry-v3
python scripts/combine_labels.py --artifact object-001 --scores results/geometry-v3 \
  --runs-dir results --output results/labels.csv
```

Each run directory needs `raw-mesh.obj`, `cameras.csv` and `run.json` from `reconstruct`.
The candidate mesh is registered to the reference by the similarity transform that maps
its aligned cameras onto the reference cameras over shared images, refined by
point-to-point ICP with uniform scale, and scored by the symmetric RMSE between surface
samples divided by the reference bounding-box diagonal (`src/bd3d_quality/evaluation.py`).
Runs whose ICP and camera scales differ by more than 2.5% are flagged. OBJ geometry only,
including negative indices and fan triangulation; input geometry is never edited. Labels
are **reference agreement**, not physical accuracy: a reference reconstruction shares its
own errors. See [methodology](METHODOLOGY.md).

The earlier rigid `geometry compare` evaluator, the image `features` commands and the
`report` tables are archived with the rest of the earlier-phase CLI in
`archive/src/bd3d_legacy/cli.py` (see `archive/README.md`).

## Architecture

- `manifests.py`, `config.py`, `provenance.py`: identity, image verification, typed settings, atomic output.
- `experiment_design.py`: deterministic image subsets with overlap/dependence metadata.
- `metashape_runner.py`, `reconstruction_metrics.py`: isolated licensed API, fresh projects, verified checkpoints, stage snapshots, raw exports, `cameras.csv`.
- `resource_monitor.py`: the watchdog that runs each reconstruction.
- `evaluation.py`: geometry evaluator v3 (mesh loading, sampling, registration, labels).
- `cli.py`: the `manifest` and `reconstruct` commands.
- `scripts/`: manifests, driver, scoring, models and figures ([index](../scripts/README.md)).
- `tests/`: provenance, corruption, reconstruction mocks, watchdog, driver, v3 scorer and label tables.
- `config/`: fully specified reconstruction profiles.
- `archive/`: earlier-phase package (`bd3d_legacy`), its tests, examples and documents.

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

Read [audit](AUDIT.md), [validation record](VALIDATION.md),
[feature definitions](../archive/docs/FEATURE_DICTIONARY.md), [attribution](ATTRIBUTION.md),
and [future statistical/ML work](../FUTURE_ML.md). Optional plots and mesh
postprocessing are deliberately outside the initial implementation. Never commit
private photos, archaeological geometry, credentials, environments or generated
research results. Publication instructions are in [GitHub setup](../archive/docs/PUBLISHING.md).

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

`archive/examples/prepare_low_resolution.py` creates lossless, EXIF-oriented derivatives
and a source-hash/derivation ledger. `experiment_design.py` defines deterministic
subsets with explicit overlap/dependence metadata. `archive/examples/run_bounded_pilot.py`
requires a successful monitored baseline, records a measured-cost budget before
execution, and runs at most three conditions sequentially. Geometry export checks
use `archive/examples/validate_exports.py` in an interpreter with Open3D installed; Open3D
is not a package dependency. The current pipeline driver is `scripts/run_watched.py`.

See [controlled experiment protocol](../archive/docs/CONTROLLED_EXPERIMENTS.md) for settings,
selection strategies, cancellation rules, pre-execution budgets and scientific
limitations. The original 38-image historical subset remains unresolved; newly
sampled 38-image conditions are explicitly distinct.

