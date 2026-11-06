# Initial research audit — 2026-11-06

## Repository boundaries

The new repository at `/Users/tristan/Documents/bd3d-reconstruction-quality` was
empty, initialized on main with no commits, uncommitted files or remotes. No useful
previous agent implementation existed to preserve. No applicable AGENTS.md files
were found in the new project or its ancestor directories.

Aphoma `/Users/tristan/Documents/ChatGPT/bd3d/Aphoma` was clean on main, tracking
origin/main; origin/bd3d-ml was inspected with read-only Git commands. Research
branch commit: `d40904c32fe0f031cdbab980d87dfa024e55a7af`. Main contained the legacy
GUI/task/Metashape processing pipeline; the research branch additionally contained
experiment metadata, profiles, metrics, image/match/geometry analyses, manifests
and methods/results documentation. No dedicated test directory was present in
the branch tree. The geometry validation script and workflow configuration were
reviewed as historical research context, not accepted as tests of the new package.

## Components and decisions

| Component | Assessment and treatment |
| --- | --- |
| `util/ExperimentMetadata.py` | Useful ordered path/hash verification and metadata conflict checks. Reimplemented with full portable condition identity and effective settings. |
| `util/ExecutionProfiles.py` | Useful named/versioned profiles. Replaced with simpler explicit alignment/dense/mesh profiles without implicit scaling, orientation or textures. |
| `util/ReconstructionMetrics.py` | Useful counts and pixel residual methods. Reimplemented without eager Metashape import or legacy Configurator and with immutable stage records. |
| `analysis/image_features.py` | Useful interpretable intensity, entropy, blur and ORB methods. Adapted concepts with documented Pillow preprocessing and a new version, not numerical interchangeability. |
| `analysis/local_match_features.py` | Useful CPU ORB/Hamming ratio matching. Reimplemented explicit one-way complete pair measurements; no assumed acquisition order or EXIF-supported order claim. |
| `analysis/geometry_quality.py` | Already SciPy/NumPy rather than Open3D on this branch. Selectively adapted OBJ/sampling/Kabsch/rigid ICP. Replaced evaluation wrapper with scale/coverage and registration rejection. |
| `analysis/report_dataset.py` | Useful experiment tables. Replaced with explicit source joins, stage columns, null semantics, duplicate rejection and incompatible-table rejection. |
| `photogrammetry/MetashapeTools.py` and task modules | Coupled to configuration, GUI/tasks, scaling/orientation and cleanup. Not imported or copied into the new runner. |
| Existing modeling scripts | Outside requested scope; neither copied nor executed. |

The adjacent `/Users/tristan/Documents/ChatGPT/bd3d` research tree was read-only.
Inspected manifest and research documentation, historical report/table structures,
and available interpreter. No original source, branch, project, image or historical
result was modified. Existing environments were not installed into or altered;
`-B` suppressed bytecode writes during the real Metashape integration checks.

## Historical ARF Dog input audit

The baseline manifest `experiments/manifests/arf_dog_baseline_50.json` declares 50
images at a locally available dataset root and explicitly excludes `_MG_4916.jpg`.
All 50 original SHA-256 entries match current files. The directory contains 51
supported photographs, and the excluded photograph exists. Thus the present 50
baseline input selection and 51-image directory are distinguishable and verified;
this does not prove historical 51-camera reconstruction provenance.

Historical projects were located through report paths at
`/Users/tristan/Documents/BD3D_Test_Output`. Read-only ZIP/XML inspection of 39
project chunks found camera counts: 0 (6 chunks), 25 (4), 30 (8), 35 (4), 40 (4),
45 (1), and 50 (12). The four inspected Test/Instrumented/FullReference project
chunks each declare 50 cameras. No 38- or 51-camera chunk was found in this set.
Zero-camera chunk records are not assumed to prove empty physical input sets.
The specific historical 38-image selection and 51-camera full project remain
unresolved. No substitute selection was invented; historical runs are not silently
migrated or merged into new records. Camera XML counts alone are not hash proof
of a historical project's exact image content.

## Environment and external services

Default Python: 3.14.7 on macOS ARM64. New project-local `.venv` created; exact
non-ML dependency versions recorded in requirements-lock.txt. Global default
Python had NumPy/Pillow/Open3D discoverable, no Metashape or pytest; no global
packages were modified. Open3D was not imported or installed for this work.

Existing read-only `venv312` interpreter: Python 3.12.14 with Metashape 2.2.1.
Initial license probe reported activation false/checkout denied. A subsequent
integration invocation obtained availability, so new real-data CPU alignment
runs became possible. This time-varying licensing state is recorded rather than
claimed as permanently blocked. Generated projects stay entirely in this new
repository's ignored results directory. No original Metashape project was opened
for writing. See VALIDATION.md for actual results and limitations.

GitHub CLI is absent. Connected GitHub tools provide repository access but no
repository-creation tool. No remote/push was attempted through alternative
credentials; the independent local repository is ready for private publication.
