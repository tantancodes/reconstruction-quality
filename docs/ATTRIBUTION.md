# Selective reuse

Source: existing Aphoma repository, `origin/bd3d-ml` at
`d40904c32fe0f031cdbab980d87dfa024e55a7af`, inspected read-only on 2026-11-06.
Its LICENSE is CC0 1.0 Universal.

`geometry_quality.py` adapts `TriangleMesh`, OBJ loading, deterministic area-weighted
sampling, Kabsch rigid transform, transform application and point-to-point ICP
from Aphoma `analysis/geometry_quality.py`. The new evaluator replaces the old
report layer, adds input checks, scale/coverage gates, reference semantics,
missing metrics for invalid comparisons, explicit units and settings validation.
The original docstring's use of “accuracy” for reference proximity is not carried
forward as a ground-truth claim. See `APHOMA_CC0.txt` for the source license.

Manifest streaming hash verification and execution profile/stage concepts were
informed by `util/ExperimentMetadata.py`, `util/ExecutionProfiles.py`, and
`util/ReconstructionMetrics.py`. New implementations remove absolute-path identity,
task queue configuration and eager Metashape imports. Image entropy, intensity
statistics and ORB ratio-match methods were informed by `analysis/image_features.py`
and `analysis/local_match_features.py`; the implementation uses explicit new feature
versions and Pillow preprocessing. No numerical compatibility with old features
is claimed. `analysis/report_dataset.py` informed experiment-level reporting;
new code rejects duplicate identities and incompatible records instead of selecting
runs silently. No ML scripts or results were copied or executed.
