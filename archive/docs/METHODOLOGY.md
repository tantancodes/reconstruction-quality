# Research methodology

The experimental condition is a particular ordered image selection and explicit
reconstruction configuration for an identified dataset/object. The execution is a
UUID-tagged run of that condition. Record object material, lighting, acquisition
geometry, camera/lens, selection method, exclusions and randomization in the dataset
description and accompanying capture notes. Condition IDs remain portable across
machines because roots and execution times are excluded from the hash.

Before execution, verify every selected path, byte count, file hash and image
readability. Store the source manifest beside the run. Create a new project and
retain initial alignment metrics, then dense and mesh metrics when requested.
Snapshots are separate immutable files, never a single mutable sparse-cloud slot.
Future refinement must introduce pre- and post-refinement snapshots before it is
allowed to change the sparse cloud; this version performs no refinement.

Alignment success requires the declared camera count, at least two aligned cameras,
at least 80% alignment and at least one valid tie point by default. These are
operational checks, not scientific evidence of an adequate reconstruction. Set
stricter criteria for a study and record them in the condition configuration.
Dense and mesh stages additionally require nonempty points/faces. Inspect visual
coverage, camera geometry and residual distributions independently.

Image features describe EXIF-oriented, RGB-to-L intensity at a specified analysis
resolution; these are not calibrated radiance or camera-resolution-invariant blur
measurements. ORB ratio-match counts are one-way descriptor matches without
geometric verification. They do not estimate physical overlap or recover camera
poses. Pair extraction is quadratic in image count.

Geometry comparisons require an explicitly identified reference and common units.
The algorithm uses area-weighted deterministic triangle-surface samples and rigid
point-to-point ICP, initialized by centroids. Registration samples and evaluation
samples use separate random streams. Both meshes share the seed within a stream:
identical topology self-comparisons have paired samples and nearly zero error.
Reordered topology does not retain this benefit. These nearest-sample distances
approximate surfaces; sampling density creates a noise floor. Repeat evaluations
at multiple sample counts/seeds to assess stability before interpreting small effects.

RMS surface radius ratio is invariant to rigid pose, but responds to partial
coverage and sampling. It is a scale/coverage warning, not a scale estimator.
Absolute disagreement >25% invalidates comparison by default. Fitness >=0.8 is
only an operational gate: local ICP can align a wrong symmetric part with high
fitness. Reaching the iteration limit also invalidates comparison. Inspect the
recorded transformation and geometry visually. There is no global registration,
nonrigid deformation, scale fitting, or silent mesh repair.

Candidate-to-reference distances measure proximity/agreement; reference-to-candidate
threshold coverage measures completeness relative to this reference. Neither is
true reconstruction accuracy without independent validated ground truth and scale
calibration. A Python API caller may label a reference validated ground truth only
with explicit evidence; this is a recorded researcher assertion, not independent
verification by software. The CLI defaults to reconstruction reference agreement.

For later analysis, report missingness and failed runs instead of selecting only
successful outputs. Keep capture conditions, execution profiles and references
stratified. Nested image subsets share inputs and are dependent observations;
repeated runs measure computational variation, not new archaeological objects.
Plan multiple objects, repeated acquisitions and preregistered subset selection.
Do not interpret one-object correlations causally. Preserve all runs but select
one run per condition explicitly when building this initial condition-level table.
