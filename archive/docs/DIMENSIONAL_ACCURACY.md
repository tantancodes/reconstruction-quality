# Preserved extension: independently measured dimensions and surfaces

This extension is available for future physical-accuracy research. The current
primary study predicts historical Metashape reprojection RMSE, an alignment
consistency measurement. Do not merge those targets or reinterpret reprojection
error as millimeter accuracy. No real physical-measurement observations have
been added by this extension.

## Ground-truth protocol

Assign object, acquisition session, subset and reconstruction run identifiers
before collecting labels. Record each physical distance with a unique measurement
ID and physical measurand ID, explicit landmark/face definitions, value and unit
(mm/cm/m), instrument ID/description, uncertainty method and date/operator/evidence
provenance. Repeated readings of one distance must be summarized with uncertainty,
not counted as separate independent checks. Software validates the supplied
records, not the authenticity of physical measurements or calibration certificates.

Reserve calibration distances and independent check distances **before** inspection
of reconstruction error. Calibration and check IDs/measurands/endpoint pairs may
not overlap. A scale bar can be a separate identified object in the scene. Use
certified bar lengths, documented control distances or an independently verified
metric coordinate system. Arbitrary OBJ/PLY coordinates are not presumed meters.
The scale is either derived from one verified calibration distance or supplied
with independent metric-scale evidence; it is never fitted to check distances.

For a box, use defined face/landmark distances for length, width and height. For a
cylinder, specify the cross-section and opposing points or fitted section used for
diameter, and the exact end planes for height. For archaeological objects, use
stable, clearly photographed landmark pairs that can be selected consistently in
the reconstruction. Matching arbitrary mesh endpoints to physical endpoints is
**manual, reviewed work**, not automatically solved by these utilities. Annotated
3D point pairs or explicitly measured reconstruction distances are accepted.

Use independent calibrated measurements and annotations for checks. Record
instrument resolution, repeatability, zero/calibration evidence, contact/landmark
ambiguity, environmental conditions and operator decisions. Expanded uncertainty
must include its coverage factor; standard uncertainty is not the same as an
unqualified ± bound. First-order propagation is provided only with an explicit
independence assumption and justification. Shared instrument bias, scale uncertainty
and shared endpoints can correlate measurements; aggregate confidence intervals are
not invented. See [NIST uncertainty guidance](https://www.nist.gov/pml/nist-technical-note-1297).

## API, schema and examples

`dimensional_accuracy.evaluate_dimensions(bundle, minimum_checks=2,
require_uncertainty=True)` consumes the structure illustrated in
`archive/examples/ground_truth/synthetic-box-dimensions.json`. Its top-level fields are
`schema_version=1`, `kind=dimensional_evaluation_input`, `data_origin`, object/session/
subset/run IDs, `ground_truth` and `reconstruction`. The ground-truth container has
an independent-physical-reference ID and physical measurement list. Each
reconstruction annotation identifies the matching measurement/object/landmarks,
coordinate units, method, value or two points, uncertainty, QC and provenance.
The reconstruction also declares geometry SHA-256, scale method/status/evidence
and uncertainty assumptions. Native Python validation enforces cross-field rules;
these examples are not claims that any actual object was measured.

The returned per-check signed residual is reconstructed minus measured distance.
Absolute error is reported in mm. Relative absolute error is
`100 * abs(reconstructed_mm - measured_mm) / measured_mm`. Mean absolute relative
error averages those percentages; the primary dimensional target is
`100 * sqrt(mean(((reconstructed_mm - measured_mm) / measured_mm)**2))`.
Calibration distances are excluded. All predeclared checks must be verified and
available, the scale must be verified, the minimum count must pass, and uncertainty
must be complete by default. Otherwise the primary label is null and blockers are
reported; partial numeric errors remain diagnostics. The default two-check minimum
is an operational gate, not proof of adequate object coverage or statistical power.

```bash
bd3d-quality dimensions evaluate archive/examples/ground_truth/synthetic-box-dimensions.json \
  --output results/synthetic-dimensional-fresh.json
bd3d-quality measured-dataset archive/examples/ground_truth/synthetic-measured-experiment.json \
  --allow-synthetic --output results/synthetic-measured-fresh
```

All example values and hash placeholders are explicitly synthetic_test, not real
observations. Default measured-data export rejects synthetic records, mixed real/
synthetic datasets, final-holdout records, reserved object IDs, extra/leaky features,
identity mismatches and incompatible settings. `--reserved-object-id` protects a
known final-test artifact even if someone mistakenly marks its record development.
The measured-experiment schema is version 2 and separate from archival tables.
It links ordered image hashes, exactly the seven pre-reconstruction predictors,
feature provenance/signatures, reconstruction configuration and embedded dimensional
input. Exports contain CSV/JSON, float32 arrays, missing masks, evaluations,
provenance and file hashes. No preprocessing or model fitting occurs. The existing
archival `X_pre`/`y_reference` schema is unchanged.

## Optional independent surface reference

`surface_accuracy.compare_surface(candidate_path, reference_path, spec)` is exposed
as `bd3d-quality surface compare CANDIDATE REFERENCE --specification SPEC.json
--output REPORT.json`. Its spec records separate candidate/reference SHA-256,
matching explicitly verified metric coordinate units, scale evidence and provenance,
a tolerance in mm, and a reference validation record. Reference types are
`independent_physical_ground_truth`, `independent_reference_scan`, and
`photogrammetry_reference`. Distances alone cannot establish full surface ground
truth. A scan is not automatically ground truth: independent acquisition,
validation evidence and uncertainty are required for independent-target eligibility.

The wrapper reuses the existing geometry sampler/rigid ICP, records transformations
and QC, and reports pooled symmetric sampled surface RMSE, median, p95, reference
completeness and candidate coverage. There is no scale optimization or deformation.
These approximate nearest-sample rather than exact surface distances. A separate
photogrammetry reference always remains reference-relative agreement. No new real
surface evaluation is performed by these documentation/tests.

## Research worksheet and future evaluation

Before photography, record object/session IDs, material/texture/shape, instrument
and calibration evidence, landmark photographs/definitions, physical readings and
uncertainty, roles (calibration/check), operators and dates. Then record camera/lens,
exposure, lighting, image count/coverage, subset selection and image hashes. After
separately authorized reconstruction, record run/settings/software, raw geometry
hashes, annotated check distances, scale evidence and independent QC.

A feasibility pilot should establish repeatable measurement and annotation methods
across several different objects and separate acquisition sessions. Vary one input
condition at a time while holding reconstruction profile and measurement panel
constant. A larger study should be sized from observed label variability, coverage,
object-held-out learning curves and uncertainty, never from an arbitrary number of
correlated rows. The coffin is only a potential held-out object: recover its exact
ID, component, dimension definitions, instrument, units, uncertainty, dates,
calibration/check roles, source photographs and matching run before using it. Keep
all of its data outside training/tuning if designated final test.

Future model comparisons may include a training-set mean, Ridge and a user-built
PyTorch MLP, using object-grouped validation, MAE/RMSE, prediction/residual plots
and frozen unseen-object evaluation. Fit imputation, scaling, selection and tuning
only within training folds. Assess photo-set selection using within-object paired
error/ranking regret against more-images and simple image-quality heuristics. This
is a plan only: no models, performance claims or PyTorch implementation exist.

Current blockers: no verified physical measurement records, no independent scans,
no manually reviewed correspondences, incomplete coffin provenance, one-object
archival dependence and resource-blocked new reconstruction. Those are retained
future-extension limitations; the historical alignment project does not require
new reconstruction or physical ground truth.
