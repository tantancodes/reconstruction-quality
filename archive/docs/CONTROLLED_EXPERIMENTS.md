# Controlled experiment protocol (2026-11-06)

The original baseline consists of 50 independently hash-verified images, excluding
_MG_4916.jpg from the available 51-image directory. The verified alignment-v2
checkpoint remains unchanged. The historical 38-image subset is unresolved: no
manifest declaring 38 inputs or 38-camera project was found in the inspected
Aphoma experiment tree and historical BD3D output projects. New 38-image subsets
must be labeled new selections, never used as substitutes for historical provenance.

## Resource protocol

No concurrent reconstruction. Before each process, inspect system pressure and
swap. A macOS watchdog samples every approximately two seconds; it terminates
only its own child process group on critical pressure (level 4), >256 MiB growth
in system-wide swap, process RSS >3 GiB, output >2 GB, or the time limit. Stable
historic swap is recorded, not attributed to Metashape. Sample-based limits can
overshoot between measurements; RSS omits compressed and swapped process memory.
Keep at least 5 GB free disk space. Retain stopped-run ledgers and raw outputs.
No changes to system settings, other applications, or original datasets.

OpenMP requests two workers but Metashape can override that request (observed
10/10 CPU workers). Therefore the watchdog uses an explicit 20% run duty cycle
(80 ms running, 320 ms suspended) on its process group, aiming at roughly two
core-equivalents on a ten-core CPU, with a sustained >300% CPU stop gate. This
limits average activity rather than instantaneous thread count. The wall deadline
includes suspension. Cancellation may leave the last saved checkpoint and vendor
lock; stopped runs require a new directory and are not silently resumed.

A complete original-raster 50-image continuation was attempted with Lowest depth
quality and four neighbors. The watchdog stopped it on system swap growth during
image loading. A 1296-pixel derivative attempt also reached the swap-growth gate.
The final fallback uses 648-pixel lossless PNG derivatives of all 50 photographs,
with a full new alignment. The derivation ledger records the original hashes,
dimensions, EXIF transpose, RGB conversion, Lanczos resizing and metadata removal.
It is a separate experimental condition. Original calibration is not reused at
changed pixel coordinates. The original alignment checkpoint is not overwritten.

Depth downscale 4 on 648x432 derivatives gives approximately 162x108 depth rasters;
this is a functionality smoke/pilot, not publication-quality archaeology geometry.
Four neighbors, fine task subdivision, two cameras per work item, no preview
textures, no trimming and CustomFaceCount=0 (no decimation). Original and derivative
conditions must not be pooled. All processing settings and versions are recorded.

## Matrix and budget, before pilot execution

The machine-readable design contains 7 conditions, with 2 planned executions per
condition (14 total): 50-image baseline; 25- and 38-image conditions, each selected
uniformly or using independent seeded random selections (71 and 72). Uniform
selection uses evenly spaced indices in parent manifest order. Manifest order is
not asserted to be physical acquisition order. Random selections use Python
Random(seed).sample, then restore parent order. Ordered image hashes define the
actual selection; seeds and parent signatures are additional audit metadata.

Only three additional pilot conditions may execute after the baseline succeeds:
25-random-71, 25-random-72 and 38-uniform. Before measurements, budget 30–300 seconds,
0.1–1 GiB process RAM and 1–100 MB outputs per run, all uncertain estimates. Limit
this pilot to 300 seconds each, 900 seconds total and 500 MB additional outputs.
Reconstruction watchdog gates still apply. Record measured baseline cost before
launching the first pilot. Stop the batch immediately on any resource-gate failure.
Do not launch the remaining 10 matrix executions automatically.

## Experimental dependencies and interpretation

All images belong to one object and capture session. Independently sampled subsets
share photographs and are statistically dependent. A rerun of the same condition
is a computational replicate, not an independent training observation. Selection
strategies may be confounded with viewpoint coverage. Use object/session-grouped
future validation, report missing outcomes and retain failed conditions. The pilot
is real research data but cannot support claims of generalization to new artifacts.

Geometric diagnostics cover raw counts, finite coordinates, surface area, bounds,
components, manifoldness and watertightness. Distance metrics compare against an
explicit reconstruction reference, with registration and scale QC. Invalid
comparisons remain invalid, with no fabricated distance targets. Self-comparisons
are evaluator checks, not independent quality labels. Independent reconstructions
may have arbitrary relative scales and orientations; no hidden rescaling is allowed.
Physical accuracy requires independently validated ground truth and scale calibration.
No modeling, PyTorch or neural-network work occurs in this phase.

## Actual execution gate outcome

The 648-pixel condition aligned all 50 images but hit the five-minute wall gate
near the end of depth processing. A verified, separately copied alignment
checkpoint permitted continuation without changing that saved run. The continuation
then hit the swap-growth gate. No additional reconstruction was launched. The
bounded pilot is therefore **not executed**, and its budget projections remain
provisional. The historical three-run audit/feature table is a separate archival
stratum, not replacement pilot evidence. See docs/VALIDATION.md for actual sampled
resource measurements, missing outputs, and the numerical registration sensitivity
check. Newly generated raw dense-cloud/mesh export validation remains a prerequisite.
