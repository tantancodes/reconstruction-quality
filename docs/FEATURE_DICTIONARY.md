# Feature dictionary (version 1)

Image measurements use Pillow EXIF transpose, RGB conversion then 8-bit `L` grayscale
(approximately 0.299 R + 0.587 G + 0.114 B). Lanczos thumbnails preserve aspect ratio
and never upscale; default maximum dimension is 1600 pixels. Definitions differ
from Aphoma's OpenCV decoding and resizing; these feature versions must not be pooled.

| Fields | Calculation / units |
| --- | --- |
| `width_px`, `height_px`, `aspect_ratio` | EXIF-oriented original dimensions; width/height |
| `size_bytes`, `sha256` | File bytes and streaming SHA-256 of original input |
| `analysis_width_px`, `analysis_height_px` | Actual analysis raster dimensions |
| `brightness_mean` | Arithmetic mean of grayscale intensity, 0–255 |
| `contrast_std` | Population standard deviation of grayscale intensity |
| `gray_p05`, `gray_median`, `gray_p95` | NumPy linear intensity percentiles |
| `contrast_p95_p05` | 95th minus 5th intensity percentile |
| `laplacian_variance` | Population variance of four-neighbor discrete Laplacian over interior pixels; intensity squared; null for <3 pixels on an axis |
| `entropy_bits` | Shannon entropy of 256-bin grayscale histogram, base 2 |
| `orb_keypoint_count` | OpenCV ORB detections, default cap 5000 at analysis resolution |
| `ratio_match_count_a_to_b` | One-way Hamming k=2 matches with d1 < 0.75*d2; no geometry verification; no descriptors gives measured zero |

Feature output stores settings, signature and software. Summary `image_*_mean`
columns are arithmetic means over nonmissing image values; `*_measured_count`
records the contributing sample count. Pair means exclude no pairs: a validated
complete pair set is required. A single-image experiment has no pair mean (null).

| Reconstruction fields | Meaning |
| --- | --- |
| `input_image_count`, `camera_count` | Manifest images, loaded Metashape cameras |
| `aligned_camera_count`, `alignment_fraction` | Non-null camera transform count; divided by manifest input count |
| `tie_point_count`, `valid_tie_point_count` | All API tie points; subset with `.valid` |
| `dense_point_count` | API `point_cloud.point_count`, null before construction |
| `mesh_vertex_count`, `mesh_face_count` | Raw model vertices/faces, null before construction |
| `reprojection_rmse_px` | sqrt(mean(norm(camera.error(point.coord, projection.coord))²)) over valid aligned-camera tie-point observations |
| `reprojection_observation_count` | Number of eligible residual observations, null if API unavailable |
| `runtime_seconds` | Wall time for that reconstruction stage, excludes checkpoint serialization |
| `size_bytes` on artifacts | Actual exported file size, not estimated model size |

Every reconstruction snapshot has `stage`, UTC timestamp, metrics and
`missing_reasons`. JSON stage files are immutable. Consolidation uses
`initial_alignment__`, `dense_reconstruction__`, `mesh_generation__` prefixes;
missing reasons remain embedded in summary source records.

| Geometry fields | Meaning |
| --- | --- |
| `scale_radius_ratio` | Candidate/reference area-sampled centered RMS radius; dimensionless |
| registration `fitness` | Fraction of registration candidate samples within correspondence_fraction * reference bounding diagonal |
| `inlier_rmse` | RMS nearest-reference-sample distance of accepted registration points, common coordinate units |
| `transform_candidate_to_reference` | 4x4 rigid transform, column-vector convention; row arrays use R transpose |
| directional `mean`, `median`, `p95`, `max`, `rmse` | Nearest-sample distances, common units |
| `chamfer_mean_squared` | (mean(candidate→reference d²)+mean(reference→candidate d²))/2; units squared |
| `candidate_coverage_fraction` | Candidate samples within threshold distance of reference |
| `reference_completeness_fraction` | Reference samples within threshold distance of candidate |
| `threshold_distance` | threshold_fraction * reference bounding diagonal, common units |

Invalid geometry comparisons have `metrics: null` and explicit flags. `status`,
identity fields, configuration/feature/software signatures, reference hashes and
run IDs are metadata, not numeric scientific features. CSV blank = missing, never
zero. Each generated report includes a field index pointing to these definitions.

Resource protocol additions: `operation_seconds` separates depth-map generation,
point-cloud generation, export, and mesh generation. `parent_checkpoint` and
`continuation_of` identify copied/inherited stages; inherited alignment times
must not be counted as processing repeated in the continuation. Resource traces
record system pressure levels, system-wide swap used, sampled process RSS, sampled
CPU, wall elapsed time and output bytes. RSS is not peak physical memory or total
process footprint; sampled maxima are lower bounds on instantaneous peaks.
