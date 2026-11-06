# Metashape integration

The adapter targets 2.2.x and records the exact installed version. Imports are lazy.
It follows [Agisoft's 2.2.1 Python API reference](https://www.agisoft.com/pdf/metashape_python_api_2_2_1.pdf)
and the vendor's [general workflow sample](https://github.com/agisoft-llc/metashape-scripts/blob/master/src/samples/general_workflow.py).

Use a Python interpreter supported by the vendor wheel; install this package in
that environment with `python -m pip install -e .` and the licensed Metashape wheel
according to [Agisoft installation instructions](https://agisoft.freshdesk.com/support/solutions/articles/31000148930-how-to-install-metashape-stand-alone-python-module).
Do not install ML extras or bypass licensing. The local 3.14 development environment
does not include Metashape. An existing read-only research interpreter has version
2.2.1; license availability varied during this implementation. See VALIDATION.md.

Set profile in a new manifest configuration: `alignment`, `dense`, or `mesh`.
Dense runs build depth maps then point cloud. Mesh runs additionally build an
arbitrary surface from depth maps with enabled interpolation and HighFaceCount,
then export raw OBJ without texture. The runner explicitly sets matching resolution,
keypoint and tiepoint limits, generic/reference preselection, depth resolution and
filter. Remaining vendor defaults are governed by the recorded API version; compare
runs on the same software build and retain the manifest. CPU-only is the default.
No automatic marker alignment, scale setting, error reduction, filtering of sparse
points, component removal, mesh cleanup, simplification, texture or UV steps occur.

The output directory stores a source manifest, mutable run-status ledger,
`project.psx` / `project.files`, immutable stage JSON and raw tie-point/dense PLY
and/or mesh OBJ. API exceptions produce a failed ledger with the exception type
and message. A successful checkpoint records complete project and artifact hashes.
Resume verifies those hashes plus identity, settings and actual loaded camera paths.
It never treats an existing file as success. Complete runs are verified no-ops;
failed/interrupted runs must start in a new directory. The interpreter's GPU/CPU
settings are restored after execution. Save/copy checkpoints only after processing
stops; modifying project files breaks future verification.

Real dense/mesh processing can take substantial memory/time. Unit mocks exercise
call flow and failure gates, not numerical reconstruction or geometry fidelity.
