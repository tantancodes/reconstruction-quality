# archive/

Code and documents kept for the record. Nothing here is needed to reproduce the README
results, and none of it is imported by `src/bd3d_quality/`.

| path | contents |
|---|---|
| `scripts/` | Superseded scripts: evaluator v2 (`rescore_similarity_icp.py`), the first GPU pilot driver (`run_gpu_pilot.py`), and the unattended 2026-11-10 run (`overnight_chain.py`, `overnight_continue.py`). The overnight scripts ran from `scripts/` and are kept as a record, not to be re-run. |
| `src/bd3d_legacy/` | Earlier-phase modules: historical alignment / reprojection-RMSE inventory and provenance checks, photo provenance, tabular and measured datasets, dimensional and surface accuracy, CPU image and match features, reporting, checkpoint cloning, the rigid `compare_geometry` evaluator (with its own rigid ICP), and the original full CLI (`cli.py`, with the `features`, `geometry compare`, `dimensions`, `measured-dataset`, `surface`, `alignment-report`, `alignment-dataset` and `report` subcommands). They import `bd3d_quality` but are not part of it. |
| `tests/` | Tests for `bd3d_legacy`. They reuse the fixtures in `tests/fixtures.py`. |
| `examples/` | Earlier-phase example scripts and the synthetic, Metashape-free smoke workflow. |
| `docs/` | Earlier-phase documents not linked from the README. |

Run the archived tests and examples from the repository root with `archive/src` on the path:

```bash
python -m pytest -q archive/tests           # conftest adds archive/src
PYTHONPATH=archive/src python archive/examples/synthetic_workflow.py results/synthetic-demo
PYTHONPATH=archive/src python -m bd3d_legacy.cli --help
```
