# Local generated results

Generated content is intentionally ignored by Git. `synthetic-demo/` is produced by
`python examples/synthetic_workflow.py results/synthetic-demo` using no private data.
It contains image fixtures, manifest, feature/match JSON, standalone self-comparison
geometry, a condition summary and a consolidated report. Select a fresh directory
for repeated invocations. Real research manifests, project files and images must
not be committed. See docs/VALIDATION.md for this implementation's actual runs.

This implementation also generated `arf-dog-50-alignment-v2/` (fresh real Metashape
alignment plus verified resume), `arf-dog-50-features.json`, `arf-dog-50-matches.json`,
`arf-dog-50-summary.json`, and `arf-dog-50-table/experiments.csv`. The first alignment
is retained separately because it exposed transient vendor-lock fingerprinting.
These private research artifacts are ignored, not portable committed fixtures.
