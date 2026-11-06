# Future statistical and model research — not implemented

Available outputs: condition identity, dataset/object ID, ordered image hashes,
explicit capture descriptions, reconstruction settings, software, input image
statistics, CPU ORB keypoint/pair counts, stage-specific alignment/tie-point/dense/
mesh measurements, residual RMSE when exposed, geometry registration diagnostics,
reference-relative agreement, status/failure records, JSON summaries and CSV tables.

Before modeling, collect multiple objects and independent acquisition sessions;
retain repeat runs, selection-family metadata and random seeds. Add independently
validated physical reference geometry where feasible. Review missingness, failed
runs and registration QC explicitly. Decide what “quality” means and whether it is
reference agreement, completeness, internal consistency or verified physical accuracy.

Stage availability matters: output geometry and late-stage diagnostics must not
leak into models intended for pre-reconstruction decisions. Group future validation
by physical object/session and related subsets, rather than randomly splitting
rows that share images. Repeated runs do not increase independent object count.
Measure sampling noise, reconstruction repeatability and reference uncertainty.

A future extension may support repeated-run tables, formal descriptive analysis,
confidence intervals, and carefully validated models under a separate user-led
research stage. This repository currently contains no predictive modeling,
training, inference, PyTorch, TensorFlow, scikit-learn or XGBoost dependency.
