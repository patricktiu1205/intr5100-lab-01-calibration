# Lab I workflow

Use the shared `intr5100` Conda environment. The current report and original numerical evidence are already included; only the report builder and read-only tests are needed for normal use.

| Purpose | Entry point | Effect |
| --- | --- | --- |
| Rebuild current report | `src/build_report.py` | Reads saved records; writes report files and build checks; compiles PDF |
| Read-only repository checks | `tests/test_repository.py` | Checks data identity and portable loading without fitting/simulation |
| Prepare human-driving inputs | `src/preprocess_stage1.py` through `preprocess_stage4.py` | Writes preprocessing artifacts |
| Historical calibration | `src/calibration/m1_instant_ls.py`, `m2_neldermead.py`, `m3_ga.py`, `m4_pso.py`, `m5_differentiable.py` | Runs the historical method implementations and writes fits |
| Historical cross-validation | `src/calibration/cross_validate.py` | Runs model evaluations and writes summaries |
| Recorded validation replay | `src/validate_historical_model.py` | Replays legacy `final_params.json` entries (currently run_1 only); retains historical sign/unit conventions |
| Group comparison | `src/analysis/analyze_group_validation.py` | Recomputes descriptive group analysis and figures |
| TGSIM pipeline | `src/tgsim/` | Prepares, fits and analyzes the supplied pairs |
| Historical HTML | `src/analysis/build_historical_report.py` | Generates a separately named historical analysis in `results/` |
| Historical review reproduction | `src/verify_historical_analysis.py` | Reproduces sign findings and fixed-parameter diagnostic trajectories; no optimization |

The fixed results and reviewed report do not require any fitting or replay script to be run again. The offline IDM and original Lab TTC sign limitations remain documented. A sign-only replay with old parameters is a diagnostic, not a corrected optimum or a new physical experiment.

`data/raw/tgsim/` contains the original `HV-HV.csv` and `HV-AV.csv`. Processed event manifests and saved outputs retain their original source-path metadata for provenance; `review/organization_manifest.json` resolves renamed paths. Older process notes in `archive/development/` describe the historical workflow and may contain Windows paths or superseded claims.

For a write-free readiness check of the legacy validation tool, use `python src/validate_historical_model.py --dry-run --run run_1`. Its `final_params.json` contains only that older run. The selected run_2 parameters remain in `results/m5_params.json`, and the current report builder reads those directly. No parameter file is rewritten to make an old pipeline appear compatible with a different run.
