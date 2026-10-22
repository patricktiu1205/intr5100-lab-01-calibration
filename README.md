# INTR 5100 Lab 01 Calibration

Car-following model calibration using human-driving records, two physical validation tests, five-group comparisons and the supplied TGSIM pairs. This repository contains the local course files, organized for standalone use.

## Current report

- [Report PDF](report/lab_01_report.pdf): seven pages, compiled and visually checked.
- [HTML version](report/lab_01_report.html) and [LaTeX source](report/source/main.tex).
- [Review findings](review/review_notes.md) and [lab instructions](docs/lab_01_guidelines.pdf).

The completed experimental records and saved fits are fixed. The historical offline model has a documented relative-speed sign discrepancy, and the original TTC calculation has a closing-sign error. The report labels fitted outputs as historical, uses traceable log summaries and states the limits on physical validation. The retained code is a reproduction record; it does not represent a corrected IDM calibration.

## Repository layout

| Folder | Contents |
| --- | --- |
| `data/raw/` | Human-driving collections and original TGSIM inputs |
| `data/preprocess/` | Retained four-stage preprocessing outputs |
| `data/validation/` | The two Group 4 physical-test records |
| `data/group_validation_raw/` | Five supplied group validation datasets |
| `data/tgsim/` | Retained TGSIM event splits and manifest |
| `src/calibration/` | Historical IDM core and five method implementations, plus cross-validation |
| `src/analysis/` | Historical metrics, group comparison and historical HTML generation |
| `src/tgsim/` | TGSIM preprocessing, fitting, analysis and plotting tools |
| `src/` | Stage tools, report builder and historical-review checker |
| `results/` | Original parameters, numerical analyses and scientific figures |
| `report/` | Current PDF/HTML, source/template/figures and older figure assets |
| `environment/` | Local copies of the supplied reference IDM and route map |
| `docs/` | Lab instructions and workflow guide |
| `review/` | Review findings, historical checks, organization and publication records |
| `tests/` | Read-only source-data and standalone-path checks |
| `archive/` | Earlier collection/results, development briefs and diagnostic logs |

Descriptive filenames use lowercase snake_case. Recorded timestamps, TGSIM pair identifiers, calibration-method IDs and stage/event filenames retain their established identifiers so metadata and source records remain interpretable. `organization_manifest.json` maps moved files. Identical scientific assets may share internal relative symlinks; no link points outside this repository.

## Shared course environment

Use the single Conda environment for all INTR 5100 work:

```sh
conda activate intr5100
```

For a standalone clone, create that same environment if needed:

```sh
conda env create -f environment.yml
conda activate intr5100
```

The repository manifest is a snapshot of the course-root `environment.yml`. In the full course workspace, maintain that root specification and sync this copy before publishing.

## Build the current report

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python src/build_report.py
```

This reads existing records, verifies protected inputs, builds HTML/LaTeX/figures and compiles the PDF with Tectonic. It does not fit parameters or simulate trajectories. Temporary compilation and plotting files are removed automatically. Use `--offline` for cached TeX resources or `--skip-pdf` for editable sources only. Keep the HTML alongside `source/figures/` when copying it.

## Verify the repository

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

The six checks verify the original source-content manifest, collection-registry hashes, eight selected events/1140 samples, TGSIM input loading, recorded validation counts and local asset links. They do not run calibration or model simulation. The report builder additionally verifies protected input hashes and builds descriptive tables directly from saved records.

## Historical tools and limitations

[Workflow guide](docs/workflow.md) identifies the stage, fitting and analysis entry points. Their original algorithms are retained for reproducibility. Running fitting, preprocessing or validation-replay scripts can produce new outputs; none is needed to rebuild the current report.

`src/analysis/build_historical_report.py` writes a separately named historical analysis. The current report is built only by `src/build_report.py`. The optional `src/verify_historical_analysis.py` reproduces the existing sign findings and fixed-parameter diagnostic trajectories; it does not refit a model, and its sign-only diagnostic is not a corrected optimum.

The source data, historical fitted parameters, model equations and deployment records were preserved during repository organization. Submission status is not assumed. Supplied course files retain their original attribution; no new redistribution license is assigned to them.

Naming convention: lowercase snake_case for descriptive files and folders; use `lab_01` / `lab_02` prefixes on lab deliverables and `YYYYMMDD` for document dates. Preserve recorded timestamps, dataset/method identifiers, standard `README.md` / `AGENTS.md` and required deployment filenames. See `review/naming_changes.json` for the rename map.
