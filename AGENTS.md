# Lab I workspace instructions

- Use the shared Conda environment `intr5100` for all Python work. Do not create a per-lab environment.
- Experimental records and historical fitted results are fixed. Preserve their file bytes, provenance and calibration/deployment distinction. Do not silently correct the historical IDM/TTC implementations or publish a new fit as the one originally deployed.
- The relative-speed and TTC sign discrepancies are documented in `review/review_notes.md`. Mathematical changes need a separate explicitly requested analysis and clearly distinguished outputs.
- Keep final reports in `report/`, report sources/figures in `report/source/`, instructions in `docs/`, review evidence in `review/`, and older development material in `archive/`.
- Use lowercase snake_case for descriptive names. Preserve recorded timestamps, data schema filenames, TGSIM pair identifiers and method IDs required by metadata.
- A standalone clone must work without a sibling course folder. Use local copies of supplied reference files in `environment/`; keep deduplication links inside the repository.
- Keep the concise Lab I report style: A4, 20-mm margins, readable 11-point serif text, black headings and clean tables/figures.
- The course-root `environment.yml` is authoritative in the full workspace. Sync this repository's copy before publishing.
- Remove task-created temporary scripts, previews, caches and build byproducts. Preserve source code, raw data, results, diagnostics and reproducibility records.

Naming convention: lowercase snake_case for descriptive files and folders; use `lab_01` / `lab_02` prefixes on lab deliverables and `YYYYMMDD` for document dates. Preserve recorded timestamps, dataset/method identifiers, standard `README.md` / `AGENTS.md` and required deployment filenames. See `review/naming_changes.json` for the rename map.
