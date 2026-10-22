# Original source data

- `collection_20260917_144342/`: first human-driving collection.
- `collection_20260922_142956/`: adopted second human-driving collection.
- `tgsim/HV-HV.csv`, `tgsim/HV-AV.csv`: original real-world pair inputs.

Human collections retain the supplied trajectory, metadata, command and event filenames. The run registry in `../run_registry.json` maps them to `run_1` and `run_2` and records original source hashes. Physical validation and five-group recordings remain separately organized under `../validation/` and `../group_validation_raw/`.

These files are original evidence. The report builder and repository tests read them without replacement or modification. Older import instructions are retained in `../../archive/development/raw_data_import_instructions.md`.
