#!/usr/bin/env python3
"""INTR5100 Lab Session I - Stage 1 data preprocessing.

Column trimming only: reads the raw `straight_trajectory.csv` of every discovered
recording session and keeps the trajectory columns needed for IDM calibration.

    run, role, t_from_start, x, y, heading, velocity

`role` is kept as-is (follower / leader): splitting per role happens in a
later stage, not here. `velocity` is the vehicle-reported reference speed and
is copied verbatim even though it differs from the position-difference speed
by roughly a factor of 3 - the unit mismatch is handled downstream.

--- run labelling -------------------------------------------------------------

A "run" is one recording session of the one human participant: the raw
collections are the sessions of that same person, so the pipeline works per run
and never mixes two of them. Every ``data/raw/collection_*`` directory is one
run and nothing is hard-coded:

  * the collections are discovered by globbing ``data/raw/collection_*`` and are
    ordered by directory name - the name carries the collection timestamp, so
    that is the chronological order;
  * the i-th collection becomes ``run_i``; a collection that already carries a
    label keeps it, and a newly dropped-in collection gets the next free number
    (numbers are never reused, so a label always means the same recording);
  * the mapping run label <-> collection directory is written to
    ``data/run_registry.json`` (together with the sha256 of the source file), so
    the numbering is traceable and stable across pipeline runs.

A collection that is registered but no longer present under ``data/raw`` is kept
in the registry under ``retired_runs`` (its number stays taken) and a directory
without a ``straight_trajectory.csv`` is listed under ``skipped_collections``
instead of being silently folded into a run label.

Outputs (one file per run) into ``data/preprocess/stage_1/``:

    run_1.csv, run_2.csv, ...

Both files are ordered by role group - the whole follower block first, then the
whole leader block - and ascending by ``t_from_start`` inside each block; rows
of the two roles are never interleaved. Outputs of the earlier revision of this
stage (the per-driver files ``driver_1.csv`` and friends) are removed if
present, so the directory holds exactly the current run files.

The script is idempotent (same input -> byte-identical output), uses only the
standard library (no pandas), and resolves every path relative to the project
root so it can be run from any working directory.

Usage:
    python src/preprocess_stage1.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# --- configuration ---------------------------------------------------------

RAW_SUBDIR = Path("data") / "raw"
COLLECTION_GLOB = "collection_*"
RAW_FILENAME = "straight_trajectory.csv"
OUTPUT_SUBDIR = Path("data") / "preprocess" / "stage_1"
REGISTRY_PATH = Path("data") / "run_registry.json"
REGISTRY_SCHEMA = "intr5100-lab1-run-registry"

# A run label is ``run_<n>`` with a positive ordinal; the ordinal is the run's
# position in the chronological order of the collections.
RUN_LABEL_RE = re.compile(r"^run_(?P<ordinal>\d+)$")

# Accepted `role` values; anything else aborts the run (never silently dropped).
ROLES = ("follower", "leader")

# Two-level output order: role group first (follower block, then leader block),
# `t_from_start` ascending inside each block.
ROLE_GROUP_ORDER = {"follower": 0, "leader": 1}

# Output column order: role/trajectory fields, prefixed with the run label.
OUTPUT_FIELDS = ("run", "role", "t_from_start", "x", "y", "heading", "velocity")

# Columns that must exist in the raw file.
REQUIRED_RAW_FIELDS = ("role", "t_from_start", "x", "y", "heading", "velocity")

# Superseded outputs of earlier revisions of this stage (the per-driver files of
# the two-collection layout and the role-split files before that). Listed
# explicitly (no globbing) so removal can never touch anything else.
LEGACY_OUTPUTS = (
    "driver_1.csv",
    "driver_2.csv",
    "driver_1_follower.csv",
    "driver_1_leader.csv",
    "driver_2_follower.csv",
    "driver_2_leader.csv",
)

# Written explicitly so output bytes do not depend on the host platform.
LINE_TERMINATOR = "\r\n"


def project_root() -> Path:
    """Project root = parent of the directory containing this script."""
    return Path(__file__).resolve().parents[1]


def rel(path: Path, root: Path) -> str:
    """Project-relative POSIX path (no absolute path in the registry)."""
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def sha256_file(path: Path) -> str:
    """Content hash of one file, recorded as the run's provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- run discovery ---------------------------------------------------------


def discover_collections(raw_dir: Path) -> list[str]:
    """Names of the ``collection_*`` directories under ``data/raw``, in name order.

    The directory name ends with the collection timestamp, so ascending name
    order is the chronological order of the sessions.
    """
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw data directory not found: {raw_dir}")
    return sorted(entry.name for entry in raw_dir.iterdir() if entry.is_dir() and entry.match(COLLECTION_GLOB))


def load_registry(path: Path) -> dict[str, object]:
    """Read ``data/run_registry.json`` (an empty skeleton when it does not exist)."""
    if not path.is_file():
        return {"runs": {}, "retired_runs": {}}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read the run registry {path}: {exc}") from None
    if not isinstance(payload, dict):
        raise ValueError(f"the run registry {path} is not a JSON object")
    runs = payload.get("runs") if isinstance(payload.get("runs"), dict) else {}
    retired = payload.get("retired_runs") if isinstance(payload.get("retired_runs"), dict) else {}
    return {"runs": dict(runs), "retired_runs": dict(retired)}


def run_ordinal(label: str) -> int:
    """Numeric part of a ``run_<n>`` label."""
    match = RUN_LABEL_RE.match(label)
    if not match:
        raise ValueError(f"not a run label: {label!r}")
    return int(match.group("ordinal"))


def assign_runs(
    collections: list[str], registry: dict[str, object]
) -> tuple[dict[str, str], dict[str, str], list[dict[str, str]], list[str]]:
    """Map every discovered collection to a run label, keeping existing labels.

    Returns ``(run_to_collection, collection_to_run, retired, skipped)``:

      * ``collections`` is the discovery order (name = chronological order);
      * a collection that already has a label in the registry keeps it, a new one
        is given the next free ``run_<n>`` in discovery order;
      * a registered collection that is not present any more is reported as
        retired (its number is not handed out again);
      * a collection without the raw file is reported as skipped.

    The function only decides labels - reading and writing is left to ``main``.
    """
    previous: dict[str, str] = {
        str(label): str(entry.get("collection"))
        for label, entry in dict(registry["runs"]).items()  # type: ignore[arg-type]
        if isinstance(entry, dict)
    }
    label_by_collection = {collection: label for label, collection in previous.items()}
    highest = max(
        [run_ordinal(label) for label in previous]
        + [run_ordinal(label) for label in dict(registry["retired_runs"])],  # type: ignore[arg-type]
        default=0,
    )

    run_to_collection: dict[str, str] = {}
    retired: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for collection in collections:
        label = label_by_collection.get(collection)
        if label is None:
            highest += 1
            label = f"run_{highest}"
        run_to_collection[label] = collection

    for label, collection in sorted(previous.items(), key=lambda item: run_ordinal(item[0])):
        if collection not in collections:
            retired.append(
                {
                    "run": label,
                    "collection": collection,
                    "reason": "collection directory is not present under data/raw",
                }
            )
    for label in run_to_collection:
        if run_ordinal(label) > highest:
            raise RuntimeError(f"internal error: {label} exceeds the highest assigned ordinal")
    return run_to_collection, {value: key for key, value in run_to_collection.items()}, retired, skipped


def gate_missing_raw(
    run_to_collection: dict[str, str], raw_dir: Path, skipped: list[dict[str, str]]
) -> dict[str, str]:
    """Drop the runs whose collection has no raw file (reported, never guessed)."""
    usable: dict[str, str] = {}
    for label, collection in sorted(run_to_collection.items(), key=lambda item: run_ordinal(item[0])):
        if (raw_dir / collection / RAW_FILENAME).is_file():
            usable[label] = collection
        else:
            skipped.append(
                {
                    "run": label,
                    "collection": collection,
                    "reason": f"no {RAW_FILENAME} in the collection directory",
                }
            )
    return usable


# --- reading and projection ------------------------------------------------


def read_raw(path: Path) -> list[dict[str, str]]:
    """Read a raw trajectory CSV verbatim (no numeric reformatting)."""
    if not path.is_file():
        raise FileNotFoundError(f"raw trajectory file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_RAW_FIELDS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing required column(s): {missing}")
        return [row for row in reader if any((v or "").strip() for v in row.values())]


def count_roles(rows: list[dict[str, str]], source: Path) -> dict[str, int]:
    """Count rows per role; unknown roles are reported (never silently dropped)."""
    counts: dict[str, int] = {role: 0 for role in ROLES}
    unknown: dict[str, int] = {}
    for row in rows:
        role = (row.get("role") or "").strip()
        if role in counts:
            counts[role] += 1
        else:
            unknown[role] = unknown.get(role, 0) + 1
    if unknown:
        raise ValueError(f"{source}: unexpected role value(s): {unknown}")
    return counts


def role_group_order(role: str) -> int:
    """Group rank of a role: follower=0, leader=1; anything else is an error."""
    try:
        return ROLE_GROUP_ORDER[role.strip()]
    except KeyError:
        raise ValueError(f"unexpected role value for ordering: {role!r}") from None


def to_output_rows(rows: list[dict[str, str]], run: str) -> list[dict[str, str]]:
    """Project onto OUTPUT_FIELDS and order by (role group, t_from_start)."""
    projected = [
        {
            "run": run,
            "role": row["role"],
            "t_from_start": row["t_from_start"],
            "x": row["x"],
            "y": row["y"],
            "heading": row["heading"],
            "velocity": row["velocity"],
        }
        for row in rows
    ]
    # Values stay verbatim strings; float() is used only as the sort key.
    # Grouping is expressed as a sort key (not by splitting rows out of the file),
    # so each run still yields exactly one output file. Python's sort is
    # stable, so ties inside a group keep their original file order.
    projected.sort(key=lambda r: (role_group_order(r["role"]), float(r["t_from_start"])))
    return projected


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write rows atomically so a failed run cannot leave a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(OUTPUT_FIELDS), lineterminator=LINE_TERMINATOR
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write the registry atomically (fixed formatting -> byte-identical reruns)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    with tmp_path.open("w", newline="", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp_path, path)


def count_data_rows(path: Path) -> int:
    """Re-read a written file and count data rows (header excluded)."""
    with path.open("r", newline="", encoding="utf-8") as fh:
        return sum(1 for _ in csv.reader(fh)) - 1


def read_header(path: Path) -> list[str]:
    """Re-read a written file and return its header field order."""
    with path.open("r", newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


def read_labels(path: Path) -> set[str]:
    """Re-read a written file and return the distinct run labels it carries."""
    with path.open("r", newline="", encoding="utf-8") as fh:
        return {row["run"] for row in csv.DictReader(fh)}


def remove_legacy_outputs(out_dir: Path) -> list[str]:
    """Delete superseded outputs of earlier revisions; return the names removed."""
    removed: list[str] = []
    for name in LEGACY_OUTPUTS:
        stale = out_dir / name
        if stale.is_file():
            stale.unlink()
            removed.append(name)
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="project root (default: directory containing this script's parent)",
    )
    args = parser.parse_args(argv)

    root = (args.root or project_root()).resolve()
    raw_dir = root / RAW_SUBDIR
    out_dir = root / OUTPUT_SUBDIR
    registry_file = root / REGISTRY_PATH

    print(f"project root : {root}")
    print(f"raw dir      : {rel(raw_dir, root)}")
    print(f"output dir   : {rel(out_dir, root)}")
    print(f"run registry : {rel(registry_file, root)}")
    print()

    collections = discover_collections(raw_dir)
    print(f"collections discovered in {rel(raw_dir, root)} ({COLLECTION_GLOB}): {len(collections)}")
    for name in collections:
        print(f"  {name}")
    print()

    registry = load_registry(registry_file)
    run_to_collection, collection_to_run, retired, skipped = assign_runs(collections, registry)
    runs = gate_missing_raw(run_to_collection, raw_dir, skipped)
    if not runs:
        print(f"\nERROR: no usable run found under {rel(raw_dir, root)}", file=sys.stderr)
        return 1

    print(f"runs to process ({len(runs)}), in run order:")
    for label, collection in runs.items():
        print(f"  {label:<8} <- {collection}")
    if retired:
        print("retired run label(s) (collection no longer present, number kept):")
        for entry in retired:
            print(f"  {entry['run']:<8} <- {entry['collection']}  ({entry['reason']})")
    if skipped:
        print("skipped collection(s):")
        for entry in skipped:
            print(f"  {entry['collection']:<32} {entry['reason']}")
    print()

    results: list[tuple[str, int]] = []
    rows_by_run: dict[str, int] = {}
    role_report: list[tuple[str, dict[str, int]]] = []
    labels_seen: dict[str, set[str]] = {}
    for run, collection in runs.items():
        source = raw_dir / collection / RAW_FILENAME
        rows = read_raw(source)
        counts = count_roles(rows, source)
        print(
            f"{run} ({rel(source, root)}): {len(rows)} data rows "
            f"({', '.join(f'{role}={counts[role]}' for role in ROLES)})"
        )
        role_report.append((run, counts))
        out_rows = to_output_rows(rows, run)
        out_path = out_dir / f"{run}.csv"
        write_csv(out_path, out_rows)
        results.append((out_path.name, len(out_rows)))
        rows_by_run[run] = len(out_rows)
        labels_seen[out_path.name] = read_labels(out_path)

    print()
    print("output row counts (data rows, header excluded):")
    width = max(len(name) for name, _ in results)
    total = 0
    for name, count in results:
        total += count
        print(f"  {name:<{width}}  {count}")
    print(f"  {'TOTAL':<{width}}  {total}")

    # Verify what actually landed on disk matches what we intended to write.
    print()
    print("on-disk verification:")
    ok = True
    for name, expected in results:
        actual = count_data_rows(out_dir / name)
        label = name[:-4]
        labels = labels_seen[name]
        status = "OK" if actual == expected and labels == {label} else "MISMATCH"
        if actual != expected or labels != {label}:
            ok = False
        print(
            f"  {name:<{width}}  expected={expected}  on_disk={actual}"
            f"  run column={sorted(labels)}  {status}"
        )

    print()
    print("role counts (from source):")
    for run, counts in role_report:
        detail = ", ".join(f"{role}={counts[role]}" for role in ROLES)
        print(f"  {run}: {detail}, total={sum(counts.values())}")

    print()
    print("header:")
    for name, _ in results:
        print(f"  {name:<{width}}  {','.join(read_header(out_dir / name))}")

    print()
    removed = remove_legacy_outputs(out_dir)
    if removed:
        print(f"removed superseded output(s) of the earlier driver layout: {', '.join(removed)}")
    else:
        print("removed superseded output(s) of the earlier driver layout: none present")
    expected_files = sorted(name for name, _ in results)
    actual_files = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    print(f"stage_1 dir contents: {', '.join(actual_files)}")

    # --- run registry ---------------------------------------------------------
    registry_payload: dict[str, object] = {
        "schema": REGISTRY_SCHEMA,
        "script": "src/preprocess_stage1.py",
        "raw_dir": rel(raw_dir, root),
        "discovery_glob": f"{rel(raw_dir, root)}/{COLLECTION_GLOB}",
        "labelling_rule": (
            "one run per data/raw/collection_* directory, numbered run_1, run_2, ... in "
            "directory-name (chronological) order; an already registered collection keeps "
            "its label and a newly added collection gets the next free number"
        ),
        "raw_filename": RAW_FILENAME,
        "stage1_file": "{run}.csv",
        "runs": {
            label: {
                "collection": collection,
                "source": rel(raw_dir / collection / RAW_FILENAME, root),
                "sha256": sha256_file(raw_dir / collection / RAW_FILENAME),
                "stage1_file": rel(out_dir / f"{label}.csv", root),
                "rows": rows_by_run[label],
            }
            for label, collection in runs.items()
        },
        "collections": {collection: label for label, collection in runs.items()},
        "retired_runs": {entry["run"]: {"collection": entry["collection"], "reason": entry["reason"]} for entry in retired},
        "skipped_collections": skipped,
    }
    write_json(registry_file, registry_payload)
    print()
    print(f"run registry written: {rel(registry_file, root)}")
    print(f"  runs: {', '.join(f'{label} <- {collection}' for label, collection in runs.items())}")
    if retired or skipped:
        print(f"  retired: {[entry['run'] for entry in retired]}  skipped: {[entry['collection'] for entry in skipped]}")

    print()
    tmp_leftovers = sorted(p.name for p in out_dir.glob("*.tmp"))
    print(f".tmp leftovers: {tmp_leftovers if tmp_leftovers else 'none'}")

    if actual_files != expected_files:
        print(f"\nERROR: unexpected stage_1 contents {actual_files} != {expected_files}", file=sys.stderr)
        return 1
    if not ok:
        print("\nERROR: row count or label mismatch between written data and disk contents.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
