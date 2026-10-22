#!/usr/bin/env python3
"""INTR5100 Lab Session I - Stage 2 data preprocessing.

Step 2 of the pipeline: drop the invalid head/tail parts of each run's
trajectory and split what is left into per-straight-line `event`s.

Input  (Stage 1 output, one file per run, follower block first / leader block
        second, ascending `t_from_start` inside each block):

    data/preprocess/stage_1/<run>.csv
    columns: run, role, t_from_start, x, y, heading, velocity

Output (rebuilt from scratch on every run):

    data/preprocess/stage_2/<run>_event_<k>.csv   one file per paired event
    data/preprocess/stage_2/stage2_summary.json      trims/segments/events report

--- step 1: invalid head/tail trimming (per run, per role) ----------------

A sample with `velocity == 0` is a standstill sample. Two kinds of standstill
run are distinguished:

* <= 3 consecutive standstill samples: a dropped reading while driving. It is
  ignored, i.e. kept, and never triggers a trim.
* > 3 consecutive standstill samples: a genuine stop (the vehicle really stood
  still for several samples). Such a run absorbs its trailing launch/brake
  transient - samples that report 0 < velocity <= 2 km/h while the position is
  unchanged - because the vehicle is demonstrably not moving yet.

Head trim: only a genuine stop run that lies inside the *unstarted head* of the
sequence may be the head cut, i.e. one whose standstill samples all precede the
first moving sample of the recording; the last such run is taken and everything
up to and including it is deleted. A genuine stop that happens after the vehicle
has demonstrably been driving is road data, not an unstarted period, and never
triggers a head trim (run_2 / follower: a 4 sample 0 km/h reading at t = 133.6 s
while the position kept changing - treating it as the head cut would have deleted
781 rows / 134 s of driving together with 7 of the run's 13 straight segments).
Tail trim: if a genuine stop run ends the sequence, everything from the last
moving sample onwards is deleted (a parking stretch is not valid driving data).

After the head trim the two vehicles are aligned: all leader rows earlier than
the follower's effective start time (the time of the follower's head-cut
boundary sample) are deleted as well. A role that did not trigger a trim itself
is left untouched, and a follower without a head trim triggers no alignment.
run_1 is the case where nothing at all is trimmed.

--- step 2: gap-based event splitting ----------------------------------------

Inside a role's sequence a gap of dt > 1.0 s starts a new segment. Normal
sampling is 0.1-0.2 s apart; the turns produce gaps of 6-8 s, so each continuous
straight-line stretch becomes exactly one segment.

Segments are numbered in time order (`event_1`, `event_2`, ...) and the follower
and leader segments are paired by ordinal. Time-overlap pairing is wrong: the
follower is on the same stretch of road as the leader but passes it at a
different moment (run_1's first event: follower t=[4.63, 6.42], leader
t=[-0.05, 0.35] - no overlap at all). Segments a role has in excess stay
unmatched and are listed in the summary.

The script is idempotent (same input -> byte-identical output; no wall-clock
timestamp is ever written), uses only the standard library, writes every file
atomically, resolves paths relative to the project root so it can be run from any
working directory, and re-reads what it wrote to verify it.

Usage:
    python src/preprocess_stage2.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

# --- configuration ---------------------------------------------------------

STAGE1_SUBDIR = Path("data") / "preprocess" / "stage_1"
OUTPUT_SUBDIR = Path("data") / "preprocess" / "stage_2"

# Run label shape of a stage 1 file (``run_1.csv``, ``run_2.csv``, ...).
RUN_LABEL_RE = re.compile(r"^run_(?P<ordinal>\d+)$")


def run_ordinal(label: str) -> int:
    """Numeric part of a ``run_<n>`` label, used as the ordering key."""
    match = RUN_LABEL_RE.match(label)
    if not match:
        raise ValueError(f"not a run label: {label!r}")
    return int(match.group("ordinal"))


# Role handling, identical to stage 1: file order is follower block, leader block.
ROLES = ("follower", "leader")

# Output column order (unchanged from stage 1).
OUTPUT_FIELDS = ("run", "role", "t_from_start", "x", "y", "heading", "velocity")

# Columns that must exist in the stage 1 input.
REQUIRED_FIELDS = ("run", "role", "t_from_start", "x", "y", "heading", "velocity")

# A standstill run longer than this many samples is a genuine stop; a shorter one
# is a dropped reading while driving and is ignored.
ISOLATED_STOP_MAX_SAMPLES = 3

# Trailing samples of a genuine stop that report at most this speed while the
# position did not change are treated as the launch/brake transient of that stop
# and absorbed into it (km/h, same unit as the `velocity` column).
STOP_LAUNCH_VELOCITY_EPS = 2.0

# Sampling gap (s) that starts a new event.
GAP_THRESHOLD_S = 1.0

ROLLING_VELOCITY_DECIMALS = 6
SUMMARY_FILENAME = "stage2_summary.json"

# Written explicitly so output bytes do not depend on the host platform.
LINE_TERMINATOR = "\r\n"


def project_root() -> Path:
    """Project root = parent of the directory containing this script."""
    return Path(__file__).resolve().parents[1]


def discover_runs(stage1_dir: Path) -> list[str]:
    """Run labels = stage 1 CSV file stems (``run_<n>``), in run order.

    Only ``run_<n>`` files are accepted (anything else in the directory is a bug
    of the stage 1 step, not something to be guessed at), and the order is the
    numeric one, so ``run_2`` comes before ``run_10`` - a plain name sort would
    not.
    """
    if not stage1_dir.is_dir():
        raise FileNotFoundError(f"stage 1 directory not found: {stage1_dir}")
    labels: list[str] = []
    for path in sorted(stage1_dir.glob("*.csv")):
        if not RUN_LABEL_RE.match(path.stem):
            raise ValueError(f"{path.name}: not a stage 1 run file (expected run_<n>.csv)")
        labels.append(path.stem)
    if not labels:
        raise FileNotFoundError(f"no stage 1 CSV found in {stage1_dir}")
    labels.sort(key=run_ordinal)
    return labels


def read_stage1(path: Path, run: str) -> dict[str, list[dict[str, str]]]:
    """Read one stage 1 file and split it into the two role blocks.

    Row order is validated: every role block must be ascending by t_from_start
    and the two blocks must not interleave (follower block first, leader second).
    """
    if not path.is_file():
        raise FileNotFoundError(f"stage 1 file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_FIELDS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing required column(s): {missing}")
        raw = [row for row in reader if any((v or "").strip() for v in row.values())]

    by_role: dict[str, list[dict[str, str]]] = {role: [] for role in ROLES}
    block_order: list[str] = []
    for row in raw:
        role = (row.get("role") or "").strip()
        if role not in by_role:
            raise ValueError(f"{path}: unexpected role value: {role!r}")
        label = (row.get("run") or "").strip()
        if label != run:
            raise ValueError(f"{path}: run column {label!r} does not match file name {run!r}")
        if not block_order or block_order[-1] != role:
            if role in block_order:
                raise ValueError(f"{path}: role block {role!r} is not contiguous")
            block_order.append(role)
        by_role[role].append(row)

    for role in ROLES:
        times = [float(r["t_from_start"]) for r in by_role[role]]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(f"{path}: {role} rows are not strictly ascending in t_from_start")
    return by_role


def as_float(row: dict[str, str], field: str) -> float:
    """Numeric view of a preserved text column."""
    try:
        return float(row[field])
    except ValueError:
        raise ValueError(f"non-numeric {field!r} value: {row[field]!r}") from None


def round6(value: float) -> float:
    """Round for the JSON report so the file stays readable and deterministic."""
    return round(value, ROLLING_VELOCITY_DECIMALS)


def zero_runs(velocities: list[float]) -> list[tuple[int, int]]:
    """Maximal runs of consecutive exactly-zero velocities as (first, last)."""
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(velocities):
        if velocities[i] == 0.0:
            j = i
            while j < len(velocities) and velocities[j] == 0.0:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def position_unchanged(rows: list[dict[str, str]], a: int, b: int) -> bool:
    """True when two consecutive samples report the same (x, y)."""
    return as_float(rows[a], "x") == as_float(rows[b], "x") and as_float(rows[a], "y") == as_float(
        rows[b], "y"
    )


def standstill_runs(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Classify every zero-velocity run of one role's sequence.

    Returns one record per run, either `kind="isolated"` (<= 3 samples, ignored)
    or `kind="stop"` (genuine stop, launch/brake transient absorbed into it).
    """
    times = [as_float(r, "t_from_start") for r in rows]
    velocities = [as_float(r, "velocity") for r in rows]
    records: list[dict[str, object]] = []
    for first, last in zero_runs(velocities):
        zero_samples = last - first + 1
        record: dict[str, object] = {
            "zero_first_index": first,
            "zero_last_index": last,
            "zero_samples": zero_samples,
            "first_index": first,
            "last_index": last,
            "t_start": round6(times[first]),
            "t_end": round6(times[last]),
        }
        if zero_samples > ISOLATED_STOP_MAX_SAMPLES:
            absorbed_end = last
            absorbed = 0
            while (
                absorbed_end + 1 < len(rows)
                and 0.0 < velocities[absorbed_end + 1] <= STOP_LAUNCH_VELOCITY_EPS
                and position_unchanged(rows, absorbed_end, absorbed_end + 1)
            ):
                absorbed_end += 1
                absorbed += 1
            record.update(
                {
                    "kind": "stop",
                    "last_index": absorbed_end,
                    "samples": absorbed_end - first + 1,
                    "absorbed_launch_samples": absorbed,
                    "t_end": round6(times[absorbed_end]),
                }
            )
        else:
            record.update({"kind": "isolated", "samples": zero_samples, "absorbed_launch_samples": 0})
        records.append(record)
    return records


def trim_role(rows: list[dict[str, str]]) -> dict[str, object]:
    """Apply the head/tail rules to one role sequence.

    Returns the kept rows (with their original indices) plus the evidence needed
    for the printed rationale and the summary file.
    """
    times = [as_float(r, "t_from_start") for r in rows]
    runs = standstill_runs(rows)
    genuine = [r for r in runs if r["kind"] == "stop"]
    n = len(rows)

    # Tail: a genuine stop that ends the sequence removes every row after the
    # last moving sample (i.e. the whole trailing standstill run).
    tail = genuine[-1] if genuine and genuine[-1]["last_index"] == n - 1 else None

    # The head cut may only come from the unstarted head of the sequence: every
    # standstill sample of the run has to sit before the first moving sample.
    # A genuine stop after the vehicle has started moving is valid driving data.
    first_motion = next(
        (index for index, row in enumerate(rows) if as_float(row, "velocity") > 0.0),
        n,
    )
    head_runs = [
        r
        for r in genuine
        if r is not tail and int(r["zero_first_index"]) < first_motion
    ]
    head = head_runs[-1] if head_runs else None

    head_end = int(head["last_index"]) if head else -1
    tail_start = int(tail["first_index"]) if tail else n

    kept = [(i, rows[i]) for i in range(head_end + 1, tail_start)]

    trims: list[dict[str, object]] = []
    if head:
        trims.append(
            {
                "kind": "start",
                "deleted_rows": head_end + 1,
                "first_index": 0,
                "last_index": head_end,
                "t_first": round6(times[0]),
                "t_last": round6(times[head_end]),
                "cut_boundary_t": round6(times[head_end]),
            }
        )
    if tail:
        trims.append(
            {
                "kind": "end",
                "deleted_rows": n - tail_start,
                "first_index": tail_start,
                "last_index": n - 1,
                "t_first": round6(times[tail_start]),
                "t_last": round6(times[n - 1]),
                "cut_boundary_t": round6(times[n - 1]),
            }
        )

    return {
        "rows": rows,
        "times": times,
        "kept": kept,
        "runs": runs,
        "head_run": head,
        "tail_run": tail,
        "trims": trims,
        "cut_boundary_t": round6(times[head_end]) if head else None,
        "kept_original_indices": [i for i, _ in kept],
    }


def apply_alignment(
    role_trim: dict[str, object], base_t: float | None
) -> tuple[list[tuple[int, dict[str, str]]], dict[str, object] | None]:
    """Delete the leading rows of a role that start before the alignment base.

    Only used for the leader, and only when the follower really triggered a head
    trim (`base_t is not None`).
    """
    if base_t is None:
        return list(role_trim["kept"]), None
    times: list[float] = role_trim["times"]  # type: ignore[assignment]
    kept: list[tuple[int, dict[str, str]]] = role_trim["kept"]  # type: ignore[assignment]
    deleted = [(i, row) for i, row in kept if times[i] < base_t]
    remaining = [(i, row) for i, row in kept if times[i] >= base_t]
    record: dict[str, object] = {
        "applied": True,
        "base_role": "follower",
        "base_t": round6(base_t),
        "deleted_rows": len(deleted),
        "first_index": deleted[0][0] if deleted else None,
        "last_index": deleted[-1][0] if deleted else None,
        "t_first": round6(times[deleted[0][0]]) if deleted else None,
        "t_last": round6(times[deleted[-1][0]]) if deleted else None,
    }
    return remaining, record


def split_segments(
    kept: list[tuple[int, dict[str, str]]], times: list[float]
) -> list[list[tuple[int, dict[str, str]]]]:
    """Split a role's kept rows into segments wherever dt > GAP_THRESHOLD_S."""
    segments: list[list[tuple[int, dict[str, str]]]] = []
    current: list[tuple[int, dict[str, str]]] = []
    for item in kept:
        if current and times[item[0]] - times[current[-1][0]] > GAP_THRESHOLD_S:
            segments.append(current)
            current = []
        current.append(item)
    if current:
        segments.append(current)
    return segments


def segment_record(
    run: str, role: str, ordinal: int, segment: list[tuple[int, dict[str, str]]], times: list[float]
) -> dict[str, object]:
    """Describe one segment: row counts, t range and source row indices."""
    indices = [i for i, _ in segment]
    return {
        "role": role,
        "event": f"{role}_event_{ordinal}",
        "rows": len(segment),
        "t_start": round6(times[indices[0]]),
        "t_end": round6(times[indices[-1]]),
        "first_index": indices[0],
        "last_index": indices[-1],
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write rows atomically so a failed run cannot leave a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(OUTPUT_FIELDS), lineterminator=LINE_TERMINATOR)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write the summary atomically (fixed formatting -> byte-identical reruns)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    with tmp_path.open("w", newline="", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp_path, path)


def rebuild_output_dir(out_dir: Path) -> list[str]:
    """Recreate the output directory from scratch; return the names removed."""
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        return []
    if (out_dir.name, out_dir.parent.name) != (OUTPUT_SUBDIR.name, OUTPUT_SUBDIR.parent.name):
        raise RuntimeError(f"refusing to rebuild unexpected directory: {out_dir}")
    removed = sorted(p.name for p in out_dir.iterdir())
    for entry in sorted(out_dir.iterdir()):
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    return removed


def sha256_file(path: Path) -> str:
    """Content hash, used as run-to-run provenance in the summary."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_written_file(path: Path, expected_rows: int, roles_expected: tuple[str, ...]) -> dict[str, object]:
    """Re-read one written event file and check rows, header, order and role blocks."""
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = tuple(reader.fieldnames or ())
        rows = list(reader)
    problems: list[str] = []
    if header != OUTPUT_FIELDS:
        problems.append(f"header {header} != {OUTPUT_FIELDS}")
    if len(rows) != expected_rows:
        problems.append(f"row count {len(rows)} != {expected_rows}")

    blocks: list[str] = []
    for row in rows:
        role = row["role"]
        if not blocks or blocks[-1] != role:
            if role in blocks:
                problems.append(f"role block {role!r} not contiguous")
            blocks.append(role)
    if tuple(blocks) != roles_expected:
        problems.append(f"role blocks {tuple(blocks)} != {roles_expected}")

    for role in ROLES:
        times = [float(r["t_from_start"]) for r in rows if r["role"] == role]
        if any(b <= a for a, b in zip(times, times[1:])):
            problems.append(f"{role} rows not strictly ascending in t_from_start")
    return {"file": path.name, "rows": len(rows), "problems": problems}


def print_trim_evidence(
    run: str, role: str, trim: dict[str, object], kept: list[tuple[int, dict[str, str]]]
) -> None:
    """Print the trimming rationale required by the brief."""
    times: list[float] = trim["times"]  # type: ignore[assignment]
    print(f"  {run} / {role}: {len(trim['rows'])} rows in, t=[{times[0]:.6f}, {times[-1]:.6f}]")
    print("    velocity==0 runs:")
    isolated_indices: list[int] = []
    # ``record`` is one zero-velocity run of this role; ``run`` above is the data
    # run label, so the two must not share a name.
    for record in trim["runs"]:  # type: ignore[union-attr]
        kind = record["kind"]
        if kind == "isolated":
            isolated_indices.extend(range(int(record["first_index"]), int(record["last_index"]) + 1))
            verdict = f"isolated (<= {ISOLATED_STOP_MAX_SAMPLES} samples) - dropped reading, IGNORED"
        else:
            verdict = "genuine stop (> %d samples)" % ISOLATED_STOP_MAX_SAMPLES
            if record["absorbed_launch_samples"]:
                verdict += (
                    f", absorbed {record['absorbed_launch_samples']} launch sample(s)"
                    f" (0 < v <= {STOP_LAUNCH_VELOCITY_EPS} km/h, position unchanged)"
                    f" -> run becomes [{record['first_index']}..{record['last_index']}]"
                )
        print(
            f"      [{record['first_index']}..{record['last_index']}]"
            f"  zero_samples={record['zero_samples']:<3} t=[{record['t_start']:.6f}, {record['t_end']:.6f}]"
            f"  -> {verdict}"
        )
    if isolated_indices:
        print(f"    ignored isolated 0 index/indices: {isolated_indices}")

    starts = [t for t in trim["trims"] if t["kind"] == "start"]  # type: ignore[union-attr]
    ends = [t for t in trim["trims"] if t["kind"] == "end"]  # type: ignore[union-attr]
    if starts:
        cut = starts[0]
        print(
            f"    start trim : delete rows {cut['first_index']}..{cut['last_index']}"
            f" ({cut['deleted_rows']} rows) t=[{cut['t_first']:.6f}, {cut['t_last']:.6f}]"
        )
        print(
            f"      basis    : last genuine stop run inside the head of the sequence"
            f" (all its standstill samples precede the first moving sample) ends at"
            f" index {cut['last_index']} (t={cut['cut_boundary_t']:.6f})"
        )
    else:
        print(
            "    start trim : none (no genuine stop run inside the head of the"
            " sequence, i.e. none before the first moving sample)"
        )
    if ends:
        cut = ends[0]
        print(
            f"    end trim   : delete rows {cut['first_index']}..{cut['last_index']}"
            f" ({cut['deleted_rows']} rows) t=[{cut['t_first']:.6f}, {cut['t_last']:.6f}]"
        )
        print(f"      basis    : sequence ends with a genuine stop run starting at index {cut['first_index']}")
    else:
        print("    end trim   : none (sequence does not end with a genuine stop run)")
    print(f"    kept: {len(kept)} rows", end="")
    if kept:
        print(f" t=[{times[kept[0][0]]:.6f}, {times[kept[-1][0]]:.6f}]")
    else:
        print()


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
    stage1_dir = root / STAGE1_SUBDIR
    out_dir = root / OUTPUT_SUBDIR
    runs = discover_runs(stage1_dir)

    print(f"project root : {root}")
    print(f"stage 1 input: {stage1_dir.relative_to(root)}")
    print(f"output dir   : {out_dir.relative_to(root)}")
    print()
    print("configuration:")
    print(f"  isolated standstill run   : <= {ISOLATED_STOP_MAX_SAMPLES} samples (ignored, kept)")
    print(f"  genuine stop run          : >  {ISOLATED_STOP_MAX_SAMPLES} samples (triggers trimming)")
    print(f"  launch transient absorbed : 0 < velocity <= {STOP_LAUNCH_VELOCITY_EPS} km/h with unchanged (x, y)")
    print(f"  event split gap           : dt > {GAP_THRESHOLD_S} s")
    print()

    removed = rebuild_output_dir(out_dir)
    if removed:
        print(f"rebuilt output dir, removed previous content: {', '.join(removed)}")
    else:
        print("rebuilt output dir: nothing to remove (directory was empty or absent)")
    print()

    summary: dict[str, object] = {
        "script": "src/preprocess_stage2.py",
        "config": {
            "isolated_stop_max_samples": ISOLATED_STOP_MAX_SAMPLES,
            "stop_launch_velocity_eps_kmh": STOP_LAUNCH_VELOCITY_EPS,
            "gap_threshold_s": GAP_THRESHOLD_S,
        },
        "inputs": {
            run: {
                "path": str((stage1_dir / f"{run}.csv").relative_to(root)).replace("\\", "/"),
                "sha256": sha256_file(stage1_dir / f"{run}.csv"),
            }
            for run in runs
        },
        "runs": {},
        "outputs": [],
    }

    written_files: list[tuple[str, int]] = []
    verification: list[dict[str, object]] = []
    total_kept_rows = 0
    total_written_rows = 0

    for run in runs:
        by_role = read_stage1(stage1_dir / f"{run}.csv", run)
        print(f"{run}: {sum(len(v) for v in by_role.values())} input rows "
              f"({', '.join(f'{role}={len(by_role[role])}' for role in ROLES)})")
        print("  trim evidence:")

        trims = {role: trim_role(by_role[role]) for role in ROLES}
        for role in ROLES:
            print_trim_evidence(run, role, trims[role], trims[role]["kept"])

        base_t = trims["follower"]["cut_boundary_t"]
        if base_t is None:
            print(
                "  start alignment: follower triggered no head trim -> leader untouched"
                " (no alignment base)"
            )
        leader_kept, alignment = apply_alignment(trims["leader"], base_t)
        if alignment is None:
            alignment = {
                "applied": False,
                "base_role": "follower",
                "base_t": None,
                "deleted_rows": 0,
                "first_index": None,
                "last_index": None,
                "t_first": None,
                "t_last": None,
            }
        else:
            print(
                f"  start alignment: leader rows with t < {base_t:.6f} (follower head-cut boundary)"
                f" deleted -> {alignment['deleted_rows']} rows"
                + (
                    f", stage1 indices {alignment['first_index']}..{alignment['last_index']}"
                    f" t=[{alignment['t_first']:.6f}, {alignment['t_last']:.6f}]"
                    if alignment["deleted_rows"]
                    else " (none)"
                )
            )

        kept_by_role = {
            "follower": list(trims["follower"]["kept"]),  # type: ignore[arg-type]
            "leader": leader_kept,
        }
        segments: dict[str, list[list[tuple[int, dict[str, str]]]]] = {}
        for role in ROLES:
            segments[role] = split_segments(kept_by_role[role], trims[role]["times"])  # type: ignore[arg-type]

        print("  segments (dt > %.1f s):" % GAP_THRESHOLD_S)
        segment_records: dict[str, list[dict[str, object]]] = {}
        for role in ROLES:
            records = []
            for ordinal, segment in enumerate(segments[role], start=1):
                rec = segment_record(run, role, ordinal, segment, trims[role]["times"])  # type: ignore[arg-type]
                records.append(rec)
                first, last = int(rec["first_index"]), int(rec["last_index"])
                print(
                    f"    {rec['event']:<16} {rec['rows']:>4} rows"
                    f"  t=[{rec['t_start']:.6f}, {rec['t_end']:.6f}]"
                    f"  (stage1 rows {first}..{last})"
                )
            segment_records[role] = records

        pairs = min(len(segments["follower"]), len(segments["leader"]))
        events: list[dict[str, object]] = []
        for ordinal in range(1, pairs + 1):
            follower_segment = segments["follower"][ordinal - 1]
            leader_segment = segments["leader"][ordinal - 1]
            rows = [row for _, row in follower_segment] + [row for _, row in leader_segment]
            name = f"{run}_event_{ordinal}.csv"
            write_csv(out_dir / name, rows)
            written_files.append((name, len(rows)))
            total_written_rows += len(rows)
            f_rec = segment_records["follower"][ordinal - 1]
            l_rec = segment_records["leader"][ordinal - 1]
            events.append(
                {
                    "event": f"event_{ordinal}",
                    "file": name,
                    "rows": len(rows),
                    "t_start": round6(min(float(f_rec["t_start"]), float(l_rec["t_start"]))),
                    "t_end": round6(max(float(f_rec["t_end"]), float(l_rec["t_end"]))),
                    "follower": {k: f_rec[k] for k in ("rows", "t_start", "t_end", "first_index", "last_index")},
                    "leader": {k: l_rec[k] for k in ("rows", "t_start", "t_end", "first_index", "last_index")},
                }
            )

        unmatched: list[dict[str, object]] = []
        for role in ROLES:
            if len(segments[role]) > pairs:
                for ordinal in range(pairs + 1, len(segments[role]) + 1):
                    rec = dict(segment_records[role][ordinal - 1])
                    rec["reason"] = "no counterpart segment in the other role"
                    unmatched.append(rec)
        unmatched.sort(key=lambda r: (ROLES.index(str(r["role"])), int(r["first_index"])))

        print(f"  pairing by ordinal: {pairs} event(s)")
        for event in events:
            print(
                f"    {event['file']:<24} {event['rows']:>4} rows"
                f"  (follower {event['follower']['rows']} + leader {event['leader']['rows']})"  # type: ignore[index]
                f"  t=[{event['t_start']:.6f}, {event['t_end']:.6f}]"
            )
        if unmatched:
            for rec in unmatched:
                print(
                    f"    UNMATCHED {rec['event']:<16} {rec['rows']:>4} rows"
                    f"  t=[{rec['t_start']:.6f}, {rec['t_end']:.6f}]  -> summary only, no CSV written"
                )
        else:
            print("    UNMATCHED: none")

        kept_total = sum(len(v) for v in kept_by_role.values())
        total_kept_rows += kept_total
        matched_rows = sum(int(e["rows"]) for e in events)  # type: ignore[arg-type]
        unmatched_rows = sum(int(r["rows"]) for r in unmatched)  # type: ignore[arg-type]
        balance = kept_total == matched_rows + unmatched_rows
        print(
            f"  row balance: kept {kept_total} = paired {matched_rows} + unmatched {unmatched_rows}"
            f"  {'OK' if balance else 'MISMATCH'}"
        )
        print()

        summary["runs"][run] = {  # type: ignore[index]
            "roles": {
                role: {
                    "input_rows": len(by_role[role]),
                    "kept_rows": len(kept_by_role[role]),
                    "zero_velocity_runs": trims[role]["runs"],
                    "ignored_isolated_zero_indices": [
                        i
                        for record in trims[role]["runs"]  # type: ignore[union-attr]
                        if record["kind"] == "isolated"
                        for i in range(int(record["first_index"]), int(record["last_index"]) + 1)
                    ],
                    "trims": trims[role]["trims"],
                    "cut_boundary_t": trims[role]["cut_boundary_t"],
                    "segments": segment_records[role],
                }
                for role in ROLES
            },
            "alignment": alignment,
            "keeping": {
                "follower_kept_rows": len(kept_by_role["follower"]),
                "leader_kept_rows": len(kept_by_role["leader"]),
                "kept_rows_total": kept_total,
                "paired_rows": matched_rows,
                "unmatched_rows": unmatched_rows,
                "balanced": balance,
            },
            "event_count": len(events),
            "events": events,
            "unmatched": unmatched,
        }

    summary["outputs"] = [
        {"file": name, "rows": rows} for name, rows in written_files
    ] + [{"file": SUMMARY_FILENAME, "rows": None}]

    # --- verification: re-read every written file --------------------------
    print("on-disk verification (re-read after writing):")
    width = max(len(name) for name, _ in written_files) if written_files else 8
    ok = True
    for name, expected in written_files:
        result = verify_written_file(out_dir / name, expected, ROLES)
        verification.append(result)
        status = "OK" if not result["problems"] else "MISMATCH"
        if result["problems"]:
            ok = False
        print(f"  {name:<{width}}  expected={expected}  on_disk={result['rows']}  {status}")
        for problem in result["problems"]:  # type: ignore[union-attr]
            print(f"      ! {problem}")
    print(f"  total written data rows: {total_written_rows}")

    summary["verification"] = verification
    summary["total"] = {
        "input_rows": sum(int(d["roles"][role]["input_rows"]) for d in summary["runs"].values() for role in ROLES),  # type: ignore[union-attr]
        "kept_rows": total_kept_rows,
        "written_rows": total_written_rows,
        "event_files": len(written_files),
    }
    write_json(out_dir / SUMMARY_FILENAME, summary)

    # --- output directory state -------------------------------------------
    print()
    tmp_leftovers = sorted(p.name for p in out_dir.glob("*.tmp"))
    print(f".tmp leftovers: {tmp_leftovers if tmp_leftovers else 'none'}")
    print(f"stage_2 dir contents: {', '.join(sorted(p.name for p in out_dir.iterdir()))}")

    expected_files = sorted([name for name, _ in written_files] + [SUMMARY_FILENAME])
    actual_files = sorted(p.name for p in out_dir.iterdir())
    if actual_files != expected_files:
        print(f"\nERROR: unexpected stage_2 contents {actual_files} != {expected_files}", file=sys.stderr)
        return 1
    if not ok:
        print("\nERROR: on-disk verification failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
