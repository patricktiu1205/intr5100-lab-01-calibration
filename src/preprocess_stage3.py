#!/usr/bin/env python3
"""INTR5100 Lab Session I - Stage 3 data preprocessing: leader/follower time alignment.

Step 3 of the pipeline: put the two vehicles of every paired `event` on one common,
equally spaced time axis and resample both of them onto it.

Input  (Stage 2 output, one file per paired event, follower block first / leader
        block second, ascending `t_from_start` inside each block):

    data/preprocess/stage_2/<run>_event_<k>.csv
    columns: run, role, t_from_start, x, y, heading, velocity

Output (rebuilt from scratch on every run):

    data/preprocess/stage_3/<run>_event_<k>.csv   aligned events
    data/preprocess/stage_3/stage3_summary.json      full per-event report

--- step 1: overlap window -----------------------------------------------------

The two vehicles sample independently (`~10 Hz` but on unrelated clocks) and they
enter / leave the same straight line at different moments, so only the intersection

    [max(follower.min, leader.min), min(follower.max, leader.max)]

holds observations of both vehicles at once. Every source row outside that window
is dropped. An event whose intersection is empty or shorter than 1.0 s is dropped
altogether (`reason = "no_overlap"`) - it cannot carry a usable paired signal
(run_1 / event_1: follower [4.63, 6.42] vs leader [-0.05, 0.35]).

--- step 2: common time axis + interpolation ----------------------------------

The grid starts at the left edge of the overlap and advances in dt = 0.1 s steps;
the right edge is appended as a closing instant when the last step falls short of
it (and skipped when it would print as that very same timestamp). Both vehicles
are then evaluated on the same instant, so the two blocks of an output file carry
literally the same `t_from_start` values.

`x`, `y` and `velocity` are linearly interpolated between the two source samples
that bracket the grid instant (an instant landing on a sample, within the 1e-9 s
float-noise tolerance, just takes that sample); heading is an angle on 0-360, so it
is unwrapped first (each step is mapped into [-180, 180) so a 0/360 crossing
becomes a short step instead of a 359 degree jump), interpolated, then normalised
back into [0, 360). Nothing is ever extrapolated: a grid instant without a
bracketing pair of source samples is dropped, never guessed.

--- step 3: discontinuity removal --------------------------------------------

At a grid instant the "interpolation span" of a role is the dt between the two
source samples surrounding it (0 when it lands exactly on a sample). A span above
0.3 s means the value there would be an average over a data hole rather than an
observation, so that instant is unusable for *both* vehicles: the whole time point
is removed from the event. The grid may therefore come out with holes; the output
keeps the surviving instants in ascending order, it does not force them to be
contiguous.

--- output --------------------------------------------------------------------

`{run}_event_{k}.csv` keeps the input columns `run,role,t_from_start,x,y,
heading,velocity` with `t_from_start` replaced by the grid instant; the follower
block comes first, the leader block second, ascending inside each block, one row
per vehicle per surviving instant. All numbers are formatted to 6 decimals.

The script uses only the standard library, is idempotent (same input ->
byte-identical output; no wall-clock timestamp is ever written), writes every file
atomically through `.tmp` + `os.replace`, rebuilds the output directory at the
start of each run, resolves paths relative to the project root so it can be run
from any working directory, and re-reads (and re-checks) everything it wrote.

Usage:
    python src/preprocess_stage3.py
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

# --- configuration ---------------------------------------------------------

STAGE2_SUBDIR = Path("data") / "preprocess" / "stage_2"
OUTPUT_SUBDIR = Path("data") / "preprocess" / "stage_3"

# Role handling, identical to stage 1/2: file order is follower block, leader block.
ROLES = ("follower", "leader")

# Output column order (unchanged from stage 2).
OUTPUT_FIELDS = ("run", "role", "t_from_start", "x", "y", "heading", "velocity")

# Columns that must exist in the stage 2 input.
REQUIRED_FIELDS = ("run", "role", "t_from_start", "x", "y", "heading", "velocity")

# Linearly interpolated channels; heading is handled separately (angle unwrapping).
INTERPOLATED_FIELDS = ("x", "y", "velocity")
HEADING_FIELD = "heading"

# Grid step of the common time axis (s).
GRID_DT_S = 0.1

# An overlap shorter than this cannot support a paired signal; the event is dropped.
MIN_OVERLAP_S = 1.0

# A grid instant whose surrounding source samples are further apart than this is
# a gap average, not an observation -> the time point is deleted for both roles.
MAX_INTERPOLATION_SPAN_S = 0.3

# A grid instant this close (s) to a source sample counts as landing on it: the
# value is taken from that sample (no interpolation) and the span is 0. Without the
# tolerance, a step like `start + 29 * 0.1` sitting 4e-16 s past a sample at 2.9 s
# would look like a point inside the neighbouring hole. Far below the printed
# resolution, so it never changes a written value beyond the 6th decimal.
SAMPLE_MATCH_TOLERANCE_S = 1e-9

OUTPUT_DECIMALS = 6
SUMMARY_FILENAME = "stage3_summary.json"

# Written explicitly so output bytes do not depend on the host platform.
LINE_TERMINATOR = "\r\n"

RUN_LABEL_RE = re.compile(r"^run_(?P<ordinal>\d+)$")
EVENT_FILE_RE = re.compile(r"^(?P<run>.+)_event_(?P<ordinal>\d+)\.csv$")


def project_root() -> Path:
    """Project root = parent of the directory containing this script."""
    return Path(__file__).resolve().parents[1]


def rel(path: Path, root: Path) -> str:
    """Project-relative POSIX path (never an absolute path in the summary)."""
    return str(path.relative_to(root)).replace("\\", "/")


def discover_events(stage2_dir: Path) -> list[tuple[str, int, Path]]:
    """Find the paired event files of stage 2 as (run, ordinal, path)."""
    if not stage2_dir.is_dir():
        raise FileNotFoundError(f"stage 2 directory not found: {stage2_dir}")
    events: list[tuple[str, int, Path]] = []
    for path in sorted(stage2_dir.glob("*.csv")):
        match = EVENT_FILE_RE.match(path.name)
        if not match:
            raise ValueError(f"{path.name}: does not look like a stage 2 event file")
        events.append((match.group("run"), int(match.group("ordinal")), path))
    if not events:
        raise FileNotFoundError(f"no stage 2 event CSV found in {stage2_dir}")
    # Numeric run order (run_2 before run_10), then the event ordinal.
    events.sort(key=lambda item: (run_ordinal(item[0]), item[1]))
    return events


def run_ordinal(label: str) -> int:
    """Numeric part of a ``run_<n>`` label, used as the ordering key."""
    match = RUN_LABEL_RE.match(label)
    return int(match.group("ordinal")) if match else 0


def read_event(path: Path, run: str) -> dict[str, list[dict[str, str]]]:
    """Read one stage 2 event file and split it into the two role blocks.

    Row order is validated: every role block must be ascending by t_from_start and
    the two blocks must not interleave (follower block first, leader second).
    """
    if not path.is_file():
        raise FileNotFoundError(f"stage 2 event file not found: {path}")
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
        if not by_role[role]:
            raise ValueError(f"{path}: {role} block is empty")
        times = [float(r["t_from_start"]) for r in by_role[role]]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(f"{path}: {role} rows are not strictly ascending in t_from_start")
    return by_role


def as_float(row: dict[str, str], field: str) -> float:
    """Numeric view of one input column."""
    try:
        return float(row[field])
    except ValueError:
        raise ValueError(f"non-numeric {field!r} value: {row[field]!r}") from None


def round6(value: float) -> float:
    """Round for the JSON report so the file stays readable and deterministic."""
    return round(value, OUTPUT_DECIMALS)


def fmt(value: float) -> str:
    """Fixed-point formatting of every written number (deterministic bytes)."""
    return f"{value:.{OUTPUT_DECIMALS}f}"


def unwrap_angles(degrees: list[float]) -> list[float]:
    """Unwrap an angle series across the 0/360 wrap-around.

    Each step is mapped into [-180, 180) before it is accumulated, so a crossing of
    the 0/360 seam becomes a short step (359 -> 1 instead of a -358 jump) while the
    result still equals the input modulo 360.
    """
    if not degrees:
        return []
    unwrapped = [degrees[0]]
    for previous, current in zip(degrees, degrees[1:]):
        delta = current - previous
        delta -= 360.0 * round(delta / 360.0)  # smallest signed representative
        if delta >= 180.0:
            delta -= 360.0
        elif delta < -180.0:
            delta += 360.0
        unwrapped.append(unwrapped[-1] + delta)
    return unwrapped


def build_series(rows: list[dict[str, str]]) -> dict[str, object]:
    """Numeric series of one role block, plus the unwrapped heading."""
    headings = [as_float(r, HEADING_FIELD) for r in rows]
    return {
        "times": [as_float(r, "t_from_start") for r in rows],
        "x": [as_float(r, "x") for r in rows],
        "y": [as_float(r, "y") for r in rows],
        "velocity": [as_float(r, "velocity") for r in rows],
        "heading": headings,
        "heading_unwrapped": unwrap_angles(headings),
    }


def interpolate(series: dict[str, object], field: str, t: float) -> tuple[float | None, float | None, str]:
    """Interpolate one channel of one role at grid instant `t`.

    Returns `(value, span, kind)` where `span` is the dt between the two source
    samples surrounding `t` (0.0 when `t` lands on a sample, up to
    SAMPLE_MATCH_TOLERANCE_S) and `kind` is `"exact"`, `"interpolated"` or
    `"out_of_range"`. `value`/`span` are None when the instant falls outside the
    role's own samples: extrapolation is forbidden.
    """
    times: list[float] = series["times"]  # type: ignore[assignment]
    values: list[float] = series[field]  # type: ignore[assignment]
    index = bisect.bisect_right(times, t)

    if index > 0 and t - times[index - 1] <= SAMPLE_MATCH_TOLERANCE_S:
        return values[index - 1], 0.0, "exact"
    if index < len(times) and times[index] - t <= SAMPLE_MATCH_TOLERANCE_S:
        return values[index], 0.0, "exact"
    if index == 0 or index == len(times):
        return None, None, "out_of_range"

    lower, upper = index - 1, index
    span = times[upper] - times[lower]
    fraction = (t - times[lower]) / span
    return values[lower] + fraction * (values[upper] - values[lower]), span, "interpolated"


def grid_instants(start: float, end: float) -> list[float]:
    """Common time axis: dt steps from `start`, plus `end` as the closing instant.

    The steps are unquantised, so the first instant *is* the left edge of the overlap
    and stays comparable with the source timestamps (quantising it first would push it
    a fraction of a microsecond outside the window and fake an out-of-range point).
    The right edge is appended whenever the last step falls short of it; when it would
    print as the very same timestamp as that step (< 1e-6 s apart) it is skipped, so a
    timestamp is never written twice.
    """
    instants: list[float] = []
    step_index = 0
    instant = start
    while instant <= end:
        instants.append(instant)
        step_index += 1
        instant = start + step_index * GRID_DT_S
    if fmt(end) != fmt(instants[-1]):
        instants.append(end)
    return instants


def align_event(
    by_role: dict[str, list[dict[str, str]]]
) -> dict[str, object]:
    """Overlap + grid + interpolation + discontinuity removal for one event."""
    series = {role: build_series(by_role[role]) for role in ROLES}
    spans = {
        role: (float(series[role]["times"][0]), float(series[role]["times"][-1]))  # type: ignore[index]
        for role in ROLES
    }

    overlap_start = max(spans[role][0] for role in ROLES)
    overlap_end = min(spans[role][1] for role in ROLES)
    duration = overlap_end - overlap_start

    outside_rows = {
        role: sum(1 for t in series[role]["times"] if t < overlap_start or t > overlap_end)  # type: ignore[union-attr]
        for role in ROLES
    }

    report: dict[str, object] = {
        "roles": {
            role: {
                "input_rows": len(by_role[role]),
                "t_start": round6(spans[role][0]),
                "t_end": round6(spans[role][1]),
                "rows_outside_overlap": outside_rows[role],
            }
            for role in ROLES
        },
        "overlap": {
            "t_start": round6(overlap_start),
            "t_end": round6(overlap_end),
            "duration_s": round6(duration),
        },
        "rows_outside_overlap_total": sum(outside_rows.values()),
    }

    if duration <= 0.0:
        report.update({"status": "deleted", "reason": "no_overlap", "detail": "empty_intersection"})
        return report
    if duration < MIN_OVERLAP_S:
        report.update(
            {
                "status": "deleted",
                "reason": "no_overlap",
                "detail": f"intersection_shorter_than_{MIN_OVERLAP_S}_s",
            }
        )
        return report

    instants = grid_instants(overlap_start, overlap_end)
    kept_instants: list[float] = []
    removed: list[dict[str, object]] = []
    values_cache: dict[float, dict[str, dict[str, float]]] = {}
    removed_by_reason: dict[str, int] = {"discontinuity": 0, "outside_source_range": 0}
    removed_by_role: dict[str, int] = {role: 0 for role in ROLES}
    span_max_seen = 0.0
    kept_span_max = 0.0

    for instant in instants:
        values: dict[str, dict[str, float]] = {}
        failing_roles: list[str] = []
        out_of_range = False
        span_max = 0.0
        for role in ROLES:
            role_values: dict[str, float] = {}
            role_span = 0.0
            role_failed = False
            for field in INTERPOLATED_FIELDS + (HEADING_FIELD,):
                value, span, kind = interpolate(
                    series[role],
                    "heading_unwrapped" if field == HEADING_FIELD else field,
                    instant,
                )
                if kind == "out_of_range" or value is None or span is None:
                    out_of_range = True
                    role_failed = True
                    break
                role_span = max(role_span, span)
                role_values[field] = value
            if role_failed:
                if role not in failing_roles:
                    failing_roles.append(role)
                continue
            span_max_seen = max(span_max_seen, role_span)
            span_max = max(span_max, role_span)
            if role_span > MAX_INTERPOLATION_SPAN_S:
                failing_roles.append(role)
            role_values[HEADING_FIELD] %= 360.0
            values[role] = role_values
        if failing_roles:
            for role in failing_roles:
                removed_by_role[role] += 1
            reason = "outside_source_range" if out_of_range else "discontinuity"
            removed_by_reason[reason] += 1
            removed.append(
                {"t_from_start": round6(instant), "reason": reason, "roles": list(failing_roles)}
            )
            continue
        kept_instants.append(instant)
        kept_span_max = max(kept_span_max, span_max)
        values_cache[round(instant, 9)] = values

    report.update(
        {
            "status": "written",
            "grid_points": len(instants),
            "grid_points_kept": len(kept_instants),
            "grid_points_removed": len(removed),
            "removed_by_reason": removed_by_reason,
            "removed_by_role": removed_by_role,
            "removed_times": removed,
            "max_interpolation_span_s": round6(span_max_seen),
            "max_interpolation_span_kept_s": round6(kept_span_max),
            "written_rows_per_role": len(kept_instants),
            "written_rows_total": 2 * len(kept_instants),
        }
    )
    report["_instants"] = kept_instants
    report["_values"] = [values_cache[round(t, 9)] for t in kept_instants]
    return report


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


def verify_written_file(
    path: Path, expected_times: list[str], expected_rows: int
) -> dict[str, object]:
    """Re-read one written event file and re-check every structural property."""
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
    if tuple(blocks) != ROLES:
        problems.append(f"role blocks {tuple(blocks)} != {ROLES}")

    times: dict[str, list[str]] = {role: [] for role in ROLES}
    for row in rows:
        times[row["role"]].append(row["t_from_start"])
        for field in ("x", "y", "heading", "velocity"):
            try:
                float(row[field])
            except ValueError:
                problems.append(f"non-numeric {field!r}: {row[field]!r}")
        heading = float(row["heading"])
        if not 0.0 <= heading < 360.0:
            problems.append(f"heading {heading} outside [0, 360)")

    for role in ROLES:
        if times[role] != sorted(times[role], key=float) or len(set(times[role])) != len(times[role]):
            problems.append(f"{role} timestamps are not strictly ascending")
    if times["follower"] != times["leader"]:
        problems.append("follower and leader timestamps differ")
    if times["follower"] != expected_times:
        problems.append("timestamps differ from the surviving grid instants")
    for role in ROLES:
        if len(times[role]) != expected_rows // 2:
            problems.append(f"{role} row count {len(times[role])} != {expected_rows // 2}")

    return {"file": path.name, "rows": len(rows), "problems": problems}


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
    stage2_dir = root / STAGE2_SUBDIR
    out_dir = root / OUTPUT_SUBDIR
    events = discover_events(stage2_dir)

    print(f"project root : {root}")
    print(f"stage 2 input: {rel(stage2_dir, root)}")
    print(f"output dir   : {rel(out_dir, root)}")
    print()
    print("configuration:")
    print(f"  grid step                 : dt = {GRID_DT_S} s")
    print(f"  grid time resolution      : {OUTPUT_DECIMALS} printed decimals (right edge appended as closing instant)")
    print(f"  minimum overlap           : {MIN_OVERLAP_S} s (shorter -> event deleted, reason no_overlap)")
    print(f"  interpolation span limit  : {MAX_INTERPOLATION_SPAN_S} s (larger -> time point deleted for both roles)")
    print()
    print(f"events found in stage_2: {len(events)}")
    print()

    removed_entries = rebuild_output_dir(out_dir)
    if removed_entries:
        print(f"rebuilt output dir, removed previous content: {', '.join(removed_entries)}")
    else:
        print("rebuilt output dir: nothing to remove (directory was empty or absent)")
    print()

    summary: dict[str, object] = {
        "script": "src/preprocess_stage3.py",
        "config": {
            "grid_dt_s": GRID_DT_S,
            "min_overlap_s": MIN_OVERLAP_S,
            "max_interpolation_span_s": MAX_INTERPOLATION_SPAN_S,
            "sample_match_tolerance_s": SAMPLE_MATCH_TOLERANCE_S,
            "output_decimals": OUTPUT_DECIMALS,
        },
        "inputs": {
            rel(path, root): {"sha256": sha256_file(path)}
            for _, _, path in events
        },
        "events": [],
        "deleted_events": [],
        "outputs": [],
        "verification": [],
        "total": {},
    }

    written_files: list[tuple[str, int, list[str]]] = []
    verification: list[dict[str, object]] = []
    total_grid_points = 0
    total_grid_removed = 0
    total_written_rows = 0

    print("per-event alignment:")
    for run, ordinal, path in events:
        by_role = read_event(path, run)
        report = align_event(by_role)
        name = f"{run}_event_{ordinal}.csv"
        input_rows = ", ".join(f"{role}={len(by_role[role])}" for role in ROLES)
        overlap = report["overlap"]  # type: ignore[assignment]
        print(
            f"  {name:<24} input rows: {input_rows:<34}"
            f"  t: follower [{report['roles']['follower']['t_start']}, {report['roles']['follower']['t_end']}]"  # type: ignore[index]
            f"  leader [{report['roles']['leader']['t_start']}, {report['roles']['leader']['t_end']}]"  # type: ignore[index]
        )
        if report["status"] == "deleted":
            print(
                f"      -> DELETED (reason: {report['reason']}, {report['detail']}):"
                f" overlap length {report['overlap']['duration_s']} s"  # type: ignore[index]
                f" < {MIN_OVERLAP_S} s; {report['rows_outside_overlap_total']} non-overlapping rows dropped"
            )
            summary["deleted_events"].append(  # type: ignore[union-attr]
                {
                    "run": run,
                    "event": f"event_{ordinal}",
                    "file": None,
                    "status": "deleted",
                    "reason": report["reason"],
                    "detail": report["detail"],
                    "input_file": rel(path, root),
                    "roles": report["roles"],
                    "overlap": report["overlap"],
                    "rows_outside_overlap_total": report["rows_outside_overlap_total"],
                }
            )
            summary["events"].append(  # type: ignore[union-attr]
                {
                    "run": run,
                    "event": f"event_{ordinal}",
                    "status": "deleted",
                    "reason": report["reason"],
                    "overlap": report["overlap"],
                }
            )
            print()
            continue

        instants: list[float] = report["_instants"]  # type: ignore[assignment]
        values: list[dict[str, dict[str, float]]] = report["_values"]  # type: ignore[assignment]
        instant_texts = [fmt(t) for t in instants]

        rows: list[dict[str, str]] = []
        for role in ROLES:
            for index, instant in enumerate(instants):
                role_values = values[index][role]
                rows.append(
                    {
                        "run": run,
                        "role": role,
                        "t_from_start": instant_texts[index],
                        "x": fmt(role_values["x"]),
                        "y": fmt(role_values["y"]),
                        "heading": fmt(role_values[HEADING_FIELD]),
                        "velocity": fmt(role_values["velocity"]),
                    }
                )
        write_csv(out_dir / name, rows)
        written_files.append((name, len(rows), instant_texts))
        total_written_rows += len(rows)
        total_grid_points += int(report["grid_points"])
        total_grid_removed += int(report["grid_points_removed"])

        print(
            f"      overlap [{overlap['t_start']}, {overlap['t_end']}]"  # type: ignore[index]
            f" = {overlap['duration_s']} s, non-overlapping rows dropped:"  # type: ignore[index]
            f" follower {report['roles']['follower']['rows_outside_overlap']},"  # type: ignore[index]
            f" leader {report['roles']['leader']['rows_outside_overlap']}"  # type: ignore[index]
        )
        print(
            f"      grid: {report['grid_points']} instants -> kept {report['grid_points_kept']},"
            f" removed {report['grid_points_removed']}"
            f" ({report['removed_by_reason']['discontinuity']} discontinuity"
            f" (follower {report['removed_by_role']['follower']}, leader {report['removed_by_role']['leader']}),"
            f" {report['removed_by_reason']['outside_source_range']} outside source range),"
            f" max span {report['max_interpolation_span_s']} s"
            f" (max span kept {report['max_interpolation_span_kept_s']} s)"
        )
        if report["removed_times"]:
            shown = ", ".join(f"{item['t_from_start']}" for item in report["removed_times"])  # type: ignore[union-attr]
            print(f"      deleted time points: {shown}")
        print(
            f"      wrote {name}: {len(rows)} rows"
            f" (follower {report['written_rows_per_role']} + leader {report['written_rows_per_role']})"
            f"  timestamps identical follower/leader: yes"
        )

        event_record = {
            "run": run,
            "event": f"event_{ordinal}",
            "file": name,
            "input_file": rel(path, root),
            "status": "written",
            "roles": report["roles"],
            "rows_outside_overlap_total": report["rows_outside_overlap_total"],
            "overlap": report["overlap"],
            "grid_points": report["grid_points"],
            "grid_points_kept": report["grid_points_kept"],
            "grid_points_removed": report["grid_points_removed"],
            "removed_by_reason": report["removed_by_reason"],
            "removed_by_role": report["removed_by_role"],
            "removed_times": report["removed_times"],
            "max_interpolation_span_s": report["max_interpolation_span_s"],
            "max_interpolation_span_kept_s": report["max_interpolation_span_kept_s"],
            "written_rows_per_role": report["written_rows_per_role"],
            "written_rows_total": report["written_rows_total"],
            "timestamps_identical": True,
            "conservation": {
                "grid_points_kept_x2_equals_rows": 2 * int(report["grid_points_kept"]) == len(rows),
                "grid_points_equals_kept_plus_removed": int(report["grid_points"])
                == int(report["grid_points_kept"]) + int(report["grid_points_removed"]),
            },
        }
        summary["events"].append(event_record)  # type: ignore[union-attr]
        print()

    print("on-disk verification (re-read after writing):")
    width = max((len(name) for name, _, _ in written_files), default=8)
    ok = True
    for name, expected_rows, expected_times in written_files:
        result = verify_written_file(out_dir / name, expected_times, expected_rows)
        verification.append(result)
        status = "OK" if not result["problems"] else "MISMATCH"
        if result["problems"]:
            ok = False
        print(
            f"  {name:<{width}}  rows={result['rows']}"
            f"  (follower/leader timestamps identical: yes)  {status}"
        )
        for problem in result["problems"]:  # type: ignore[union-attr]
            print(f"      ! {problem}")
    print(f"  total written data rows: {total_written_rows}")
    print(f"  total surviving grid instants (x2 = rows): {total_written_rows // 2}")
    print(f"  total removed grid instants: {total_grid_removed}")

    summary["verification"] = verification
    kept_x2 = 2 * sum(int(e["grid_points_kept"]) for e in summary["events"] if e["status"] == "written")  # type: ignore[union-attr]
    summary["total"] = {
        "input_event_files": len(events),
        "written_event_files": len(written_files),
        "deleted_event_files": len(summary["deleted_events"]),  # type: ignore[arg-type]
        "grid_points": total_grid_points,
        "grid_points_kept": total_written_rows // 2,
        "grid_points_removed": total_grid_removed,
        "written_rows": total_written_rows,
        "conservation_ok": kept_x2 == total_written_rows
        and total_grid_points == total_written_rows // 2 + total_grid_removed,
    }
    summary["outputs"] = [{"file": name, "rows": rows} for name, rows, _ in written_files] + [
        {"file": SUMMARY_FILENAME, "rows": None}
    ]
    write_json(out_dir / SUMMARY_FILENAME, summary)

    print()
    tmp_leftovers = sorted(p.name for p in out_dir.glob("*.tmp"))
    print(f".tmp leftovers: {tmp_leftovers if tmp_leftovers else 'none'}")
    print(f"stage_3 dir contents: {', '.join(sorted(p.name for p in out_dir.iterdir()))}")

    expected_files = sorted([name for name, _, _ in written_files] + [SUMMARY_FILENAME])
    actual_files = sorted(p.name for p in out_dir.iterdir())
    if actual_files != expected_files:
        print(f"\nERROR: unexpected stage_3 contents {actual_files} != {expected_files}", file=sys.stderr)
        return 1
    if not ok:
        print("\nERROR: on-disk verification failed.", file=sys.stderr)
        return 1
    if not summary["total"]["conservation_ok"]:  # type: ignore[index]
        print("\nERROR: point conservation failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
