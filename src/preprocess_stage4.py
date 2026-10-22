#!/usr/bin/env python3
"""INTR5100 Lab Session I - Stage 4 data preprocessing: merge into the calibration table.

Step 4 of the pipeline: collapse every aligned event of stage 3 into the four-column
table the calibration/identification work consumes:

    time, fv, lv, dhw

Input  (Stage 3 output - one file per paired event, follower block first / leader
        block second, both blocks on the same instants):

    data/preprocess/stage_3/<run>_event_<k>.csv
    columns: run, role, t_from_start, x, y, heading, velocity

Output (rebuilt from scratch on every run):

    data/preprocess/stage_4/<run>_event_<k>.csv   calibration tables
    data/preprocess/stage_4/stage4_summary.json      full per-event report

--- step 0: pairing ------------------------------------------------------------

Stage 3 guarantees the two blocks carry literally the same instants, so row `i` of
the follower block and row `i` of the leader block are one time point. That is
re-checked here instead of assumed: the two blocks must have the same row count and
their raw `t_from_start` lists must match element by element, or the file is
rejected. Each block must be strictly ascending in `t_from_start`.

--- step 1: time --------------------------------------------------------------

`time` is the stage 3 timestamp rounded to the nearest 0.1 s (half up). The data
already sits on a 0.1 s grid, so this is a *display precision normalisation only*:
the raw timestamp is kept internally and is what every dt below is computed from, so
a rounded value can never feed a division. The stage 3 right edge can be an extra
closing instant that is off-grid by a few hundredths of a second, so two rounded
times may collide (the closing instant has no grid cell of its own). The output
table is keyed by `time`, so a duplicate is not allowed to survive: scanning in time
order, a point whose rounded `time` equals the `time` of the row above it is DROPPED
- the earlier point (the one on the grid cell) is kept and the off-grid closing
instant goes. One cell therefore yields exactly one row and the written `time`
column is strictly increasing, hence unique. The rule is generic: it follows from
the rounded times alone and applies to every event - no event number, no known
collision and no raw-timestamp threshold is hard-coded. Every dropped point is
reported with its raw timestamp, its event, the row above it and the gap that
collapsed in `dropped_time_duplicates` (action: dropped) and on stdout. This removal
is the FIRST step of the merge, so the point set every later step works on already
excludes the dropped row: it can neither take part in a speed window nor contribute a
`dhw` value, and the table cannot contain a number derived from a row it does not
show.

--- step 2: dhw ---------------------------------------------------------------

`dhw` is the forward distance to the leading vehicle along the follower's heading,
with no vehicle length subtracted:

    dhw = (x_l - x_f) * cos(h_f) + (y_l - y_f) * sin(h_f)

`h_f` is the follower heading in radians (0 deg -> +x, 90 deg -> +y, 180 deg -> -x,
270 deg -> -y - the same convention the earlier stages use), so `dhw` is the
projection of the leader-follower separation onto the follower's direction of travel.
It is evaluated on the points that survive step 1.

--- step 3: row removal -------------------------------------------------------

One further rule drops a time point, evaluated on the points that survive step 1, in
time order:

  * non-positive `dhw` - the leader projects behind the follower (tracking glitch or
    an overtake), which the calibration table cannot use, so the point goes, listed
    under `deleted_anomalies`.

Because this test needs `dhw`, it runs after step 2 and before the speeds of step 4,
so an anomaly is removed from the very set the speeds are computed on.

--- step 4: velocities --------------------------------------------------------

`fv` belongs to the follower, `lv` to the leader, and both are derived from the
position track - never from the reported `velocity` column of stage 3, which is
discarded because it disagrees with the position-difference speed by roughly a
factor of three. Unit: coordinate units per second (`(x, y)` units / s). The speeds
are computed LAST, on exactly the points that steps 1 and 3 kept.

For an interior point the speed is the chord over the central window,

    v_i = |P_{i+1} - P_{i-1}| / (t_{i+1} - t_{i-1})

i.e. the displacement between the two neighbouring points divided by 2*dt. The dt
uses the raw timestamps, so a hole in the stage 3 grid (the alignment step can leave
one) yields the real, larger interval instead of a fake 0.1 s. The two end points
have no second neighbour and use the one-sided difference over the single adjacent
interval,

    v_0 = |P_1 - P_0| / (t_1 - t_0)      v_{n-1} = |P_{n-1} - P_{n-2}| / (t_{n-1} - t_{n-2})

Both formulas reduce to the plain 0.1 s difference wherever the grid is contiguous
*and* no point has been removed. `i +/- 1` always means the neighbouring *retained*
point, so a removed point never enters a window: at the end of a series whose last
point was dropped the interior formula collapses to the one-sided
one and the last row is a plain forward/backward difference over the retained pairs.
The rule is unchanged - interior central, ends one-sided, dt from raw timestamps -
only the point set it is applied to is the retained one.

--- output -------------------------------------------------------------------

`{run}_event_{k}.csv` carries `time,fv,lv,dhw`, one row per surviving time point,
`time` strictly increasing (hence unique - step 1) and printed with 1 decimal (the
0.1 s normalisation), the three values with 6 decimals. Numbers are formatted so the
bytes only depend on the input.

The script uses only the standard library, is idempotent (same input ->
byte-identical output; no wall-clock timestamp is ever written), writes every file
atomically through `.tmp` + `os.replace`, rebuilds the output directory at the start
of each run, resolves paths relative to the project root so it can be run from any
working directory, and re-reads (and re-checks) everything it wrote.

Usage:
    python src/preprocess_stage4.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

# --- configuration ---------------------------------------------------------

STAGE3_SUBDIR = Path("data") / "preprocess" / "stage_3"
OUTPUT_SUBDIR = Path("data") / "preprocess" / "stage_4"

# Role handling, identical to stage 1/2/3: file order is follower block, leader block.
ROLES = ("follower", "leader")

# Columns that must exist in the stage 3 input (velocity is read for validation only
# and never copied: stage 4 recomputes speeds from the position track).
REQUIRED_FIELDS = ("run", "role", "t_from_start", "x", "y", "heading", "velocity")

# Calibration table columns.
OUTPUT_FIELDS = ("time", "fv", "lv", "dhw")

# `time` is the stage 3 timestamp normalised to this grid (s) and printed with this
# many decimals.
TIME_STEP_S = 0.1
TIME_DECIMALS = 1

# Reason label of a row removed because its rounded `time` repeats the row above it.
DROPPED_TIME_DUPLICATE_REASON = "duplicate_rounded_time"

# fv / lv / dhw printed decimals (same resolution the earlier stages write).
OUTPUT_DECIMALS = 6

SUMMARY_FILENAME = "stage4_summary.json"

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


def discover_events(stage3_dir: Path) -> list[tuple[str, int, Path]]:
    """Find the aligned event files of stage 3 as (run, ordinal, path)."""
    if not stage3_dir.is_dir():
        raise FileNotFoundError(f"stage 3 directory not found: {stage3_dir}")
    events: list[tuple[str, int, Path]] = []
    for path in sorted(stage3_dir.glob("*.csv")):
        match = EVENT_FILE_RE.match(path.name)
        if not match:
            raise ValueError(f"{path.name}: does not look like a stage 3 event file")
        events.append((match.group("run"), int(match.group("ordinal")), path))
    if not events:
        raise FileNotFoundError(f"no stage 3 event CSV found in {stage3_dir}")
    # Numeric run order (run_2 before run_10), then the event ordinal.
    events.sort(key=lambda item: (run_ordinal(item[0]), item[1]))
    return events


def run_ordinal(label: str) -> int:
    """Numeric part of a ``run_<n>`` label, used as the ordering key."""
    match = RUN_LABEL_RE.match(label)
    return int(match.group("ordinal")) if match else 0


def as_float(row: dict[str, str], field: str, path: Path) -> float:
    """Numeric view of one input column."""
    try:
        return float(row[field])
    except (TypeError, ValueError):
        raise ValueError(f"{path}: non-numeric {field!r} value: {row[field]!r}") from None


def read_event(path: Path, run: str) -> dict[str, list[dict[str, str]]]:
    """Read one stage 3 event file and split it into the two role blocks.

    Validates the shape the merge relies on: both role blocks present and contiguous
    (follower first), strictly ascending raw timestamps, equal row counts and
    element-wise identical follower/leader timestamps.
    """
    if not path.is_file():
        raise FileNotFoundError(f"stage 3 event file not found: {path}")
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
        times = [as_float(r, "t_from_start", path) for r in by_role[role]]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(f"{path}: {role} rows are not strictly ascending in t_from_start")

    if len(by_role["follower"]) != len(by_role["leader"]):
        raise ValueError(
            f"{path}: follower/leader row counts differ "
            f"({len(by_role['follower'])} vs {len(by_role['leader'])})"
        )
    follower_times = [as_float(r, "t_from_start", path) for r in by_role["follower"]]
    leader_times = [as_float(r, "t_from_start", path) for r in by_role["leader"]]
    if follower_times != leader_times:
        raise ValueError(f"{path}: follower and leader timestamps differ (stage 3 must pair them)")
    return by_role


def round6(value: float) -> float:
    """Round for the JSON report so the file stays readable and deterministic."""
    return round(value, OUTPUT_DECIMALS)


def fmt(value: float) -> str:
    """Fixed-point formatting of a written value (deterministic bytes)."""
    return f"{value:.{OUTPUT_DECIMALS}f}"


def quantise_time(value: float, step: float = TIME_STEP_S) -> float:
    """Round a timestamp to the nearest `step` (half up), without float drift.

    `floor(v/step + 0.5) * step` is half-up rather than Python's round-half-even, so
    a timestamp sitting exactly on `.X5` always moves up; the product can carry float
    noise (3.7000000000000002), which the 1-decimal formatting removes.
    """
    return math.floor(value / step + 0.5) * step


def fmt_time(value: float) -> str:
    """`time` column text: the 0.1 s grid, 1 decimal."""
    return f"{value:.{TIME_DECIMALS}f}"


def positions(rows: list[dict[str, str]], path: Path) -> tuple[list[float], list[float], list[float]]:
    """(x, y, raw t) of one role block."""
    return (
        [as_float(r, "x", path) for r in rows],
        [as_float(r, "y", path) for r in rows],
        [as_float(r, "t_from_start", path) for r in rows],
    )


def speed_series(
    rows: list[dict[str, str]], path: Path, indices: list[int] | None = None
) -> tuple[list[float], list[tuple[int, int]]]:
    """Position-difference speed of one role, one value per retained time point.

    `rows` is the full role block and `indices` the ascending grid indices of the
    points that survive the removal rules; the speeds are evaluated on exactly that
    sequence (`indices=None` means every row). A neighbour is therefore the
    neighbouring *retained* point: a removed row can never enter a window, and the
    central difference collapses to a one-sided one at the edge of a removal.

    Interior points: chord between the two neighbours divided by their interval
    (central difference, i.e. / (2*dt) on a contiguous grid). End points: the
    one-sided difference over the single adjacent interval. All dt come from the raw
    stage 3 timestamps, never from the rounded `time` column.

    Returns the speeds and, for the conservation check, the window each speed was
    built from as its two grid indices (lower, upper).
    """
    xs, ys, times = positions(rows, path)
    order = list(range(len(xs))) if indices is None else list(indices)
    count = len(order)
    if count < 2:
        raise ValueError(f"{path}: a role with fewer than 2 retained points has no speed")
    speeds: list[float] = []
    windows: list[tuple[int, int]] = []
    for position in range(count):
        if position == 0:
            lower, upper = order[0], order[1]
        elif position == count - 1:
            lower, upper = order[count - 2], order[count - 1]
        else:
            lower, upper = order[position - 1], order[position + 1]
        dt = times[upper] - times[lower]
        if dt <= 0.0:
            raise ValueError(f"{path}: non-positive dt at grid index {order[position]} ({dt})")
        distance = math.hypot(xs[upper] - xs[lower], ys[upper] - ys[lower])
        speeds.append(distance / dt)
        windows.append((lower, upper))
    return speeds, windows
def dhw_series(
    follower_rows: list[dict[str, str]], leader_rows: list[dict[str, str]], path: Path
) -> list[float]:
    """Forward distance to the leader along the follower heading (no car length).

    dhw = (x_l - x_f)*cos(h_f) + (y_l - y_f)*sin(h_f), `h_f` in radians.
    """
    fx, fy, _ = positions(follower_rows, path)
    lx, ly, _ = positions(leader_rows, path)
    headings = [as_float(r, "heading", path) for r in follower_rows]
    values: list[float] = []
    for i in range(len(fx)):
        radians = math.radians(headings[i])
        values.append((lx[i] - fx[i]) * math.cos(radians) + (ly[i] - fy[i]) * math.sin(radians))
    return values


def median(values: list[float]) -> float:
    """Median of a non-empty series (kept dependency-free)."""
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def build_table(by_role: dict[str, list[dict[str, str]]], path: Path) -> dict[str, object]:
    """Time normalisation + duplicate drop, then dhw, anomalies and velocities.

    One fixed order, so every number in the table comes from the same point set:

      step 1  round `time` to the 0.1 s grid and drop the later point of a collision;
      step 2  `dhw` on the points that survive step 1;
      step 3  drop the non-positive-`dhw` points (anomalies);
      step 4  `fv`/`lv` on the points that survive steps 1 and 3.

    A removed point therefore never feeds a surviving row: no speed window spans it.
    """
    follower_rows, leader_rows = by_role["follower"], by_role["leader"]
    raw_times = [as_float(r, "t_from_start", path) for r in follower_rows]

    rounded_times = [quantise_time(t) for t in raw_times]

    # --- step 1: rounded-time normalisation, the later point of a collision goes ---
    time_kept: list[int] = []
    dropped: list[dict[str, object]] = []
    last_kept: int | None = None
    for index in range(len(raw_times)):
        if last_kept is not None and fmt_time(rounded_times[index]) == fmt_time(rounded_times[last_kept]):
            # Same 0.1 s cell as the row above: this is the off-grid closing instant,
            # so it is the one that goes (the grid point above is kept). Raw timestamps
            # ascend, so equal rounded times are always adjacent.
            dropped.append(
                {
                    "index": index,
                    "t_from_start": round6(raw_times[index]),
                    "time": fmt_time(rounded_times[index]),
                    "previous_index": last_kept,
                    "previous_t_from_start": round6(raw_times[last_kept]),
                    "gap_s": round6(raw_times[index] - raw_times[last_kept]),
                    "action": "dropped",
                    "reason": DROPPED_TIME_DUPLICATE_REASON,
                }
            )
            continue
        time_kept.append(index)
        last_kept = index

    # --- step 2: dhw on the points that survive step 1 ---
    dhw_time_kept = dhw_series(
        [follower_rows[i] for i in time_kept], [leader_rows[i] for i in time_kept], path
    )

    # --- step 3: drop the non-positive dhw points (tracking glitch or overtake) ---
    kept: list[int] = []
    kept_positions: list[int] = []
    deleted: list[dict[str, object]] = []
    for position, index in enumerate(time_kept):
        if dhw_time_kept[position] <= 0.0:
            # No fv/lv on an anomaly: under this order the speeds are computed on the
            # surviving points only, so a removed point simply has none.
            deleted.append(
                {
                    "index": index,
                    "time": fmt_time(rounded_times[index]),
                    "t_from_start": round6(raw_times[index]),
                    "dhw": round6(dhw_time_kept[position]),
                    "reason": "dhw_not_positive",
                }
            )
            continue
        kept.append(index)
        kept_positions.append(position)

    # --- step 4: velocities on the points that survive steps 1 and 3 ---
    fv, fv_windows = speed_series(follower_rows, path, kept)
    lv, lv_windows = speed_series(leader_rows, path, kept)
    dhw = [dhw_time_kept[position] for position in kept_positions]
    if not (len(fv) == len(lv) == len(dhw) == len(kept)):
        raise RuntimeError(f"{path}: speed series length does not match the retained rows")
    retained = set(kept)
    window_endpoints = {index for window in fv_windows + lv_windows for index in window}
    if not window_endpoints <= retained:
        leaked = sorted(window_endpoints - retained)
        raise RuntimeError(f"{path}: speed window used removed point(s) {leaked}")

    rows: list[dict[str, str]] = [
        {
            "time": fmt_time(rounded_times[i]),
            "fv": fmt(fv[k]),
            "lv": fmt(lv[k]),
            "dhw": fmt(dhw[k]),
        }
        for k, i in enumerate(kept)
    ]
    time_texts = [row["time"] for row in rows]
    # Numeric, not lexical: "9.9" < "10.0" as text but not as a time.
    time_values = [float(text) for text in time_texts]
    if any(b <= a for a, b in zip(time_values, time_values[1:])):
        raise RuntimeError(f"{path}: written time column is not strictly increasing")

    def span(values: list[float]) -> dict[str, float]:
        return {"min": round6(min(values)), "max": round6(max(values))} if values else {}

    return {
        "rows": rows,
        "time_texts": time_texts,
        "grid_points": len(raw_times),
        "grid_points_time_kept": len(time_kept),
        "grid_points_kept": len(kept),
        "grid_points_deleted": len(deleted),
        "grid_points_dropped": len(dropped),
        "kept_grid_indices": kept,
        "speed_window_endpoints": sorted(window_endpoints),
        "speed_values_equal_rows": len(rows) == len(fv) == len(lv) == len(dhw),
        "speed_windows_use_kept_points_only": window_endpoints <= retained,
        "deleted_anomalies": deleted,
        "dropped_time_duplicates": dropped,
        "raw_time_range": {"min": round6(raw_times[0]), "max": round6(raw_times[-1])},
        "grid_time_range": {
            "min": fmt_time(rounded_times[0]),
            "max": fmt_time(rounded_times[-1]),
        },
        "time_range": (
            {"min": time_texts[0], "max": time_texts[-1]} if time_texts else {}
        ),
        # fv/lv/dhw all exist on the retained points only, so the ranges below are
        # retained-point ranges by construction (`dhw_range_kept` is kept as an alias
        # for readers of the previous summary shape).
        "dhw_range": span(dhw),
        "dhw_range_kept": span(dhw),
        "dhw_median": round6(median(dhw)) if dhw else None,
        "fv_range": span(fv),
        "lv_range": span(lv),
        "fv_mean": round6(sum(fv) / len(fv)) if fv else None,
        "lv_mean": round6(sum(lv) / len(lv)) if lv else None,
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


def verify_written_file(path: Path, expected_rows: list[dict[str, str]]) -> dict[str, object]:
    """Re-read one written table and re-check every structural property."""
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = tuple(reader.fieldnames or ())
        rows = list(reader)
    problems: list[str] = []
    if header != OUTPUT_FIELDS:
        problems.append(f"header {header} != {OUTPUT_FIELDS}")
    if len(rows) != len(expected_rows):
        problems.append(f"row count {len(rows)} != {len(expected_rows)}")
    if rows != expected_rows:
        for index, (written, expected) in enumerate(zip(rows, expected_rows)):
            if written != expected:
                problems.append(f"row {index} differs from the computed table: {written} != {expected}")
                break

    duplicate_times = 0
    strictly_increasing = True
    previous: float | None = None
    for row in rows:
        try:
            time_value = float(row["time"])
            fv = float(row["fv"])
            lv = float(row["lv"])
            dhw = float(row["dhw"])
        except ValueError:
            problems.append(f"non-numeric value in row {row}")
            continue
        if previous is not None and time_value <= previous:
            strictly_increasing = False
            if time_value == previous:
                duplicate_times += 1
                problems.append(
                    f"duplicate time {row['time']} survived (rows must be unique)"
                )
            else:
                problems.append(f"time {row['time']} is not ascending")
        previous = time_value
        steps = time_value / TIME_STEP_S
        if abs(steps - round(steps)) > 1e-6:
            problems.append(f"time {row['time']} is not on the {TIME_STEP_S} s grid")
        if fv < 0.0 or lv < 0.0:
            problems.append(f"negative speed in row {row}")
        if dhw <= 0.0:
            problems.append(f"non-positive dhw survived: {row['dhw']}")
    return {
        "file": path.name,
        "rows": len(rows),
        "duplicate_time_values": duplicate_times,
        "time_strictly_increasing": strictly_increasing,
        "problems": problems,
    }


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
    stage3_dir = root / STAGE3_SUBDIR
    out_dir = root / OUTPUT_SUBDIR
    events = discover_events(stage3_dir)

    print(f"project root : {root}")
    print(f"stage 3 input: {rel(stage3_dir, root)}")
    print(f"output dir   : {rel(out_dir, root)}")
    print()
    print("configuration:")
    print(f"  time grid                 : {TIME_STEP_S} s (rounded half up, {TIME_DECIMALS} printed decimal)")
    print(f"  dt source                 : raw stage 3 timestamps (rounded `time` never feeds a division)")
    print("  merge order               : (1) drop rounded-time duplicates -> (2) dhw -> (3) drop dhw<=0 -> (4) speeds")
    print("  speeds                    : position difference, central difference inside, one-sided at ends; neighbours are retained points only")
    print("  dhw                       : (x_l - x_f)*cos(h_f) + (y_l - y_f)*sin(h_f), no car length subtracted")
    print("  time duplicates           : the later point of a rounded-time collision is dropped, `time` stays strictly increasing")
    print("  reported velocity column  : discarded (stage 4 recomputes every speed)")
    print(f"  value decimals            : {OUTPUT_DECIMALS}")
    print()
    print(f"events found in stage_3: {len(events)}")
    print()

    removed_entries = rebuild_output_dir(out_dir)
    if removed_entries:
        print(f"rebuilt output dir, removed previous content: {', '.join(removed_entries)}")
    else:
        print("rebuilt output dir: nothing to remove (directory was empty or absent)")
    print()

    summary: dict[str, object] = {
        "script": "src/preprocess_stage4.py",
        "config": {
            "time_step_s": TIME_STEP_S,
            "time_decimals": TIME_DECIMALS,
            "output_decimals": OUTPUT_DECIMALS,
            "speed_method": "central_difference_interior_one_sided_ends",
            "speed_source": "position_difference",
            "dt_source": "raw_stage3_timestamps",
            "dhw_formula": "(x_l - x_f)*cos(h_f) + (y_l - y_f)*sin(h_f)",
            "dhw_anomaly_rule": "delete time point when dhw <= 0",
            "time_duplicate_rule": "drop the later point when its rounded time repeats the row above it",
            "removal_order": "rounded_time_duplicate_drop_then_dhw_anomaly_drop_then_speeds",
            "speed_window_source": "retained_points_only",
            "velocity_column_used": False,
        },
        "inputs": {rel(path, root): {"sha256": sha256_file(path)} for _, _, path in events},
        "events": [],
        "dropped_time_duplicates": [],
        "outputs": [],
        "verification": [],
        "total": {},
    }

    written: list[tuple[str, dict[str, object]]] = []
    verification: list[dict[str, object]] = []
    total_rows = 0
    total_deleted = 0
    total_dropped = 0

    print("per-event merge:")
    for run, ordinal, path in events:
        by_role = read_event(path, run)
        table = build_table(by_role, path)
        name = f"{run}_event_{ordinal}.csv"
        rows: list[dict[str, str]] = table["rows"]  # type: ignore[assignment]
        dropped: list[dict[str, object]] = table["dropped_time_duplicates"]  # type: ignore[assignment]
        dropped_records = [
            {"event": f"event_{ordinal}", "file": name, **item} for item in dropped
        ]

        print(
            f"  {name:<24} pairs: {table['grid_points']:<5}"
            f" out rows: {table['grid_points_kept']:<5}"
            f" deleted: {table['grid_points_deleted']:<3}"
            f" dropped: {table['grid_points_dropped']}"
        )
        print(
            f"      raw t [{table['raw_time_range']['min']}, {table['raw_time_range']['max']}]"  # type: ignore[index]
            f"  -> time [{table['time_range']['min']}, {table['time_range']['max']}] s"  # type: ignore[index]
        )
        print(
            f"      dhw [{table['dhw_range']['min']}, {table['dhw_range']['max']}]"  # type: ignore[index]
            f" (median {table['dhw_median']}), no car length subtracted"
            f"  | fv [{table['fv_range']['min']}, {table['fv_range']['max']}]"  # type: ignore[index]
            f" mean {table['fv_mean']}"
            f"  | lv [{table['lv_range']['min']}, {table['lv_range']['max']}]"  # type: ignore[index]
            f" mean {table['lv_mean']}"
        )
        if table["deleted_anomalies"]:
            for item in table["deleted_anomalies"]:  # type: ignore[union-attr]
                print(f"      DELETED anomaly at {item['time']} s (dhw = {item['dhw']})")
        else:
            print("      deleted anomalies: none (no dhw <= 0)")
        if dropped:
            for item in dropped:
                print(
                    f"      DROPPED duplicate time: t {item['t_from_start']} -> {item['time']} s"
                    f" (row above at t {item['previous_t_from_start']}, gap {item['gap_s']} s,"
                    f" off-grid closing instant)"
                )
        else:
            print("      dropped duplicate times: none (every rounded time is unique)")
        print()

        write_csv(out_dir / name, rows)
        written.append((name, table))
        total_rows += len(rows)
        total_deleted += int(table["grid_points_deleted"])
        total_dropped += int(table["grid_points_dropped"])

        summary["events"].append(  # type: ignore[union-attr]
            {
                "run": run,
                "event": f"event_{ordinal}",
                "file": name,
                "input_file": rel(path, root),
                "input_sha256": sha256_file(path),
                "grid_points": table["grid_points"],
                "rows": len(rows),
                "time_kept_points": table["grid_points_time_kept"],
                "deleted_points": table["grid_points_deleted"],
                "dropped_points": table["grid_points_dropped"],
                "deleted_anomalies": table["deleted_anomalies"],
                "dropped_time_duplicates": dropped_records,
                "time_range": table["time_range"],
                "grid_time_range": table["grid_time_range"],
                "raw_time_range": table["raw_time_range"],
                "dhw_range": table["dhw_range"],
                "dhw_range_kept": table["dhw_range_kept"],
                "dhw_median": table["dhw_median"],
                "fv_range": table["fv_range"],
                "lv_range": table["lv_range"],
                "fv_mean": table["fv_mean"],
                "lv_mean": table["lv_mean"],
                "conservation": {
                    "rows_plus_deleted_plus_dropped_equals_grid_points": len(rows)
                    + int(table["grid_points_deleted"])
                    + int(table["grid_points_dropped"])
                    == int(table["grid_points"]),
                    # Speed side of the new order: exactly one speed per written row and
                    # every speed window endpoint is a retained point (nothing derived
                    # from a removed point can survive in the table).
                    "speed_values_equal_rows": bool(table["speed_values_equal_rows"]),
                    "speed_windows_use_kept_points_only": bool(
                        table["speed_windows_use_kept_points_only"]
                    ),
                },
            }
        )
        summary["dropped_time_duplicates"].extend(dropped_records)  # type: ignore[union-attr]

    print("on-disk verification (re-read after writing):")
    width = max((len(name) for name, _ in written), default=8)
    ok = True
    for name, table in written:
        rows = table["rows"]  # type: ignore[assignment]
        result = verify_written_file(out_dir / name, rows)
        verification.append(result)
        if result["problems"]:
            ok = False
        print(
            f"  {name:<{width}}  rows={result['rows']:<5}"
            f" duplicate time values={result['duplicate_time_values']}"
            f" strictly increasing={result['time_strictly_increasing']}"
            f"  {'OK' if not result['problems'] else 'MISMATCH'}"
        )
        for problem in result["problems"]:  # type: ignore[union-attr]
            print(f"      ! {problem}")
    print(f"  total written rows: {total_rows}")
    print(f"  total deleted anomaly points: {total_deleted}")
    print(f"  total dropped duplicate-time points: {total_dropped}")

    summary["verification"] = verification
    summary["total"] = {
        "input_event_files": len(events),
        "written_event_files": len(written),
        "rows": total_rows,
        "deleted_points": total_deleted,
        "dropped_points": total_dropped,
        "conservation_ok": all(
            record["conservation"]["rows_plus_deleted_plus_dropped_equals_grid_points"]  # type: ignore[index]
            for record in summary["events"]  # type: ignore[union-attr]
        )
        and sum(record["rows"] for record in summary["events"]) == total_rows  # type: ignore[union-attr,arg-type]
        and sum(record["dropped_points"] for record in summary["events"]) == total_dropped  # type: ignore[union-attr,arg-type]
        and all(
            record["conservation"]["speed_values_equal_rows"]  # type: ignore[index]
            and record["conservation"]["speed_windows_use_kept_points_only"]  # type: ignore[index]
            for record in summary["events"]  # type: ignore[union-attr]
        ),
    }
    summary["outputs"] = [{"file": name, "rows": len(table["rows"])} for name, table in written] + [  # type: ignore[arg-type]
        {"file": SUMMARY_FILENAME, "rows": None}
    ]
    write_json(out_dir / SUMMARY_FILENAME, summary)

    print()
    tmp_leftovers = sorted(p.name for p in out_dir.glob("*.tmp"))
    print(f".tmp leftovers: {tmp_leftovers if tmp_leftovers else 'none'}")
    print(f"stage_4 dir contents: {', '.join(sorted(p.name for p in out_dir.iterdir()))}")

    expected_files = sorted([name for name, _ in written] + [SUMMARY_FILENAME])
    actual_files = sorted(p.name for p in out_dir.iterdir())
    if actual_files != expected_files:
        print(f"\nERROR: unexpected stage_4 contents {actual_files} != {expected_files}", file=sys.stderr)
        return 1
    if not ok:
        print("\nERROR: on-disk verification failed.", file=sys.stderr)
        return 1
    if not summary["total"]["conservation_ok"]:  # type: ignore[index]
        print("\nERROR: row conservation failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())