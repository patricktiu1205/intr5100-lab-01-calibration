#!/usr/bin/env python3
"""Historical validation replay for entries in results/final_params.json.

The retained final_params.json contains run_1 only. The selected run_2 fit lives
in m5_params.json and is read by the current report builder; it is not silently
substituted into this older pipeline. The historical IDM/TTC sign conventions
remain documented in review/review_notes.md.

Dry-run example (imports/paths/parameters only; no output files):
    python src/validate_historical_model.py --dry-run --run run_1

With --validation-raw, this command runs the four preprocessing stages in an
isolated validation tree and replays the legacy parameter entry. It can write
new derived outputs. --report invokes the separately named historical HTML
builder. These operations are not required to rebuild the current report and
are not independent validation of a corrected IDM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --- import bootstrap (the project is a set of scripts, not a package) --------

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # src/validate_historical_model.py -> lab1
_SRC = PROJECT_ROOT / "src"
_ANALYSIS = _SRC / "analysis"
_CALIBRATE = _SRC / "calibration"
for _entry in (str(_SRC), str(_ANALYSIS), str(_CALIBRATE)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import idm_core as core  # noqa: E402  (after the path fix, like the calibration scripts)
import metrics as mt  # noqa: E402
import preprocess_stage1 as s1  # noqa: E402
import preprocess_stage2 as s2  # noqa: E402
import preprocess_stage3 as s3  # noqa: E402
import preprocess_stage4 as s4  # noqa: E402

SCRIPT = "src/validate_historical_model.py"
STAGE_MODULES = {1: s1, 2: s2, 3: s3, 4: s4}
# Which module constant names the input directory of each stage. Stage 1 has no
# input directory: it reads one raw collection file, handled by run_stage1().
STAGE_INPUT_ATTR = {2: "STAGE1_SUBDIR", 3: "STAGE2_SUBDIR", 4: "STAGE3_SUBDIR"}
STAGE_SUMMARY_FILES = {
    2: "stage2_summary.json",
    3: "stage3_summary.json",
    4: "stage4_summary.json",
}
PREPROCESS_SUBDIR = Path("data") / "preprocess"
VALIDATION_PREFIX = "validation_"
HUMAN_STAGE_DIRS = tuple(PREPROCESS_SUBDIR / f"stage_{index}" for index in (1, 2, 3, 4))

# Stages 2..4 accept only ``run_<n>`` labels, so a validation label such as
# ``152115`` is mapped to the stage label ``run_152115`` inside its own tree.
STAGE_LABEL_RE = re.compile(r"^run_(?P<ordinal>\d+)$")


def stage_label_for(label: str) -> str:
    """Label of the stage 1..4 tables inside a validation tree (``run_<n>`` shape)."""
    return label if STAGE_LABEL_RE.match(label) else f"run_{label}"


# --- paths and hashing --------------------------------------------------------


def rel(path: Path, root: Path) -> str:
    """Project-relative POSIX path (reports never carry absolute paths)."""
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def sha256_file(path: Path) -> str:
    """Content hash of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hashes(root: Path, subdirs) -> dict[str, str]:
    """``{relative path: sha256}`` of every file under the given subdirectories."""
    hashes: dict[str, str] = {}
    for subdir in subdirs:
        directory = root / subdir
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                hashes[rel(path, root)] = sha256_file(path)
    return hashes


def validation_base(root: Path, run: str) -> Path:
    """``data/preprocess/validation_<run>`` - the isolated output tree."""
    return root / PREPROCESS_SUBDIR / f"{VALIDATION_PREFIX}{run}"


def validation_stage_dir(root: Path, run: str, stage: int) -> Path:
    """``data/preprocess/validation_<run>/stage_<n>``."""
    return validation_base(root, run) / f"stage_{stage}"


def _guard(base: Path, path: Path) -> None:
    """Refuse to touch anything that is not inside a ``validation_*`` tree."""
    if not base.name.startswith(VALIDATION_PREFIX) or base.parent.name != "preprocess":
        raise RuntimeError(f"refusing to use {path}: {base} is not a validation tree")
    if path != base and base not in path.parents:
        raise RuntimeError(f"refusing to use {path}: not inside {base}")


def _clear_dir(path: Path, base: Path) -> list[str]:
    """Empty one stage directory (validated path), returning the names removed."""
    _guard(base, path)
    if not path.exists():
        return []
    removed = sorted(entry.name for entry in path.iterdir())
    for entry in sorted(path.iterdir()):
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    return removed


# --- stage 1 (function-level reuse) ------------------------------------------


def run_stage1(root: Path, run: str, raw_file: Path, out_dir: Path) -> dict[str, object]:
    """Trim the raw collection to the stage 1 shape, reusing the stage 1 functions.

    ``preprocess_stage1.main()`` has the two human collections hard-coded in
    ``RUN_SOURCES``, so the validation run calls its processing functions
    directly instead. The functions themselves - reading, role counting, column
    projection and ordering, atomic writing - are the originals, unmodified.
    """
    rows = s1.read_raw(raw_file)
    counts = s1.count_roles(rows, raw_file)
    out_rows = s1.to_output_rows(rows, run)
    out_path = out_dir / f"{run}.csv"
    s1.write_csv(out_path, out_rows)
    on_disk = s1.count_data_rows(out_path)
    header = list(s1.read_header(out_path))
    print(f"[stage 1] {rel(raw_file, root)}: {len(rows)} raw rows "
          f"({', '.join(f'{role}={counts[role]}' for role in s1.ROLES)})")
    print(f"[stage 1] wrote {rel(out_path, root)}: {len(out_rows)} rows "
          f"(on disk {on_disk}, header {','.join(header)})")
    if on_disk != len(out_rows):
        raise RuntimeError(f"stage 1 row count mismatch: {on_disk} != {len(out_rows)}")
    if header != list(s1.OUTPUT_FIELDS):
        raise RuntimeError(f"stage 1 output header is not the stage 1 header: {header}")
    return {
        "input_file": rel(raw_file, root),
        "input_rows": len(rows),
        "role_counts": counts,
        "output_file": rel(out_path, root),
        "output_rows": len(out_rows),
        "on_disk_rows": on_disk,
        "header": header,
    }


# --- stages 2..4 (original main(), rebound directories) ----------------------


def run_stage(root: Path, module, stage: int, input_dir: Path, out_dir: Path) -> int:
    """Call one stage's own ``main()`` with its directory constants rebound.

    The stage script is used exactly as it runs standalone: its trimming, its
    alignment, its grid, its merge, its summary JSON, its stdout report and its
    own on-disk verification are the originals. Only the two path constants are
    swapped for the validation ones, and they are restored afterwards, so a later
    in-process call cannot inherit the validation paths.
    """
    input_attr = STAGE_INPUT_ATTR[stage]
    saved = {input_attr: getattr(module, input_attr), "OUTPUT_SUBDIR": module.OUTPUT_SUBDIR}
    try:
        setattr(module, input_attr, input_dir.relative_to(root))
        module.OUTPUT_SUBDIR = out_dir.relative_to(root)
        print(f"[stage {stage}] {input_attr}={rel(input_dir, root)}  "
              f"OUTPUT_SUBDIR={rel(out_dir, root)}")
        code = module.main(["--root", str(root)])
    finally:
        for attr, value in saved.items():
            setattr(module, attr, value)
    print(f"[stage {stage}] exit code {code}")
    if code != 0:
        raise RuntimeError(f"stage {stage} failed with exit code {code}")
    return int(code)


def read_summary(out_dir: Path, filename: str) -> dict:
    """Read a stage summary JSON (the provenance the report quotes)."""
    path = out_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"stage summary not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# --- self-checks --------------------------------------------------------------


def check_validation_tables(stage4_dir: Path, run: str) -> dict[str, object]:
    """Structural checks of the produced validation tables.

    At least two samples, strictly increasing ``time``, positive ``dhw``,
    non-negative speeds and the header ``time,fv,lv,dhw`` - the properties the
    calibration side relies on. Each check is a boolean in the returned record, so
    the artifact says exactly what was verified.
    """
    files = sorted(stage4_dir.glob(f"{run}_event_*.csv"))
    problems: list[str] = []
    per_event: dict[str, object] = {}
    for path in files:
        frame = pd.read_csv(path)
        times = frame["time"].to_numpy(dtype=float)
        record = {
            "rows": int(len(frame)),
            "header_ok": tuple(frame.columns) == core.INPUT_COLUMNS,
            "time_strictly_increasing": bool(np.all(np.diff(times) > 0.0)),
            "positive_spacing": bool(np.all(frame["dhw"].to_numpy(dtype=float) > 0.0)),
            "non_negative_speeds": bool(
                np.all(frame["fv"].to_numpy(dtype=float) >= 0.0)
                and np.all(frame["lv"].to_numpy(dtype=float) >= 0.0)
            ),
            "at_least_two_samples": bool(len(frame) >= 2),
        }
        for key in (
            "header_ok",
            "time_strictly_increasing",
            "positive_spacing",
            "non_negative_speeds",
            "at_least_two_samples",
        ):
            if not record[key]:
                problems.append(f"{path.name}: {key} is false")
        per_event[path.name] = record
    if not files:
        problems.append(f"no {run}_event_*.csv found in {stage4_dir}")
    return {"events": per_event, "n_events": len(files), "problems": problems, "ok": not problems}


def check_prediction_seam(
    stage4_dir: Path,
    run_label: str,
    run: dict[str, object],
    theta,
    method: str,
) -> dict[str, object]:
    """The two RMSEs of the reused regression code must equal idm_core's own.

    ``metrics.regression_metrics`` (the cross-validation code) recomputes the
    errors from the residual series while ``idm_core.event_metrics`` reports them
    itself; on the same simulation the two must agree to floating-point noise.
    That seam is what would silently break if either side were changed, so it is
    checked mechanically instead of by eye.

    ``run_label`` is the run the tables belong to, ``run`` the evaluated metric
    block (the naming of ``metrics.evaluate_run``) - the two must not share a name.
    """
    checks: dict[str, bool] = {}
    for name, block in run["events"].items():
        frame = pd.read_csv(stage4_dir / f"{run_label}_{name}.csv")
        reference = core.event_metrics(core.to_event_data(frame, name), theta, method)
        checks[f"{name}_spacing_rmse_agrees"] = bool(
            math.isclose(float(block["prediction"]["spacing"]["rmse"]),
                         float(reference["spacing_rmse"]), rel_tol=1e-9, abs_tol=1e-9)
        )
        checks[f"{name}_speed_rmse_agrees"] = bool(
            math.isclose(float(block["prediction"]["speed"]["rmse"]),
                         float(reference["speed_rmse"]), rel_tol=1e-9, abs_tol=1e-9)
        )
        checks[f"{name}_n_samples_agrees"] = bool(
            int(block["prediction"]["n_samples"]) == int(reference["n_samples"])
        )
    return {
        "per_event": checks,
        "all_match": all(checks.values()) if checks else False,
        "note": (
            "metrics.regression_metrics recomputes the errors from the residual series "
            "and must reproduce the RMSEs reported by idm_core.event_metrics for the "
            "same simulation (tolerance 1e-9)"
        ),
    }


# --- dry run ------------------------------------------------------------------


def dry_run(root: Path, run: str, raw_dir: Path | None) -> int:
    """Check every import, path and parameter that does not need data.

    Returns 0 when nothing is broken (missing validation data is a WARNING, not a
    failure: it does not exist yet by definition) and 1 when a real problem was
    found.
    """
    checks: list[tuple[str, str, str]] = []  # (severity, name, detail)

    def record(ok: bool, name: str, detail: str, warn_only: bool = False) -> None:
        severity = "ok" if ok else ("warn" if warn_only else "FAIL")
        checks.append((severity, name, detail))

    print("=" * 78)
    print(f"DRY RUN - {SCRIPT}")
    print("=" * 78)
    print(f"project root : {root}")
    print(f"run          : {run}")
    print(f"python       : {sys.version.split()[0]} ({sys.executable})")
    print()

    # 1. imports
    record(
        True,
        "imports: preprocessing stages",
        ", ".join(f"preprocess_stage{index}" for index in sorted(STAGE_MODULES)),
    )
    record(True, "imports: idm_core", f"L_VEH={core.L_VEH:g} DELTA={core.DELTA:g} method={core.DEFAULT_METHOD}")
    record(True, "imports: metrics", f"{len(mt.thresholds()['derivation'])} threshold group(s)")
    try:
        import cross_validate  # noqa: F401  (the import cost is part of the check)

        record(True, "imports: cross_validate (regression metrics)", "imported")
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        record(False, "imports: cross_validate (regression metrics)", f"{type(exc).__name__}: {exc}")
    try:
        import matplotlib

        record(True, "imports: matplotlib", f"{matplotlib.__version__} (backend {matplotlib.get_backend()})")
    except Exception as exc:  # noqa: BLE001
        record(False, "imports: matplotlib (figure rendering of the report)", f"{type(exc).__name__}: {exc}")
    try:
        import build_historical_report  # noqa: F401

        record(True, "imports: build_historical_report", "imported")
    except Exception as exc:  # noqa: BLE001
        record(False, "imports: build_historical_report", f"{type(exc).__name__}: {exc}")

    # 2. calibration artefacts / parameters
    try:
        available = mt.runs_with_final_params(root)
        record(run in available, "parameters: final_params.json",
               f"runs with a parameter vector: {available}")
        theta, entry = mt.theta_for_run(run, root)
        record(
            True,
            f"parameters: theta of {run}",
            f"{core.format_theta(theta)}  (method {entry.get('method')}, "
            f"within bounds {core.inside_bounds(theta)})",
        )
    except Exception as exc:  # noqa: BLE001
        record(False, "parameters: final_params.json", f"{type(exc).__name__}: {exc}")

    cv_path = root / core.RESULTS_SUBDIR / "cv_results.json"
    record(
        cv_path.is_file(),
        "parameters: cv_results.json (report section 3)",
        rel(cv_path, root) if cv_path.is_file() else "missing",
    )

    # 3. human data (the reference side of every comparison)
    try:
        grouped = core.load_events(root)
        record(
            run in grouped,
            "human tables: data/preprocess/stage_4",
            "events per run: " + ", ".join(f"{name}={len(events)}" for name, events in grouped.items()),
        )
    except Exception as exc:  # noqa: BLE001
        record(False, "human tables: data/preprocess/stage_4", f"{type(exc).__name__}: {exc}")

    # 4. validation input
    if raw_dir is None:
        record(False, "validation input", "--validation-raw not given (the run needs it)", warn_only=True)
    else:
        raw_file = raw_dir / s1.RAW_FILENAME
        if not raw_file.is_file():
            record(False, "validation input",
                   f"{rel(raw_file, root)} absent - expected while no validation run exists yet",
                   warn_only=True)
        else:
            try:
                rows = s1.read_raw(raw_file)
                counts = s1.count_roles(rows, raw_file)
                record(True, "validation input",
                       f"{rel(raw_file, root)}: {len(rows)} rows, "
                       f"{ {role: counts[role] for role in s1.ROLES} }")
            except Exception as exc:  # noqa: BLE001
                record(False, "validation input", f"{type(exc).__name__}: {exc}")

    # 5. output plan
    base = validation_base(root, run)
    try:
        for stage in sorted(STAGE_MODULES):
            _guard(base, validation_stage_dir(root, run, stage))
        record(True, "output plan: validation tree", f"{rel(base, root)}/stage_1..4 (isolated)")
        record(
            base.is_relative_to(root / PREPROCESS_SUBDIR),
            "output plan: isolation",
            "inside data/preprocess and prefixed 'validation_' - cannot collide with the human stages",
        )
    except Exception as exc:  # noqa: BLE001
        record(False, "output plan: validation tree", f"{type(exc).__name__}: {exc}")

    out_json = root / core.RESULTS_SUBDIR / f"validation_{run}.json"
    record(True, "output plan: run summary", rel(out_json, root))
    record(
        not out_json.with_name(out_json.name + ".tmp").exists(),
        "repository state: no .tmp leftovers",
        "none",
    )

    # 6. report
    print("checks:")
    width = max(len(name) for _severity, name, _detail in checks)
    failures = 0
    for severity, name, detail in checks:
        if severity == "FAIL":
            failures += 1
        print(f"  [{severity:<4}] {name:<{width}}  {detail}")
    print()
    print("commands once the validation data exists:")
    print(f"  python src/validate_historical_model.py --validation-raw <dir> --run {run}")
    print(f"  python src/analysis/build_historical_report.py --run {run}")
    print("  (or both at once: append --report to the pipeline command)")
    print()
    if failures:
        print(f"DRY RUN: {failures} failure(s) - the pipeline is not ready.", file=sys.stderr)
        return 1
    print("DRY RUN: all checks passed (missing validation data is expected and reported as a warning).")
    return 0


# --- main ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="project root (default: inferred)")
    parser.add_argument("--run", required=True, help="run whose calibrated parameters the validation run replays")
    parser.add_argument(
        "--validation-raw",
        type=Path,
        default=None,
        help=f"raw validation collection directory (must contain {s1.RAW_FILENAME})",
    )
    parser.add_argument("--method", default=core.DEFAULT_METHOD, choices=list(core.METHODS),
                        help="integration scheme of the IDM replay (the calibration default)")
    parser.add_argument("--dry-run", action="store_true",
                        help="check imports/paths/parameters, write nothing")
    parser.add_argument("--report", action="store_true", help="run the HTML report generator afterwards")
    parser.add_argument("--demo", action="store_true",
                        help="mark the report as a DEMO and use the demo output filename")
    parser.add_argument("--out", type=Path, default=None, help="HTML output path (implies --report)")
    args = parser.parse_args(argv)

    root = core.resolve_root(args.root)
    raw_dir = args.validation_raw
    if raw_dir is not None and not raw_dir.is_absolute():
        raw_dir = (root / raw_dir).resolve()

    if args.dry_run:
        return dry_run(root, args.run, raw_dir)

    if raw_dir is None:
        parser.error("--validation-raw is required unless --dry-run is used")

    started = time.perf_counter()
    available = mt.runs_with_final_params(root)
    if args.run not in available:
        parser.error(f"--run {args.run} has no entry in final_params.json (available: {available})")
    theta, final_entry = mt.theta_for_run(args.run, root)

    print("=" * 78)
    print(f"{SCRIPT} - validation run pipeline")
    print("=" * 78)
    print(f"project root    : {root}")
    print(f"run          : {args.run}")
    print(f"validation raw  : {rel(raw_dir, root)}")
    print(f"validation tree : {rel(validation_base(root, args.run), root)}/stage_1..4")
    print(f"IDM parameters  : {core.format_theta(theta)}   (method {final_entry.get('method')}, "
          f"delta {core.DELTA:g}, L_VEH {core.L_VEH:g})")
    print(f"integration     : {args.method}, first measured (s, v) as the initial state")
    print()

    # --- isolation evidence: hash the human tree before the run ---------------
    before = tree_hashes(root, HUMAN_STAGE_DIRS)
    print(f"human stage trees hashed before the run: {len(before)} file(s)")
    print()

    # --- 1..4 preprocessing ---------------------------------------------------
    base = validation_base(root, args.run)
    _guard(base, base)
    for stage in sorted(STAGE_MODULES):
        out_dir = validation_stage_dir(root, args.run, stage)
        removed = _clear_dir(out_dir, base)
        if removed:
            print(f"[stage {stage}] cleared {len(removed)} previous file(s) from {rel(out_dir, root)}")
    print()

    raw_file = raw_dir / s1.RAW_FILENAME
    if not raw_file.is_file():
        print(f"ERROR: the validation collection has no {s1.RAW_FILENAME}: {raw_file}", file=sys.stderr)
        return 1

    stage1_info = run_stage1(root, args.run, raw_file, validation_stage_dir(root, args.run, 1))
    print()
    for stage in (2, 3, 4):
        run_stage(
            root,
            STAGE_MODULES[stage],
            stage,
            validation_stage_dir(root, args.run, stage - 1),
            validation_stage_dir(root, args.run, stage),
        )
        print()

    # --- 5 IDM replay + metrics ----------------------------------------------
    stage4_dir = validation_stage_dir(root, args.run, 4)
    stage4_subdir = PREPROCESS_SUBDIR / f"{VALIDATION_PREFIX}{args.run}" / "stage_4"
    run = mt.evaluate_run(root, args.run, stage4_subdir, "validation", args.method)
    print(f"[replay] {run['n_events']} validation event(s) simulated with the {args.run} parameters")
    mt.print_run(run)
    print()

    checks = check_validation_tables(stage4_dir, args.run)
    seam = check_prediction_seam(stage4_dir, args.run, run, theta, args.method)
    divergence = [name for name, block in run["events"].items() if not block["prediction"]["feasible"]]
    deviations = int(run["prediction_aggregate"]["total_s_violations"])

    # --- isolation evidence: hash the human tree after the run ----------------
    after = tree_hashes(root, HUMAN_STAGE_DIRS)
    changed = sorted(
        set(before) ^ set(after)
        | {path for path in set(before) & set(after) if before[path] != after[path]}
    )
    isolation = {
        "human_files_hashed": len(before),
        "changed_or_added": changed,
        "unchanged": not changed,
        "note": "sha256 of every file under data/preprocess/stage_1..4, before and after the run",
    }
    print(f"[isolation] human stage trees unchanged: {isolation['unchanged']} "
          f"({len(before)} file(s) hashed, {len(changed)} changed)")

    summaries = {
        stage: read_summary(validation_stage_dir(root, args.run, stage), filename)
        for stage, filename in STAGE_SUMMARY_FILES.items()
    }
    print(f"[summaries] stage 2 events={summaries[2]['total']['event_files']}, "
          f"stage 3 written={summaries[3]['total']['written_event_files']} "
          f"(deleted {summaries[3]['total']['deleted_event_files']}), "
          f"stage 4 rows={summaries[4]['total']['rows']}")
    print()

    self_checks = {
        "validation_tables": checks,
        "prediction_seam": seam,
        "no_diverged_event": not divergence,
        "diverged_events": divergence,
        "s_violations_total": deviations,
        "human_tree_untouched": isolation,
        "all_passed": bool(checks["ok"] and seam["all_match"] and not divergence and isolation["unchanged"]),
    }

    payload: dict[str, object] = {
        "script": SCRIPT,
        "stage": "validation run: stage 1..4 preprocessing + IDM replay + metrics",
        "run": args.run,
        "validation_raw": rel(raw_dir, root),
        "validation_tree": rel(base, root),
        # True when this run used --demo, i.e. when the "validation" tables are a
        # real collection standing in for a validation run. Recorded so the report
        # can warn instead of letting demo data pass as a validation result.
        "demo": bool(args.demo),
        "method": args.method,
        "L_VEH": core.L_VEH,
        "delta": core.DELTA,
        "theta": core.theta_to_dict(theta),
        "theta_source": {
            "file": f"{core.RESULTS_SUBDIR.as_posix()}/final_params.json",
            "method": final_entry.get("method"),
            "cv_score": final_entry.get("cv_score"),
        },
        "stage1": stage1_info,
        "stage_summaries": {
            f"stage_{stage}": {
                "file": rel(validation_stage_dir(root, args.run, stage) / filename, root),
                "total": summaries[stage].get("total"),
            }
            for stage, filename in STAGE_SUMMARY_FILES.items()
        },
        "run": run,
        "thresholds": mt.thresholds(),
        "self_checks": self_checks,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    out_json = core.results_dir(root) / f"validation_{args.run}.json"
    core.write_json_atomic(out_json, payload)

    print(f"[self-checks] tables ok={checks['ok']}, seam ok={seam['all_match']}, "
          f"no diverged event={not divergence}, human tree untouched={isolation['unchanged']}")
    print(f"written: {rel(out_json, root)}  ({payload['elapsed_seconds']:.1f} s)")

    if args.report or args.out is not None:
        import build_historical_report

        report_argv = ["--root", str(root), "--run", args.run, "--method", args.method]
        if args.out is not None:
            report_argv += ["--out", str(args.out)]
        if args.demo:
            report_argv += [
                "--demo",
                "DEMO - no validation run exists yet. The validation tables of this report come "
                f"from the human collection {payload['validation_raw']} (the other run's "
                f"recording), pushed through the same stage 1..4 pipeline as if it were the "
                f"{args.run} validation run. The preprocessing, the IDM replay, the metrics and "
                "the figures are the real code paths; only the role of the validation input is "
                "simulated, so the model-transfer error visible in section 4 is expected (the "
                f"parameters of {args.run} are being applied to a trajectory it was never fitted "
                "to) and is not a defect of the pipeline.",
            ]
        code = build_historical_report.main(report_argv)
        if code != 0:
            print(f"ERROR: the HTML report generator returned {code}", file=sys.stderr)
            return code

    if not self_checks["all_passed"]:
        print(f"WARNING: a self-check of the validation run failed - see {rel(out_json, root)}",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
