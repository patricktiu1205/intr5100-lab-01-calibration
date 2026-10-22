#!/usr/bin/env python3
"""INTR5100 Lab Session I - self-contained HTML report of the whole study.

One file, no external resource: every figure is a matplotlib PNG embedded as a
base64 data URI and the styling is plain CSS in a ``<style>`` block, so the report
opens offline in any browser (and prints to PDF from there).

Sections:
  1. experiment overview: the IDM model, its equations and every parameter;
  2. data processing: the stage 1 -> 4 pipeline, the stage configuration and the
     row counts / event inventory read from the stage summary JSONs;
  3. calibration: the five methods, their cross-validation budgets, the CV
     ranking with MSE / RMSE / MAE / MAPE / R2 of spacing and speed, the final
     parameter vector of the run and the meaning of each parameter;
  4. validation: measured versus simulated spacing and speed, one figure per
     event, plus the prediction metric tables of both runs;
  5. driving style: efficiency / comfort / safety metrics of the human and the
     validation run, compared in a table and in a radar / bar chart, with short
     conclusions derived from the numbers;
  6. limitations.

Inputs are the artifacts the earlier steps already produced - ``results/
final_params.json``, ``results/cv_results.json``, the stage summary JSONs, the
stage 4 tables of both runs and (when it exists) ``results/validation_<run>
.json`` - so this script never recomputes a calibration and never invents a
number. When no validation run exists yet the report is still generated: section
4 and the validation half of section 5 then say so explicitly instead of showing
an empty table.

Usage:
    python src/analysis/build_historical_report.py --run run_1
    python src/analysis/build_historical_report.py --run run_1 --demo "why this is a demo"
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

# --- import bootstrap (the project is a set of scripts, not a package) --------

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/analysis/... -> lab1
_SRC = PROJECT_ROOT / "src"
_ANALYSIS = _SRC / "analysis"
_CALIBRATE = _SRC / "calibration"
for _entry in (str(_SRC), str(_ANALYSIS), str(_CALIBRATE)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # no display: the figures are rendered to memory
import matplotlib.pyplot as plt  # noqa: E402

import idm_core as core  # noqa: E402
import metrics as mt  # noqa: E402

SCRIPT = "src/analysis/build_historical_report.py"
RESULTS_SUBDIR = core.RESULTS_SUBDIR
HUMAN_STAGE4_SUBDIR = core.STAGE4_SUBDIR
CV_FILENAME = "cv_results.json"
FINAL_PARAMS_FILENAME = "final_params.json"
DEFAULT_OUTPUT = "historical_analysis.html"
DEMO_OUTPUT = "historical_analysis_demo.html"
PREPROCESS_SUBDIR = Path("data") / "preprocess"

HUMAN_COLOR = "#1f4e79"
VALIDATION_COLOR = "#c0504d"
MEASURED_COLOR = "#111111"

METHOD_ORDER = ("M1", "M2", "M3", "M4", "M5")
# Relative change above which the automatic conclusions call a difference material.
MATERIALITY_PERCENT = 5.0


# --- small helpers ------------------------------------------------------------


def rel(path: Path, root: Path) -> str:
    """Project-relative POSIX path."""
    return core.rel(path, root)


def read_json(path: Path) -> dict:
    """Read one JSON artifact (fails loudly: a missing input is not a zero)."""
    if not path.is_file():
        raise FileNotFoundError(f"required input not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_csv_rows(path: Path) -> int | None:
    """Data rows of a CSV file (header excluded), or None when it is absent."""
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def esc(text) -> str:
    """Minimal HTML escaping for interpolated text."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def num(value, decimals: int = 3, dash: str = "n/a") -> str:
    """Fixed-point cell text; ``n/a`` when the value does not exist."""
    if value is None:
        return dash
    try:
        number = float(value)
    except (TypeError, ValueError):
        return esc(value)
    if not np.isfinite(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:,.{decimals}f}"


def signed(value, decimals: int = 3) -> str:
    """Signed fixed-point cell text (for differences)."""
    if value is None:
        return "n/a"
    number = float(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:+,.{decimals}f}"


def pct(value, decimals: int = 1) -> str:
    """Signed percentage cell text."""
    if value is None:
        return "n/a"
    return f"{float(value):+.{decimals}f}%"


def table(headers, rows, caption: str = "", cls: str = "", aligns=None) -> str:
    """Render a styled HTML table.

    ``headers`` is either a flat list of column titles (one header row) or a list
    of such rows whose entries are ``(text, colspan)`` pairs, which gives a grouped
    header (the spans of the first row must add up to the number of leaf columns,
    i.e. to the number of cells the data rows carry). ``aligns`` is a per-column
    string of ``l`` / ``r``; by default everything but the first column is
    right-aligned.
    """
    if headers and isinstance(headers[0], list):
        header_rows: list[list[tuple[str, int]]] = [
            [
                (entry[0], int(entry[1])) if isinstance(entry, tuple) else (str(entry), 1)
                for entry in row
            ]
            for row in headers
        ]
        columns = sum(span for _text, span in header_rows[-1])
    else:
        header_rows = [[(str(text), 1) for text in headers]]
        columns = len(headers)
    if aligns is None:
        aligns = "l" + "r" * (columns - 1)
    for row in header_rows:
        if sum(span for _text, span in row) != columns:
            raise ValueError(f"a header row spans a different number of columns than the table ({columns})")
    if len(aligns) != columns:
        raise ValueError(f"aligns has {len(aligns)} entries, the table has {columns} columns")
    for values in rows:
        if len(values) != columns:
            raise ValueError(f"row has {len(values)} cells, the table has {columns} columns")

    out = [f'<table class="{cls}">' if cls else "<table>"]
    if caption:
        out.append(f"<caption>{caption}</caption>")
    out.append("<thead>")
    for row in header_rows:
        cells = "".join(f'<th colspan="{int(span)}">{esc(text)}</th>' for text, span in row)
        out.append(f"<tr>{cells}</tr>")
    out.append("</thead><tbody>")
    for values in rows:
        cells = "".join(
            f'<td class="{"num" if align == "r" else "txt"}">{value}</td>'
            for value, align in zip(values, aligns)
        )
        out.append(f"<tr>{cells}</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def figure(png_base64: str, caption: str, label: str = "") -> str:
    """One embedded figure with its caption."""
    prefix = f"<strong>{esc(label)}</strong> " if label else ""
    return (
        f'<figure><img src="data:image/png;base64,{png_base64}" alt="{esc(caption)}">'
        f"<figcaption>{prefix}{esc(caption)}</figcaption></figure>"
    )


def _png(fig) -> str:
    """Render one matplotlib figure to a base64 PNG and close it."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# --- figures ------------------------------------------------------------------


def fig_event_comparison(name: str, side: str, prediction: dict) -> str:
    """Measured versus simulated spacing and follower speed of one event."""
    observed = prediction["observed"]
    simulated = prediction["simulated"]
    time = np.asarray(observed["time_s"], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.6), sharex=True)
    axis = axes[0]
    axis.plot(time, observed["spacing_cu"], color=MEASURED_COLOR, linewidth=1.6, label="measured")
    axis.plot(time, simulated["spacing_cu"], color=VALIDATION_COLOR, linewidth=1.4, linestyle="--",
              label="IDM simulation")
    axis.set_ylabel("spacing s = dhw - 250  (cu)")
    axis.set_title(f"{side} {name}: spacing (top) and follower speed (bottom), IDM replay")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="best")
    axis = axes[1]
    axis.plot(time, observed["speed_cu_s"], color=MEASURED_COLOR, linewidth=1.6, label="measured")
    axis.plot(time, simulated["speed_cu_s"], color=VALIDATION_COLOR, linewidth=1.4, linestyle="--",
              label="IDM simulation")
    axis.plot(time, observed["leader_speed_cu_s"], color="#7f7f7f", linewidth=1.0, alpha=0.8,
              label="measured leader speed (model input)")
    axis.set_xlabel("time (s)")
    axis.set_ylabel("speed  (cu/s)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="best")
    metrics_block = prediction
    axes[0].text(
        0.995, 0.04,
        f"spacing RMSE {num(metrics_block['spacing']['rmse'], 2)} cu, "
        f"R2 {num(metrics_block['spacing']['r2'], 3)}",
        transform=axes[0].transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc"),
    )
    axes[1].text(
        0.995, 0.04,
        f"speed RMSE {num(metrics_block['speed']['rmse'], 2)} cu/s, "
        f"R2 {num(metrics_block['speed']['r2'], 3)}",
        transform=axes[1].transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc"),
    )
    fig.tight_layout()
    return _png(fig)


def fig_cv_scores(run: str, aggregate: dict, best_method: str) -> str:
    """Cross-validation score of the five methods (lower is better)."""
    methods = [
        method for method in METHOD_ORDER
        if aggregate["per_method"].get(method, {}).get("score") is not None
    ]
    scores = [float(aggregate["per_method"][method]["score"]) for method in methods]
    order = np.argsort(scores)
    fig, axis = plt.subplots(figsize=(6.4, 3.2))
    colors = ["#1f4e79" if methods[index] == best_method else "#9db7d0" for index in order]
    axis.bar([methods[index] for index in order], [scores[index] for index in order], color=colors)
    for position, index in enumerate(order):
        axis.text(position, scores[index], f"{scores[index]:.3f}", ha="center", va="bottom", fontsize=8)
    axis.set_ylabel("CV score (lower is better)")
    axis.set_title(f"{run}: leave-one-event-out cross-validation score")
    axis.grid(axis="y", alpha=0.3)
    axis.set_ylim(0.0, max(scores) * 1.18)
    fig.tight_layout()
    return _png(fig)


def fig_radar(human_scores: dict, validation_scores: dict) -> str:
    """Three-axis radar of the normalised efficiency / comfort / safety scores."""
    labels = ["Efficiency", "Comfort", "Safety"]
    keys = ["efficiency", "comfort", "safety"]
    angles = np.linspace(0.0, 2.0 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, axis = plt.subplots(figsize=(5.4, 4.6), subplot_kw=dict(polar=True))
    for label, scores, color in (
        ("human run", human_scores, HUMAN_COLOR),
        ("validation run", validation_scores, VALIDATION_COLOR),
    ):
        values = [float(scores[key]) for key in keys]
        closed = values + values[:1]
        axis.plot(angles, closed, color=color, linewidth=1.8, label=label)
        axis.fill(angles, closed, color=color, alpha=0.12)
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(labels, fontsize=10)
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks([0.25, 0.5, 0.75, 1.0])
    axis.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    axis.grid(alpha=0.4)
    axis.set_title("Driving-style scores (normalised, higher is better)", fontsize=10, pad=16)
    axis.legend(loc="upper right", bbox_to_anchor=(1.34, 1.12), fontsize=8)
    fig.tight_layout()
    return _png(fig)


def fig_style_bars(labels, human_values, validation_values) -> str:
    """Grouped bars of the normalised sub-indicators of the style scores."""
    positions = np.arange(len(labels), dtype=float)
    width = 0.38
    fig, axis = plt.subplots(figsize=(9.0, 3.6))
    axis.bar(positions - width / 2.0, human_values, width, color=HUMAN_COLOR, label="human run")
    axis.bar(positions + width / 2.0, validation_values, width, color=VALIDATION_COLOR,
             label="validation run")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=8, rotation=18, ha="right")
    axis.set_ylim(0.0, 1.08)
    axis.set_ylabel("normalised score (0-1)")
    axis.set_title("Style sub-indicators (higher is better, formulas in the table above)", fontsize=10)
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _png(fig)


# --- html assembly ------------------------------------------------------------


CSS = """
:root { --ink:#1a1a1a; --muted:#5a5a5a; --line:#d6d6d6; --accent:#1f4e79; --warn:#8a5a00; }
* { box-sizing: border-box; }
body { margin:0; padding:32px 20px 64px; background:#f7f7f5; color:var(--ink);
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size:15px; line-height:1.55; }
main { max-width: 980px; margin: 0 auto; background:#fff; padding:36px 44px 52px;
  border:1px solid var(--line); box-shadow:0 1px 3px rgba(0,0,0,.06); }
h1 { font-size:25px; margin:0 0 4px; line-height:1.25; }
h2 { font-size:19px; margin:36px 0 10px; padding-bottom:6px; border-bottom:2px solid var(--accent); }
h3 { font-size:16px; margin:24px 0 8px; color:#12395c; }
h4 { font-size:14px; margin:18px 0 6px; color:#12395c; }
p { margin:8px 0; }
ul, ol { margin:8px 0 8px 22px; padding:0; }
li { margin:3px 0; }
.subtitle { color:var(--muted); font-size:14px; margin-bottom:18px; }
table { border-collapse:collapse; width:100%; margin:10px 0 6px; font-size:12.5px; }
caption { caption-side:top; text-align:left; color:var(--muted); font-size:12.5px; padding-bottom:5px; }
th, td { border:1px solid var(--line); padding:4px 7px; }
thead th { background:#eef2f6; font-weight:600; text-align:center; }
tbody tr:nth-child(even) td { background:#fafbfc; }
td.num { text-align:right; font-variant-numeric: tabular-nums; }
td.txt { text-align:left; }
figure { margin:16px 0; text-align:center; }
figure img { max-width:100%; border:1px solid var(--line); background:#fff; }
figcaption { color:var(--muted); font-size:12.5px; margin-top:5px; text-align:left; }
code, pre { font-family: "Cascadia Mono", Consolas, "Courier New", monospace; font-size:12.5px; }
pre { background:#f4f6f8; border:1px solid var(--line); border-left:3px solid var(--accent);
  padding:9px 12px; overflow-x:auto; margin:8px 0; }
code { background:#f4f6f8; padding:1px 4px; border:1px solid #e6e6e6; }
.box { border:1px solid var(--line); border-left:4px solid var(--accent); background:#f8fafc;
  padding:10px 14px; margin:12px 0; }
.note { background:#fdf8ec; border:1px solid #ecd9a8; border-left:4px solid var(--warn);
  padding:10px 14px; margin:12px 0; }
.demo { background:#fff4d6; border:2px solid #d9a300; padding:12px 16px; margin:0 0 22px; }
.demo strong { color:#8a5a00; }
.kv { display:grid; grid-template-columns: 260px 1fr; gap:2px 14px; font-size:13.5px; margin:10px 0; }
.kv div { padding:3px 0; border-bottom:1px dotted #e6e6e6; }
.kv .k { color:var(--muted); }
.toc { font-size:13.5px; columns:2; margin:6px 0 0 18px; }
.eq { background:#f4f6f8; border:1px solid var(--line); padding:10px 14px; margin:10px 0;
  font-family: "Cambria Math", Cambria, Georgia, serif; font-size:14.5px; }
.small { font-size:12.5px; color:var(--muted); }
footer { max-width:980px; margin:16px auto 0; color:var(--muted); font-size:12px; }
"""


def _kv(pairs) -> str:
    """Definition-style key/value block."""
    items = "".join(
        f'<div class="k">{esc(key)}</div><div>{value}</div>' for key, value in pairs
    )
    return f'<div class="kv">{items}</div>'


def _stage_rows(human_summaries, human_run: str, root: Path):
    """Per-stage row counts of the human run, read from the stage summary JSONs."""
    h2_, h3_, h4_ = (human_summaries.get(stage) for stage in (2, 3, 4))
    rows = []
    if h2_:
        kept = sum(
            int(h2_["runs"][human_run]["roles"][role]["kept_rows"])
            for role in ("follower", "leader")
        )
        rows.append(("stage 1", "raw trajectory trimmed to 7 columns per role",
                     f"{count_csv_rows(root / PREPROCESS_SUBDIR / 'stage_1' / f'{human_run}.csv')}"))
        rows.append(("stage 2", "standstill head/tail trim, event split at dt &gt; 1.0 s",
                     f"{kept} kept rows in {h2_['runs'][human_run]['event_count']} event(s)"))
    if h3_:
        rows.append(("stage 3", "overlap window, 0.1 s common grid, interpolation, discontinuity removal",
                     f"{h3_['total']['written_rows']} rows / {h3_['total']['written_event_files']} event(s), "
                     f"{h3_['total']['deleted_event_files']} deleted"))
    if h4_:
        rows.append(("stage 4", "merge to the calibration table <code>time, fv, lv, dhw</code>",
                     f"{h4_['total']['rows']} rows"))
    return rows


# --- report builder -----------------------------------------------------------


def build(
    root: Path,
    run: str,
    validation_subdir: Path,
    method: str,
    out_path: Path,
    demo_note: str | None = None,
) -> dict[str, object]:
    """Assemble the report and write it; returns the self-check record."""
    figures: list[str] = []
    cv = read_json(root / RESULTS_SUBDIR / CV_FILENAME)
    final = read_json(root / RESULTS_SUBDIR / FINAL_PARAMS_FILENAME)
    human_summaries = {
        stage: read_json(root / PREPROCESS_SUBDIR / f"stage_{stage}" / f"stage{stage}_summary.json")
        for stage in (2, 3, 4)
    }
    validation_base = root / validation_subdir.parent
    validation_summaries: dict[int, dict] = {}
    for stage in (2, 3, 4):
        path = validation_base / f"stage_{stage}" / f"stage{stage}_summary.json"
        if path.is_file():
            validation_summaries[stage] = read_json(path)
    has_validation = (root / validation_subdir).is_dir()
    validation_run_json = root / RESULTS_SUBDIR / f"validation_{run}.json"
    validation_artifact = read_json(validation_run_json) if validation_run_json.is_file() else None

    human = mt.evaluate_run(root, run, HUMAN_STAGE4_SUBDIR, "human", method, keep_series=True)
    validation = (
        mt.evaluate_run(root, run, validation_subdir, "validation", method, keep_series=True)
        if has_validation
        else None
    )
    conclusions = (
        mt.text_conclusions(human, validation, MATERIALITY_PERCENT) if validation else []
    )
    thresholds = mt.thresholds()

    run_final = final["runs"][run]
    theta = core.as_theta(run_final["theta_vector"])
    aggregate = cv["runs"][run]

    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(
        "<title>INTR5100 Lab Session I - IDM calibration and validation report "
        f"({esc(run)})</title>"
    )
    parts.append(f"<style>{CSS}</style></head><body><main>")

    # --- header ---------------------------------------------------------------
    parts.append("<h1>INTR5100 Lab Session I &mdash; IDM Calibration and Validation</h1>")
    parts.append(
        f'<div class="subtitle">Run <strong>{esc(run)}</strong> &middot; '
        f'identified with method <strong>{esc(run_final["method"])}</strong> &middot; '
        f"validation run: "
        + ("<strong>present</strong>" if has_validation else "<strong>not available yet</strong>")
        + "</div>"
    )
    if demo_note:
        parts.append(
            '<div class="demo"><strong>DEMO REPORT.</strong> '
            f"{esc(demo_note)} The numbers are computed by the real pipeline on the real "
            "files; only the role of the validation input is simulated.</div>"
        )
    parts.append(
        '<div class="box"><strong>How to produce this report.</strong> One command does the '
        "whole job once the validation collection exists (a directory with the same layout as a "
        "human collection, i.e. <code>straight_trajectory.csv</code>):"
        "<pre>python src/validate_historical_model.py --validation-raw &lt;collections-dir&gt; "
        f"--run {esc(run)} --report</pre>"
        "The two steps can also be run separately:<pre>"
        f"python src/validate_historical_model.py --validation-raw &lt;collections-dir&gt; --run {esc(run)}\n"
        f"python src/analysis/build_historical_report.py --run {esc(run)}</pre>"
        "The pipeline writes <code>data/preprocess/validation_&lt;run&gt;/stage_1..4</code> and "
        f"<code>results/validation_{esc(run)}.json</code>; the report generator reads those "
        "artifacts and writes <code>results/historical_analysis.html</code>.</div>"
    )
    parts.append(
        '<div class="toc">'
        "<a href=\"#s1\">1. Experiment and model</a> &middot; "
        "<a href=\"#s2\">2. Data processing</a> &middot; "
        "<a href=\"#s3\">3. Calibration</a> &middot; "
        "<a href=\"#s4\">4. Validation</a> &middot; "
        "<a href=\"#s5\">5. Driving style</a> &middot; "
        "<a href=\"#s6\">6. Limitations</a></div>"
    )

    # --- 1. overview ----------------------------------------------------------
    parts.append('<h2 id="s1">1. Experiment and model</h2>')
    parts.append(
        f"<p>A human driver follows a leader vehicle along the straight sections of a closed "
        f"test route. The collected trajectories are cleaned and merged into per-event tables "
        f"(<code>time, fv, lv, dhw</code>) and an Intelligent Driver Model (IDM) is fitted to "
        f"<strong>{esc(run)}</strong>'s driving with five different identification methods. The "
        f"winning parameter vector is used in a TA-driven <strong>validation run</strong> on the "
        f"same route; this report compares the validation data with the human data and with the "
        f"model's reproduction of both.</p>"
    )
    parts.append('<div class="eq">')
    parts.append("a<sub>IDM</sub>(s, v, &Delta;v) = a &middot; [ 1 &minus; (v/v<sub>0</sub>)<sup>&delta;</sup> "
                 "&minus; (s<sup>*</sup>/s)<sup>2</sup> ]<br>")
    parts.append("s<sup>*</sup> = s<sub>0</sub> + max(0, v&middot;T + v&middot;&Delta;v / (2&radic;(a&middot;b)))"
                 "&nbsp;&nbsp;&nbsp;&Delta;v = v<sub>leader</sub> &minus; v<br>")
    parts.append("ds/dt = v<sub>leader</sub> &minus; v&nbsp;&nbsp;&nbsp;&nbsp;dv/dt = a<sub>IDM</sub>(s, v, &Delta;v)"
                 f"&nbsp;&nbsp;&nbsp;&nbsp;s = dhw &minus; L<sub>VEH</sub>, L<sub>VEH</sub> = {core.L_VEH:g} cu, "
                 f"&delta; = {core.DELTA:g} (fixed)</div>")
    parts.append(
        "<p>The leader is an input rather than a state: the simulation starts from the first "
        "measured sample of the event and is fed the measured leader speed at every integration "
        f"node, integrated with <strong>{esc(method.upper())}</strong> on the real time deltas. "
        "Because the leader never drifts, any error reported below is an error of the "
        "follower model alone.</p>"
    )
    bounds = final.get("parameter_bounds", {})
    parameter_rows = [
        ("v<sub>0</sub>", "desired (free-flow) speed", f"{num(theta[0], 4)}", "cu/s",
         f"[{num(bounds.get('v0', [None])[0], 1)}, {num(bounds.get('v0', [None, None])[1], 1)}]"),
        ("T", "safe time headway", f"{num(theta[1], 4)}", "s",
         f"[{num(bounds.get('T', [None])[0], 1)}, {num(bounds.get('T', [None, None])[1], 1)}]"),
        ("s<sub>0</sub>", "standstill gap", f"{num(theta[2], 4)}", "cu",
         f"[{num(bounds.get('s0', [None])[0], 1)}, {num(bounds.get('s0', [None, None])[1], 1)}]"),
        ("a", "maximum acceleration", f"{num(theta[3], 4)}", "cu/s&sup2;",
         f"[{num(bounds.get('a', [None])[0], 1)}, {num(bounds.get('a', [None, None])[1], 1)}]"),
        ("b", "comfortable deceleration", f"{num(theta[4], 4)}", "cu/s&sup2;",
         f"[{num(bounds.get('b', [None])[0], 1)}, {num(bounds.get('b', [None, None])[1], 1)}]"),
    ]
    parts.append(table(
        ["symbol", "meaning", f"value for {run}", "unit", "bounds", "at a bound"],
        [
            [f"<strong>{symbol}</strong>", meaning, value, unit, bound,
             "yes" if core.PARAM_NAMES[index] in (run_final.get("parameters_at_bound") or []) else "no"]
            for index, (symbol, meaning, value, unit, bound) in enumerate(parameter_rows)
        ],
        caption="Table 1. The five identified IDM parameters (delta and L_VEH are fixed by the brief, "
                f"not identified). Fitted by {run_final['method']}; a parameter sitting on a bound "
                "means the fit wanted to leave the box.",
        aligns="lll" + "r" * 3,
    ))
    parts.append(
        '<p class="small"><strong>Units.</strong> Every quantity stays in the coordinate units (cu) '
        "of the recorded trajectory table: <code>x</code>, <code>y</code>, <code>dhw</code> and the "
        "spacing in cu, the speeds in cu/s, the accelerations in cu/s&sup2;. One vehicle length is "
        f"L<sub>VEH</sub> = {core.L_VEH:g} cu by the lab brief. No metre/second conversion is "
        "applied anywhere, because none is documented for this dataset (the vehicle-reported "
        "<code>velocity</code> column is discarded by stage 4 for exactly that reason: it disagrees "
        "with the position-difference speed by roughly a factor of three). The style thresholds of "
        "section 5 are therefore stated in the same units, and the one threshold that would need a "
        "physical conversion is also given in a unit-free (sigma-relative) form.</p>"
    )
    parts.append(_kv([
        ("identifying method", f"{esc(run_final['method'])} &mdash; {esc(run_final['method_description'])} "
                               f"(CV rank {run_final['method_rank_in_cv']}, score {num(run_final['cv_score'], 4)})"),
        ("final parameter vector", f"{esc(core.format_theta(theta))}"),
        ("full-set normalised cost J", f"{num(run_final['total_error'], 6)} "
                                       f"({esc(run_final['total_error_method'])})"),
        ("events / samples used for the fit", f"{run_final['n_events']} / {run_final['n_samples']}"),
        ("mean RMSE on the fit set",
         f"spacing {num(run_final['mean_spacing_rmse'], 2)} cu, "
         f"speed {num(run_final['mean_speed_rmse'], 2)} cu/s"),
        ("integration / seed", f"{esc(cv.get('integration_method'))} / {esc(cv.get('seed'))}"),
    ]))

    # --- 2. data processing ---------------------------------------------------
    parts.append('<h2 id="s2">2. Data processing</h2>')
    parts.append(
        "<p>Both runs go through the same four preprocessing stages of the project. Every stage "
        "writes a summary JSON next to its tables and re-reads what it wrote; the numbers below are "
        "quoted from those summaries, not recomputed here.</p>"
    )
    stage_rows = _stage_rows(human_summaries, run, root)
    validation_rows: list[tuple[str, str, str]] = []
    if validation_summaries:
        v2 = validation_summaries[2]
        v3 = validation_summaries[3]
        v4 = validation_summaries[4]
        kept = sum(int(v2["runs"][run]["roles"][role]["kept_rows"]) for role in ("follower", "leader"))
        stage1_rows = (
            validation_artifact["stage1"]["output_rows"]
            if validation_artifact
            else count_csv_rows(validation_base / "stage_1" / f"{run}.csv")
        )
        validation_rows = [
            ("stage 1", "raw trajectory trimmed to 7 columns per role", f"{stage1_rows}"),
            ("stage 2", "standstill head/tail trim, event split at dt &gt; 1.0 s",
             f"{kept} kept rows in {v2['runs'][run]['event_count']} event(s)"),
            ("stage 3", "overlap window, 0.1 s common grid, interpolation, discontinuity removal",
             f"{v3['total']['written_rows']} rows / {v3['total']['written_event_files']} event(s), "
             f"{v3['total']['deleted_event_files']} deleted"),
            ("stage 4", "merge to the calibration table <code>time, fv, lv, dhw</code>",
             f"{v4['total']['rows']} rows"),
        ]
    parts.append(table(
        ["stage", "operation", f"human run ({run})", "validation run"],
        [
            [stage, operation, human_cell,
             validation_rows[index][2] if validation_rows else "<em>no run yet</em>"]
            for index, (stage, operation, human_cell) in enumerate(stage_rows)
        ],
        caption="Table 2. Row counts of the four preprocessing stages. The validation column is "
                f"filled from the summaries of data/preprocess/validation_{run}/stage_2..4.",
        aligns="llll",
    ))
    h2_cfg = human_summaries[2]["config"]
    h3_cfg = human_summaries[3]["config"]
    h4_cfg = human_summaries[4]["config"]
    parts.append(_kv([
        ("stage 2 configuration",
         f"genuine stop &gt; {h2_cfg['isolated_stop_max_samples']} zero-velocity samples, launch "
         f"transient absorbed below {h2_cfg['stop_launch_velocity_eps_kmh']:g} km/h, "
         f"event split at dt &gt; {h2_cfg['gap_threshold_s']} s"),
        ("stage 3 configuration",
         f"grid dt = {h3_cfg['grid_dt_s']} s, minimum overlap {h3_cfg['min_overlap_s']} s, "
         f"interpolation span limit {h3_cfg['max_interpolation_span_s']} s"),
        ("stage 4 configuration",
         f"speeds from the position track ({esc(h4_cfg['speed_method'])}), "
         f"dt from {esc(h4_cfg['dt_source'])}, dhw = {esc(h4_cfg['dhw_formula'])}"),
        ("discarded input column",
         "the vehicle-reported <code>velocity</code> column (it disagrees with the "
         "position-difference speed by about a factor of three)"),
    ]))

    def event_rows(summaries: dict) -> list[list[str]]:
        rows: list[list[str]] = []
        for event in summaries[4]["events"]:
            rows.append([
                f"{esc(event['event'])}",
                f"{event['rows']}",
                f"{num(event['time_range'].get('min'), 1)}&ndash;{num(event['time_range'].get('max'), 1)}",
                f"{num(event['dhw_range'].get('min'), 1)}&ndash;{num(event['dhw_range'].get('max'), 1)}",
                f"{num(event['fv_range'].get('min'), 1)}&ndash;{num(event['fv_range'].get('max'), 1)}",
                f"{num(event['fv_mean'], 1)}",
                f"{num(event['lv_range'].get('min'), 1)}&ndash;{num(event['lv_range'].get('max'), 1)}",
                f"{num(event['lv_mean'], 1)}",
                f"{event['deleted_points'] + event['dropped_points']}",
            ])
        return rows

    human_event_rows = event_rows(human_summaries)
    validation_event_rows = event_rows(validation_summaries) if validation_summaries else []
    parts.append(table(
        ["event", "samples", "t (s)", "dhw (cu)", "fv (cu/s)", "fv mean", "lv (cu/s)", "lv mean", "points removed"],
        [[f"<em>{esc(run)} &mdash; human</em>", "", "", "", "", "", "", "", ""]] + human_event_rows
        + [[f"<em>{esc(run)} &mdash; validation</em>", "", "", "", "", "", "", "", ""]]
        + validation_event_rows,
        caption="Table 3. Event inventory of both runs (stage 4 summary). "
                "&ldquo;points removed&rdquo; counts non-positive-dhw anomalies and rounded-time "
                "duplicates dropped by the merge.",
        aligns="l" + "r" * 8,
    ))
    deleted = human_summaries[3].get("deleted_events") or []
    if deleted:
        parts.append(
            "<p class=\"small\">Events dropped by stage 3 (no usable leader/follower overlap): "
            + ", ".join(
                f"{esc(item['run'])}/{esc(item['event'])} "
                f"({esc(item['reason'])}: {esc(item['detail'])}, overlap "
                f"{num(item['overlap'].get('duration_s'), 2)} s of the required "
                f"{h3_cfg['min_overlap_s']:g} s)"
                for item in deleted
            )
            + ".</p>"
        )

    # --- 3. calibration -------------------------------------------------------
    parts.append('<h2 id="s3">3. Calibration</h2>')
    parts.append(
        "<p>Five identification methods were compared by leave-one-event-out cross-validation: "
        "each fold trains on the run's remaining events and then <em>predicts</em> the held-out "
        "event, so the comparison is on data the fit never saw. The ranking score is "
        "<code>mean(spacing RMSE)/scale_s + mean(speed RMSE)/scale_v + 1.0 &times; &Sigma;violations</code>, "
        "with the scales the pooled standard deviations of the run's measured spacing and speed.</p>"
    )
    method_rows = []
    per_method = aggregate["per_method"]
    budget_cv = cv["budgets"]["cross_validation"]
    budget_full = cv["budgets"]["final_retrain"]
    for method_key in METHOD_ORDER:
        entry = per_method.get(method_key) or {}
        regression = entry.get("regression") or {}
        spacing = regression.get("spacing") or {}
        speed = regression.get("speed") or {}
        method_rows.append([
            f"<strong>{esc(method_key)}</strong>" + (" &#9733;" if method_key == run_final["method"] else ""),
            esc(entry.get("method_description", "")),
            esc(budget_cv.get(method_key, {}).get("mode", "-")),
            esc(budget_full.get(method_key, {}).get("mode", "-")),
            num(entry.get("score"), 4),
            num(entry.get("mean_spacing_rmse"), 2),
            num(entry.get("mean_speed_rmse"), 2),
            f"{entry.get('n_feasible_folds', '-')}/{entry.get('n_folds', '-')}",
        ])
    parts.append(table(
        ["method", "search", "CV budget", "final budget", "CV score", "mean s RMSE (cu)",
         "mean v RMSE (cu/s)", "feasible folds"],
        method_rows,
        caption="Table 4. The five methods and their cross-validation results for "
                f"{esc(run)} (&#9733; = the selected method). Lower scores are better. "
                "The global methods run at half budget inside the cross-validation because CV "
                "multiplies the number of optimisations by the number of folds; the final retrain "
                "always uses the full budget.",
        aligns="llll" + "r" * 4,
    ))
    regression_rows = []
    for row in aggregate["ranking"]:
        method_key = row["method"]
        entry = per_method[method_key]
        spacing = (entry.get("regression") or {}).get("spacing") or {}
        speed = (entry.get("regression") or {}).get("speed") or {}
        regression_rows.append([
            f"{row['rank']}",
            f"<strong>{esc(method_key)}</strong>",
            num(entry.get("score"), 4),
            num(spacing.get("mean_mse"), 1),
            num(spacing.get("mean_rmse"), 2),
            num(spacing.get("mean_mae"), 2),
            num(spacing.get("mean_mape_percent"), 2),
            num(spacing.get("pooled_r2"), 3),
            num(speed.get("mean_mse"), 1),
            num(speed.get("mean_rmse"), 2),
            num(speed.get("mean_mae"), 2),
            num(speed.get("mean_mape_percent"), 2),
            num(speed.get("pooled_r2"), 3),
            f"{row['s_violations_sum']}",
        ])
    parts.append(table(
        [
            [("", 3), ("spacing s = dhw &minus; 250 cu", 5), ("speed (cu/s)", 5), ("", 1)],
            ["rank", "method", "CV score", "MSE", "RMSE", "MAE", "MAPE %", "R&sup2;",
             "MSE", "RMSE", "MAE", "MAPE %", "R&sup2;", "violations"],
        ],
        regression_rows,
        caption="Table 5. Held-out regression metrics of every ranked method for "
                f"{esc(run)}. Errors and their means are in the original units; R&sup2; is pooled "
                "from the fold residual sums (1 &minus; &Sigma;SS_res/&Sigma;SS_tot, each fold about "
                "its own measured mean) because a mean of per-fold R&sup2; is not a coefficient of "
                "determination. MAPE skips samples whose measurement is exactly zero; none were "
                f"skipped on this data (total skipped: {cv['self_checks']['mape_skipped_points_total']}).",
        aligns="ll" + "r" * 12,
    ))
    parts.append(figure(
        fig_cv_scores(run, aggregate, run_final["method"]),
        f"Cross-validation score of the five methods for {run}; the selected method "
        "is highlighted (lower is better).",
        "Figure 1.",
    ))
    figures.append("cv")
    excluded = aggregate.get("excluded_methods") or []
    if excluded:
        parts.append(
            '<p class="small">Methods excluded from the ranking (no feasible fold): '
            + "; ".join(
                f"<strong>{esc(item['method'])}</strong> (" + ", ".join(
                    f"{esc(fold['fold'])}: {esc(fold['reason'])}" for fold in item["infeasible_folds"]
                ) + ")"
                for item in excluded
            )
            + ".</p>"
        )
    parts.append(
        "<p><strong>Why the cross-validation winner is not the lowest-cost fit.</strong> "
        "The ranking above is built on held-out events, whereas the normalised cost "
        "<code>J</code> of the fit set rewards the position of the optimum of the whole set. "
        "On this data the two orderings disagree, which is the reason the selection was made by "
        "cross-validation in the first place.</p>"
    )
    fit_rows = []
    for method_key in METHOD_ORDER:
        reference = (aggregate.get("calibration_set_reference") or {}).get(method_key) or {}
        if not reference.get("available"):
            continue
        fit_rows.append([
            f"<strong>{esc(method_key)}</strong>",
            num(reference.get("total_error"), 6),
            num(reference.get("mean_spacing_rmse"), 2),
            num(reference.get("mean_speed_rmse"), 2),
            f"<code>{esc(reference.get('source'))}</code>",
        ])
    if fit_rows:
        parts.append(table(
            ["method", "fit-set J (all events)", "mean s RMSE (cu)", "mean v RMSE (cu/s)", "source"],
            fit_rows,
            caption="Table 6. The same methods fitted on all events (the published Step A/B runs, "
                    "read from their JSON files - not recomputed). J is the equal-weight mean of the "
                    "per-event normalised costs, so it is comparable across methods but not with the "
                    "CV score of Table 5.",
            aligns="l" + "r" * 3 + "l",
        ))
    final_regression = run_final.get("regression") or {}
    if final_regression:
        parts.append(table(
            ["series", "MSE", "RMSE", "MAE", "MAPE %", "R&sup2; (pooled)"],
            [
                ["spacing (cu)", num(final_regression["spacing"].get("mean_mse"), 1),
                 num(final_regression["spacing"].get("mean_rmse"), 2),
                 num(final_regression["spacing"].get("mean_mae"), 2),
                 num(final_regression["spacing"].get("mean_mape_percent"), 2),
                 num(final_regression["spacing"].get("pooled_r2"), 3)],
                ["speed (cu/s)", num(final_regression["speed"].get("mean_mse"), 1),
                 num(final_regression["speed"].get("mean_rmse"), 2),
                 num(final_regression["speed"].get("mean_mae"), 2),
                 num(final_regression["speed"].get("mean_mape_percent"), 2),
                 num(final_regression["speed"].get("pooled_r2"), 3)],
            ],
            caption=f"Table 7. Regression metrics of the final {esc(run_final['method'])} "
                    f"parameters on all {run_final['n_events']} events they were fitted to "
                    "(fit-set numbers, lower MSE/RMSE/MAE and higher R&sup2; are better).",
            aligns="l" + "r" * 5,
        ))

    # --- 4. validation --------------------------------------------------------
    parts.append('<h2 id="s4">4. Validation run</h2>')
    if validation is None:
        parts.append(
            '<div class="note"><strong>No validation run available yet.</strong> '
            f"<code>{esc(rel(root / validation_subdir, root))}</code> does not exist, so this "
            "section has nothing to compare. As soon as the TA's validation collection is in place, "
            "run<pre>"
            f"python src/validate_historical_model.py --validation-raw &lt;collections-dir&gt; "
            f"--run {esc(run)} --report</pre>"
            "and this section will fill with the measured-versus-simulated figures and the "
            "prediction tables of both runs.</div>"
        )
    else:
        source = validation_artifact["validation_raw"] if validation_artifact else rel(root / validation_subdir.parent, root)
        parts.append(
            "<p>The validation collection <code>" + esc(source) + "</code> is processed by exactly "
            "the same four stages as the human data, which puts it in the isolated tree "
            f"<code>{esc(rel(root / validation_subdir.parent, root))}/stage_1..4</code>. The "
            f"parameters of {esc(run)} are then replayed event by event: initial state = the "
            "event's first measured sample, leader input = the measured leader speed, RK4 on the "
            "real time deltas.</p>"
        )
        if validation_artifact and validation_artifact.get("demo"):
            parts.append(
                '<div class="note"><strong>These validation tables come from a DEMO run.</strong> '
                f"<code>{esc(rel(validation_run_json, root))}</code> was produced with "
                "<code>--demo</code>, i.e. the collection named above is a real recording used to "
                "stand in for the TA's validation run. Re-run the pipeline without "
                "<code>--demo</code> on the real validation collection before quoting any number "
                "of this section.</div>"
            )

        def metric_rows(run: dict) -> list[list[str]]:
            rows: list[list[str]] = []
            for name, block in run["events"].items():
                prediction = block["prediction"]
                rows.append([
                    f"{esc(name)}",
                    f"{prediction['n_samples']}",
                    num(prediction["spacing"]["mse"], 1),
                    num(prediction["spacing"]["rmse"], 2),
                    num(prediction["spacing"]["mae"], 2),
                    num(prediction["spacing"]["mape_percent"], 2),
                    num(prediction["spacing"]["r2"], 3),
                    num(prediction["speed"]["mse"], 1),
                    num(prediction["speed"]["rmse"], 2),
                    num(prediction["speed"]["mae"], 2),
                    num(prediction["speed"]["mape_percent"], 2),
                    num(prediction["speed"]["r2"], 3),
                    f"{prediction['s_violations']}",
                ])
            aggregate_block = run["prediction_aggregate"]
            rows.append([
                "<em>aggregate (equal-weight mean)</em>",
                f"{aggregate_block['n_samples']}",
                num(aggregate_block["spacing_mse"], 1),
                num(aggregate_block["spacing_rmse"], 2),
                num(aggregate_block["spacing_mae"], 2),
                num(aggregate_block["spacing_mape_percent"], 2),
                "&mdash;",
                num(aggregate_block["speed_mse"], 1),
                num(aggregate_block["speed_rmse"], 2),
                num(aggregate_block["speed_mae"], 2),
                num(aggregate_block["speed_mape_percent"], 2),
                "&mdash;",
                f"{aggregate_block['total_s_violations']}",
            ])
            return rows

        parts.append(table(
            [
                [("", 2), ("spacing s = dhw &minus; 250 cu", 5), ("speed (cu/s)", 5), ("", 1)],
                ["event", "n", "MSE", "RMSE", "MAE", "MAPE %", "R&sup2;",
                 "MSE", "RMSE", "MAE", "MAPE %", "R&sup2;", "s&le;0"],
            ],
            metric_rows(validation) + [[f'<em>{esc(run)} &mdash; human run (same metrics, for reference)</em>',
                                        "", "", "", "", "", "", "", "", "", "", "", ""]]
            + metric_rows(human),
            caption="Table 8. Prediction metrics of the IDM replay, validation run first and the "
                    "human run underneath (identical model, identical conventions). The aggregate row "
                    "is the equal-weight mean over that run's events; R&sup2; is per event because a "
                    "mean of R&sup2; is not a coefficient of determination, and &ldquo;s&le;0&rdquo; "
                    "counts simulated spacing violations.",
            aligns="ll" + "r" * 11,
        ))
        for name, block in validation["events"].items():
            parts.append(figure(
                fig_event_comparison(name, "validation", block["prediction"]),
                f"Validation {name}: measured versus simulated spacing (top) and follower speed "
                "(bottom) with the parameters of "
                f"{run} ({core.format_theta(theta)}).",
                f"Figure {len(figures) + 1}.",
            ))
            figures.append(f"validation-{name}")
        for name, block in human["events"].items():
            parts.append(figure(
                fig_event_comparison(name, "human", block["prediction"]),
                f"Human {name}: the same replay on the data the parameters were fitted to, i.e. "
                "the fit-quality counterpart of the validation figure above.",
                f"Figure {len(figures) + 1}.",
            ))
            figures.append(f"human-{name}")

    # --- 5. driving style -----------------------------------------------------
    parts.append('<h2 id="s5">5. Driving style</h2>')
    parts.append(
        "<p>The style metrics are computed on the measured trajectories only, so the human run and "
        "the validation run are described by the same yardstick. They are grouped into "
        "<em>efficiency</em>, <em>comfort</em> and <em>safety</em>. Acceleration and jerk come from "
        "the measured speed series by central differences with one-sided ends (the convention stage 4 "
        "uses for its speeds), so a 0.1 s sampling grid makes them the most noise-sensitive metrics "
        "of the three groups.</p>"
    )
    derivation_rows = [
        ["stop speed", f"v &le; {num(thresholds['stop_speed_cu_s'], 1)} cu/s",
         esc(thresholds["derivation"]["stop_speed_cu_s"])],
        ["acceleration limits", ", ".join(f"{num(limit, 0)} cu/s&sup2;"
                                        for limit in thresholds["accel_abs_limits_cu_s2"]),
         esc(thresholds["derivation"]["accel_abs_limits_cu_s2"])],
        ["acceleration sigma levels", ", ".join(f"{num(level, 1)}&sigma;<sub>a</sub>"
                                                for level in thresholds["accel_sigma_levels"]),
         esc(thresholds["derivation"]["accel_sigma_levels"])],
        ["TTC thresholds", ", ".join(f"{num(value, 1)} s" for value in thresholds["ttc_thresholds_s"]),
         esc(thresholds["derivation"]["ttc_thresholds_s"])],
        ["spacing violation", f"s &le; {num(thresholds['spacing_violation_cu'], 0)} cu",
         esc(thresholds["derivation"]["spacing_violation_cu"])],
        ["reference minimum gap", f"{num(thresholds['min_spacing_warn_cu'], 0)} cu",
         esc(thresholds["derivation"]["min_spacing_warn_cu"])],
        ["score scales", ", ".join(f"{esc(key)} = {num(value, 0)}"
                                  for key, value in thresholds["score_scales"].items()),
         "values at which the normalised 0-1 style scores saturate (see the radar chart)"],
    ]
    parts.append(table(
        ["quantity", "value", "basis"],
        derivation_rows,
        caption="Table 9. Every threshold used by the style metrics, with the reason it was chosen. "
                "All are module constants of src/analysis/metrics.py and are listed here so no "
                "comparison rests on an undocumented number.",
        aligns="lll",
    ))

    style_fields = [
        ("Efficiency", "mean speed", "mean_speed_cu_s", 2, "cu/s", "higher"),
        ("Efficiency", "speed achievement mean(v)/v0", "speed_achievement", 3, "", "closer to 1"),
        ("Efficiency", "total driving time", "duration_total_s", 1, "s", "context"),
        ("Efficiency", "stop fraction (v &le; threshold)", "stop_fraction", 4, "", "lower"),
        ("Comfort", "acceleration mean", "accel_mean_cu_s2", 2, "cu/s&sup2;", "context"),
        ("Comfort", "acceleration std", "accel_std_cu_s2", 2, "cu/s&sup2;", "lower"),
        ("Comfort", "acceleration RMS", "accel_rms_cu_s2", 2, "cu/s&sup2;", "lower"),
        ("Comfort", "share |a| &gt; 200 cu/s&sup2;", "frac_accel_gt_200_cu_s2", 4, "", "lower"),
        ("Comfort", "share |a| &gt; 2&sigma;<sub>a</sub>", "frac_accel_gt_2sigma", 4, "", "lower"),
        ("Comfort", "jerk RMS", "jerk_rms_cu_s3", 1, "cu/s&sup3;", "lower"),
        ("Comfort", "jerk max |j|", "jerk_abs_max_cu_s3", 1, "cu/s&sup3;", "lower"),
        ("Safety", "minimum spacing", "spacing_min_cu", 1, "cu", "higher"),
        ("Safety", "minimum TTC", "ttc_min_s", 3, "s", "higher"),
        ("Safety", "share TTC &lt; 1.5 s", "ttc_1.5_share", 4, "", "lower"),
        ("Safety", "share TTC &lt; 3.0 s", "ttc_3.0_share", 4, "", "lower"),
        ("Safety", "spacing violations (s &le; 0)", "spacing_violation_count", 0, "", "lower"),
    ]
    human_style = human["style_aggregate"]
    validation_style = validation["style_aggregate"] if validation else None
    style_rows = []
    for group, label, field, decimals, unit, direction in style_fields:
        human_value = human_style.get(field)
        validation_value = validation_style.get(field) if validation_style else None
        delta = (
            float(validation_value) - float(human_value)
            if validation_value is not None and human_value is not None
            else None
        )
        relative = (
            mt.relative_change(human_value, validation_value)
            if validation_style
            else None
        )
        style_rows.append([
            esc(group), label,
            f"{num(human_value, decimals)} {unit}".strip(),
            f"{num(validation_value, decimals)} {unit}".strip() if validation_style else "&mdash;",
            signed(delta, decimals) if validation_style else "&mdash;",
            pct(relative) if validation_style else "&mdash;",
            esc(direction),
        ])
    parts.append(table(
        ["group", "indicator", f"human ({run})", "validation", "difference", "relative", "better"],
        style_rows,
        caption="Table 10. Driving-style comparison. Run-level values are sample-weighted means "
                "over the events of that run (minima stay minima, maxima stay maxima, counts and "
                "durations are summed), so a short event cannot outvote a long one. The difference "
                "column is validation minus human.",
        aligns="ll" + "r" * 4 + "l",
    ))

    def style_event_rows(run: dict) -> list[list[str]]:
        rows: list[list[str]] = []
        for name, block in run["events"].items():
            style = block["style"]
            rows.append([
                f"{esc(name)}", f"{style['n_samples']}",
                num(style["mean_speed_cu_s"], 1), num(style["speed_achievement"], 3),
                num(style["duration_s"], 1), num(style["stop_fraction"], 4),
                num(style["accel_rms_cu_s2"], 1), num(style["jerk_rms_cu_s3"], 0),
                num(style["spacing_min_cu"], 1), num(style["ttc_min_s"], 3),
                num(style["ttc_1.5_share"], 4), f"{style['spacing_violation_count']}",
            ])
        return rows

    parts.append(table(
        ["event", "n", "mean v (cu/s)", "achievement", "duration (s)", "stop frac.",
         "a RMS (cu/s&sup2;)", "jerk RMS", "s min (cu)", "TTC min (s)", "TTC&lt;1.5", "viol."],
        [[f"<em>{esc(run)} &mdash; human</em>", "", "", "", "", "", "", "", "", "", "", ""]]
        + style_event_rows(human)
        + ([[f"<em>{esc(run)} &mdash; validation</em>", "", "", "", "", "", "", "", "", "", "", ""]]
           + style_event_rows(validation) if validation else []),
        caption="Table 11. Per-event style metrics of both runs (the values Table 10 aggregates).",
        aligns="ll" + "r" * 10,
    ))

    if validation:
        scores_human = human["style_scores"]
        scores_validation = validation["style_scores"]
        parts.append(table(
            ["axis", "human", "validation", "difference", "definition"],
            [
                [esc(axis_name), num(scores_human[axis_key], 3), num(scores_validation[axis_key], 3),
                 signed(float(scores_validation[axis_key]) - float(scores_human[axis_key]), 3),
                 esc(definition)]
                for axis_name, axis_key, definition in (
                    ("Efficiency", "efficiency", "clamp(mean(v)/v0) - how close the run stays to the run's own free speed"),
                    ("Comfort", "comfort", "1 - clamp(accel RMS / 200 cu/s^2) - less violent motion is better"),
                    ("Safety", "safety", "0.5*clamp(s_min/250 cu) + 0.5*clamp(TTC_min/3 s), forced to 0 by any s<=0"),
                )
            ],
            caption="Table 12. The three normalised style scores of the radar chart, with their formulas.",
            aligns="l" + "r" * 3 + "l",
        ))
        parts.append(figure(
            fig_radar(scores_human, scores_validation),
            "Normalised efficiency / comfort / safety scores of the two runs (formulas in "
            "Table 12).",
            "Figure %d." % (len(figures) + 1),
        ))
        figures.append("radar")
        sub_labels = list(scores_human["sub_scores"].keys())
        parts.append(figure(
            fig_style_bars(
                sub_labels,
                [float(scores_human["sub_scores"][key]) for key in sub_labels],
                [float(scores_validation["sub_scores"][key]) for key in sub_labels],
            ),
            "The sub-indicators behind the three scores (all normalised to 0-1, higher is better; "
            "the formulas are in Table 12). A run can be better on an aggregate axis while being "
            "worse on one of its sub-indicators.",
            "Figure %d." % (len(figures) + 1),
        ))
        figures.append("bars")
        parts.append("<h3>Automatic conclusions</h3>")
        parts.append(
            '<p class="small">Generated from the numbers above with a '
            f"{MATERIALITY_PERCENT:g} percent materiality band: a relative change smaller than "
            "that is reported as agreement rather than as a difference.</p>"
        )
        parts.append("<ul>" + "".join(f"<li>{esc(line)}</li>" for line in conclusions) + "</ul>")
    else:
        parts.append(
            "<p>The validation half of the comparison, the radar chart and the automatic conclusions "
            "appear here as soon as a validation run exists.</p>"
        )

    # --- 6. limitations -------------------------------------------------------
    parts.append('<h2 id="s6">6. Limitations</h2>')
    at_bound = run_final.get("parameters_at_bound") or []
    limitation_items = [
        f"<strong>Few events.</strong> {run} contributes {run_final['n_events']} event(s) and the "
        f"cross-validation therefore has {run_final['n_events']} fold(s); one event cannot be "
        "averaged away, so a single unusual manoeuvre moves the ranking. Every fold trains on "
        "the remaining events only, which for this run is a very small training set.",
        "<strong>Event length.</strong> The events are short (from "
        + ", ".join(f"{num(block['style']['duration_s'], 1)} s"
                    for block in human["events"].values())
        + " in the human run), so acceleration, jerk and TTC statistics rest on a few hundred "
        "samples at 0.1 s spacing.",
        "<strong>Unit system.</strong> All quantities are in coordinate units; no documented "
        "conversion to metres exists for this dataset and the vehicle-reported velocity column "
        "disagrees with the position-difference speed by about a factor of three, so it is "
        "discarded. Physical thresholds are therefore partly given as multiples of the run's own "
        "standard deviation (see Table 9).",
        "<strong>Parameter bounds.</strong> " + (
            "The identified vector sits on a bound for " + ", ".join(f"<code>{esc(name)}</code>" for name in at_bound)
            + ", which means the fit wanted to leave the box: the true optimum of the cost is "
            "outside the admissible parameter region and the reported numbers are therefore "
            "constrained, not unconstrained, estimates."
            if at_bound else
            "No identified parameter sits on a bound for this run."
        ),
        "<strong>Run-1 consistency.</strong> The cross-validation of run_1 is visibly less "
        "consistent than that of run_1 (its mean spacing RMSE and its worst-fold RMSE are "
        "several times larger), and several of its folds were infeasible, which is itself evidence "
        "that one event of that run is not described by the parameters the others imply.",
        "<strong>Leader is an input.</strong> The simulation is fed the measured leader speed, so "
        "the follower model is never asked to predict the leader; any leader-follower interaction "
        "error is invisible here by construction.",
        "<strong>Metrics from position differences.</strong> Speeds are position differences and "
        "accelerations and jerks are their differences, so quantisation noise grows with each "
        "derivative; the acceleration-based comfort columns are consequently the most uncertain "
        "of the report and are why the sigma-relative threshold is reported next to the absolute one.",
        "<strong>Single validation run.</strong> A validation run is one realisation: without "
        "repetitions there is no way to separate a systematic difference between the human and the "
        "validation run from the day-to-day spread of a single run.",
    ]
    if demo_note:
        limitation_items.insert(
            0,
            "<strong>This report is a DEMO.</strong> " + esc(demo_note)
            + " The pipeline and the metrics are the real ones; the validation input is not.",
        )
    parts.append("<ul>" + "".join(f"<li>{item}</li>" for item in limitation_items) + "</ul>")

    parts.append(
        '<p class="small">Generated by <code>' + esc(SCRIPT) + "</code> from "
        + esc(cv.get("script", "results/cv_results.json")) + ", "
        + esc(rel(root / RESULTS_SUBDIR / mt.FINAL_PARAMS_FILENAME, root)) + ", "
        + "the stage summary JSONs under <code>" + esc(rel(root / PREPROCESS_SUBDIR, root)) + "</code>"
        + (f" and <code>{esc(rel(root / validation_subdir.parent, root))}</code>" if has_validation else "")
        + (f", plus <code>{esc(rel(validation_run_json, root))}</code>" if validation_artifact else "")
        + ". Figures are rendered by matplotlib and embedded as base64 PNG; the file is fully "
        "self-contained and needs no network access. No wall-clock timestamp is written into it, so "
        "two runs on the same artifacts are byte-identical.</p>"
    )
    parts.append("</main>")
    parts.append(
        "<footer>INTR5100 Lab Session I &middot; IDM calibration and validation &middot; "
        f"{esc(run)} &middot; generated by {esc(SCRIPT)}</footer>"
    )
    parts.append("</body></html>")

    html = "\n".join(parts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    checks = {
        "output": rel(out_path, root),
        "bytes": out_path.stat().st_size,
        "figures_rendered": len(figures),
        "images_embedded": html.count("<img "),
        "tables": html.count("<table"),
        "unresolved_placeholders": html.count("{{") + html.count("}}"),
        "external_resources": (
            html.count("http://") + html.count("https://") + html.count("src=\"//")
        ),
        "sections": html.count("<h2 "),
    }
    checks["ok"] = bool(
        checks["bytes"] > 0
        and checks["images_embedded"] == checks["figures_rendered"]
        and checks["unresolved_placeholders"] == 0
        and checks["external_resources"] == 0
        and checks["sections"] == 6
    )
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="project root (default: inferred)")
    parser.add_argument("--run", required=True, help="run label (run_1 / run_2)")
    parser.add_argument("--validation-dir", type=Path, default=None,
                        help="validation run directory (default: data/preprocess/validation_<run>)")
    parser.add_argument("--method", default=core.DEFAULT_METHOD, choices=list(core.METHODS))
    parser.add_argument("--out", type=Path, default=None, help="output HTML path")
    parser.add_argument("--demo", default=None, metavar="NOTE",
                        help="mark the report as a DEMO and default to the demo output filename")
    args = parser.parse_args(argv)

    root = core.resolve_root(args.root)
    run = args.run
    if run not in mt.runs_with_final_params(root):
        parser.error(f"--run {run} has no entry in final_params.json "
                     f"(available: {mt.runs_with_final_params(root)})")

    if args.validation_dir is not None:
        candidate = args.validation_dir
        if candidate.is_absolute() and candidate.is_relative_to(root):
            candidate = candidate.relative_to(root)
        if (root / candidate / "stage_4").is_dir():
            validation_subdir = candidate / "stage_4"
        else:
            validation_subdir = candidate
    else:
        validation_subdir = (
            PREPROCESS_SUBDIR / f"{mt.VALIDATION_PREFIX}{run}" / "stage_4"
        )

    out_path = args.out
    if out_path is None:
        out_path = root / RESULTS_SUBDIR / (DEMO_OUTPUT if args.demo else DEFAULT_OUTPUT)
    elif not out_path.is_absolute():
        out_path = root / out_path

    print(f"project root     : {root}")
    print(f"run              : {run}")
    print(f"human tables     : {rel(root / HUMAN_STAGE4_SUBDIR, root)}")
    print(f"validation tables: {rel(root / validation_subdir, root)}"
          + ("" if (root / validation_subdir).is_dir() else "  (absent - validation section will say so)"))
    print(f"output           : {rel(out_path, root)}")
    checks = build(root, run, validation_subdir, args.method, out_path, args.demo)
    print()
    print("self-checks:")
    for key in ("output", "bytes", "figures_rendered", "images_embedded", "tables",
                "unresolved_placeholders", "external_resources", "sections"):
        print(f"  {key:<24} {checks[key]}")
    print(f"  {'all checks passed':<24} {checks['ok']}")
    if not checks["ok"]:
        print("ERROR: the report self-check failed.", file=sys.stderr)
        return 1
    print()
    print(f"written: {rel(out_path, root)} ({checks['bytes']} bytes, "
          f"{checks['figures_rendered']} embedded figure(s), {checks['tables']} table(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
