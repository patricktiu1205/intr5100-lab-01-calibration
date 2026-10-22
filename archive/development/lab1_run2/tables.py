"""Emit the markdown evidence tables of the run_2 build report from the artefacts.

Read-only: nothing is written next to the pipeline outputs. Usage:
    python tables.py > tables.md
"""
import csv
import glob
import json
import os
import statistics as st


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fmt(value, decimals=4):
    if value is None:
        return "n/a"
    return f"{float(value):.{decimals}f}"


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    # tables.py lives in <lab1>/.coding-dept/lab1-run2, so the project root is two
    # levels up; an explicit LAB1_ROOT still wins.
    root = os.path.abspath(os.path.join(root, "..", ".."))
    root = os.environ.get("LAB1_ROOT", root)

    out = []
    s2 = load(os.path.join(root, "data/preprocess/stage_2/stage2_summary.json"))
    s3 = load(os.path.join(root, "data/preprocess/stage_3/stage3_summary.json"))
    s4 = load(os.path.join(root, "data/preprocess/stage_4/stage4_summary.json"))

    # --- run_2 stage 2 events -------------------------------------------------
    out.append("<!-- run_2 stage-2 event list -->")
    out.append("| event | stage-2 rows | follower | leader | t start | t end | duration s | stage-3 | stage-4 rows |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    deleted3 = {
        (e.get("run"), e.get("event")): e.get("detail")
        for e in s3["deleted_events"]
    }
    for e in s2["runs"]["run_2"]["events"]:
        name = e["event"]
        f4 = os.path.join(root, f"data/preprocess/stage_4/run_2_{name}.csv")
        n4 = sum(1 for _ in open(f4, encoding="utf-8")) - 1 if os.path.isfile(f4) else 0
        status = "deleted (no_overlap: %s)" % deleted3.get(("run_2", name)) if ("run_2", name) in deleted3 else "kept"
        out.append(
            f"| {name} | {e['rows']} | {e['follower']['rows']} | {e['leader']['rows']} "
            f"| {e['t_start']:.3f} | {e['t_end']:.3f} | {e['t_end'] - e['t_start']:.1f} "
            f"| {status} | {n4} |"
        )
    out.append("")

    # --- run_2 stage 3 deletions ---------------------------------------------
    out.append("<!-- run_2 stage-3 deletions -->")
    out.append("| event | reason | detail | overlap (s) | rows dropped |")
    out.append("|---|---|---|---|---|")
    for e in s3["deleted_events"]:
        if e.get("run") != "run_2":
            continue
        out.append(
            f"| {e['event']} | {e['reason']} | {e.get('detail')} "
            f"| {fmt((e.get('overlap') or {}).get('duration_s'), 3)} "
            f"| {e.get('rows_outside_overlap_total')} |"
        )
    out.append("")

    # --- pooled scales --------------------------------------------------------
    out.append("<!-- pooled scales of the two runs -->")
    out.append("| run | stage-4 events | stage-4 rows | scale_s (pooled sd of s=dhw-250) | scale_v (pooled sd of fv) | mean dhw |")
    out.append("|---|---|---|---|---|---|")
    for run in ("run_1", "run_2"):
        s_vals, v_vals, h_vals = [], [], []
        files = sorted(glob.glob(os.path.join(root, f"data/preprocess/stage_4/{run}_event_*.csv")))
        for path in files:
            with open(path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    s_vals.append(float(row["dhw"]) - 250.0)
                    v_vals.append(float(row["fv"]))
                    h_vals.append(float(row["dhw"]))
        out.append(
            f"| {run} | {len(files)} | {len(s_vals)} | {st.pstdev(s_vals):.6f} | {st.pstdev(v_vals):.6f} | {st.fmean(h_vals):.1f} |"
        )
    out.append("")

    # --- CV ranking -----------------------------------------------------------
    cv = load(os.path.join(root, "results/cv_results.json"))
    for run in ("run_1", "run_2"):
        agg = cv["runs"][run]
        out.append(f"<!-- {run} CV ranking -->")
        out.append(
            f"scale_s={agg['scale_s']:.6f} scale_v={agg['scale_v']:.6f}, "
            f"folds={len(agg['folds'])} ({', '.join(agg['folds'])})"
        )
        out.append("| rank | m | s_RMSE | s_MAE | s_MAPE% | s_R2 | v_RMSE | v_MAE | v_MAPE% | v_R2 | viol | score |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for row in agg["ranking"]:
            pm = agg["per_method"][row["method"]]
            reg = pm["regression"]
            out.append(
                f"| {row['rank']} | {row['method']} | {fmt(reg['spacing']['mean_rmse'])} "
                f"| {fmt(reg['spacing']['mean_mae'])} | {fmt(reg['spacing']['mean_mape_percent'], 3)} "
                f"| {fmt(reg['spacing']['pooled_r2'])} | {fmt(reg['speed']['mean_rmse'])} "
                f"| {fmt(reg['speed']['mean_mae'])} | {fmt(reg['speed']['mean_mape_percent'], 3)} "
                f"| {fmt(reg['speed']['pooled_r2'])} | {row['s_violations_sum']} | {fmt(row['score'], 6)} |"
            )
        out.append("")

    # --- selection ------------------------------------------------------------
    fin = load(os.path.join(root, "results/final_params.json"))
    sel = fin["selected_for_validation"]
    out.append("<!-- selection comparison table -->")
    out.append("| run | method | normalized score | score (incl. viol penalty) | s RMSE | s RMSE / sigma_s | v RMSE | v RMSE / sigma_v | s MAPE% | v MAPE% | s violations | folds |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in sel["comparison_table"]:
        out.append(
            f"| {row['run']} | {row['method']} | {fmt(row['normalized_score'], 6)} "
            f"| {fmt(row['score_with_violation_penalty'], 6)} | {fmt(row['mean_spacing_rmse'])} "
            f"| {fmt(float(row['mean_spacing_rmse']) / float(row['scale_s']))} | {fmt(row['mean_speed_rmse'])} "
            f"| {fmt(float(row['mean_speed_rmse']) / float(row['scale_v']))} "
            f"| {fmt(row['spacing_mape_percent'], 3)} | {fmt(row['speed_mape_percent'], 3)} "
            f"| {row['s_violations_sum']} | {row['n_feasible_folds']}/{row['n_folds']} |"
        )
    out.append("")
    out.append(f"selected_for_validation: {sel['selected_run']} / {sel['method']}")
    out.append(f"theta: {json.dumps(sel['theta'])}")
    out.append(f"reason: {sel['reason']}")
    out.append("")

    # --- final params per run -------------------------------------------------
    out.append("<!-- final parameters per run -->")
    out.append("| run | method | theta (v0,T,s0,a,b) | full-set J | mean spacing RMSE | mean speed RMSE | at bound |")
    out.append("|---|---|---|---|---|---|---|")
    for run, entry in fin["runs"].items():
        theta = entry["theta"]
        out.append(
            f"| {run} | {entry['method']} | "
            + ", ".join(f"{theta[k]:.6f}" for k in ("v0", "T", "s0", "a", "b"))
            + f" | {entry['total_error']:.6f} | {entry['mean_spacing_rmse']:.4f} "
            f"| {entry['mean_speed_rmse']:.4f} | {', '.join(entry['parameters_at_bound']) or 'none'} |"
        )
    out.append("")

    # --- per-event fit metrics of the final parameters ------------------------
    for run, entry in fin["runs"].items():
        out.append(f"<!-- {run} per-event metrics of the final theta -->")
        out.append("| event | n | spacing RMSE | speed RMSE | J_e | s viol |")
        out.append("|---|---|---|---|---|---|")
        for name, m in entry["events"].items():
            out.append(
                f"| {name} | {m['n_samples']} | {fmt(m['spacing_rmse'])} | {fmt(m['speed_rmse'])} "
                f"| {fmt(m['j_total'])} | {m['s_violations']} |"
            )
        out.append("")

    print("\n".join(out))


if __name__ == "__main__":
    main()
