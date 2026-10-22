"""Build a report from retained Lab I records without fitting or simulating.

Use the shared intr5100 Conda environment. Calibration JSONs and all experiment
records are read-only inputs. Only report sources, report figures and the
current HTML report are written.
"""
from pathlib import Path
import os
import argparse
import shutil
import subprocess
import sys
import tempfile
import hashlib
import html
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/source"
FIG = OUT / "figures"
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pdf", action="store_true", help="Build HTML/LaTeX without invoking Tectonic")
    parser.add_argument("--offline", action="store_true", help="Use cached Tectonic resources only")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="intr5100-lab1-report-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG.mkdir(parents=True, exist_ok=True)
        RESULTS = ROOT / "results"
        protected = sorted(p for folder in ["data", "results", "src", "archive", "environment"]
                           for p in (ROOT / folder).rglob("*") if p.is_file()
                           and "__pycache__" not in p.parts and p.name != ".DS_Store")
        def digest(p):
            return hashlib.sha256(p.read_bytes()).hexdigest()
        before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
        load = lambda name: json.loads((RESULTS / name).read_text())
        params = load("m5_params.json")["runs"]["run_2"]
        groups = load("group_validation_analysis.json")
        tgsim = load("tgsim_calibration.json")
        descriptive = load("tgsim_presentation_stats.json")
        colors = ["#356ba0", "#d17c35", "#4a8972", "#8a4d7b", "#77733b"]
        plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                             "axes.spines.right": False, "savefig.bbox": "tight"})

        def commands(folder):
            f = pd.read_csv(folder / "commands.csv")
            return f[f.role == "follower"].sort_values("t_from_start").reset_index(drop=True)

        def summarize(f):
            a = f[f["mode"] == "idm_straight"]
            flags = f.emergency.to_numpy() > 0
            starts = flags & ~np.r_[False, flags[:-1]]
            episodes = [dict(start=float(row.t_from_start), reason=str(row.reason),
                             spacing=float(row.spacing))
                        for row in f.loc[starts].itertuples()]
            return dict(n=len(f), n_active=len(a), minimum=float(f.spacing.min()),
                        active_minimum=float(a.spacing.min()), active_median=float(a.spacing.median()),
                        flags=int(flags.sum()), active_flags=int((a.emergency > 0).sum()),
                        episodes=episodes,
                        reasons={str(k): int(v) for k, v in f.loc[flags, "reason"].value_counts().items()})

        runs, summary, metadata = {}, {}, {}
        for label in ["152115", "155015"]:
            folder = ROOT / f"data/validation/idm_test_20260922_{label}"
            runs[label] = commands(folder)
            summary[label] = summarize(runs[label])
            metadata[label] = json.loads((folder / "metadata.json").read_text())

        fig, axes = plt.subplots(2, 2, figsize=(9.4, 4.5), sharex="col")
        for col, label in enumerate(runs):
            f = runs[label]
            active = f["mode"] == "idm_straight"
            axes[0, col].plot(f.t_from_start, f.spacing, color="#b6bdc5", lw=.8, label="All modes")
            axes[0, col].plot(f.t_from_start, f.spacing.where(active), color=colors[col], lw=1.3,
                              label="IDM straight")
            for reason, marker, color in [("spacing_emergency", "x", "#b33030"),
                                          ("center_sensor_emergency", "o", "#c48626")]:
                mask = (f.emergency > 0) & (f.reason == reason)
                if mask.any():
                    axes[0, col].scatter(f.loc[mask, "t_from_start"], f.loc[mask, "spacing"],
                                         s=17, marker=marker, color=color, zorder=4,
                                         label="Spacing flag" if marker == "x" else "Sensor flag")
            threshold = metadata[label]["idm_params"]["emergency_spacing"]
            axes[0, col].axhline(threshold, color="#777", ls=":", lw=1)
            axes[0, col].set(title=f"Run {label} | emergency setting {threshold:g} cu",
                             ylabel="Logged bumper gap (cu)", ylim=(0, 1100))
            axes[0, col].legend(fontsize=7, loc="upper left")
            axes[1, col].plot(f.t_from_start, f.v_cmd, color=colors[col], lw=1)
            axes[1, col].set(xlabel="Time from run start (s)",
                             ylabel="Sent speed command (record units)", ylim=(0, 85))
            for ax in axes[:, col]:
                ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(FIG / "validation_records.pdf")
        fig.savefig(FIG / "validation_records.png", dpi=160)
        plt.close(fig)

        group_frames, group_summary, group_meta = {}, {}, {}
        for i in range(1, 6):
            label = f"group_{i}"
            folder = ROOT / f"data/group_validation_raw/{label}"
            group_frames[label] = commands(folder)
            group_summary[label] = summarize(group_frames[label])
            group_meta[label] = json.loads((folder / "metadata.json").read_text())
        fig, ax = plt.subplots(figsize=(9.4, 2.6))
        for i, (name, f) in enumerate(group_frames.items()):
            ax.plot(f.t_from_start, f.spacing.where(f["mode"] == "idm_straight"),
                    label=f"Group {i+1}", color=colors[i], lw=1.1)
        ax.axhline(80, color="#777", ls=":", lw=1, label="Common 80-cu reference")
        ax.set(xlabel="Time from each run start (s)", ylabel="Active logged gap (cu)")
        ax.legend(ncol=3, fontsize=8, loc="upper left")
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(FIG / "group_active_gaps.pdf")
        fig.savefig(FIG / "group_active_gaps.png", dpi=160)
        plt.close(fig)

        # Shared content drives the PDF source and HTML, preventing divergent versions.
        tex, web = [], []
        def escape(s):
            return str(s).translate(str.maketrans({
                "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_",
                "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}"}))
        def cell(s):
            return "".join(("$" + bit + "$") if i % 2 else escape(bit)
                           for i, bit in enumerate(str(s).split("$")))
        def hcell(s):
            s = html.escape(str(s)).replace("$", "")
            s = re.sub(r"_\{([^}]+)\}|_([A-Za-z0-9]+)", lambda m: "<sub>"+(m[1] or m[2])+"</sub>", s)
            return s
        def section(title):
            tex.append(r"\section{" + escape(title) + "}")
            web.append("<h2>" + html.escape(title) + "</h2>")
        def sub(title):
            tex.append(r"\subsection*{" + escape(title) + "}")
            web.append("<h3>" + html.escape(title) + "</h3>")
        def para(s):
            tex.append(escape(s) + "\n")
            web.append("<p>" + html.escape(s) + "</p>")
        def equation(t, h):
            tex.append("\\[\n" + t + "\n\\]")
            web.append('<div class="equation">' + h + "</div>")
        def table(headers, rows, caption):
            n = len(headers)
            tex.append(r"\begin{table}[H]\centering\small" + "\n" +
                       r"\caption{" + escape(caption) + "}\n" +
                       r"\begin{tabular}{" + "l" + "r"*(n-1) + "}\n" +
                       r"\toprule" + "\n" + " & ".join(map(cell, headers)) + r" \\" + "\n" +
                       r"\midrule" + "\n" +
                       "\n".join(" & ".join(map(cell, row)) + r" \\" for row in rows) +
                       "\n" + r"\bottomrule\end{tabular}\end{table}")
            web.append('<table><caption>'+html.escape(caption)+'</caption><thead><tr>'+
                       "".join("<th>"+hcell(x)+"</th>" for x in headers)+"</tr></thead><tbody>"+
                       "".join("<tr>"+"".join("<td>"+hcell(x)+"</td>" for x in row)+"</tr>" for row in rows)+
                       "</tbody></table>")
        def figure(name, caption, width=r"\linewidth"):
            tex.append(r"\begin{figure}[H]\centering\includegraphics[width="+width+
                       "]{figures/"+name+r".pdf}\caption{"+escape(caption)+r"}\end{figure}")
            web.append('<figure><img src="source/figures/'+name+'.png" alt="'+
                       html.escape(caption)+'"><figcaption>'+html.escape(caption)+"</figcaption></figure>")
        def page():
            tex.append(r"\newpage")
            web.append('<hr class="pagebreak">')

        tex.append(r"""\begin{center}
        {\LARGE\bfseries Car-Following Model Calibration}\\[5pt]
        {\large INTR 5100 \quad Lab Session I}\\[5pt]
        PakHin TIU \quad Xuqian JU\\[3pt]
        HKUST(GZ) \quad Fall 2026
        \end{center}""")
        web.append("<h1>Car-Following Model Calibration</h1><p class='byline'>INTR 5100 · Lab Session I<br>PakHin TIU · Xuqian JU<br>HKUST(GZ) · Fall 2026</p>")
        sub("Overview")
        para("This experiment combined human-driving data collection, microscopic-model calibration and two physical following tests on the SC2I scaled testbed. The second test recorded larger minimum gaps and no emergency flags. However, a retrospective code review found a relative-speed sign discrepancy in the calibration model, while the tests also differed in their safety setting. The retained results therefore document the completed procedure and observed behavior; they do not establish independent validation of the intended IDM.")
        section("Experiment setup and data preparation")
        para("A prescribed-speed leader was followed first by a human-controlled scaled vehicle and then by an automated follower. The driver used the simulator's camera view, steering wheel and pedals. During automated testing, the supplied program applied IDM following on eligible straight-road intervals and bypass commands elsewhere. Consequently, active following and whole-run behavior are reported separately.")
        para("Two human-driving collections were retained: September 17 and September 22. The second collection, run 2, supplied the selected calibration data. Its straight-zone records span approximately 304 seconds, including two laps. Short segments without simultaneous leader and follower coverage were excluded.")
        table(["Processing stage", "Retained run-2 data"], [
            ["Trim and sort raw straight-zone records", "3435 rows"],
            ["Split and pair continuous segments", "13 candidate events"],
            ["Synchronize on a 0.1-second grid", "8 overlapping events"],
            ["Prepare calibration inputs", "1140 paired samples"],
        ], "Data reduction for the adopted human-driving collection.")
        para("Synchronization used the temporal intersection of each paired segment, linear interpolation and a maximum permitted source span of 0.3 seconds. Five candidate events lacked overlap. Duplicate rounded times and nonpositive projected headways were removed with recorded exclusions. Genuine within-event stopping was retained rather than silently treated as bad data.")
        para("Calibration speeds were derived from positions. Headway was projected along the follower heading and a platform vehicle-length constant of 250 coordinate units was subtracted. This historical projection convention differs from forward route-distance spacing in the command logs; the two measures are not pooled. The position-derived/reported-speed discrepancy was approximately 2.8 during cruising. Physical length scale and command-response gain remain unresolved, so testbed distances are reported in coordinate units (cu).")

        page()
        section("Calibration method and historical outputs")
        sub("Intended model and implementation difference")
        equation(r"\dot s=v_L-v,\qquad \dot v=a\left[1-(v/v_0)^4-(s^*/s)^2\right],",
                 "ds/dt = v<sub>L</sub> − v; &nbsp; dv/dt = a[1 − (v/v<sub>0</sub>)<sup>4</sup> − (s*/s)<sup>2</sup>]")
        equation(r"s^*=s_0+\max\left(0,vT+\frac{v(v-v_L)}{2\sqrt{ab}}\right).",
                 "s* = s<sub>0</sub> + max[0, vT + v(v − v<sub>L</sub>)/(2√(ab))]")
        para("The calibration code instead supplied leader-minus-follower speed to the positive dynamic-gap term. The same discrepancy occurs in its NumPy evaluation, differentiable training graph and TGSIM extension. The supplied TA controller uses the intended sign. Thus the offline calibration model and the physical controller were not the same acceleration law. All fitted values and errors below are retained historical outputs of that offline implementation, not corrected IDM estimates.")
        sub("Fitting procedure")
        para("The trajectory-based methods used the normalized objective below. Instantaneous least squares instead minimized acceleration residuals; its reported trajectory errors were evaluated on the same events for comparison.")
        equation(r"""J=\frac1E\sum_{e=1}^{E}\left\{\frac1{n_e}\sum_{k=1}^{n_e}
        \left[\frac{(s^{\rm sim}_{ek}-s^{\rm obs}_{ek})^2}{\sigma_{s,e}^2}
        +\frac{(v^{\rm sim}_{ek}-v^{\rm obs}_{ek})^2}{\sigma_{v,e}^2}\right]
        +\lambda N_{{\rm viol},e}\right\}.""",
                 "J = mean over events {mean[(spacing error/σ<sub>s,e</sub>)² + (speed error/σ<sub>v,e</sub>)²] + violation penalty}")
        para("Each trajectory starts from its first measured spacing and follower speed, with measured leader speed as input. Events have equal objective weight. Four reported methods were instantaneous acceleration least squares, genetic algorithm, particle swarm and differentiable trajectory fitting. The differentiable method used bounded sigmoid parameters and multistart Adam; the retained run-2 configuration used 10 starts, 300 steps and learning rate 0.05. Training used Euler integration; final candidate evaluation used RK4. The acceleration exponent was fixed at four.")
        method_rows = []
        for label, filename in [("Instantaneous least squares", "m1_params.json"),
                                ("Genetic algorithm", "m3_params.json"),
                                ("Particle swarm", "m4_params.json"),
                                ("Differentiable fitting", "m5_params.json")]:
            item = load(filename)["runs"]["run_2"]
            method_rows.append([label, f'{item["mean_spacing_rmse"]:.2f}',
                                f'{item["mean_speed_rmse"]:.2f}'])
        table(["Method", "Mean gap RMSE (cu)", "Mean speed RMSE (cu/s)"], method_rows,
              "Retained in-sample errors, averaged equally over the eight events.")
        table(["Parameter set", "$v_0$", "$T$", "$s_0$", "$a$", "$b$"], [
            ["Historical selected fit", "123.358", "2.091", "28.723", "13.974", "185.159"],
            ["First physical test", "123.36", "2.09", "28.72", "13.97", "185.16"],
            ["Second physical test", "41.12", "2.09", "28.72", "13.97", "185.16"],
        ], "Historical fit and actually configured physical-test parameters; units follow their respective input channels.")
        para("The differentiable result was selected during the experiment. Similar fitting errors among the trajectory methods are historical comparisons on a shared implementation; they cannot establish correct model structure, global optimality, generalization or physically meaningful braking parameters.")

        page()
        section("Fit quality and validation design")
        sub("Measured and model-generated calibration trajectories")
        rows = []
        for name, ev in params["events"].items():
            rows.append([name.replace("event_", ""), str(ev["n_samples"]),
                         f'{ev["spacing_rmse"]:.2f}', f'{ev["speed_rmse"]:.2f}'])
        table(["Event", "Samples", "Gap RMSE (cu)", "Speed RMSE (cu/s)"], rows,
              "Historical RK4 reconstruction errors from the selected run-2 fit.")
        para("The trajectory comparison is summarized by its saved event residuals. Event 1 has substantially larger gap error than the other events and also exceeds its constant-mean gap baseline of 40.92 cu. The eight-event mean gap RMSE is 39.45 cu, but this average conceals the launch-event mismatch. The event table uses the retained machine-readable fit results consistently rather than the unreconciled values in earlier report drafts.")
        para("These trajectories reconstruct the same events used for fitting, initialized from each event's measured first state. They are therefore in-sample reconstruction evidence. They are not an independent test of generalization, and shared errors between training and evaluation code do not validate the intended IDM equation.")
        sub("Physical-test configuration")
        table(["Recorded setting", "Run 152115", "Run 155015"], [
            ["Desired speed parameter", "123.36", "41.12"],
            ["Emergency-spacing setting (cu)", "80", "20"],
            ["Minimum-spacing floor (cu)", "80", "80"],
            ["Leader / curve speed settings", "40 / 35", "40 / 35"],
            ["Follower command records", str(summary["152115"]["n"]), str(summary["155015"]["n"])],
            ["IDM-straight command records", str(summary["152115"]["n_active"]), str(summary["155015"]["n_active"])],
        ], "Recorded configurations; the safety setting also changed between tests.")
        para("The second test divided the desired-speed parameter by three, leaving the other four fitted parameters at their configured values. Its emergency-spacing threshold also changed. Command cadence was approximately 0.2 seconds. The tests were separate runs with different evolving states, not matched repetitions differing in a single isolated factor.")
        para("A trustworthy out-of-sample trajectory-error table would require an agreed state/command unit mapping and a model matching the implemented controller and vehicle response. The retained record does not establish those conditions. No new validation prediction RMSE is asserted here. The following section instead reports directly traceable physical observations, with the sign and unit limitations stated explicitly.")

        page()
        section("Recorded physical-test outcomes")
        table(["Command-log statistic", "Run 152115", "Run 155015"], [
            ["Whole-run minimum gap (cu)", f'{summary["152115"]["minimum"]:.2f}', f'{summary["155015"]["minimum"]:.2f}'],
            ["IDM-straight minimum gap (cu)", f'{summary["152115"]["active_minimum"]:.2f}', f'{summary["155015"]["active_minimum"]:.2f}'],
            ["IDM-straight median gap (cu)", f'{summary["152115"]["active_median"]:.2f}', f'{summary["155015"]["active_median"]:.2f}'],
            ["Spacing-emergency flagged rows", *[str(summary[x]["reasons"].get("spacing_emergency", 0)) for x in runs]],
            ["Center-sensor flagged rows", *[str(summary[x]["reasons"].get("center_sensor_emergency", 0)) for x in runs]],
            ["Distinct flagged episodes", *[str(len(summary[x]["episodes"])) for x in runs]],
        ], "Follower command-log observations. An episode is a consecutive block of flagged follower records.")
        figure("validation_records",
               "Recorded gap and sent command in each independent run. Colored gap segments identify IDM-straight operation; grey includes bypass. Markers show logged flag reasons. Commands are not measured physical speeds.")
        para("Run 152115 contains three spacing-emergency episodes starting at approximately 13.25, 26.86 and 128.27 seconds, plus two center-sensor episodes starting at 106.66 and 140.06 seconds. Its 17 flagged command rows are not 17 independent incidents. Sensor flags are retained as recorded; the logs alone do not establish whether an external obstacle or a sensor fault caused them.")
        para("Run 155015 has no emergency flags and maintains an active-control gap above 149 cu, exceeding both its own 20-cu emergency setting and the first run's 80-cu setting. This is a favorable observation in that run. It does not prove general safety or isolate a unit conversion as the cause of improvement. The whole-run minimum is lower because it occurs outside IDM-straight operation.")

        page()
        section("Five-group parameter and behavior comparison")
        rows = []
        for i in range(1, 6):
            p = group_meta[f"group_{i}"]["idm_params"]
            rows.append([str(i), *[f'{p[k]:.2f}' for k in ["v0", "T", "s0", "a", "b"]]])
        table(["Group", "$v_0$", "$T$", "$s_0$", "$a$", "$b$"], rows,
              "Reported/deployed parameter sets in the five supplied validation datasets.")
        para("Group 1 combines low desired speed with the largest desired time gap and standstill gap. Group 2 has the largest desired speed and acceleration parameters; Group 5 has the shortest time-gap parameter. Group 4's desired speed is its deployed 41.12, not the original fitted 123.36. These differences suggest nominal policy differences, but fitting methods, data, channel conventions and initial states prevent a causal driver-style ranking from parameter values alone.")
        rows = []
        for i in range(1, 6):
            s = group_summary[f"group_{i}"]
            rows.append([str(i), f'{s["active_minimum"]:.1f}', f'{s["active_median"]:.1f}',
                         str(s["active_flags"]), str(len(s["episodes"]))])
        table(["Group", "Active min (cu)", "Active median (cu)", "Active flagged rows", "All-mode episodes"],
              rows, "Common mode definitions applied to each follower command log.")
        figure("group_active_gaps",
               "IDM-straight gaps only, with breaks at bypass intervals. Each run has its own elapsed-time axis and initial state; aligned clock time is not a matched disturbance.")
        para("Group 1 has the largest active median gap; Group 4 has the next largest. Group 2 records zero gap and spacing-emergency intervention, while Group 5 reaches 44.4 cu without a native flag because its emergency setting is 20 cu. All five datasets use that 20-cu setting. A common gap reference is therefore more informative than comparing flag absence alone.")
        para("Group 4 shows wider recorded margins than Groups 2, 3 and 5 in this scenario. No physical comfort winner is claimed: differences of logged model acceleration are control-demand diagnostics, not measured ride jerk after actuator dynamics and safety overrides. One run per group, mode switching and unequal starting gaps limit broader conclusions.")

        page()
        section("Comparison with the real-world TGSIM pairs")
        para("The supplied HV-HV and HV-AV trajectories provide a descriptive real-world comparison. Retained preprocessing yields 10 events and 4251 samples for HV-HV, and 8 events and 4370 samples for HV-AV, at 0.1-second sampling. The historical analysis treated coordinates and vehicle lengths as metres and speed as metres per second; position-difference speeds are consistent with the supplied speed channel.")
        para("The historical gap estimator was Euclidean separation minus the leader's length. The original documentation alternated between center and front-bumper reference points, so that geometrical assumption still requires confirmation. The following errors and parameter values preserve that estimator and the historical reversed-sign implementation; they are not newly estimated standard-IDM results.")
        rows = []
        for pair, item in tgsim["pairs"].items():
            p = item["theta"]
            rows.append([pair, *[f'{p[k]:.3f}' for k in ["v0", "T", "s0", "a", "b"]]])
        table(["Pair", "$v_0$", "$T$", "$s_0$", "$a$", "$b$"], rows,
              "Retained real-world fitting outputs under the historical implementation.")
        rows = []
        for pair, item in tgsim["pairs"].items():
            a = item["aggregate"]
            rows.append([pair, f'{a["pooled_spacing_rmse"]:.3f}', f'{a["pooled_spacing_mae"]:.3f}',
                         f'{a["pooled_speed_rmse"]:.3f}', f'{a["pooled_speed_mae"]:.3f}'])
        table(["Pair", "Gap RMSE (m)", "Gap MAE (m)", "Speed RMSE (m/s)", "Speed MAE (m/s)"],
              rows, "Historical errors pooled over retained samples; all columns use the same aggregation.")
        para("These raw fitted parameters cannot be directly compared numerically with the scaled platform's coordinate-unit parameters. Even dimensionless combinations would retain the model-sign and reference-point assumptions. The desired-speed and braking estimates approach their search-box limits, while observed speeds are far below the fitted desired speed; strong parameter interpretations are not justified by these fits.")
        rows = []
        for pair in ["HV-HV", "HV-AV"]:
            gap = descriptive["metrics"]["spacing_mean"]["pairs"][pair]
            speed = descriptive["metrics"]["speed_mean"]["pairs"][pair]
            rows.append([pair, f'{gap["all_events"]["event_median"]:.2f}',
                         f'{gap["interaction_subset"]["event_median"]:.2f}',
                         f'{speed["all_events"]["event_median"]:.2f}'])
        table(["Pair", "All-event gap (m)", "Interaction gap (m)", "All-event speed (m/s)"],
              rows, "Median of event-mean observed values; interaction subset excludes mean gaps above 20 m.")
        para("The two sparse HV-AV events have mean gaps of 28.10 and 41.23 m. They increase pooled fitting difficulty and aggregate spacing without representing the same interaction regime as dense following. Removing them is a sensitivity analysis, not proof that the model is correct. The observed gap medians in interacting events are similar across the two pairs. Repeated events from one pair are not independent drivers, so no population-level AV safety or comfort advantage is inferred.")

        page()
        section("Discussion and conclusions")
        sub("What the completed experiment supports")
        para("The retained record demonstrates an end-to-end workflow: human following data were collected, synchronized and used to produce a selected parameter set; two physical automated-following tests were then recorded. The second test has larger active gaps and no logged emergency flags. Across five group datasets, parameter settings and observed gaps differ substantially. The real-world pairs further illustrate how interaction regime and aggregation affect comparisons.")
        sub("What the retained results cannot establish")
        para("The offline calibration law differs from the intended and deployed IDM. Its reconstruction errors therefore describe a historical implementation rather than independent validation of the intended model. The sign error is a separate limitation from the approximately 2.8 speed-channel discrepancy. Neither successful numerical fitting nor improvement in a later physical run proves that units were the sole cause of the first run's behavior.")
        para("A consistent future unit ledger should distinguish position scale, reported velocity, returned speed commands and actuator response. The supplied route map is already available, but does not explicitly establish a physical scale. Its coordinates alone cannot resolve that scale without documentation or a known physical measurement. A fully consistent parameter conversion cannot be treated as experimentally validated from the current records.")
        para("The historical Lab I TTC function also reversed the closing-speed sign. Its affected TTC values and derived safety scores are excluded here. Physical acceleration, jerk, end-to-end latency and fuel use were not independently validated; no comfort, delay or energy ranking is inferred from command cadence or model acceleration. Mode-dependent logged gap statistics provide a more direct account of the observed tests.")
        sub("Conclusions")
        para("The main practical result is the favorable recorded behavior of the second configuration within its particular test. The main methodological result is the importance of checking the implemented model, channel units, safety settings and metric definitions before interpreting calibration quality as physical validation. The historical fitted values and deployment records are preserved. No recalibration, new physical trial or corrected optimum is introduced in this report.")
        para("Stronger future evidence would require a corrected and independently checked model implementation, documented state/command units, reliable trajectory-derived dynamics and repeated matched tests. Those steps are prospective work; the current conclusions remain restricted to the observations available.")
        sub("Sources and data provenance")
        para("[1] INTR 5100, Lab Session I Guidelines: Car-Following Model Calibration. [2] Lab Session I report template. [3] Supplied human and validation trajectory files, command logs and configuration metadata. [4] Retained fitting outputs: m1_params.json, m3_params.json, m4_params.json and m5_params.json; TGSIM calibration and presentation-statistics records. [5] Retrospective code review and read-only verification records accompanying this report.")
        para("All physical-test statistics in this revision are descriptive summaries of existing follower command logs. All fitted parameters and fit errors are taken from retained results. Raw experimental files, historical model equations and machine-readable scientific results were preserved.")

        header = r"""\documentclass[11pt,a4paper]{article}
        \usepackage[margin=20mm]{geometry}
        \usepackage[T1]{fontenc}
        \usepackage{lmodern}
        \usepackage{amsmath,amssymb,graphicx,booktabs,float,array,caption}
        \usepackage[hidelinks]{hyperref}
        \setlength{\parindent}{0pt}
        \setlength{\parskip}{5pt}
        \setlength{\tabcolsep}{5pt}
        \emergencystretch=2em
        \renewcommand{\arraystretch}{1.12}
        \captionsetup{font=small,labelfont=bf}
        \begin{document}
        """
        (OUT / "main.tex").write_text(header + "\n\n".join(tex) + "\n\\end{document}\n")
        css = """body{font:17px/1.55 Georgia,serif;color:#202833;max-width:1000px;margin:45px auto;padding:0 28px}
        h1,h2,h3{font-family:system-ui,sans-serif;line-height:1.25}h1{text-align:center}
        h2{border-bottom:1px solid #ccd3db;padding-bottom:8px;margin-top:40px}
        h3{font-size:1.05em}.byline{text-align:center;color:#586574}
        table{border-collapse:collapse;width:100%;font-size:.9em;margin:25px 0}
        th,td{padding:8px 10px;border-bottom:1px solid #d9dfe5;text-align:right}
        th:first-child,td:first-child{text-align:left}thead{background:#f2f5f7}
        caption,figcaption{font-size:.88em;color:#52616e;text-align:left;padding:8px 0}
        figure{margin:25px 0}img{width:100%;height:auto}.equation{padding:15px;background:#f5f7f9;text-align:center}
        .pagebreak{border:0;border-top:1px solid #d9dfe5;margin:45px 0}
        @media print{body{font-size:11pt;margin:0;max-width:none}.pagebreak{break-before:page;border:0;margin:0}
        table,figure{break-inside:avoid}h2{margin-top:0}}
        """
        html_doc = "<!doctype html><html lang='en'><meta charset='utf-8'><title>Lab I Car-Following Model Calibration</title><style>"+css+"</style><body>"+"\n".join(web)+"</body></html>"
        (ROOT / "report/lab_01_report.html").write_text(html_doc)
        assert all(digest(ROOT / name) == h for name, h in before.items())
        check = dict(scope="Report-only revision; no fitting or simulation was run.",
                     protected_file_count=len(before), protected_files_unchanged=True,
                     historical_parameters= params["theta"], validation_statistics=summary,
                     group_statistics=group_summary, protected_sha256=before)
        (ROOT / "review/report_build_checks.json").write_text(json.dumps(check, indent=2)+"\n")
        print(f"Report sources and HTML built. Verified {len(before)} data/result files unchanged.")

        if not args.skip_pdf:
            compiler = shutil.which("tectonic") or str(Path(sys.executable).with_name("tectonic"))
            with tempfile.TemporaryDirectory(prefix="intr5100-lab1-tex-") as build_dir:
                command = [compiler, "--outdir", build_dir, "main.tex"]
                if args.offline:
                    command.insert(1, "--only-cached")
                subprocess.run(command, cwd=OUT, check=True)
                shutil.copyfile(Path(build_dir) / "main.pdf", ROOT / "report/lab_01_report.pdf")
            print("Compiled report/lab_01_report.pdf; temporary compiler files removed.")


if __name__ == "__main__":
    main()
