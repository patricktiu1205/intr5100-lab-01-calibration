STATUS: COMPLETE (structural verification passed; local PDF compile NOT possible — no TeX engine on this machine)

# Build report — Overleaf-ready LaTeX rebuilt to match the CORRECTED report

Brief: `briefs/latex_regen.md`
Project: `C:\Users\ptiu1\Documents\PhDStudy\INTR5100\lab1`
Single source of truth: `results/lab1_template_report.html` (4,249,683 bytes; 26 `<img>` base64; 12 tables)
Deliverable: `latex/` (Overleaf-ready, uploadable as-is)

---------------------------------------------------------------------------
## 0. Summary

Re-extracted all 26 figures from the corrected HTML **in document order** into
`latex/figures/fig_01..26.png` (all 26 now unique — this fixed the stale duplicate at
`fig_14`), verified the figure order against the brief, re-checked every caption / heading /
table / paragraph of the existing `main.tex` against the corrected report, and fixed the only
two captions that had drifted (FIG 12 and FIG 14). Then cleaned `latex/` down to exactly
`main.tex` + `figures/` (26 PNG). No packages installed, git untouched.

---------------------------------------------------------------------------
## 1. Files changed / created

| Path | Change |
|---|---|
| `latex/figures/fig_01.png … fig_26.png` | RE-EXTRACTED in document order from the corrected HTML (base64 → PNG). All 26 unique; `fig_14.png` was previously an erroneous duplicate of `fig_17.png`. |
| `latex/main.tex` | EDITED — 2 caption blocks fixed (FIG 12, FIG 14). Size 46,559 bytes, sha256 `7c8d6a7ff71df4dd3cc739e973f0ccf483a88ef0e0e3326d3b343b63ebbc25e7`. |
| `latex/audit_dump.txt` | DELETED (brief: `latex/` must contain only `main.tex` + `figures/`). |
| `build-report.md` (project root) | UPDATED — this report (prior report now in §9 Appendix). |

`latex/english_source.txt` was **already absent** (deleted by the previous build). No other
project file under `data/`, `src/`, `results/`, `docs/`, `briefs/` was created or modified.

Final `latex/` tree:
```
latex/
├── main.tex
└── figures/            (26 PNG: fig_01.png … fig_26.png, all unique)
```
No PDF, because no TeX engine is installed locally (see §4a).

---------------------------------------------------------------------------
## 2. What was rebuilt and why

The corrected report changed the **figure order** and the caption of one figure. The previous
`main.tex` still carried the old caption at FIG 12 and a stale trigger-type caption at FIG 14,
and its `figures/fig_14.png` was a byte-identical duplicate of `fig_17.png`.

### 2.1 Figure extraction (document order → 1:1 with the brief)

Parsed `results/lab1_template_report.html` with a stdlib `html.parser` pass, walked the
`<body>` in document order, and decoded each `<img src="data:image/png;base64,…">` in turn to
`latex/figures/fig_NN.png`. Result matches the brief exactly:

```
01-02 closed-loop spacing / speed reconstruction
03-04 validation runs 152115 (raw) / 155015 (v0/3)
05-09 overlap with human driving (distributions, time-normalised, speed tracking,
      lateral, safety)
10-12 group validation (spacing trajectories, speed trajectories, trigger-type stacked bars)
13    scale-ratio gamma evidence
14    counterfactual unit-policy rollouts
15    actual 152115 spacing with emergency overrides
16-26 TGSIM extension (11 figures)
```

Order/key figures were visually confirmed (see §4c): fig_12 = stacked bars of
proximity vs sensor trigger type by group; fig_13 = gamma≈2.8 histogram + scatter;
fig_14 = three-policy rollout spacing trajectories; fig_15 = 152115 spacing collapse.

### 2.2 `main.tex` — content diff against the corrected report

Extracted the English text of the corrected report (zh `<span class="zh">` stripped, the 20
stray `'`+newline source artifacts removed) into a linear reference dump, then aligned every
caption, heading, table and paragraph against `main.tex`. Findings:

- **26/26 figure captions align** with the corrected report after the two fixes below
  (the only residual textual diffs — FIG 12 `\texttt{commands.csv}`, FIG 25 `$v_0/\sqrt{b\,s_0}$`
  — are LaTeX math/formatting, not content).
- **Headings**: 7 `\section`, 14 `\subsection`, 10 `\subsubsection` mirror the report exactly.
- **All 12 tables reproduced** (booktabs); every source data cell is present.
- **All paragraphs / notes / keys / list items present**; no source prose missing, no invented prose.

Two caption edits applied (the only real drift):

| Figure | Old caption | New caption (from corrected English figcap) |
|---|---|---|
| FIG 12 | "Safety and efficiency behaviour: safety-layer episodes (left) and run duration (right)…" | "Safety-layer episodes classified by trigger type from commands.csv. Red: proximity triggers (emergency at spacing ≤250); blue: sensor transient flags … group_1 = 0 proximity + 4 sensor (spacing 942–1974); group_2 = 1 genuine proximity trigger at spacing 0–5.5 plus one center-sensor episode at spacing ~338; groups 3/4/5 = none. Only group_2 shows real close-range risk." |
| FIG 14 | (trigger-type caption misplaced here) | "Counterfactual rollouts of the same IDM under three unit policies: consistent units remain safe, mixed units collapse to the emergency threshold, and mixed units with v0/3 recover. Only the unit convention changes while the formula and parameters stay fixed …" |

### 2.3 `main.tex` requirements (unchanged, verified)

- `\documentclass[11pt]{article}`, UTF-8; package set exactly the brief's Overleaf-default list:
  `inputenc, fontenc, geometry, graphicx, booktabs, array, amsmath, amssymb, float, caption`
  with `hyperref` loaded **last** (`[hidelinks]`).
- Authors **PakHin TIU** and **Xuqian JU**; title `Experiment I: Car-Following Model Calibration`;
  course line `INTR 5100 Traffic Flow Theory, Lab Session I; HKUST(GZ), Fall 2026.`; title page.
- Figures `\begin{figure}[H]\centering\includegraphics[width=\linewidth]{figures/fig_NN.png}`,
  captions auto-numbered (no literal "Fig. N" embedded).
- Full `amsmath` typesetting: IDM (`align`: `a_IDM = a[1 − (v/v0)^δ − (s*/s)^2]`,
  `s* = s0 + max(0, vT + vΔv/(2√(ab)))`, `δ=4`); scale relations and conversion (`equation`);
  dimensionless combos `ṽ0 = v0/√(b s0)`, `ã = a/b`, `T̃ = T√(b/s0)`, `l̃ = L_veh/s0`;
  cost `J = Σ_e[(1/n_e)Σ_t(s_sim−s_obs)²/σ_s² + (1/n_e)Σ_t(v_sim−v_obs)²/σ_v²]`.

---------------------------------------------------------------------------
## 3. Commands actually executed (with real output excerpts)

### 3.1 Extract the 26 base64 figures in document order
```
$ python extract.py latex/figures
fig_01.png 171367 c7702f142113 png
fig_02.png 199431 e2cdfd33e9a9 png
...
fig_14.png 99042 1f99201838c9 png      <-- previously 292950 (duplicate of fig_17)
...
fig_26.png 96302 e2fbbdcdcba3 png
total images: 26
```

### 3.2 Caption alignment (corrected HTML vs main.tex) — after fixes
```
$ python diff3.py
source caps: 26 tex figure captions: 26
FIG 01 OK
...
FIG 11 OK
FIG 12 DIFF   # only formatting: source "commands.csv" vs tex \texttt{commands.csv}
FIG 13 OK
FIG 14 OK
FIG 15 OK
...
FIG 25 DIFF   # only formatting: source "v0/root(b s0)" vs tex $v_0/\sqrt{b\,s_0}$
FIG 26 OK
```

### 3.3 Structural verification (`verify_tex.py`, all PASS)
```
includegraphics: 26
fig env begin/end: 26 26
table env begin/end: 12 12
tabular begin/end: 12 12
document begin/end: 1 1
itemize begin/end: 2 2
align begin/end: 1 1
equation begin/end: 4 4
titlepage begin/end: 1 1
final brace depth: 0 min depth: 0
literal 'Fig.': 0
[[ marker: 0
includegraphics paths: 26 missing: []
unique paths: 26
figure captions: 26
```
Non-ASCII scan of `main.tex`: `{}` (pure ASCII — no `² ³ ≤ ≥ → Δ σ δ γ ≈ √ × – —` leftovers).

### 3.4 `latex/` shrink + uniqueness
```
$ rm -f audit_dump.txt english_source.txt
$ ls -la latex/
drwxr-xr-x  figures
-rw-r--r--  46559  main.tex
$ ls latex/figures/*.png | wc -l        -> 26
$ md5sum latex/figures/*.png | sort -u | wc -l   -> 26   (all unique)
$ find latex -type f ! -name '*.png' ! -name 'main.tex'   -> (none)
```

---------------------------------------------------------------------------
## 4. Verification results

### 4a. TeX engine availability — NONE (no PDF produced)
`which pdflatex`, `which tectonic`, `which latexmk`, `which xelatex` → all not found on
Windows; `wsl.exe -- bash -lc 'ls /usr/bin/*latex*; dpkg -l | grep texlive'` → empty;
no MiKTeX/TeX Live under `%LOCALAPPDATA%\Programs` or `C:\texlive`. Installing packages is
forbidden, so **no compile log tail is available**; the structural checks of §3.3 were run
instead, as the brief permits. The source uses only standard Overleaf packages, so it is
expected to compile on Overleaf directly.

### 4b. Structural checks — PASS
26 `\includegraphics` (all paths resolve, in order, unique), 26 figure envs, 12 table envs,
12 tabular envs, 1 document env, 2 itemize, 1 align, 4 equation — all begin/end balanced;
braces balanced (final depth 0, min depth 0); no literal `Fig.`, no `[[` markers, ASCII-only.

### 4c. Visual spot-check of the corrected figures 12–15 (all correct)
- fig_12 — "Follower safety-layer episodes by trigger type" stacked bars (group_1 = 4 sensor, group_2 = 1 proximity + 1 sensor, groups 3–5 = 0). OK
- fig_13 — steady-cruise scale ratio γ ≈ 2.8 histogram (left) + position-derived-vs-reported speed scatter (right). OK
- fig_14 — "predicted spacing" under three unit policies: consistent units / mixed units / mixed units, v0/3. OK
- fig_15 — actual 152115 spacing collapse with emergency-override markers. OK

---------------------------------------------------------------------------
## 5. Assumptions and deviations (all content-preserving)

1. **No local TeX engine** → PDF not produced; structural checks used instead (brief-permitted).
2. **Stray `'` line-break artifacts** present in the source HTML (20 of them, e.g. `the '<NL> calibrated model`) were removed during extraction/rendering; the shipped `main.tex` reads "the calibrated model". Pre-existing in the previous build too.
3. **`\texttt{commands.csv}` (FIG 12)** and `$…$` math in captions differ cosmetically from the HTML plain text; meaning is identical.
4. `\texorpdfstring{}` retained in the two math-bearing headings so hyperref bookmarks stay clean.
5. Figure file names keep lowercase underscores (`fig_01.png`) as the brief requires (kernel `\includegraphics` supports this on modern TeX Live / Overleaf).
6. The prior build's note about a "Fig. 26 / Fig. 15 numbering" mismatch is now obsolete: the corrected report puts the trigger-type figure at position 12 and the counterfactual rollouts at position 14, and `main.tex` matches.

## 6. Remaining issues / open items

- Cannot visually confirm the rendered PDF without a TeX engine. On Overleaf: upload `latex/`, set main document `main.tex`.
- Everything else in the brief is met and structurally verified.

## 7. Services / processes

None started. No daemons, no background processes. Nothing to stop.

## 8. Suggestions (not done — outside brief scope)

- Keep a copy of the input HTML hash alongside the LaTeX build (e.g. a one-line `PROVENANCE` note) so future figure re-extractions are provably against the same report revision. Not added, because the brief forbids extra files in `latex/`.

===========================================================================
## 9. APPENDIX — previous build report (superseded, kept for history)

STATUS: COMPLETE (structural verification passed; local PDF compile NOT possible — no TeX engine on this machine)

# Build report — Overleaf-ready LaTeX version of the Lab I report

Brief: `briefs/latex_report.md`
Project: `C:\Users\ptiu1\Documents\PhDStudy\INTR5100\lab1`
Deliverable: `latex/` (Overleaf-ready, uploadable as-is)

---------------------------------------------------------------------------
## 1. Files changed / created

| Path | Change |
|---|---|
| `latex/main.tex` | CREATED — full Overleaf-ready LaTeX source (46,527 bytes, sha256 `a52b6039a1547d4f1cd5fd32abf0236ec7148b0b19ea8ee7ebaf0ffdf6599672`) |
| `latex/english_source.txt` | DELETED (per brief §Verification: content fully incorporated) |
| `build-report.md` (project root) | CREATED — this report |

Project root is outside `latex/`. The brief says "do not modify anything outside `latex/`";
this report is an **additive** new file (no existing project file under `data/`, `src/`,
`results/`, `docs/` was touched) and is required by the coder contract. Nothing else in the
project was created or modified. `latex/` itself was left containing exactly what the brief
demands (see §4).

Final `latex/` tree:
```
latex/
├── main.tex
└── figures/            (26 PNG: fig_01.png … fig_26.png)
```
No PDF, because no TeX engine is installed locally (see §3).

---------------------------------------------------------------------------
## 2. What was built

`main.tex` reproduces the entire `english_source.txt` report as a single self-contained
`pdflatex` document:

- **Document class / layout**: `article`, `11pt`, `geometry` (a4paper, margin=1in), UTF-8.
- **Packages** (all standard / Overleaf-default; exactly the brief's set):
  `inputenc, fontenc, geometry, graphicx, booktabs, array, amsmath, amssymb, float, caption, hyperref`.
  `hyperref` is loaded **last** (asserted in the automated check). No custom `.sty`, no `\input`.
- **Title page**: title `Experiment I: Car-Following Model Calibration`; authors **PakHin TIU**
  and **Xuqian JU**; course line `INTR 5100 Traffic Flow Theory, Lab Session I; HKUST(GZ), Fall 2026.`;
  plus the source's platform / instructor / TA / university lines (source lines 3–6 all used).
- **Heading structure mirrors the source exactly**: 7 `\section`, 14 `\subsection`, 10 `\subsubsection`
  (counts verified). Source `### 6.1 How the extension was performed` is written as
  `\subsection{How the extension was performed}` so LaTeX auto-numbers it as **6.1** (no doubled number).
- **12 tables** (`booktabs`: `\toprule/\midrule/\bottomrule/\addlinespace/\cmidrule`), every source table
  reproduced including header rows:
  1. IDM parameters / meanings / search bounds
  2. Four-stage data-processing pipeline
  3. Four calibration methods (RMSE)
  4. Closed-loop reconstruction error per event
  5. Two physical validation runs
  6. Cross-group driving-style comparison (10 columns, `\shortstack` headers)
  7. Counterfactual unit-policy rollouts
  8. Fully consistent parameter conversion
  9. Dimensionless combinations
  10. TGSIM calibrated parameters (v0,T,s0,a,b,J)
  11. TGSIM calibration error (RMSE/MAE)
  12. TGSIM driving-style metrics (two-row grouped header, `\multicolumn` + `\cmidrule`)
- **26 figures**, `\begin{figure}[H]\centering\includegraphics[width=\linewidth]{figures/fig_NN.png}`,
  `fig_01.png … fig_26.png` in document order, 1:1 with the 26 `[[IMAGE]]` markers, captions from the
  source, numbered automatically by LaTeX.
- **Mathematics** (`amsmath`): IDM acceleration and dynamic gap as an `align` block
  (`a_IDM = a[1 − (v/v0)^δ − (s*/s)^2]`, `s* = s0 + max(0, vT + vΔv/(2√(ab)))`, `δ=4`); the three
  scale relations (`equation`); the parameter conversion (`equation`); the dimensionless
  combinations `ṽ0 = v0/√(b s0)`, `ã = a/b`, `T̃ = T√(b/s0)`, `l̃ = L_veh/s0` (`equation`); and the
  calibration cost `J` as a sum over events with normalised σ_s / σ_v terms (`equation`).
- **Figures/tables 14–15 note**: the source places *"Fig. 26"* (safety-layer episode classification)
  as the 14th image and *"Fig. 15"* (152115 spacing with overrides) as the 15th. Their content is
  preserved verbatim, but since the brief requires **automatic in-order numbering**, the redundant
  `"Fig. N."` label prefix was stripped and LaTeX numbers them Figure 14 and Figure 15 respectively.

---------------------------------------------------------------------------
## 3. Verification actually performed

### 3a. TeX engine availability — NONE (so no PDF was produced)

Probe (real output):
```
$ command -v pdflatex || echo "pdflatex: NOT FOUND"
pdflatex: NOT FOUND
$ command -v tectonic || echo "tectonic: NOT FOUND"
tectonic: NOT FOUND
$ wsl.exe -- bash -lc 'command -v pdflatex || echo "wsl pdflatex: NOT FOUND"'
wsl pdflatex: NOT FOUND
```
Also searched: `which pdflatex tectonic latexmk xelatex` (none), `find` under
`%LOCALAPPDATA%` and `C:\Program Files` for `pdflatex*` (none), WSL `dpkg -l | grep texlive`
(empty). No MiKTeX/TeX Live install exists on the host. Per the brief, installing packages is
forbidden, so **no PDF log tail is available**; instead the structural checks below were run.
The document is written to the brief's package/tooling contract (standard Overleaf packages only),
so it should compile on Overleaf out of the box.

### 3b. Structural checks (all PASS)

Script: `C:\Users\ptiu1\AppData\Local\hermes\profiles\coder\cache\scratch\final_check.py`
plus `check_us2.py` (math-aware underscore scan) and `check_content.py` (content presence).

```
--- NOTES ---
 - ASCII-only source: OK
 - braces balanced (max nesting 3)
 - figures=26 tables=12 captions=38 includegraphics=26 (all files exist, in order)
 - packages: inputenc, fontenc, geometry, graphicx, booktabs, array, amsmath,amssymb, float, caption, hyperref
 - latex/ entries: ['figures', 'main.tex']
--- FAILS ---
 none
EXIT=0
```

What that check asserts (implemented, not eyeballed):
- **Balanced environments**: per-environment begin/end counts AND stack nesting order —
  no mismatch, nothing unclosed.
- **Balanced braces**: depth returns to 0 (escaped `\{`/`\}` ignored).
- **Every `\includegraphics` path exists**: 26/26 resolve to `latex/figures/fig_01..26.png`,
  in exact document order (asserted by name, not just count).
- **26 `figure` environments**, **12 `table` environments**, **38 `\caption{}`**
  (26 figures + 12 tables).
- **No unresolved markers / entities**: no `[[IMAGE]]`, `&lt;`, `&gt;`, `&sup2;`, `&radic;`,
  `&middot;`, `&#771;`, full-width colon `：`.
- **No unicode math tokens**: no `≈ ≤ ≥ → Δ σ ∑ √ × — – ² ³ δ γ` anywhere.
- **ASCII-only bytes**: the whole file is pure ASCII (UTF-8 encoded) → no `inputenc` failures.
- **Legal underscores only** (math-aware scan): 0 illegal; the 26 underscores inside
  `figures/fig_NN.png` are file-name characters handled by the modern LaTeX kernel.
- **hyperref loaded last**.

Content presence check — 93 source strings/values asserted present in `main.tex`
(all 0 missing), e.g. `3{,}435`, `1{,}787`, `1{,}648`, `133.63/133.94`, `727`,
`e2, e4, e5, e10, e11`, `1{,}140`, `123.36/2.091/28.72/13.97/185.16`, `51.72/40.14/41.38/39.45`,
`170.8/831.4/830.2`, `152115/155015`, `60.8/149.2/80`, `17 emergency`, `29.63/418.8/2.27`,
`5.1\%`, `15.39/34.42`, `0.91/0.41`, `208/149/61`, `group\_1…group\_5`, `1562/1485.84`,
`center\_sensor\_emergency`, `942–1974`, `338`, `23.84`, `2.80 ± 0.1`, `737`, `250.000000`,
`311`, `0.69`, `44.06/5.85/4.99/8.43`, `1.692/0.075/5.309/8.705`, `6e68\_map.yaml`,
`4{,}251/4{,}370`, `9.26/1.05`, `4.97/11.18`, `28.1/41.2`, `8.21`, `r=-0.18/p=0.57`,
`4.77/r=-0.43/p=0.15`, `r=0/p=1.00`, `Treibir \S21.7.3`, `b≈7.8`, `HV--HV/HV--AV`,
`SC$^2$I`, `Mann--Whitney`, `3.0 $\to$ 0`, `Treiber \& Kesting … p.300`.

Required-content spot-checks from the brief §6 are all present verbatim in meaning:
`center_sensor_emergency` details; group counts (group_1 0/4, group_2 1/1, groups 3–5 none);
and the caveats ("different physical cars/IPs per run and one run per group, so these are
descriptive style orderings, not inferential claims").

### 3c. Files / hash evidence

```
$ ls -la latex/
drwxr-xr-x  figures
-rw-r--r--  46527  main.tex

$ ls latex/figures/ | wc -l
26

$ sha256sum latex/main.tex
a52b6039a1547d4f1cd5fd32abf0236ec7148b0b19ea8ee7ebaf0ffdf6599672  latex/main.tex
```

---------------------------------------------------------------------------
## 4. `latex/` shipped-folder contract — satisfied

`latex/` contains only `main.tex` and `figures/` (26 PNG). `english_source.txt` has been deleted.
No PDF, because no compile was possible (§3a). Nothing else was added to the folder
(no `.aux/.log`, no build-report, no stray files).

---------------------------------------------------------------------------
## 5. Assumptions and deviations (all content-preserving)

1. **No local TeX engine** → PDF not produced; structural checks run instead, as the brief permits.
2. **Figure numbering**: captions auto-numbered in order per brief §4; the source's redundant
   `"Fig. N."` prefixes were dropped from caption text (see §2 for the Fig-26/Fig-15 ordering note).
3. **Caption vs. body split**: for markers whose source paragraph continues from the figure
   description into discussion (markers for source Figs 2, 3, 4, 9, 13, 14/26, 15), the descriptive
   sentences are the `\caption` and the remaining discussion sentences follow the figure as body
   text. **All content is retained** — nothing omitted, nothing invented.
4. **CJK duplicate labels dropped**: the TGSIM driving-style table's Chinese duplicates
   (平均车速, 行驶占比, 平均间距, 最小间距, 加速度 RMS, jerk RMS（平滑）) were removed because they
   duplicate the English labels and CJK cannot be typeset with `pdflatex` under the allowed package
   set. The English metric names (and all numbers) are preserved.
5. **Full-width colon** `：` (source lines 15, 65) → ASCII `:`.
6. **Source typo preserved verbatim**: "Treibir \S21.7.3" kept as in the source (not silently corrected).
7. **Unicode → ASCII LaTeX**: SC²I→`SC$^2$I`, `≈ ≤ ≥ → Δ σ γ δ ∑ √ ± × · – —` etc. mapped to their
   macros; en-dashes → `--`, em-dashes → `---`.
8. `\texorpdfstring{}` used in the two math-bearing headings (`Selected parameters: …`,
   `Run 155015 --- v0/3`) to keep hyperref PDF bookmarks clean.
9. **Figure file names keep the lowercase underscores** (`fig_01.png`) as the brief specifies;
   underscores in file names are supported by the LaTeX kernel since TeX Live 2019 (grffile merged
   into the kernel) and by Overleaf. This could not be compile-verified locally (§3a).
10. Package set limited to the brief's list plus the two core encoding packages
    (`inputenc`, `fontenc`), which are standard and Overleaf-default.

## 6. Remaining issues / open items

- **Cannot guarantee the PDF visually** without a TeX engine. On Overleaf this should compile
  directly (upload `latex/`; main document `main.tex`).
- If the target TeX distribution were pre-2019 (it will not be on Overleaf), the 26
  `figures/fig_NN.png` names would each need escaping/`grffile`. Flagged only for completeness.
- Everything else in the brief is met and structurally verified.

## 7. Services / processes

None started. No daemons, no background processes. Nothing to stop.

## 8. Suggestions (not done — outside brief scope)

- A companion `README` for the Overleaf folder was deliberately **not** added (brief forbids extra files).
- If the authors later want the Fig-14/Fig-15 numbering to match the source labels exactly,
  switch to manual `\caption` numbering — but that contradicts brief §4 (auto-number in order).
