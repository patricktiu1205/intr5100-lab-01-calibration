# Brief: Produce Overleaf-ready LaTeX version of the Lab I report (English)

## Project
`C:\Users\ptiu1\Documents\PhDStudy\INTR5100\lab1`
Output folder: `latex/` — this folder will be zipped and uploaded to Overleaf as-is.

## Content source (single source of truth)
`latex/english_source.txt` — the full English text of the latest report, linearly
ordered with `# / ## / ### / ####` headings, table rows (` | ` separated), and
`[[IMAGE]]` markers. The paragraph(s) immediately following each marker are that
figure's caption text. Use ALL of it; do not invent or omit content.

## Figures
`latex/figures/fig_01.png` … `fig_26.png` in document order, matching the 26
`[[IMAGE]]` markers 1:1.

## Requirements
1. `main.tex` — compiles with **pdflatex**, UTF-8. Use only standard packages
   available on Overleaf by default: `article`, `graphicx`, `booktabs`,
   `amsmath,amssymb`, `geometry`, `float`, `caption`, `hyperref` (last), maybe
   `array`. No exotic/custom .sty files.
2. Title page / preamble:
   - Title: `Experiment I: Car-Following Model Calibration`
   - Authors: **PakHin TIU** and **Xuqian JU**
   - Course line: INTR 5100 Traffic Flow Theory, Lab Session I;
     HKUST(GZ), Fall 2026.
3. Section structure exactly mirrors the source headings (sections/subsections).
   Tables with `booktabs`; reproduce every table including header rows, using
   `\hline`/booktabs rules. Units in captions/headers.
4. Figures: `\begin{figure}[H]\centering\includegraphics[width=\linewidth]`
   (two-up side by side acceptable where the source clearly presents paired
   panels; otherwise one per figure), captions from the source, numbered
   automatically in order.
5. Typeset all mathematical content properly with `amsmath` — at minimum:
   - IDM: a_IDM = a[ 1 − (v/v0)^δ − (s*/s)^2 ],
     s* = s0 + v T + v Δv/(2√(ab)), δ = 4
   - the calibration cost J (sum over events, σ_s, σ_v normalised terms)
   - dimensionless combinations ṽ0 = v0/√(b s0), ã = a/b, T̃ = T√(b/s0)
   Do not leave unicode math tokens in the PDF body; map: SC²I → SC$^2$I,
   ≈, ≤, ≥, →, Δ, σ, ∑, sub/sup, etc.
6. The `center_sensor_emergency` details, group counts (group_1 0/4, group_2 1/1,
   groups 3–5 none), and caveats must survive verbatim in meaning.
7. Clean formatting: reasonable margins (geometry), 11pt, no raw HTML,
   no `[[IMAGE]]` markers, no leftover source markup.

## Verification (do before finishing)
- If a TeX engine is available locally (check `pdflatex` or `tectonic`), compile
  to PDF and fix all errors; report the log tail. If none is available, say so
  explicitly and instead run structural checks: balanced environments, every
  `\includegraphics` path exists, 26 includegraphics, all tables balanced, and
  no unresolved unicode/marker tokens.
- Do not install packages, no git, do not modify anything outside `latex/`.
- After `main.tex` is final, DELETE `latex/english_source.txt` from the shipped
  folder (content is fully incorporated).
- Ensure `latex/` contains only: `main.tex`, `figures/` (26 PNG), and PDF if a
  local compile produced one.
