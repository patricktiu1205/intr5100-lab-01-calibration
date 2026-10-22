Rebuild Overleaf-ready LaTeX (pdflatex, English) to match the corrected report results/lab1_template_report.html.

Current correct figure order (extract each <img> base64 in document order into latex/figures/fig_01..26.png):
1-2 closed-loop spacing/speed; 3-4 validation runs 152115/155015; 5-9 overlap (distributions, time-norm, speed tracking, lateral, safety); 10-12 group validation (spacing trajectories, speed trajectories, trigger-type stacked bars); 13 scale-ratio gamma; 14 counterfactual rollouts; 15 actual 152115 spacing; 16-26 TGSIM 11 figures.

main.tex requirements identical to prior brief: article 11pt, only Overleaf-default packages (graphicx,booktabs,array,amsmath,amssymb,float,caption,geometry,hyperref last), figures [H] with captions taken from the English figcaps, all tables reproduced with booktabs, full amsmath typesetting of IDM / cost J / dimensionless combos / parameter conversion, authors PakHin TIU and Xuqian JU, course/title lines as before. Number captions automatically; no literal 'Fig. N' numbers embedded.

Final latex/ must contain only main.tex and figures/ (26 PNG). Remove any other files (audit_dump.txt, english_source.txt, extra PNG duplicates). Do not install packages or touch git. Structural verification: 26 includegraphics, balanced env/braces, no [[FIG]]/unicode-math leftovers.
