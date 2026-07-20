# Publication figure: the video-QA evaluation pipeline

Camera-ready LaTeX/TikZ version of the two HTML artifacts describing every
configuration knob between a VideoMME row and a reported accuracy.

| File | What it is |
| --- | --- |
| `eval_pipeline.tex` | The figure itself, as an `\input`-able TikZ picture. Vector, no external assets, native width 13.56 cm (fits the 5.5 in ICLR/NeurIPS column without scaling). |
| `knob_table.tex` | Companion appendix table: the full knob inventory, ordered by how much each moves the reported number. Needs `booktabs`. |
| `eval_pipeline_standalone.tex` | Wrapper that compiles the figure to a tightly cropped one-page PDF for `\includegraphics`. |
| `demo_paper.tex` | Minimal host document showing figure + table + a ready-to-edit caption in a 5.5 in column. |
| `eval_pipeline_standalone.pdf`, `demo_paper.pdf` | Pre-built output. |

## Using it in a paper

Preamble:

```latex
\usepackage{tikz}
\usetikzlibrary{positioning,calc,arrows.meta}
\usepackage{xcolor,amsmath,booktabs}
```

Body:

```latex
\begin{figure}[t]
  \centering
  \input{figures/eval_pipeline}
  \caption{...}          % see demo_paper.tex for a drafted caption
  \label{fig:eval-pipeline}
\end{figure}
```

Alternatively, `\includegraphics[width=\linewidth]{figures/eval_pipeline_standalone.pdf}`
if you would rather not carry TikZ into your build.

## Building

```bash
pdflatex -output-directory=figures figures/eval_pipeline_standalone.tex
pdflatex -output-directory=figures figures/demo_paper.tex
```

Only `tikz`, `xcolor`, `amsmath`, `booktabs` and the psnfss fonts are required —
deliberately no `standalone`, `tcolorbox` or `pdfcrop`, so it builds on a minimal
TeX Live (the standalone wrapper resizes the page to the figure box itself).

## Design notes

- **Native size, no scaling.** Nothing is wrapped in `\resizebox`, so every
  label keeps the point size it was set at (5.5 pt body, 6.8 pt titles). If you
  scale the figure, the type will fall below the legibility floor.
- **Grayscale-safe.** Hue encodes the sub-system (source / vision / text /
  inference / scoring), but stage order, the accent rule and the numbering carry
  the same information, so a b/w print loses nothing.
- **Two levels of detail.** The figure carries three knobs per stage plus the
  pitfalls; the complete inventory lives in `knob_table.tex` so the figure stays
  readable at column width.
- Content is taken from the two source artifacts, i.e. from
  `run_video_qa_hf.sh`, `VIDEOMME_VLLM_ARGS.md`, `LMMS_EVAL_VIDEO_QA_NOTES.md`
  and `VIDEO_BENCHMARKS_REFERENCE.md` as of this checkout
  (vLLM 0.11.0, qwen-vl-utils 0.0.14, Qwen2.5-VL-7B-Instruct).
