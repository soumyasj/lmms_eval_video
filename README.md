# lmms_eval_video

A fork of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for measuring how much the
evaluation setup (frames, decoder, resolution, backend, batching) moves video-benchmark scores,
and the home of **Meta-VideoBench (`metav`)**, a 1000-item video benchmark built to rank models
in minutes instead of hours.

- [FIXES.md](./FIXES.md): what was changed in `lmms_eval/` and why.
- [README_upstream.md](./README_upstream.md): the original lmms-eval README (install, all tasks).
- [VIDEO_BENCHMARKS_REFERENCE.md](./VIDEO_BENCHMARKS_REFERENCE.md): the video benchmarks as they behave here.

## Meta-VideoBench (`metav`)

Five video benchmarks recombined into one evaluation set of **1000 questions on 788 videos**
(none longer than 60 min, at most 3 questions per video), in-tree as the task `metav`:

| source | full set | in `metav` | subtask |
|---|---|---|---|
| Video-MME | 2700 | 312 | `metav_videomme` |
| LongVideoBench (val, video) | 1337 | 240 | `metav_longvideobench_val_v` |
| LVBench | 1549 | 146 | `metav_lvbench` |
| VideoMMMU (perception / comprehension / adaptation) | 300 / 300 / 300 | 60 / 60 / 60 | `metav_video_mmmu_{perception,comprehension,adaptation}` |
| VSI-Bench | 5130 | 122 | `metav_vsibench` |

How it was built: questions that three models answer correctly from **black frames** were removed
(the set needs the video), and every question category of every source is covered. Each subtask
re-uses its source task's visual loader, parser and metric, so an item is scored exactly as in its
benchmark. Scores are lower than on the sources by construction (the
easy items are gone): compare models by rank and by `metav` score, not against published numbers.

Files: `lmms_eval/tasks/metav/data/metav.jsonl` (the items: question, options, answer, video,
duration, source ids, categories), `metav_ids.json` (selected ids per source), `metav_prompts.json`
(the prompt text of every item, materialized once).

```bash
# any wrapper, like any lmms_eval task
python -m lmms_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_num_frames=32 \
       --tasks metav --batch_size 1 --log_samples --output_path logs/metav_try
python eval_scripts/score_metav.py --write logs/metav_try/*/*_results.json     # metav score = macro over the 5 sources
```

`metav` is also a `--task` / `TASKS` value for every sweep below. Prompt knobs:
`METAV_PROMPT_VARIANT`, `METAV_POST_PROMPT`, `METAV_OPTION_STYLE` (see `lmms_eval/tasks/metav/utils.py`).

## Setup

- Run every command from the repository root.
- Environments: the runners execute in the current environment unless you point `ENV_A`
  (transformers 4.57 / vllm 0.11.0: Qwen2.5-VL, Qwen3-VL, Gemma 3), `ENV_A_BI` (same with vllm 0.11.1,
  used for batch-invariant vLLM runs) and `ENV_B` (transformers 5.16 / vllm 0.28: Qwen3.5, InternVL,
  GLM-V) at the `bin` dirs of three envs, or `LMMS_EVAL_ENV_BIN=<env>/bin` at one env for everything.
- `HF_HOME` defaults to `~/.cache/huggingface`; point it at the cache that holds the benchmark videos.
  Gated datasets need `HF_TOKEN` in the environment.
- `--model` takes an HF id or an alias (`qwen25vl_7b`, `qwen3vl_8b`, `qwen35_9b`, `internvl35_8b`,
  `glm46v_flash`, `gemma3_12b`). The runner family is inferred from the id; fine-tunes need
  `--family` (e.g. `Video-R1/Video-R1-7B --family qwen25vl`).
- Every sweep takes `--dry_run` (print the commands) and resumes: a run whose results file exists is skipped
  (`RESUME=0` redoes it). Results land in `logs/<sweep>/<task>/<model>/<exp>/<run>/` with a
  `config.json` and lmms_eval's `*_results.json` / samples.

## Experiment 1: parameter sweep

```bash
bash eval_scripts/sweep_params.sh --model qwen25vl_7b --task videomme            # all experiments
bash eval_scripts/sweep_params.sh --model qwen3vl_8b --task metav --exp 1,5,7
```

One model, one task, one knob at a time (base: 4 frames, hf, batch 1, seed 22):

| exp | varies | tells you |
|---|---|---|
| exp1 | hf vs vLLM (frames 4-128, 3 seeds) | whether the backend changes answers |
| exp2 | batch size 1-32 | whether batching is score-neutral |
| exp3 / exp3a | decoder (decord, torchvision, torchcodec) / random frame positions | whether the decoder or the sampled positions matter |
| exp4 | 1 fps capped vs uniform at the same count | frame selection |
| exp5 | 4-128 frames | accuracy vs frame budget |
| exp6 | pixels per frame (InternVL: tiles) | accuracy vs resolution |
| exp7 | `.mp4` vs the same frames as JPEGs | decoder + video-processor path vs the model |
| exp8 | temperature 0.7-1.0, 3 seeds | sampling sensitivity |

Knobs a family's wrapper cannot vary are skipped with a printed reason. Grids are environment
variables (`NFRAMES`, `SEEDS`, `MAX_PIXELS`, ...), listed in the script header.

## Experiment 2: frame-budget protocol

```bash
bash eval_scripts/sweep_frames.sh --model qwen25vl_7b                       # TASKS default: videomme mvbench tempcompass_multi_choice vsibench mmvu_val
TASKS="videomme metav" bash eval_scripts/sweep_frames.sh --model internvl35_8b
```

exp5 (raw `.mp4`) and exp7 (pre-extracted frames) at 4, 8, 16, 32, 64, 128 frames on the hf backend,
one seed, for every task in `TASKS`. Gives the accuracy-vs-frames curve of a model and, per frame
count, the gap between the wrapper's video path and plain frames. GLM and Gemma get exp7 only.

## Precomputed frames (needed by exp7)

```bash
python eval_scripts/make_frames_task.py --task videomme            # hours for long benchmarks; use a batch job, --workers N
python eval_scripts/make_frames_task.py --task metav
python eval_scripts/make_frames_task.py --task vsibench --limit 3   # smoke test
```

For a task `T` (or a group) this extracts 128 uniform frames per video into `data/frames/videos/`,
writes `data/frames/T_frames<N>.parquet` for N = 4 ... 128, and creates the tasks `T_frames<N>`
next to `T`'s YAML (same prompt and metric, frames instead of the video). `data/frames/` is
gitignored; regenerate it on a new machine. Both sweeps pick the variants up automatically.

## Experiment 3: the vLLM video fix

```bash
bash eval_scripts/sweep_vllm_fix.sh --model Qwen/Qwen2.5-VL-7B-Instruct --task metav
```

Stock vLLM sends a video as N separate images; the `vllm_fix` plugin sends the real video (see
FIXES.md). Runs hf (reference), stock vLLM and `vllm_fix` at `NFRAMES` (default 4) over batch sizes
1-32 and seeds 22, 42, 72. Compare `vllm` vs `vllm_fix` for the size of the artefact and both vs `hf`.
Qwen2.5-VL only.

## Results

```bash
python eval_scripts/collect_results.py                    # one table per sweep / task / model / experiment
python eval_scripts/collect_results.py --csv runs.csv
```

Single runs without a sweep: call a runner in `eval_scripts/runners/` directly, e.g.
`bash eval_scripts/runners/run_video_qa_hf_multigpu.sh --tasks videomme --nframes 8 --limit 50`
(`--help` lists the flags).
