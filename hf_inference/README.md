# hf_inference — Standalone Qwen2.5-VL VideoMME evaluation

A self-contained script to evaluate **Qwen2.5-VL** on a VideoMME subset using
plain HuggingFace `transformers` + `qwen_vl_utils`, **outside** the full
`lmms_eval` harness. Useful for debugging prompts, frame sampling, and scoring
without spinning up the whole evaluation framework.

## Files

| File | Purpose |
|------|---------|
| `qwenvl.py`      | Entry point: model loading, inference, and the eval loop. |
| `utils.py`       | Everything else: prompt styles/building, CLI parsing, seeding, video-path resolution, answer extraction, scoring & reporting. |
| `run_qwenvl.sh`  | Runner with ready-made command blocks — uncomment one and run. |
| `README.md`      | This file. |

## Input

A JSON list of VideoMME questions, e.g.
`random_debug/videomme_subset_200.json`. Each entry needs at least:

```json
{
  "videoID": "fFjv93ACGo8",
  "question": "....",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "answer": "C",
  "duration": "short", "domain": "Knowledge", "task_type": "Counting Problem"
}
```

Videos are looked up as `<video-root>/<videoID>.{mp4,MP4,mkv}`. Default
`--video-root` is `$HF_HOME/videomme/data` (matching the lmms_eval task).

## What the script does

1. Loads the JSON questions.
2. Builds the multiple-choice prompt (see **Prompt styles** below).
3. Runs Qwen2.5-VL on the corresponding video (greedy decoding, `max_new_tokens=16`).
4. Extracts the predicted option letter using lmms_eval's own MCQ parser
   (`lmms_eval/tasks/_task_utils/mcq_extract.py`, loaded directly by file path).
5. Scores against the gold answer and reports accuracy — overall and broken
   down by `duration` / `domain` / `task_type`.
6. Optionally writes per-question predictions + a summary to `--output` (JSON).

## Prompt styles (`--prompt-style`)

| Style | Option prompt | Post prompt | System message |
|-------|---------------|-------------|----------------|
| `lmms` (default) | "...based on the video **and the subtitles**..." | `\nAnswer with the option's letter from the given choices directly.` | `You are a helpful assistant.` |
| `standalone`     | "...based on the video." | `The best answer is:` | none |

- **`lmms`** reproduces exactly what `lmms_eval` feeds `qwen2_5_vl` on the
  `videomme` task (text from `tasks/videomme/utils.py` `videomme_doc_to_text`
  with the `default` kwargs, plus the `simple/qwen2_5_vl.py` system prompt).
- **`standalone`** is this script's original frames-only prompt.

## Frame sampling (`--fps` vs `--nframes`)

`qwen_vl_utils` accepts **either** a target fps **or** a fixed frame count
(not both):

- `--fps 1.0` (default): samples ≈ `duration_seconds` frames, clamped to
  `[4, 768]`, rounded to an even number.
- `--nframes N`: samples a fixed `N` frames uniformly across the whole video.

To match `lmms_eval --model_args ...,max_num_frames=4`, use `--nframes 4`.

> **Note:** lmms_eval's `simple/qwen2_5_vl.py` samples in two stages — it first
> decodes at 2 fps, then `np.linspace`-subsamples down to `max_num_frames` and
> force-includes the last frame. `--nframes N` here samples `N` frames directly
> in one stage. The frame **count** matches; individual timestamps can differ by
> a fraction of a second. Frame content is near-identical for practical runs.

## All flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--json`          | `random_debug/videomme_subset_200.json` | Input questions JSON. |
| `--video-root`    | `$HF_HOME/videomme/data` | Directory of `<videoID>.mp4` files. |
| `--model`         | `Qwen/Qwen2.5-VL-7B-Instruct` | HF model id / path. |
| `--prompt-style`  | `lmms` | `lmms` or `standalone` (see above). |
| `--fps`           | `1.0` | Target sampling fps (ignored if `--nframes` set). |
| `--nframes`       | `None` | Fixed frame count per video. |
| `--max-pixels`    | `151200` (360×420) | Max pixels per frame. Use `602112` to match lmms_eval. |
| `--max-new-tokens`| `16` | Generation length. |
| `--seed`          | `42` | Seeds `random`/`numpy`/`torch` + deterministic cuDNN. |
| `--flash-attn`    | off | Use `flash_attention_2` (needs `flash-attn`). |
| `--limit`         | `None` | Only evaluate the first N questions (smoke test). |
| `--output`        | `None` | Write predictions + summary JSON here. |

## Quick start

```bash
# Uncomment one block in run_qwenvl.sh, then:
bash hf_inference/run_qwenvl.sh

# Or call directly, e.g. mirror the lmms_eval setting:
python hf_inference/qwenvl.py \
  --prompt-style lmms --nframes 4 --max-pixels 602112 \
  --output logs/videomme/lmms_nframes4.json
```

## Output

Printed: a running per-question log and a final accuracy summary. If `--output`
is given, a JSON file with:

```json
{
  "summary": {
    "model": "...", "prompt_style": "lmms",
    "sampling": {"nframes": 4}, "max_pixels": 602112,
    "num_questions": 200, "evaluated": 200, "correct": 123,
    "overall_accuracy": 0.615,
    "breakdown": { "duration/short": {"correct": ..., "total": ..., "accuracy": ...}, ... }
  },
  "records": [ { "question_id": "001-1", "raw_pred": "...", "pred_answer": "C",
                 "answer": "C", "correct": true, ... }, ... ]
}
```

## Comparison against lmms_eval

To reproduce the reference command

```bash
python -m lmms_eval --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=602112,max_num_frames=4 \
  --tasks videomme --batch_size 1 --log_samples --output_path ./logs/videomme/subset200_hf
```

the closest standalone equivalent is:

```bash
python hf_inference/qwenvl.py \
  --prompt-style lmms --nframes 4 --max-pixels 602112 \
  --output logs/videomme/lmms_nframes4.json
```

Remaining (minor) differences: the two-stage frame sampling described above.
