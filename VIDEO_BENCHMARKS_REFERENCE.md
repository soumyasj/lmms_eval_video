# Video benchmark reference (lmms-eval)

Per-benchmark defaults as they exist **in this checkout**: task names, the exact prompt the model receives, generation defaults, whether an LLM judge is involved, metrics, and dataset-specific gotchas.

Companion doc: [`VIDEOMME_VLLM_ARGS.md`](./VIDEOMME_VLLM_ARGS.md) for the model/CLI side.

---

## Read this first

**Frame count is not a task setting.** Every video task here hands the model a *video path*; the number of frames is decided entirely by the model wrapper (`nframes` / `fps` / `max_frame_num` in `--model_args`). There is no per-benchmark frame default in the task YAMLs. The `frame_num: 32` entries you'll see in a few `*_w_subtitle.yaml` files are used **only to align subtitle text with frame timestamps** inside `doc_to_text` — they do not change how many frames the model sees. Two exceptions worth knowing:

- **SeedBench** ships pre-extracted frames as PIL images in the dataset, so frame count is fixed by the data, not by you.
- **LongVideoBench `_i` variants** use `dataset_kwargs.max_num_frames` (default 16) to compute subtitle interleaving timestamps — again, a prompt-construction number, not a sampling one.

Because `--model vllm` decodes video into frames client-side (see `VIDEOMME_VLLM_ARGS.md` §4), whatever you pass as `nframes` is what every benchmark below gets.

**`max_new_tokens` in these YAMLs is a floor, not a cap, on the vllm path.** `_select_max_new_tokens` takes `max(task_value, model_value)` and the chat vllm class defaults to 4096. A task asking for 16 will still generate up to 4096 unless you pass `max_new_tokens=16` in `--model_args`. This matters most for the 16-token MCQ benchmarks (most of this list).

**Judge environment variables** (shared by every judged task):

| Var | Default | Used by |
|---|---|---|
| `API_TYPE` | `openai` | all judged tasks (`openai`, `azure_openai`, `local`, `bedrock`, async variants) |
| `OPENAI_API_KEY` | `YOUR_API_KEY` | openai provider |
| `OPENAI_API_URL` | `https://api.openai.com/v1/chat/completions` | openai provider |
| `MODEL_VERSION` | `gpt-4o-2024-11-20` | activitynetqa, videochatgpt, mmvu |
| `AZURE_API_KEY`, `AZURE_ENDPOINT`, `API_VERSION` | – / – / `2024-02-15-preview` | azure provider |
| `LLM_JUDGE_URL` | `http://localhost:8000/v1/chat/completions` | `API_TYPE=local` (self-hosted judge) |

Note the `metadata.gpt_eval_model_name: gpt-3.5-turbo-0613` lines in activitynetqa/videochatgpt YAMLs are **dead** — the code reads `MODEL_VERSION` instead and defaults to `gpt-4o-2024-11-20`.

**Not present in this repo:** MMBench-Video, MMOU, VideoSIAH-Eval. No task, alias, or YAML matches them; they'd need to be added.

---

## Summary table

| Benchmark | Task name(s) | Samples | `max_new_tokens` | Judge | Primary metric |
|---|---|---|---|---|---|
| MVBench | `mvbench` (group, 20 subtasks) | 4,000 (200 × 20) | 16 | no | `mvbench_accuracy` |
| VideoMME | `videomme`, `videomme_w_subtitle`, `videomme_long`, `videomme_long_w_subtitle` | 2,700 (**local: 200 subset**) | 16 | no¹ | `videomme_perception_score` |
| TempCompass | `tempcompass` (group, 4 subtasks) | 7,540 | unset (greedy) | **yes** — gpt-3.5-turbo-1106 | `avg_accuracy` + 5 dims |
| MLVU | `mlvu_dev`, `mlvu_test` | 2,174 dev / 502 test | 16 | no | `mlvu_percetion_score` |
| LongVideoBench | `longvideobench_val_v`, `_val_i`, `_test_v`, `_test_i` | 1,337 val / test held-out | 32 | no | `lvb_acc` |
| VideoMathQA | `videomathqa_{mcq,mbin}[_cot][_w_subtitle]` (8) | 2,100 (420 mcq + 1,680 mbin) | 16 / **8096** (cot) | no | `videomathqa_perception_score` |
| Video-MMMU | `video_mmmu_{perception,comprehension,adaptation}` | 900 (300 each) | 1024 | no | `mmmu_acc` |
| MMVU-Val | `mmvu_val` | 1,000 | 1024 | **yes** (open-ended fallback) | `accuracy` |
| VSI-Bench | `vsibench`, `vsibench_debiased`, `vsibench_pruned` | 5,130 / 2,362 / 2,768 | 16 | no | `vsibench_overall` |
| MINERVA | `minerva` | 1,341 | 32 | no | `minerva_acc` |
| SciVideoBench | `scivideobench` | 1,000 | 16 | no | `scivideobench_acc` |
| ActivityNetQA | `activitynetqa` | 8,000 | 64 | **yes** — every sample | `gpt_eval_accuracy`, `gpt_eval_score` |
| EgoSchema | `egoschema`, `egoschema_subset` | 5,031 full / 500 subset | unset (greedy) | no | `submission` (+ `score` on subset) |
| PerceptionTest | `perceptiontest_val_mc`, `perceptiontest_test_mc` | 19,140 val / test held-out | unset (greedy) | no | `accuracy` (val) / `submission` (test) |
| SeedBench | `seedbench`, `seedbench_lite`, `seedbench_ppl` | 17,990 (image+video) | unset (`until: ASSISTANT:`) | no | `seed_all`, `seed_image`, `seed_video` |
| VideoChatGPT | `videochatgpt_gen`, `videochatgpt_temporal`, `videochatgpt_consistency` | 1,996 / 499 / 998 | 1024 | **yes** — every sample | `gpt_eval_score_*` |
| NExT-QA | `nextqa_mc_test`, `nextqa_oe_test`, `nextqa_oe_val` | 8,564 mc / 9,178 oe-test / 5,343 oe-val | unset (greedy) | no | `exact_match` (mc) |
| LVBench | `lvbench` | 1,549 | 16 | no | `lvbench_score` |
| VideoEval-Pro | `videoevalpro` | 1,289 | 16 | **yes** — every sample | `videoevalpro_score` |
| JumpScore | `JumpScore` | 189 | 1024 | no | `jumpscore_map`, `jumpscore_score` |
| Charades-STA | `temporal_grounding_charades` | 3,720 | 50 | no | `charades_sta_mIOU`, `IOU@{3,5,7}` |
| Video-Holmes | `video_holmes`, `video_holmes_test`, `video_holmes_reasoning` | 1,837 | 64 | no | `video_holmes_accuracy` + 7 dims |
| WorldSense | `worldsense`, `worldsense_w_subtitle` | 3,172 | 1024 | no | `worldsense_score` |
| VSI-Super | `vsisuper_count_*`, `vsisuper_recall_*` (9) | 400 count / 300 recall | 16 | no | `mra` / `accuracy` |
| TVBench | `tvbench` (group, 10 subtasks) | 2,525 | 32 | no | `tvbench_acc` |

¹ except the `videomme_convert_mcq_oe` variant, which calls `gpt-4o-mini`.

### LLaVA-OneVision-2 reference settings (upstream `llava-onevision2` branch)

**Reference origin:** [`EvolvingLMMs-Lab/lmms-eval@llava-onevision2`](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/llava-onevision2), HEAD `3997a60c` ("feat: add tuned online codec fallback path"). These are the *published reproduction* settings for `lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct` — treat this branch as the untouched upstream baseline, not as something to edit locally. Values below were read from `examples/llava_onevision2_repro/{run_frames,run_codec}.sh` and the README tables on that branch, not from this checkout.

**Shared across every run** (both launchers): `trust_remote_code=True`, `attn_implementation=flash_attention_2`, `messages_format=timestamp`, **`fps=1`**, `--batch_size 1`, `accelerate launch --num_processes=8`. Note `fps=1` is constant everywhere — the per-benchmark tuning happens entirely in `max_num_frames` and the pixel budget.

**Frames backend** (`video_backend=frames`, uniform sampling). `MP` sets **both** `min_pixels` and `max_pixels` to the same value — i.e. a fixed per-frame resolution, not a range:

```bash
TASK=<task> F=<frames> MP=<pixels> [IL=1] bash examples/llava_onevision2_repro/run_frames.sh
```

| Benchmark (`TASK`) | `F` (`max_num_frames`) | `MP` (`min`=`max_pixels`) | `IL` | Reported score |
|---|---:|---:|:---:|---:|
| `ov2_videomme_short_wo_sutitle` | 128 | 321,489 | – | 81.56 (perception) |
| `ov2_videomme_medium_wo_sutitle` | 256 | 136,900 | – | 72.56 (perception) |
| `ov2_videomme_long_wo_sutitle` | 640 | 102,400 | – | 62.33 (perception) |
| `videomme_short_interleaved_subtitle` | 128 | 233,289 | 1 | 83.33 (perception) |
| `videomme_medium_interleaved_subtitle` | 448 | 128,164 | 1 | 78.00 (perception) |
| `videomme_long_interleaved_subtitle` | 640 | 84,100 | 1 | 69.22 (perception) |
| `lvbench` | 768 | 84,100 | – | 55.46 |
| `mlvu_dev` | 512 | 72,900 | – | 76.62 (perception) |
| `videoeval_pro` | 768 | 84,100 | – | 61.45 (overall) |
| `vsibench` | 128 | 153,664 | – | 70.94 (overall) |
| `timelens_activitynet` | 128 | 153,664 | – | mIOU 53.75 |
| `timelens_charades` | 128 | 153,664 | – | mIOU 53.49 |
| `timelens_qvhighlights` | 128 | 153,664 | – | mIOU 66.43 |
| `videommev2_interleaved_subtitle` | 64 | 330,000 | 1 | 18.34 |

**Codec backend** (`video_backend=codec`, canvas-packed video tokens). Here `TC` sets `max_num_frames` **and** `codec_target_canvas`; pixels are fixed defaults `min_pixels=100352` / `max_pixels=313600` (overridable via `MIN_PX`/`MAX_PX`), *not* per-benchmark:

```bash
TASK=<task> TC=<canvases> [TS=2] [IL=1] bash examples/llava_onevision2_repro/run_codec.sh
```

| Benchmark (`TASK`) | `TC` (canvases) | `TS` (`timestamp_decimals`) | `IL` | Reported score |
|---|---:|:---:|:---:|---:|
| `videommev2_interleaved_subtitle` | 64 | 1 | 1 | 19.89 |
| `JumpScore` | 128 | 2 | – | mAP 0.7549 |
| `timelens_activitynet` | 64 | 1 | – | mIOU 51.23 |
| `timelens_charades` | 64 | 1 | – | mIOU 50.09 |
| `timelens_qvhighlights` | 64 | 1 | – | mIOU 63.53 |

**`IL=1`** is not a model arg — it exports `LMMS_IL_NOPREFIX=1` and `LMMS_IL_FILTER_NOISE=1`, used for the interleaved-subtitle task variants.

**Notes for anyone porting these to this checkout:**

- **The pixel budgets are far below Qwen's.** OV2 runs 72,900–330,000 px/frame against 128–768 frames; the Qwen2.5-VL path here caps at 602,112 px/frame with 4–32 frames. The two families trade resolution against frame count in opposite directions — copying `max_pixels` across model families is meaningless.
- **`MP` is set as a fixed point** (`min_pixels == max_pixels`), which pins every frame to one resolution. The Qwen wrapper instead takes a `[min,max]` range and lets `smart_resize` pick — so an OV2 `MP` value is not a drop-in `--max_pixels`.
- **Task names differ from this checkout.** `ov2_videomme_*`, `videomme_*_interleaved_subtitle`, `videommev2_*`, `timelens_*`, and `videoeval_pro` (vs. local `videoevalpro`) are branch-specific. Only `lvbench`, `mlvu_dev`, `vsibench`, and `JumpScore` name-match here.
- **These runs assume 8 GPUs and flash-attn.** At `F=768` × 84,100 px they are not reproducible on a single 40GB card; see [`VIDEOMME_VLLM_ARGS.md`](./VIDEOMME_VLLM_ARGS.md) for the memory arithmetic.

### Sample counts — detail (per config / split)

Counts are the exact row counts the tasks load, verified against HuggingFace's datasets-server (`/size`) at query time, except where noted. Subtitle / CoT variants (`_w_subtitle`, `_cot`) **re-prompt the same rows** — they do not add samples.

| Benchmark | Split | Breakdown | Total |
|---|---|---|---|
| MVBench | train | 20 subtasks × 200 each | **4,000** |
| VideoMME | test | upstream 2,700; **this checkout uses `videomme_subset_200.parquet` → 200** | 2,700 / **200 local** |
| TempCompass | test | multi-choice 1,580 · yes_no 2,453 · caption_matching 1,503 · captioning 2,004 | **7,540** |
| MLVU | test | dev 2,174 · test 502 | 2,174 / 502 |
| LongVideoBench | validation / test | val 1,337 (counted from local `lvb_val.json`); test split has no ground truth | 1,337 / — |
| VideoMathQA | test | mcq 420 · multi_binary (mbin) 1,680 | **2,100** |
| Video-MMMU | test | perception 300 · comprehension 300 · adaptation 300 (gated; from dataset card) | **900** |
| MMVU-Val | validation | — | **1,000** |
| VSI-Bench | test | full 5,130 · debiased 2,362 · pruned 2,768 | 5,130 / 2,362 / 2,768 |
| MINERVA | — (JSON URL) | counted from `minerva.json` | **1,341** |
| SciVideoBench | test | — | **1,000** |
| ActivityNetQA | test | — | **8,000** |
| EgoSchema | test | GENERATION (full) 5,031 · Subset 500 (MC / MC_PPL configs also 5,031) | 5,031 / 500 |
| PerceptionTest | validation / test | val mc 19,140; test split is gated + held-out (submission only) | 19,140 / — |
| SeedBench | test | 17,990 total across all 12 dims; `seed_video` is the video-dimension subset only | **17,990** |
| VideoChatGPT | test | Generic 1,996 · Temporal 499 · Consistency 998 | **3,493** |
| NExT-QA | test / val | MC test 8,564 · OE test 9,178 · OE val 5,343 (OE train 37,523 unused) | 8,564 / 9,178 / 5,343 |
| LVBench | train | — | **1,549** |
| VideoEval-Pro | test | — | **1,289** |
| JumpScore | test | — | **189** |
| Charades-STA | test | — | **3,720** |
| Video-Holmes | test | counted from `test_Video-Holmes.json` (datasets-server pending) | **1,837** |
| WorldSense | test | from dataset card (`num_examples`) | **3,172** |
| VSI-Super | test | Count 400 (split across 10/30/60/120-min buckets) · Recall 300 (10/30/60/120/240-min) | 400 / 300 |
| TVBench | train | 10 subtasks, uneven: action_count 536 · action_sequence 437 · action_antonym 320 · moving_direction 232 · object_shuffle 225 · egocentric_sequence 200 · scene_transition 185 · action_localization 160 · object_count 148 · unexpected_action 82 | **2,525** |

Notes on gated/unverifiable entries: `longvideobench/LongVideoBench`, `lmms-lab/VideoMMMU`, and `lmms-lab/PerceptionTest_Test` are gated on the Hub, so their `/size` calls returned 401. LongVideoBench val was counted from a local copy of the annotation file; Video-MMMU came from the public dataset card; PerceptionTest test is both gated and label-held-out, so no count is asserted.

---

## MVBench

- **Tasks:** group `mvbench` → `mvbench_action_sequence`, `_moving_count`, `_action_prediction`, `_episodic_reasoning`, `_action_antonym`, `_action_count`, `_scene_transition`, `_object_shuffle`, `_object_existence`, `_fine_grained_pose`, `_unexpected_action`, `_moving_direction`, `_state_change`, `_object_interaction`, `_character_order`, `_action_localization`, `_counterfactual_inference`, `_fine_grained_action`, `_moving_attribute`, `_egocentric_navigation`.
- **Dataset:** `OpenGVLab/MVBench`, `revision: video`, `cache_dir: mvbench_video`, `create_link: True`. Split is **`train`** (not test).
- **Generation:** `max_new_tokens: 16`, `temperature: 0`, `top_p: 1.0`, `num_beams: 1`, `do_sample: false`.
- **Judge:** none. Answer matching via `mcq_acc` (VQA-style punctuation/article normalization).
- **Prompt** (`mvbench_doc_to_text`):

```
Question:<question>
Option:
(A) <opt0>
(B) <opt1>
...
Only give the best option.
```

- **Gotchas:** each subtask reads a different video folder via `DATA_LIST[sub_task]`; `clevrer` and `star` fall back to a `data0613/` subdirectory. `mvbench_episodic_reasoning` has a commented-out `mvbench_frames_doc_to_visual` — that subtask's data is a *directory of frames*, so the default video reader may fail on it.

## VideoMME

- **Tasks:** `videomme`, `videomme_w_subtitle`, `videomme_long`, `videomme_long_w_subtitle`. Variants in subdirectories: `videomme_number_option`, `videomme_gt_none_option`, `videomme_no_visual`, `videomme_convert_mcq_oe`, `videomme_revert_oe_mcq`, `videomme_video_only_abcd`, `videomme_random_choice`.
- **Dataset:** upstream `lmms-lab/Video-MME`. **In this checkout `videomme.yaml` is modified** to `dataset_path: parquet` pointing at `random_debug/videomme_subset_200.parquet`. Videos resolve to `$HF_HOME/videomme/data/<videoID>.mp4`.
- **Generation:** `max_new_tokens: 16`, `temperature: 0`, `top_p: 1.0`.
- **Judge:** none for the main tasks. `videomme_convert_mcq_oe` calls **`gpt-4o-mini`** to convert MCQ answers to open-ended form.
- **Prompt** (`videomme_doc_to_text`):

```
Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.
<question>
A. <opt>
B. <opt>
C. <opt>
D. <opt>

Answer with the option's letter from the given choices directly.
```

- **Gotchas:** the fixed preamble mentions "and the subtitles" **even for the no-subtitle `videomme` task** — that's upstream behavior, not a local bug. `cluster_key: videoID` groups questions from one video for clustered standard-error computation. A `qwen3_vl` prompt profile exists (`format: qwen3_vl`, `pre_prompt: "Question: "`).

## TempCompass

- **Tasks:** group `tempcompass` → `tempcompass_multi_choice`, `tempcompass_yes_no`, `tempcompass_caption_matching`, `tempcompass_captioning`.
- **Dataset:** `lmms-lab/TempCompass`, split `test`, cache `tempcompass` (videos under `<cache>/videos`).
- **Generation:** no `generation_kwargs` → falls back to `{until: [fewshot_delimiter], do_sample: False}`. On the vllm path that means **`max_new_tokens` = the model default (4096)**.
- **Judge: yes — hardcoded `gpt-3.5-turbo-1106`** (`temperature: 1.0`, `max_tokens: 128`, `presence_penalty: 1`). Not configurable via `MODEL_VERSION`; edit `tempcompass/utils.py` to change it. Requires `OPENAI_API_KEY`.
  - MC / yes-no / caption-matching: judge is a **fallback**, only when hand-written regex fails to match a letter.
  - Captioning: judge on **every** sample.
- **Prompt:** `pre_prompt + question + post_prompt`, where `post_prompt` is keyed by subtask:

| Subtask | post_prompt |
|---|---|
| multi-choice | `\nPlease directly give the best option:` |
| yes_no | `\nPlease answer yes or no:` |
| caption_matching | `\nPlease directly give the best option:` |
| captioning | `""` (empty) |

- **Gotchas:** `HF_HOME` is read with `os.environ["HF_HOME"]` (hard `KeyError` if unset, unlike other tasks which use `getenv` with a default). Metrics are reported per temporal dimension: `speed`, `direction`, `action`, `order`, `attribute_change`.

## MLVU

- **Tasks:** `mlvu_dev` (`sy1998/MLVU_dev`), `mlvu_test` (`sy1998/MLVU_Test`).
- **Generation:** `max_new_tokens: 16`, `temperature: 0`, `top_p: 1.0`.
- **Judge:** none. Letter extraction via the shared `_task_utils/mcq_extract.extract_mcq_answer` with choices A–D.
- **Prompt:** `pre_prompt + question + post_prompt` with the default post_prompt:

```
\nOnly give the best option.\nBest option: (
```

- **Gotchas:** the metric name has an upstream typo — `mlvu_percetion_score`. The trailing `(` in the prompt is deliberate (forces the model to complete a letter), but interacts badly with chat models that ignore prefill. A `plm` profile drops the `Best option: (` suffix.

## LongVideoBench

- **Tasks:** `longvideobench_val_v`, `longvideobench_val_i`, `longvideobench_test_v`, `longvideobench_test_i` (+ `no_visual`, `random_choice` variants). `_v` = video input, `_i` = interleaved subtitles.
- **Dataset:** `longvideobench/LongVideoBench`, cache `datasets/longvideobench`, split `validation` / `test`.
- **Generation:** `max_new_tokens: 32`, `temperature: 0`, `do_sample: False`.
- **Judge:** none.
- **Prompt:**

```
<question>
A. <option0>
B. <option1>
... (up to 5; entries equal to "N/A" are skipped)
Answer with the option's letter from the given choices directly.
```

- **Gotchas:** the `_i` variants read `dataset_kwargs.max_num_frames` (default **16**) to compute subtitle timestamps and interleave subtitle text into the prompt prefix. If your model samples a different frame count, the interleaved timestamps won't line up with the frames you actually feed it. Test-split answers are held out.

## VideoMathQA

- **Tasks:** `videomathqa_mcq`, `videomathqa_mbin`, each × `_cot` × `_w_subtitle` → 8 total. `mcq` = 5-way (A–E), `mbin` = binary (A/B).
- **Dataset:** `MBZUAI/VideoMathQA`, `token: False`, cache `videomathqa`, split `test`.
- **Generation:** `max_new_tokens: 16` for direct tasks, **`8096` for the `_cot` tasks**, `temperature: 0`, `top_p: 1.0`.
- **Judge:** none.
- **Prompt** (direct, mcq):

```
Select the best answer to the following multiple-choice question based on the video. Respond with the letter (A, B, C, D or E) of the correct option.
<question>
<options>

Answer with the option's letter (A, B, C, D or E) from the given choices directly.
```

CoT post_prompt instead: `First please perform reasoning, and think step by step to provide best answer to the following question with the option's letter (A, B, C, D or E) from the given choices.`

- **Gotchas:** the option-count preamble is chosen at runtime from `len(doc["options"])` (2 → "A or B"). The `_w_subtitle` variants carry `frame_num: 32` / `768` — subtitle alignment only, see the note at the top.

## Video-MMMU

- **Tasks:** `video_mmmu_perception`, `video_mmmu_comprehension`, `video_mmmu_adaptation`, plus `adaptation_question_only` and `no_visual` / `number_option` / `gt_none_option` / `random_choice` variants.
- **Dataset:** `lmms-lab/VideoMMMU`, cache `video_mmmu`, split `test`.
- **Generation:** `max_new_tokens: 1024` (no temperature specified → wrapper default `temperature=0`, `top_p=0.95`).
- **Judge:** none — `evaluate_videommmu` is rule-based despite the `judge_dict` variable name.
- **Prompts:** perception/comprehension and adaptation differ.
  - perception/comprehension: `<question>\n<options>` + `\nPlease ignore the Quiz question in last frame of the video.`
  - adaptation: `You should watch and learn the video content. Then apply what you learned to answer the following multi-choice question. The image for this question is at the end of the video.\n<question>\n<options>` (open-ended swaps in `answer the following open-ended question. ...`).
- **Gotchas:** the adaptation track deliberately puts the question image in the **last frame of the video** — if your frame sampling drops the final frame, adaptation scores collapse. `nframes` sampling via `qwen_vl_utils` uses `linspace(0, total-1, n)`, which does include the last frame; the simple-mode `encode_video` explicitly appends it. Verify before trusting adaptation numbers with low frame counts.

## MMVU-Val

- **Task:** `mmvu_val`. (There's also a stray `mmvu/mmvu_val_cot copy.yaml` — a filename with a literal space, clearly unintentional.)
- **Dataset:** `lmms-lab/MMVU`, cache `mmvu`, split `validation`.
- **Generation:** `max_new_tokens: 1024`, **`temperature: 0.7`** (unusual — every other benchmark here is 0), `top_p: 1.0`, `do_sample: false`. Note `do_sample` is dropped on the vllm path, so temperature 0.7 **does** take effect and makes runs non-deterministic.
- **Judge: yes, hybrid.** Rule-based first; if that fails **and** the question is open-ended, falls back to the LLM judge (`lmms_eval.llm_judge`, `MODEL_VERSION`, default `gpt-4o-2024-11-20`). Multiple-choice questions are **never** judged.
- **Prompt** (multiple-choice, 5-way; built from module constants, not YAML):

```
Question:<question>
A: <a>
B: <b>
C: <c>
D: <d>
E: <e>
Visual Information: processed video
Do not generate any intermediate reasoning process. Answer directly with the option letter from the
given choices.
```

Open-ended: `Question:<question>\nVisual Information: processed video\nDo not generate any intermediate reasoning process. Directly output the final answer.`

- **Gotchas:** `lmms_eval_specific_kwargs` pre/post are **ignored** — the prompt is fully hardcoded in `utils.py`. CoT prompt variants exist in the module (`multiple_choice_prompt_cot`, `open_ended_prompt_cot`) reachable via `mmvu_doc_to_text_cot`.

## VSI-Bench

- **Tasks:** `vsibench` (`dataset_name: full`), `vsibench_debiased`, `vsibench_pruned`, plus `multi_image_input/` variants.
- **Dataset:** `nyu-visionx/VSI-Bench`, cache `vsibench`, split `test`.
- **Generation:** `max_new_tokens: 16`, `temperature: 0`, `top_p: 1.0`.
- **Judge:** none. Numeric answers scored with **MRA** (Mean Relative Accuracy), not exact match.
- **Prompt:** depends on question type.
  - Numeric (NA types): `These are frames of a video.\n<question>\nPlease answer the question using a single word or phrase.`
  - Multiple-choice (MCA types): `These are frames of a video.\n<question>\nOptions:\n<options>\nAnswer with the option's letter from the given choices directly.`
- **Metrics:** `vsibench_overall` plus per-type: `object_counting_mra`, `object_abs_distance_mra`, `object_rel_distance_accuracy`, `object_rel_direction_accuracy`, `object_size_estimation_mra`, `room_size_estimation_mra`, `route_planning_accuracy`, `obj_appearance_order_accuracy`.
- **Gotchas:** `pre_prompt` defaults to `"These are frames of a video."` via an `or` fallback, so setting it to `""` does **not** remove it — you must set a non-empty string. `LMMS_EVAL_SHUFFLE_DOCS` shuffles the dataset. The `gpt4v` / `gemini_api` profiles harden the numeric instruction to `Do not response anything other than a single number!`.

## MINERVA

- **Task:** `minerva`, tagged `video_qa`.
- **Dataset:** `dataset_path: json`, loaded directly from a URL — `https://huggingface.co/datasets/lmms-lab-eval/minerva/resolve/main/minerva.json`. **Needs network access at load time**; there is no local cache path configured.
- **Generation:** `max_new_tokens: 32`, `temperature: 0`, `do_sample: false`.
- **Judge:** none.
- **Prompt:** `<question>\nA. <choice>\nB. <choice>\n...` + `\nAnswer with the option's letter from the given choices directly.`

## SciVideoBench

- **Task:** `scivideobench`, tagged `video_qa`.
- **Dataset:** `groundmore/scivideobench`, cache `scivideobench`, split `test`.
- **Generation:** `max_new_tokens: 16`, `temperature: 0`, `top_p: 1.0`.
- **Judge:** none.
- **Prompt:** `<question>` + `\nAnswer with the option's letter from the given choices directly.`
- **Gotchas:** options run **A–J** (10-way), so the `gpt4v` profile spells out `Answer the question with A, B, C, D, E, F, G, H, I or J.` A `post_prompt_cot` is defined in the YAML but the default `doc_to_text` doesn't use it — you'd need to swap `post_prompt` to enable CoT.

## ActivityNetQA

- **Task:** `activitynetqa`.
- **Dataset:** `lmms-lab/ActivityNetQA`, cache `activitynetqa`, split `test`.
- **Generation:** `max_new_tokens: 64`, `until: ["ASSISTANT:"]`, `image_aspect_ratio: original`, `temperature: 0`.
- **Judge: yes — on every sample.** `MODEL_VERSION` (default `gpt-4o-2024-11-20`) rates each QA pair for correctness (yes/no) and a 0–5 score. **This benchmark cannot be run without a working judge API.**
- **Prompt:** `pre_prompt + <question> + " Answer the question using a single word or phrase."`
- **Metrics:** `gpt_eval_accuracy` (fraction judged correct), `gpt_eval_score` (mean 0–5).
- **Gotchas:** answers are single-word ground truth; the judge is what makes short free-form answers scorable. `NUM_SECONDS_TO_SLEEP=5` between retries.

## EgoSchema

- **Tasks:** `egoschema` (full, `dataset_name: GENERATION`), `egoschema_subset`, plus `_mcppl` loglikelihood variants.
- **Dataset:** `lmms-lab/egoschema`, cache `egoschema`, split `test`.
- **Generation:** none specified → greedy default. On vllm that means **4096 max tokens** unless you override.
- **Judge:** none.
- **Prompt:** `<question>` + each option on its own line + `\nAnswer with the option's letter from the given choices directly.` (The post_prompt is force-set inside `doc_to_text` when the doc has an `option` field, overriding the YAML.)
- **Gotchas:** **`egoschema` (full) only produces a `submission` file — there is no accuracy metric**, because the full-set labels are held out for the leaderboard. Use **`egoschema_subset`** if you want a number; it reports both `submission` and `score`. An `aria` prompt profile adds `Please answer the question about the video:\n`.

## PerceptionTest

- **Tasks:** `perceptiontest_val_mc`, `perceptiontest_test_mc`, plus `_mcppl` loglikelihood variants.
- **Dataset:** `lmms-lab/PerceptionTest_Val` (val) / `lmms-lab/PerceptionTest_Test` (test), caches `perceptiontest_val` / `perceptiontest_test`.
- **Generation:** none specified → greedy default (4096 on vllm).
- **Judge:** none.
- **Prompt:** `pre_prompt + <question>\nA. <opt>\nB. <opt>\nC. <opt> + post_prompt` — and **both pre and post default to empty**, so the model gets no answer-format instruction at all. Expect verbose answers and matching failures unless you add a `vllm:` profile with a post_prompt.
- **Gotchas:** `perceptiontest_test_mc` emits **`submission` only** (held-out labels). Use `perceptiontest_val_mc` for an `accuracy` number.

## SeedBench

- **Tasks:** `seedbench`, `seedbench_lite` (`lmms-lab/LMMs-Eval-Lite`), `seedbench_ppl` (loglikelihood), `seedbench_ko`, `seedbench/reasoning`.
- **Dataset:** `lmms-lab/SEED-Bench`, split `test`. **No `video: True`** — see below.
- **Generation:** `until: ["ASSISTANT:"]`, `image_aspect_ratio: original`. No `max_new_tokens` → 4096 on vllm.
- **Judge:** none. Prediction is truncated to its first character if longer than 1.
- **Prompt:**

```
<question>
A. <choice_a>
B. <choice_b>
C. <choice_c>
D. <choice_d>
Answer with the option's letter from the given choices directly.
```

- **Gotchas:** `seed_doc_to_visual` returns `[image.convert("RGB") for image in doc["image"]]` — **the dataset ships pre-extracted frames as images**. The video dimensions (`seed_video`) are evaluated from those fixed frames, so your `nframes` setting has **no effect** on this benchmark. Metrics split into `seed_image`, `seed_video`, `seed_all`.

## VideoChatGPT

- **Tasks:** `videochatgpt_gen` (`Generic`), `videochatgpt_temporal`, `videochatgpt_consistency`.
- **Dataset:** `lmms-lab/VideoChatGPT`, cache `videochatgpt`, split `test`.
- **Generation:** `max_new_tokens: 1024`, `temperature: 0`, `top_p: 1.0`.
- **Judge: yes — on every sample.** `MODEL_VERSION` (default `gpt-4o-2024-11-20`). **Unrunnable without a judge API.**
- **Prompt:** `pre_prompt + <question> + post_prompt`, both **empty by default** — the raw question is the whole prompt. That's intentional: this is a free-form captioning/QA benchmark.
- **Metrics:** `gpt_eval_score_correctness`, `gpt_eval_score_detailed_orientation`, `gpt_eval_score_context` (generic); temporal and consistency tasks have their own score keys. `videochatgpt_consistency` asks two questions per video and scores answer consistency.

## NExT-QA

- **Tasks:** group `nextqa` → `nextqa_mc_test`, `nextqa_oe_test`, `nextqa_oe_val`.
- **Dataset:** `lmms-lab/NExTQA`, cache `nextqa`, `load_package: True`.
- **Generation:** none specified → greedy default (4096 on vllm).
- **Judge:** none. MC scored with `exact_match`.
- **Prompt (MC):** `<question>\nA. <a0>\nB. <a1>\nC. <a2>\nD. <a3>\nE. <a4>` — **no pre/post prompt is configured anywhere**, so there's no "answer with the letter" instruction. With `exact_match` scoring this punishes any model that answers in a sentence. Add a `vllm:` block under `lmms_eval_specific_kwargs` if you want comparable numbers.

## LVBench

- **Task:** `lvbench` (+ `no_visual`, `random_choice` variants).
- **Dataset:** `lmms-lab/LVBench`, cache `lvbench`, split **`train`**.
- **Generation:** `max_new_tokens: 16` only (no temperature → wrapper defaults `temperature=0`, `top_p=0.95`).
- **Judge:** none. Metric `lvbench_score`, aggregation `mean`.
- **Prompt:** `pre_prompt + <question> + "\nAnswer the question with the option letter"` (no trailing period — upstream).
- **Gotchas:** LVBench videos are hour-scale. At low `nframes` this is effectively a blind-guessing benchmark.

## VideoEval-Pro

- **Task:** `videoevalpro`.
- **Dataset:** `TIGER-Lab/VideoEval-Pro`, cache `videoevalpro`, split `test`.
- **Generation:** `max_new_tokens: 16`, `temperature: 0`, `top_p: 1.0`.
- **Judge: yes — on every sample.** Hardcoded class `GPT4oJudge(model_name="gpt-4o-2024-11-20")`, instantiated with `openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])` — a **hard `KeyError` if `OPENAI_API_KEY` is unset**, and it does not honor `API_TYPE` / `OPENAI_API_URL`. Retries up to 10× with 2s delay.
- **Prompt:** `pre_prompt + <question> + " Keep the answer short and concise."`
- **Gotchas:** free-form short answers judged for semantic equivalence; `max_new_tokens: 16` is tight for that.

## JumpScore

- **Task:** `JumpScore` (capitalized — that's the literal task name).
- **Dataset:** `lmms-lab-encoder/JumpScore`, cache `jumpscore`, `create_link: true`, split `test`.
- **Generation:** `max_new_tokens: 1024`, `temperature: 0`, `top_p: 1.0`.
- **Judge:** none.
- **Prompt:** `pre_prompt + <question> + post_prompt`, both empty by default — the dataset's question already contains the timestamp instruction.
- **Metrics:** `jumpscore_map`, `jumpscore_score`.
- **Gotchas:** defines `doc_to_messages`, so it works natively on the chat path. Has its own `video_cache_dir: jumpscore` in `lmms_eval_specific_kwargs`.

## Charades-STA (mIoU)

- **Task:** **`temporal_grounding_charades`** — note the task name does not contain "charades_sta"; the *directory* is `charades_sta`.
- **Dataset:** `lmms-lab/charades_sta`, cache `charades_sta`, split `test`.
- **Generation:** `max_new_tokens: 50`, `temperature: 0`, `top_p: 1.0`.
- **Judge:** none — IoU computed against parsed start/end times.
- **Prompt:** `pre_prompt + <caption> + post_prompt`, i.e.:

```
Please find the visual event described by a sentence in the video, determining its starting and ending times. The format should be: 'The event happens in the start time - end time'. For example, The event 'person turn a light on' happens in the 24.3 - 30.4 seonds. Now I will give you the textual sentence: <caption>Please return its start time and end time.
```

- **Metrics:** `charades_sta_IOU@3`, `IOU@5`, `IOU@7`, `charades_sta_mIOU`.
- **Gotchas:** the prompt has an upstream typo (`seonds`) and no separator between the caption and the post_prompt (`...sentence: <caption>Please return...`). The task is `doc["caption"]`-driven, not `doc["question"]`. Timestamp parsing is regex-based, so prompt drift breaks scoring silently.

## Video-Holmes

- **Tasks:** `video_holmes`, `video_holmes_test`, `video_holmes_reasoning`.
- **Dataset:** `TencentARC/Video-Holmes`, cache `video_holmes`, `data_files: {test: test_Video-Holmes.json, train: train_Video-Holmes.json}`, split `test`.
- **Generation:** `max_new_tokens: 64`, `temperature: 0`, `top_p: 1.0`.
- **Judge:** none.
- **Prompt:** `pre_prompt + <question> + "\nAnswer with the option's letter from the given choices directly."` A `qwen3_vl` profile exists (`format: qwen3_vl`, `pre_prompt: "Question: "`, `post_prompt: "Answer with the option letter only."`).
- **Metrics:** `video_holmes_accuracy` plus seven reasoning dimensions — `SR` (social reasoning), `IMC` (intention & motive chaining), `TCI` (temporal causal inference), `TA` (timeline analysis), `MHR` (multimodal hint reasoning), `PAR` (physical anomaly reasoning), `CTI` (core theme inference).
- **Gotchas:** `doc_to_target` is `"Answer"` (capital A). Defines `doc_to_messages`, so it's chat-path native.

## WorldSense

- **Tasks:** `worldsense`, `worldsense_w_subtitle`.
- **Dataset:** `lmms-lab/worldsense`, cache `WorldSense` (capitalized), split `test`.
- **Generation:** `max_new_tokens: 1024`, `temperature: 0`, `top_p: 1.0`.
- **Judge:** none — `parse_multi_choice_response` does the extraction.
- **Prompt** (`worldsense`, the audio-framing variant):

```
Carefully watch this video and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question.
These are the frames of a video and the corresponding audio. Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
<question>
<candidate A>
<candidate B>
...
```

The `_w_subtitle` variant swaps in `FRAMES_TMPL_SUB`, which embeds the SRT text and uses `frame_num: 32` to pick which subtitle lines to include.

- **Gotchas:** the default prompt claims the model receives **"the corresponding audio"** — the vllm path sends frames only, so this is a lie to the model. WorldSense is an audio-visual benchmark; scores from a video-only model are not comparable to the paper. There's an unused `FRAMES_TMPL_NOSUB` if you want honest framing.

## VSI-Super

- **Tasks:** `vsisuper_count_{10,30,60,120}mins`, `vsisuper_recall_{10,30,60,120,240}mins`. Tags: `vsisuper`, `vsisuper_count`, `vsisuper_recall`. There are also `count_streaming/` variants (`vsc_streaming_*`).
- **Datasets:** `nyu-visionx/VSI-SUPER-Count` and `nyu-visionx/VSI-SUPER-Recall`; caches `vsisuper_count` / `vsisuper_recall`; split `test`. Length buckets come from `process_docs` filters on `x["split"]` (count) / `x["type"]` (recall).
- **Generation:** `max_new_tokens: 16`, `temperature: 0`, `top_p: 1.0`.
- **Judge:** none.
- **Prompts** (hardcoded in the utils, `lmms_eval_specific_kwargs` are ignored):
  - Count: `These are frames of a video.\n<question>\nPlease answer the question using a single word or phrase.`
  - Recall: `<question>\nOptions:\n<options>\nAnswer with the option's letter from the given choices directly.`
- **Metrics:** count → `mra` (Mean Relative Accuracy); recall → `accuracy`.
- **Gotchas:** the 120/240-minute buckets are far beyond any practical frame budget — decode time and prompt length are the real constraints, not accuracy.

## TVBench

- **Tasks:** group `tvbench` → `tvbench_action_antonym`, `_action_count`, `_action_localization`, `_action_sequence`, `_egocentric_sequence`, `_moving_direction`, `_object_count`, `_object_shuffle`, `_scene_transition`, `_unexpected_action`.
- **Dataset:** `FunAILab/TVBench`, cache `tvbench`, split **`train`**.
- **Generation:** `max_new_tokens: 32`, `temperature: 0`, `do_sample: false`.
- **Judge:** none. Metric `tvbench_acc`, aggregation `mean`.
- **Prompt:**

```
<question>
A. <candidate>
B. <candidate>
...
Answer with the option letter only.
```

- **Gotchas:** `_safe_get` looks for the question under `question` / `prompt` / `query`, so the schema is tolerant. TVBench is explicitly designed so that temporally-blind models score near chance — it's a good control against frame-count-insensitive results.

---

## Cross-cutting notes

**Benchmarks that need a judge API to produce any number at all:** ActivityNetQA, VideoChatGPT, VideoEval-Pro. Partial/fallback judges: TempCompass (captioning always; others on regex miss), MMVU-Val (open-ended on rule-based miss).

**Benchmarks that emit a submission file instead of a score:** `egoschema` (full), `perceptiontest_test_mc`. Use `egoschema_subset` / `perceptiontest_val_mc` for local numbers.

**Benchmarks with no answer-format instruction in the default prompt** (expect parsing losses on chatty models): NExT-QA MC, PerceptionTest, VideoChatGPT (by design), JumpScore (by design).

**Benchmarks whose prompts ignore `lmms_eval_specific_kwargs`** (you must edit `utils.py` to change them): MMVU-Val, VSI-Super, SeedBench.

**Non-`test` splits:** MVBench (`train`), LVBench (`train`), TVBench (`train`), LongVideoBench val (`validation`), MMVU (`validation`), PerceptionTest val (`validation`).

**Model-specific prompt profiles** are keyed by the `--model` string. Since you run `--model vllm`, only a block literally named `vllm:` would apply — otherwise you get `default:`. Existing profiles you could copy from: `qwen3_vl`, `llava_vid`, `gpt4v`, `gemini_api`, `plm`, `aria`, `xcomposer2_4khd`.

**Verification method:** everything above was read from the YAMLs and `utils.py` in this checkout (`git log -1` → `3b614543`), not from the benchmarks' papers. Where upstream defaults differ from the paper protocol (VideoMME's subtitle preamble, WorldSense's audio claim, MLVU's prefill suffix), the code is what runs.
