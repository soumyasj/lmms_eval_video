# Datasets

Every video task in this checkout: where the data comes from, which version, which split, what the model is asked,
how it is scored. Everything below was read from the task YAMLs and `utils.py` files in `lmms_eval/tasks/`, so it
describes what runs, not what a benchmark's paper says.

## How a video task loads its data

- **Questions** come from the Hugging Face dataset named by `dataset_path` / `dataset_name` / `test_split` in the
  task YAML, through `datasets.load_dataset`. The **revision** column below is the dataset commit the paper's numbers
  were produced with (`HF_HOME/hub/datasets--<org>--<name>/snapshots/<revision>`); pin it with
  `dataset_kwargs: {revision: <hash>}` if a dataset changes upstream.
- **Videos** are downloaded once by lmms_eval's `video: True` mechanism: the whole dataset repo is snapshotted, its
  archives extracted (or the snapshot linked) at `$HF_HOME/<cache_dir>`, and the task's `doc_to_visual` returns
  `[path to the .mp4]`. Set `HF_HOME` to a shared, persistent location; `LMMS_EVAL_DATASETS_CACHE` (the arrow cache)
  is put inside it by the runners.
- **The frame count is not a task setting.** Every task hands the model a video path; the number of frames, their
  positions, the decoder and the resolution are decided by the model wrapper from `--model_args`, i.e. by the
  config of `eval_scripts/run_eval.py`. The `frame_num` entries in a few `*_w_subtitle.yaml` files only align
  subtitle text with timestamps inside the prompt.
- **`max_new_tokens` is a floor, not a cap, on the vLLM path**: the chat vLLM wrapper takes `max(task, model)` and
  defaults to 4096. The runners pass the task's value through `--max_new_tokens` where it matters.
- **Judges.** Tasks marked "judge" call an OpenAI-compatible model for answers the regex cannot parse (or for every
  answer). Shared variables: `API_TYPE` (`openai` | `azure_openai` | `local`), `OPENAI_API_KEY`, `OPENAI_API_URL`,
  `MODEL_VERSION` (default `gpt-4o-2024-11-20`), `LLM_JUDGE_URL` for a self-hosted judge.
  `TEMPCOMPASS_DISABLE_JUDGE=1` scores unparsed TempCompass answers as wrong instead of calling a judge.
- **Pre-extracted frames.** `python eval_scripts/make_frames_task.py --task <T>` creates `<T>_frames<N>` (N = 4 ... 128):
  the same rows and prompt, `doc_to_visual` returns N JPEGs sampled uniformly ahead of time. `input: frames` in a
  config selects them.

## Meta-Video

| | |
|---|---|
| task | `metav` (group) = `metav_videomme`, `metav_longvideobench_val_v`, `metav_lvbench`, `metav_video_mmmu_perception`, `metav_video_mmmu_comprehension`, `metav_video_mmmu_adaptation`, `metav_vsibench` |
| dataset | [`soumyasj/Meta-Video`](https://huggingface.co/datasets/soumyasj/Meta-Video), config `meta_v`, split `test`, 1,000 rows; videos under `videos/<source>/` (791 files, 70 GB); cache dir `meta_video` |
| columns | `id`, `source`, `source_task`, `source_id`, `video`, `duration_s`, `duration_bucket`, `question`, `options`, `answer`, `answer_type`, `categories` (JSON), `blind_correct_models` |
| composition | Video-MME 312, LongVideoBench 240, VideoMMMU 60 / 60 / 60, LVBench 146, VSI-Bench 122 |
| questions | 937 multiple choice (564 x 4 options, 185 x 5, 157 x 10, 31 other), 63 numerical (VSI-Bench); random guessing 20.3, majority letter 23.3 |
| videos | 791, up to 60 min: 335 short (< 5 min), 281 medium (5 to 20), 175 long (20 to 60) |
| prompt | the source task's prompt, materialized per item and variant in `lmms_eval/tasks/metav/data/metav_prompts.json`; variants `default`, `qwen3_vl` |
| generation | the source task's `generation_kwargs` (16 tokens for Video-MME / LVBench / VSI-Bench, 32 LongVideoBench, 1024 VideoMMMU; temperature 0) |
| metric | the source task's metric per subtask; Meta-Video score = macro average over the five sources (`eval_scripts/score_metav.py`) |
| judge | none |

Construction: black-frame runs of Qwen2.5-VL-7B-Instruct, Qwen3-VL-8B-Instruct and InternVL3.5-8B; every question
at least two of them answered correctly without the video was removed (41 % of Video-MME, 42 % of LongVideoBench,
17 % of LVBench, 25 to 31 % of VideoMMMU, 27 % of VSI-Bench); stratified selection over the source categories with a
repair pass that keeps the model ordering of the full benchmarks at every frame budget; at most 3 questions per
video; no video over 60 minutes. The items with their selection metadata ship in `lmms_eval/tasks/metav/data/metav.jsonl`.

### Non-blind splits of the source benchmarks

[`soumyasj/video-benchmarks-nonblind`](https://huggingface.co/datasets/soumyasj/video-benchmarks-nonblind): the same
filter applied to whole benchmarks. One config per lmms_eval task (29: the seven above except VSI-Bench, plus
`mmvu_val`, `scivideobench`, `tempcompass_multi_choice` and the 20 `mvbench_*` subtasks), split `nonblind`, original
columns plus the three models' black-frame scores. Point a task at it with
`dataset_path: soumyasj/video-benchmarks-nonblind`, `dataset_name: <task>`, `test_split: nonblind`.

| config | original | nonblind | | config | original | nonblind |
|---|---|---|---|---|---|---|
| videomme | 2,700 | 1,601 | | mmvu_val | 1,000 | 644 |
| longvideobench_val_v | 1,337 | 773 | | scivideobench | 1,000 | 824 |
| lvbench | 1,549 | 1,281 | | tempcompass_multi_choice | 1,580 | 946 |
| video_mmmu_perception / comprehension / adaptation | 300 each | 206 / 221 / 226 | | mvbench (20 subtasks) | 4,000 | 2,742 |

## The benchmarks used in the paper

| benchmark | task(s) | dataset (revision) | split | rows | videos | tokens | metric | judge |
|---|---|---|---|---|---|---|---|---|
| Video-MME | `videomme` (+ `_w_subtitle`, `_long`) | `lmms-lab/Video-MME` (`ead1408f`) | test | 2,700 (900 videos x 3) | `$HF_HOME/videomme/data/<videoID>.mp4` | 16 | `videomme_perception_score` (accuracy; short / medium / long) | no |
| LongVideoBench | `longvideobench_val_v` (`_val_i` interleaves subtitles) | `longvideobench/LongVideoBench` (`60d1c89c`, gated) | validation | 1,337 | `$HF_HOME/datasets/longvideobench/videos/` | 32 | `lvb_acc` | no |
| LVBench | `lvbench` | `lmms-lab/LVBench` (`a2ee023e`) | **train** | 1,549 (103 videos, hour scale) | `$HF_HOME/lvbench/` | 16 | `lvbench_score` | no |
| VideoMMMU | `video_mmmu_perception`, `_comprehension`, `_adaptation` | `lmms-lab/VideoMMMU` (`d1c35ac9`, gated), configs `Perception` / `Comprehension` / `Adaptation` | test | 300 each | `$HF_HOME/video_mmmu/<discipline>/` | 1024 | `mmmu_acc` | no (rule based) |
| VSI-Bench | `vsibench` (config `full`), `vsibench_debiased`, `vsibench_pruned` | `nyu-visionx/VSI-Bench` (`d7cb1a39`) | test | 5,130 / 2,362 / 2,768 | `$HF_HOME/vsibench/{scannet,scannetpp,arkitscenes}/` | 16 | `vsibench_overall` (accuracy for MC types, mean relative accuracy for numeric types, mean over 8 categories) | no |
| MVBench | `mvbench` (group of 20) | `OpenGVLab/MVBench`, revision `video` (`a776e554`) | **train** | 200 x 20 = 4,000 | per-subtask folders in the snapshot (`$HF_HOME/mvbench_video`) | 16 | `mvbench_accuracy` | no |
| TempCompass | `tempcompass_multi_choice` (group `tempcompass` also has `yes_no`, `caption_matching`, `captioning`) | `lmms-lab/TempCompass` (`c8a67d88`) | test | 1,580 (7,540 all four) | `$HF_HOME/tempcompass/videos/` | task default (greedy) | `avg_accuracy` + action / direction / speed / order / attribute_change | fallback `gpt-3.5-turbo-1106` on regex miss; captioning always |
| MMVU | `mmvu_val` | `lmms-lab/MMVU` (`7537bc8a`) | validation | 1,000 (MC + open ended) | `$HF_HOME/mmvu/videos/<discipline>/` | 1024 | `accuracy` | open-ended on rule miss (`MODEL_VERSION`) |
| MLVU | `mlvu_dev`, `mlvu_test` | `sy1998/MLVU_dev` (`96207eb9`), `sy1998/MLVU_Test` (`4a9826be`) | test | 2,174 / 502 | `$HF_HOME/mlvu/` | 16 | `mlvu_percetion_score` (sic) | no |
| SciVideoBench | `scivideobench` | `groundmore/scivideobench` (`2c868c0f`) | test | 1,000 (10-way MC) | `$HF_HOME/scivideobench/` | 16 | `scivideobench_acc` | no |
| VideoMathQA | `videomathqa_{mcq,mbin}[_cot][_w_subtitle]` | `MBZUAI/VideoMathQA` (`702cdd48`) | test | 420 mcq + 1,680 mbin | `$HF_HOME/videomathqa/` | 16 (8096 cot) | `videomathqa_perception_score` | no |

"tokens" is the task's `max_new_tokens`; temperature is 0 everywhere except MMVU (0.7 in the YAML; the runners pass
`temperature: 0` for the protocol). Datasets marked gated need `HF_TOKEN`.

### Prompts the model receives

The text below `<question>` / `<options>` is exactly what `doc_to_text` builds for the `default` prompt block.
Model-specific blocks (`lmms_eval_specific_kwargs`) are keyed by the **registered model name**: `--model qwen3_vl`
gets a task's `qwen3_vl:` block (Video-MME has one), `--model vllm` always gets `default:`. Meta-Video materializes
its prompts to avoid exactly this.

| task | prompt |
|---|---|
| Video-MME | `Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.` `<question>` `<A. ... D.>` `Answer with the option's letter from the given choices directly.` (the preamble mentions subtitles even without them, upstream behaviour) |
| LongVideoBench | `<question>` `A. ... E.` (options equal to `N/A` skipped) `Answer with the option's letter from the given choices directly.` |
| LVBench | `<question with (A) ... (D) lines>` `Answer the question with the option letter` |
| VideoMMMU | perception / comprehension: `<question>` `<options>` `Please ignore the Quiz question in last frame of the video.`; adaptation: `You should watch and learn the video content. Then apply what you learned to answer the following multi-choice question. The image for this question is at the end of the video.` `<question>` `<options>`. The adaptation question image is the **last frame**: a sampler that drops it collapses the score (uniform `linspace` keeps it). |
| VSI-Bench | numeric: `These are frames of a video.` `<question>` `Please answer the question using a single word or phrase.`; MC: `These are frames of a video.` `<question>` `Options:` `<options>` `Answer with the option's letter from the given choices directly.` |
| MVBench | `Question:<question>` `Option:` `(A) ...` `Only give the best option.` |
| TempCompass (MC) | `<question>` `Please directly give the best option:` |
| MMVU | `Question:<question>` `A: ... E:` `Visual Information: processed video` `Do not generate any intermediate reasoning process. Answer directly with the option letter from the given choices.` (hard-coded in `utils.py`; open-ended: `... Directly output the final answer.`) |
| MLVU | `<question>` `Only give the best option.` `Best option: (` (the trailing `(` is upstream's prefill trick) |
| SciVideoBench | `<question with A ... J>` `Answer with the option's letter from the given choices directly.` |
| VideoMathQA (mcq) | `Select the best answer to the following multiple-choice question based on the video. Respond with the letter (A, B, C, D or E) of the correct option.` `<question>` `<options>` `Answer with the option's letter (A, B, C, D or E) from the given choices directly.` |

### Things to know per dataset

- **Video-MME**: `cluster_key: videoID` groups the three questions of a video for clustered standard errors. The
  `_w_subtitle` variants read `$HF_HOME/videomme/subtitle/<videoID>.srt`.
- **LongVideoBench**: the `_i` variants compute subtitle timestamps from `dataset_kwargs.max_num_frames` (16); if the
  model samples a different count the interleaved timestamps do not match the frames. Test-split answers are held out.
- **LVBench**: videos are hour scale; at a few frames the task is close to blind guessing. The question text already
  contains the option lines.
- **VSI-Bench**: `pre_prompt` falls back to `These are frames of a video.` when set to an empty string. Numeric answers
  are parsed from the first whitespace token; `metav/utils.py` adds a fallback that reads the number out of a sentence.
- **MVBench**: each subtask has its own video folder (`clevrer` and `star` under `data0613/`); `mvbench_episodic_reasoning`
  ships a directory of frames instead of a video.
- **TempCompass**: `HF_HOME` must be set (the task reads it without a default). The judge model is hard-coded in
  `tempcompass/utils.py`.
- **MMVU**: `lmms_eval_specific_kwargs` are ignored, the prompt is hard-coded; multiple-choice questions are never judged.
- **MLVU**: metric name has an upstream typo, `mlvu_percetion_score`.

### Task-level changes

- `metav/`: new task; response cleaning (`<|begin_of_box|>` markers, `<think>` blocks) before the source parsers;
  a numeric fallback for VSI-Bench items; an aggregate that tolerates missing question types on partial runs.
- `precomputed_frames.py` + generated `<task>_frames<N>.yaml`: frames-as-images variants of any video task.
- `tempcompass/utils.py`: `TEMPCOMPASS_DISABLE_JUDGE=1`.
- `api/task.py`: an existing `create_link` symlink counts as a cached dataset (no Hub round trip per run).

## Other video tasks in the tree

Present and runnable with the same configs, not used in the paper. Rows, tokens and judges as in the YAMLs.

| benchmark | task(s) | dataset | split | rows | tokens | metric | judge |
|---|---|---|---|---|---|---|---|
| MINERVA | `minerva` | `lmms-lab-eval/minerva` (JSON URL, network at load time) | - | 1,341 | 32 | `minerva_acc` | no |
| EgoSchema | `egoschema` (submission only), `egoschema_subset` | `lmms-lab/egoschema` | test | 5,031 / 500 | greedy | `score` (subset) | no |
| PerceptionTest | `perceptiontest_val_mc`, `perceptiontest_test_mc` (submission) | `lmms-lab/PerceptionTest_Val` / `_Test` (gated) | validation / test | 19,140 | greedy | `accuracy` | no |
| NExT-QA | `nextqa_mc_test`, `nextqa_oe_test`, `nextqa_oe_val` | `lmms-lab/NExTQA` | test / val | 8,564 / 9,178 / 5,343 | greedy | `exact_match` (no answer-format instruction in the prompt) | no |
| ActivityNetQA | `activitynetqa` | `lmms-lab/ActivityNetQA` | test | 8,000 | 64 | `gpt_eval_accuracy`, `gpt_eval_score` | every sample |
| VideoChatGPT | `videochatgpt_gen`, `_temporal`, `_consistency` | `lmms-lab/VideoChatGPT` | test | 1,996 / 499 / 998 | 1024 | `gpt_eval_score_*` | every sample |
| VideoEval-Pro | `videoevalpro` | `TIGER-Lab/VideoEval-Pro` | test | 1,289 | 16 | `videoevalpro_score` | every sample (`OPENAI_API_KEY` required) |
| SeedBench | `seedbench`, `seedbench_lite` | `lmms-lab/SEED-Bench` | test | 17,990 | greedy | `seed_video`, `seed_image`, `seed_all` | no; frames ship in the dataset, `nframes` has no effect |
| JumpScore | `JumpScore` | `lmms-lab-encoder/JumpScore` | test | 189 | 1024 | `jumpscore_map`, `jumpscore_score` | no |
| Charades-STA | `temporal_grounding_charades` | `lmms-lab/charades_sta` | test | 3,720 | 50 | `charades_sta_mIOU`, `IOU@{3,5,7}` | no |
| Video-Holmes | `video_holmes`, `_test`, `_reasoning` | `TencentARC/Video-Holmes` | test | 1,837 | 64 | `video_holmes_accuracy` + 7 dimensions | no |
| WorldSense | `worldsense`, `_w_subtitle` | `lmms-lab/worldsense` | test | 3,172 | 1024 | `worldsense_score` (audio-visual benchmark; the prompt claims audio the model never gets) | no |
| VSI-Super | `vsisuper_count_*`, `vsisuper_recall_*` | `nyu-visionx/VSI-SUPER-Count` / `-Recall` | test | 400 / 300 | 16 | `mra` / `accuracy` | no |
| TVBench | `tvbench` (group of 10) | `FunAILab/TVBench` | train | 2,525 | 32 | `tvbench_acc` | no |

Benchmarks that cannot produce a number without a judge API: ActivityNetQA, VideoChatGPT, VideoEval-Pro. Benchmarks
that emit a submission file instead of a score: `egoschema` (full), `perceptiontest_test_mc`.
