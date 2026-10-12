# Fixes and deviations from upstream lmms-eval

This repository is a fork of [EvolvingLMMs-Lab/lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
(base: upstream commit `3b614543`, "feat: add CC-OCR benchmark (#1372)"). Everything that differs from
upstream is listed here, grouped by where it lives. The [README](./README.md) only explains how to run
the experiments; the upstream README is kept verbatim as [README_upstream.md](./README_upstream.md).

Every fix below was found because a sweep produced a number that could not be right. Most of them
did not crash: they silently changed what the model saw, or scored a broken run as a low number.

---

## 1. Core (`lmms_eval/`)

### 1.1 Video decoding

**Full-video decode OOM on the torchvision backend** — `lmms_eval/models/model_utils/qwen/video_reader_patch.py` (new)
`qwen_vl_utils`' torchvision backend calls `torchvision.io.read_video`, which decodes the *entire*
clip into one uint8 tensor before subsampling, regardless of how few frames were requested.
Video-MME has 70k–90k-frame clips; that tensor alone is 100–250 GB and kills the node whatever
`--nframes` says. The patch replaces `VIDEO_READER_BACKENDS["torchvision"]` with a seek-based
reader that decodes only the selected frames, using the same frame-index formula as the original
(verified to match decord / torchcodec). It is imported for its side effect by the Qwen2.5-VL
wrappers (`models/chat/qwen2_5_vl.py`, `models/simple/qwen2_5_vl.py`). decord and torchcodec were
already seek-based and are untouched.

**`--nframes` larger than the clip** — `models/chat/qwen2_5_vl.py`, `models/chat/qwen3_vl.py`, `models/chat/vllm.py`
`qwen_vl_utils` treats `nframes` as exact and raises `nframes should in interval [2, total_frames]`
on clips shorter than the request (MVBench has clips with < 64 frames). The Qwen2.5-VL chat wrapper
already probed the clip with decord; the Qwen3-VL and vLLM wrappers now do the same
(`_clamp_nframes`, floored to a multiple of 2). Because the reader that actually decodes
(torchcodec / torchvision) can report a *different* frame count than the decord probe, all three
wrappers additionally step `nframes` down by 2 and retry when the error still occurs. The vLLM
wrapper probes with the backend `FORCE_QWENVL_VIDEO_READER` selects, so the probe and the decode
agree whenever possible.

**1-fps sampling ignored the frame cap** — `models/chat/qwen2_5_vl.py`
In `fps` mode the cap was never forwarded, so `max_num_frames` had no effect and hour-long videos
expanded without bound. Fixed by `video_kwargs["max_frames"] = self.max_num_frames`.

**InternVL short clips** — `models/chat/internvl_hf.py`
transformers' `sample_frames` raises instead of clamping when `num_frames` exceeds a clip's length;
the wrapper now clamps to the shortest video of the batch (`_probe_video_metadata`).

### 1.2 InternVL video crash that scored ~0 % — `models/chat/internvl_hf.py`

transformers' `InternVLVideoProcessor` resizes video frames to a generic 384×384. With ViT
`patch_size=14` that is a 27×27 patch grid, and `pixel_shuffle` needs an *even* grid to fold 2×2
blocks, so `.view()` raised `RuntimeError: shape [4, 27, 13, 2048] is invalid` on **every** video
sample. `generate_until` catches per-sample exceptions and logs them, so the run continued and
reported near-0 % instead of failing. Fixed with a `video_size` constructor argument that defaults
to the model's own native resolution (`config.vision_config.image_size[0]`, 448 for InternVL3/3.5),
applied through `videos_kwargs["size"]`. Original write-up: [`lmms_eval/models/chat/CHANGES.md`](./lmms_eval/models/chat/CHANGES.md).

### 1.3 Frame sampler (which frames, not how many) — `models/simple/qwen2_5_vl.py`, `models/chat/qwen2_5_vl.py`

New model args `frame_sampler=uniform|random`, `frame_sampler_seed=<int>` and
`frame_pool_factor` (chat path). `random` draws `max_num_frames` distinct positions out of the
decoded frames and keeps them in temporal order; the RNG is seeded per video
(`blake2b(seed:video_path)`) so a video gets the same frames at a given seed regardless of batching,
sharding or request order. On the chat path the reader is asked for `frame_pool_factor ×
max_num_frames` frames to sample from, and `frames_indices` in the video metadata is remapped so the
timestamps the processor writes into the prompt stay correct. `uniform` is the unchanged default.
Used by exp3a of `sweep_params.sh` (Qwen2.5-VL only; the Qwen3-VL wrapper asserts on unknown kwargs).

### 1.4 Run bookkeeping

**Self-describing result files** — `lmms_eval/__main__.py`, `lmms_eval/loggers/evaluation_tracker.py`
`--log_samples_suffix <tag>` is also used as a filename prefix, so a run writes
`<tag>_<date>_results.json` / `<tag>_<date>_samples_<task>.jsonl`. The sweep scripts set the tag to
the run configuration; the collector only needs the file to exist.

**Dataset cache symlink thrash** — `lmms_eval/api/task.py`
`os.path.exists` follows symlinks, so a valid `create_link` symlink from a previous run already
counts as cached. The old condition re-resolved the Hub snapshot on every run (a network round trip
per task, rate-limited on the cluster); now only a dangling link triggers it.

**TempCompass judge escape hatch** — `lmms_eval/tasks/tempcompass/utils.py`
`TEMPCOMPASS_DISABLE_JUDGE=1` makes `get_eval_result()` return immediately instead of calling an
OpenAI-compatible judge for responses the regex could not parse. Unparsed responses then score 0,
so accuracy is a slight underestimate; each sample records `match_success`, whose rate bounds the
error. Does not cover TempCompass captioning, which is judge-only.

### 1.5 Tasks

**Meta-VideoBench** — `lmms_eval/tasks/metav/` (new; see the README for usage)
`metav` (1000 items, group of `metav_<source>`): a compute-aware subset of Video-MME, LongVideoBench,
LVBench, VideoMMMU and VSI-Bench, one subtask per source, all re-using the in-tree source tasks'
visual loaders, parsers and metrics. Item ids and materialized
prompts ship in `lmms_eval/tasks/metav/data/`. Task-level fixes that live only there
(`metav/utils.py`):

- `clean_response`: strips `<|begin_of_box|>…<|end_of_box|>` (GLM) and a leading `<think>…</think>`
  block before the source parser sees the text. VSI-Bench reads the first whitespace token as a
  number, so even a boxed bare `210` scored zero.
- VSI-Bench numeric fallback: only when the stock parse fails, the number is pulled out of the
  sentence (preferring one after "answer"/"is"/"=" and, for chain-of-thought replies, the text after
  the last `Answer:`). A parse that already succeeds is never touched.
- `vsibench_aggregate_overall` tolerant of missing question types (partial runs, `--limit`) instead
  of a `KeyError`; identical to the in-tree aggregate on a full run.

**Precomputed-frame task variants** — `lmms_eval/tasks/precomputed_frames.py` (new) + generated `<task>_frames<N>.yaml`
`eval_scripts/make_frames_task.py` writes, for any video task, a variant that feeds N JPEG frames
(sampled ahead of time) instead of the `.mp4`; the YAML `include`s the source task and swaps only the
dataset and `doc_to_visual`, so prompt and metric are identical by construction. The earlier
per-benchmark `*_doc_to_visual_frames` helpers in `videomme/`, `vsibench/`, `mmvu/`, `tempcompass/`,
`videommmu/`, `mvbench/utils.py` are still present but no longer referenced.

`lmms_eval/protocol.py` differs from upstream by a comment only.

---

## 2. Runner-level workarounds (`eval_scripts/runners/`)

These are things the model wrappers get wrong that could be fixed from the outside, so they live in
the runner scripts and plugins rather than in `lmms_eval/`.

- **vLLM ignored `--nframes`.** The chat `vllm` wrapper defaults `nframes=32` and `max_frame_num`
  alone does not change the per-request frame count. Every runner passes both.
- **Batch-invariant vLLM + Qwen-VL vision tower.** `VLLM_BATCH_INVARIANT=1` forces `FLASH_ATTN`
  globally, whose bundled kernel lacks the vision tower's head dim (80); the runners pin the vision
  encoder to SDPA (`mm_encoder_attn_backend=TORCH_SDPA`) and keep the LM on the batch-invariant path.
  The mode needs vllm ≥ 0.11.1 (a second env, `ENV_A_BI`); the runners fail loudly on an older vllm
  instead of silently running a non-invariant job under an invariant label. Qwen3.5 cannot run
  batch-invariant at all (its Gated DeltaNet layers are unsupported); its runner refuses the flag.
- **Qwen3-VL on vLLM at 64+ frames** hit the engine assertion `Expected a cached item for mm_hash`;
  the runner disables the multimodal processor cache (`mm_processor_cache_gb=0`).
- **Thinking models.** Qwen3.5 thinks by default; the hf runner passes `enable_thinking=false`, and
  because the vLLM wrapper cannot forward `chat_template_kwargs`, the runner writes a copy of the
  checkpoint's chat template with `enable_thinking` preset to false
  (`chat_templates/generated/`). GLM-4.6V gets the same treatment via `chat_templates/glm4v_nothink.jinja`.
- **InternVL tiles.** The stock video processor ignores `max_patches` (always one 448 tile per
  frame). On vLLM the runner pins tiles through `mm_processor_kwargs={"max_patches":M,"min_patches":m}`;
  on hf the out-of-tree model `internvl_hf_tiled` (`runners/internvl_tiled/`) reproduces the model
  card's `load_video(max_num=M)`: every sampled frame goes through the *image* tiler, with a
  `Frame{i}: <img>` prompt. That is the only hf path where the tile budget changes a video run.
- **Wrappers that cannot take a frame-budgeted video.** `glm4v` and `gemma3` accept frame lists but
  apply no frame budget to a raw `.mp4`; their runners refuse raw-video tasks so a run can never
  silently use an uncontrolled number of frames. Use the `<task>_frames<N>` variants for them.
- **Caches.** Every script exports `HF_HOME` (default `~/.cache/huggingface`) and points
  `LMMS_EVAL_DATASETS_CACHE` inside it: the arrow build cache used to live in `/tmp`,
  which is node-local and reaped, after which `datasets` believed the build existed and failed to
  mmap it.

### 2.1 vLLM backend never sends a video — plugin `vllm_fix` (`runners/vllm_fix/`)

lmms_eval's stock `--model vllm` decodes every video **client-side** (`qwen_vl_utils`) and sends the
sampled frames to vLLM as N separate images. Qwen2.5-VL therefore sees N × `<|image_pad|>`: no
2-frame temporal merge, no video position ids, no timestamps. Every "raw video" vLLM run in the
literature-style comparison is in fact a frames-as-images run, and the `.mp4` condition degenerates
into the frames condition. `vllm_fix` forwards the `.mp4` itself as a `file://` URL
(`ChatMessages.to_openai_messages(pass_video_url=True)`), so vLLM decodes it server-side through the
model's own video processor: one `<|video_pad|>` block, the input the hf backend builds. The plugin
also sets what that path depends on — `allowed_local_media_path`, `media_io_kwargs.video.num_frames`
(from `nframes`), `mm_processor_kwargs.max_pixels/min_pixels` — and registers a torchcodec video
loader that uses `qwen_vl_utils`' index formula, so the frames match the hf backend's. `fps`
sampling is not available on this path. Validated on Qwen2.5-VL only (the processor kwargs are
Qwen-VL's). `sweep_vllm_fix.sh` measures the effect.

---

## 3. Known limitations that would need further core changes

- **Prompt blocks are keyed by the registered model name.** `lmms_eval_specific_kwargs` is resolved
  by model name, so `--model qwen3_vl` gets a task's `qwen3_vl:` prompt while the same checkpoint
  under `--model vllm` falls back to `default:`. An hf-vs-vllm comparison on such a task (Video-MME
  has a `qwen3_vl:` block) also changes the prompt. Meta-VideoBench sidesteps this with materialized
  prompts selected by `prompt_variant`.
- `max_new_tokens` in a task YAML acts as a floor, not a cap, on the vLLM path.
- Multi-GPU in the runners is data parallelism (one replica per GPU); models that do not fit on one
  GPU need `--backend vllm --tensor_parallel_size N`.
