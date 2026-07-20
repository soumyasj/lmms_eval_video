# Configurable arguments for `--model vllm --tasks videomme`

Reference for every knob that affects this run:

```bash
python -m lmms_eval \
  --model vllm \
  --model_args model=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=602112,max_model_len=32768,gpu_memory_utilization=0.9,max_num_seqs=8,nframes=4,max_frame_num=4,limit_mm_per_prompt={\"video\":1} \
  --tasks videomme \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix qwen25vl_4frames_run1_mseq8 \
  --output_path ./logs/videomme_subset
```

Environment this was written against: vLLM `0.11.0`, qwen-vl-utils `0.0.14`, conda env `/mnt/lustre/work/kuehne/kqr828/.conda/lmms_eval`.

---

## 1. Which code actually runs

`--model vllm` is resolved through `lmms_eval/models/__init__.py` → `registry_v2.py`. A model id can have both a *simple* and a *chat* implementation, and **the chat implementation wins by default**:

| Requested | Class used | File |
|---|---|---|
| `--model vllm` | `VLLM` (chat) | `lmms_eval/models/chat/vllm.py` |
| `--model vllm --force_simple` | `VLLM` (simple) | `lmms_eval/models/simple/vllm.py` |
| `--model vllm_generate` | `VLLMGenerate` | `lmms_eval/models/chat/vllm_generate.py` |

So your run uses `chat/vllm.py`, which subclasses `simple/vllm.py` for engine setup but overrides `generate_until`.

Two things worth knowing before you touch these:

- **`--force_simple` is currently broken for vllm.** `lmms_eval/models/simple/vllm.py:521` calls `SamplingParams(**params)` but `params` is never defined in that scope, so it raises `NameError` on the first batch.
- **`vllm_generate` sends the real video** to vLLM (via `LLM.generate` with `multi_modal_data`) instead of pre-decoded frames. It's the path intended for Qwen3-VL. `vllm` (chat) decodes frames client-side and sends them as images — see §4.

---

## 2. CLI arguments (`lmms_eval/__main__.py`)

Everything below is a top-level flag you can add to the command.

### Model / task selection
| Flag | Default | Notes |
|---|---|---|
| `--model` | `hf` | `vllm`, `vllm_generate`, `qwen2_5_vl`, … |
| `--model_args` | `""` | Comma-separated `k=v`; see §3. Nested commas inside `{...}` are handled by `_smart_comma_split`. |
| `--tasks` | – | Comma-separated names, a YAML path, a directory of YAMLs, or `list` / `list_groups` / `list_tags` / `list_subtasks`. |
| `--include_path` | `None` | Extra directory to scan for task YAMLs. |
| `--force_simple` | off | Force the simple implementation (broken for vllm, see §1). |
| `--config` | `""` | YAML file of these same CLI args; explicit CLI flags override it. Supports an `env:` block to set environment variables. |

### Batching / execution
| Flag | Default | Notes |
|---|---|---|
| `--batch_size` / `-b` | `1` | How many docs are encoded and handed to `client.chat()` per call. See the `max_num_seqs` note in §6. |
| `--max_batch_size` | `None` | Only used with `--batch_size auto`. |
| `--device` | `None` | Ignored by the vllm path (vLLM manages devices). |
| `--limit` | `None` | `-1`/omitted = all; `0 < x < 1` = fraction; `>= 1` = absolute count. |
| `--offset` | `0` | Start index per task; pairs with `--limit` for sharding a run. |
| `-n` / `--repeats` / `--num_samples` | `1` | >1 enables k-samples mode (EA, CA, IV, CR stability metrics). |
| `--max_tokens` | `None` | Global token budget; the eval stops gracefully when exceeded. |
| `--seed` | `0,1234,1234,1234` | `random,numpy,torch,fewshot`. A single int sets all four. Note the vLLM engine seed is hardcoded to `1` in `simple/vllm.py:239`. |

### Generation
| Flag | Default | Notes |
|---|---|---|
| `--gen_kwargs` | `""` | e.g. `temperature=0,top_p=1.0,max_new_tokens=16`. **Overrides the task YAML's `generation_kwargs`** (`evaluator.py:414`). Only three keys survive to vLLM — see §5. |
| `--num_fewshot` | `None` | N/A for videomme. |
| `--fewshot_as_multiturn` | off | Requires `--apply_chat_template` and `num_fewshot > 0`. |
| `--apply_chat_template` | off | The vllm chat path applies the template inside vLLM anyway. |
| `--system_instruction` | `None` | Prepends a system message. |
| `--reasoning_tags` | `[["<think>","</think>"],["<analysis>","</analysis>"]]` | Tag pairs stripped out for reasoning extraction. |

### Logging / output
| Flag | Default | Notes |
|---|---|---|
| `--output_path` | `None` | Required with `--log_samples` / `--predict_only`. |
| `--log_samples` | off | Per-sample outputs. |
| `--log_samples_suffix` | `model_outputs` | Becomes `filename_prefix` on the evaluation tracker. |
| `--predict_only` / `-x` | off | Implies `--log_samples`; skips metric computation. |
| `--show_config` | off | Dump the resolved config. |
| `--write_out` / `-w` | off | Deprecated. |
| `--verbosity` | `INFO` | `DEBUG` also re-raises exceptions instead of swallowing them. |
| `--timezone` | `Asia/Singapore` | Affects the datetime in output paths. |
| `--wandb_args` | `""` | e.g. `project=lmms-eval,job_type=eval`. |
| `--wandb_log_samples` | off | |
| `--hf_hub_log_args` | `""` | e.g. `hub_results_org=...,push_results_to_hub=True`. |
| `--use_cache` / `-c` | `None` | Response cache dir or `.db` file. |
| `--cache_requests` | `None` | `true` / `refresh` / `delete` — caches request *building*, not responses. |
| `--baseline` | `None` | JSONL path, `hf://user/repo`, or a preset (e.g. `qwen25vl`) for a paired t-test. |
| `--check_integrity` | off | Runs the task test suite. |
| `--trust_remote_code` | off | For HF dataset loading (separate from the model-args one). |
| `--process_with_media` | off | Keep media in the dataset during processing. |
| `--agentic_trace_mode` | `basic` | `basic` / `full`. |

### Power analysis (planning, exits before eval)
`--power-analysis`, `--effect-size` (0.03), `--alpha` (0.05), `--power` (0.80), `--correlation` (0.5), `--std-a`, `--std-b`.

### Subcommands
`python -m lmms_eval <sub>` where `<sub>` ∈ `eval, tasks, models, ui, serve, power, version, tui, mcp`. The bare `--model X --tasks Y` form you use is the legacy flat form and still works.

---

## 3. `--model_args` — consumed by the wrapper

These are eaten by `chat/vllm.py` + `simple/vllm.py` and never reach the vLLM engine:

| Arg | Default | Meaning |
|---|---|---|
| `model` | `Qwen/Qwen2.5-VL-3B-Instruct` | HF id or local path. Also forwarded to `LLM(model=...)`. |
| `tensor_parallel_size` | `1` | GPUs per replica. |
| `data_parallel_size` | `1` | **Global** world-size value, not per-node. >1 requires torchrun/accelerate; single-process launch raises. |
| `gpu_memory_utilization` | `0.8` | Forwarded to `LLM`. |
| `batch_size` | `1` | Overridden by the `--batch_size` CLI flag in practice. |
| `max_frame_num` | `768` (chat) / `32` (simple) | Becomes `max_frames` in `video_kwargs`. **No-op when `nframes` is set** — see §4. |
| `nframes` | `32` | Fixed frame count per video. |
| `fps` | `None` | If set, `nframes` is *not* sent and sampling becomes fps-based. Mutually exclusive with `nframes`. |
| `max_pixels` | `1605632` | Per-frame pixel budget → `smart_resize`. |
| `min_image_pixels` | `28` | Sent as `min_pixels`. In simple mode it also drives the Qwen2/2.5-VL minimum-side resize. |
| `max_new_tokens` | `4096` (chat) / `1024` (simple) | **Acts as a floor, not a cap** — see §5. |
| `trust_remote_code` | `True` | |
| `chat_template` | `None` | A `.jinja`/`.jinja2`/`.j2` file path, a path containing a separator, or a raw template string. |
| `disable_log_stats` | `False` | When `False`, TTFT/TPOT/throughput are logged after each task. |
| `image_first` | `False` | Simple mode only: put images before the text. |

Any value that starts with `{` and ends with `}` is `json.loads`-ed automatically (`simple/vllm.py:194-199`), which is how `limit_mm_per_prompt={"video":1}` and `hf_overrides={...}` work.

---

## 4. Video → frames: how the frame args actually combine

`chat/vllm.py:74-83` builds:

```python
video_kwargs = {"max_pixels": self.max_pixels, "min_pixels": self.min_image_pixels, "max_frames": self.max_frame_num}
if self.fps is not None: video_kwargs["fps"] = self.fps
else:                    video_kwargs["nframes"] = self.nframes
```

and passes it to `qwen_vl_utils.fetch_video`, which returns decoded frames. `lmms_eval/protocol.py:121-129` then converts **each frame to a base64 `image_url`**. vLLM therefore sees N images, never a video.

Consequences for your command:

- **`max_frame_num=4` does nothing.** `smart_nframes` only reads `max_frames` on the fps branch (`if "nframes" in ele: nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)`). With `nframes=4` set, `max_frames` is ignored. Harmless, just noise.
- **`limit_mm_per_prompt={"video":1}` is a no-op.** No video ever reaches the engine. The meaningful knob is `limit_mm_per_prompt={"image":4}` (match your `nframes`). Unspecified modalities default to **999** on the V1 engine (`MultiModalConfig.get_limit_per_prompt`), so images are currently unbounded.
- **`max_pixels=602112` is exactly the ceiling.** qwen-vl-utils caps video frames at `VIDEO_MAX_TOKEN_NUM * 28 * 28 = 768 * 784 = 602112`. Anything larger is clamped and logs `The given max_pixels[...] exceeds limit[...]`. So you're at max resolution per frame; lower it to cut tokens.
- `nframes` is rounded to a multiple of `FRAME_FACTOR = 2`.

Keys `fetch_video` accepts (all can be surfaced via `video_kwargs` if you extend the model): `nframes`, `fps`, `min_frames`, `max_frames`, `min_pixels`, `max_pixels`, `total_pixels`, `resized_height`, `resized_width`, `video_start`, `video_end`.

---

## 5. Sampling params — the important gotcha

`_build_sampling_params_dict` (`simple/vllm.py:352`) passes **only three** things to `SamplingParams`:

```python
{"max_tokens": ..., "temperature": ..., "top_p": ...}
```

So `videomme.yaml`'s `num_beams: 1` and `do_sample: false` are silently dropped. `top_p: 0` is rewritten to `1.0`. Defaults if the task omits them: `temperature=0`, `top_p=0.95`.

And `_select_max_new_tokens` (`simple/vllm.py:330`):

```python
return max(request_max_new_tokens, self.max_new_tokens)
```

It takes the **max**, not the task value. `videomme.yaml` asks for `max_new_tokens: 16`, but the chat class defaults to `4096`, so **your run generates up to 4096 tokens per answer** for what should be a single letter. Fix either way:

```bash
--model_args ...,max_new_tokens=16          # max(16, 16) = 16
# or
--gen_kwargs max_new_tokens=16              # still max(16, 4096) = 4096 — does NOT help alone
```

Only the `model_args` value lowers the floor. This is likely your biggest easy speedup.

---

## 6. `--model_args` — passed through to vLLM `LLM(...)`

Every `**kwargs` key lands in `LLM()` (`simple/vllm.py:232-241`), i.e. any of vLLM 0.11.0's 139 `EngineArgs` fields. The ones that matter here:

| Arg | Notes |
|---|---|
| `max_model_len` | Context window. With 4 frames @ 602112 px you're nowhere near 32768 — you can shrink it to buy KV cache. |
| `max_num_seqs` | Concurrent sequences. **Capped by your `--batch_size`**: with `batch_size=1` only one request is in flight, so `max_num_seqs=8` buys nothing. Raise `--batch_size` to ≥ 8 to use it. |
| `limit_mm_per_prompt` | Per-modality item cap. Use `{"image": <nframes>}` for this path. |
| `dtype` | `auto`, `bfloat16`, `float16`, … |
| `kv_cache_dtype` | e.g. `fp8` to roughly halve KV cache. |
| `quantization` | `fp8`, `awq`, `gptq`, … |
| `enforce_eager` | Skip CUDA graph capture (faster startup, slower decode). |
| `enable_prefix_caching` | Big win when the prompt prefix repeats. |
| `enable_chunked_prefill` | Helps with long multimodal prefills. |
| `swap_space`, `cpu_offload_gb`, `num_gpu_blocks_override`, `block_size`, `gpu_memory_utilization`, `kv_cache_memory_bytes` | Memory tuning. |
| `mm_processor_kwargs`, `media_io_kwargs`, `mm_processor_cache_gb`, `disable_mm_preprocessor_cache`, `mm_encoder_tp_mode`, `skip_mm_profiling`, `interleave_mm_strings`, `video_pruning_rate` | Multimodal handling. |
| `hf_overrides`, `override_generation_config`, `generation_config`, `rope_scaling`, `rope_theta`, `revision`, `tokenizer`, `tokenizer_mode`, `hf_token` | Model/config overrides. |
| `pipeline_parallel_size`, `distributed_executor_backend`, `enable_expert_parallel` | Parallelism. Note the wrapper force-sets `distributed_executor_backend="external_launcher"` under multi-process accelerate. |
| `speculative_config`, `compilation_config`, `cuda_graph_sizes`, `scheduling_policy`, `max_num_batched_tokens` | Perf. |
| `seed` | Overridden to `1` by the wrapper — passing it will raise a duplicate-kwarg `TypeError`. |
| `disable_log_stats`, `trust_remote_code`, `tensor_parallel_size`, `data_parallel_size`, `gpu_memory_utilization`, `model` | Also set explicitly by the wrapper — don't pass twice. |

Full list for this exact vLLM version:

```bash
python -c "from vllm.engine.arg_utils import EngineArgs; import dataclasses; print('\n'.join(f.name for f in dataclasses.fields(EngineArgs)))"
```

---

## 7. Task-side configuration for videomme

### Available task names
`videomme`, `videomme_w_subtitle`, `videomme_long`, `videomme_long_w_subtitle`, `videomme_number_option`, `videomme_gt_none_option`, `videomme_no_visual`, `videomme_convert_mcq_oe`, `videomme_revert_oe_mcq`, `videomme_video_only_abcd`, `videomme_random_choice`.

### Your local `videomme.yaml` is modified
It currently points at a local parquet subset rather than the hub:

```yaml
dataset_path: parquet
dataset_kwargs:
  data_files:
    test: .../random_debug/videomme_subset_200.parquet
```

To go back to the full set: `dataset_path: lmms-lab/Video-MME` and drop `data_files`.

Videos are resolved by `videomme_doc_to_visual` to `$HF_HOME/videomme/data/<videoID>.mp4` (falls back to `.MP4`, `.mkv`, then `sys.exit`s). So **`HF_HOME` decides where videos are read from**, and `dataset_kwargs.cache_dir` (`videomme`) is the subdirectory name.

### YAML fields you can change
| Field | Current | Notes |
|---|---|---|
| `generation_kwargs` | `max_new_tokens: 16, temperature: 0, top_p: 1.0, num_beams: 1, do_sample: false` | Only the first three reach vLLM, and `max_new_tokens` is a floor (§5). |
| `lmms_eval_specific_kwargs` | `default: {pre_prompt: "", post_prompt: "\nAnswer with the option's letter..."}` | Keyed by `--model` value. **Add a `vllm:` block to give the vllm path its own prompt** — otherwise `default` is used. A `format: qwen3_vl` key switches to the alternate prompt builder. |
| `model_specific_generation_kwargs` | not set | Same keying, merged into `generation_kwargs` (`api/task.py:897-904`). |
| `cluster_key` | `videoID` | Clustered SE — questions from one video are treated as correlated. |
| `metric_list` | `videomme_perception_score` | |
| `test_split`, `doc_to_visual`, `doc_to_text`, `doc_to_target`, `process_results` | | Standard overrides. |

---

## 8. Environment variables

| Var | Default | Effect |
|---|---|---|
| `HF_HOME` | `~/.cache/huggingface/` | Video lookup root for videomme (§7). |
| `WORKERS` | `32` | Thread pool for per-request encoding in the vllm models. |
| `LMMS_IMAGE_ENCODE_FORMAT` | `PNG` | `JPEG`/`JPG`/`WEBP` shrink the base64 payload a lot — worth trying for frame-heavy runs. |
| `LMMS_IMAGE_JPEG_QUALITY` | `85` | Only for JPEG/JPG/WEBP. |
| `LMMS_WATCHDOG_DIR` | unset | Writes per-rank heartbeat JSON (`encode_start`/`chat_start`/`chat_done`/`batch_done`/`error`) — useful for diagnosing hangs. |
| `FORCE_QWENVL_VIDEO_READER` | auto | `torchcodec` / `decord` / `torchvision`. Auto-detect prefers torchcodec. |
| `MODEL_SEQ_LEN` | `128000` | qwen-vl-utils `total_pixels` default budget. |
| `TORCHCODEC_NUM_THREADS` | `8` | Video decode threads. |
| `NUM_SECONDS_TO_SLEEP` | `5` | API retry backoff (unused on this path). |
| `VLLM_WORKER_MULTIPROC_METHOD` | forced to `spawn` | Set by the wrapper. |
| `LMMS_EVAL_PLUGINS` | unset | Deprecated plugin loading; prefer entry points. |
| `VERBOSITY` | from `--verbosity` | |

---

## 9. Suggested corrected command

```bash
python -m lmms_eval \
  --model vllm \
  --model_args model=Qwen/Qwen2.5-VL-7B-Instruct,\
max_pixels=602112,\
nframes=4,\
max_new_tokens=16,\
max_model_len=32768,\
gpu_memory_utilization=0.9,\
max_num_seqs=8,\
limit_mm_per_prompt={\"image\":4},\
enable_prefix_caching=True \
  --tasks videomme \
  --batch_size 8 \
  --log_samples \
  --log_samples_suffix qwen25vl_4frames_run1_mseq8 \
  --output_path ./logs/videomme_subset
```

Changes from the original and why:

1. `max_new_tokens=16` — caps generation at 16 instead of 4096 (§5). Biggest win.
2. `--batch_size 8` — lets `max_num_seqs=8` actually do something (§6).
3. `limit_mm_per_prompt={"image":4}` instead of `{"video":1}` — the video limit is a no-op on this path (§4).
4. Dropped `max_frame_num=4` — ignored when `nframes` is set (§4).
5. `enable_prefix_caching=True` — videomme prompts share a long fixed prefix.

Verify #1 and #2 against a short `--limit 8` run before committing to a full sweep.
