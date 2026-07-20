# lmms-eval — Video QA Internals & Accuracy-Affecting Factors

> Living notes on how `lmms-eval` runs video question-answering benchmarks (VideoMME etc.)
> with the vLLM chat backend and Qwen2.5-VL, plus **every factor we've found that can change
> the reported accuracy**. Update this file whenever we conclude something new — especially
> anything that would explain a difference in results between two runs.

Last updated: 2026-07-17

---

## 0. TL;DR — the things most likely to change your accuracy

If two runs disagree, check these first (details in §5):

| # | Factor | Where | Effect |
|---|--------|-------|--------|
| 1 | `nframes` (how many frames sampled) | `--model_args nframes=` → `chat/vllm.py:82` | Fewer frames = less temporal info. `nframes=2` = **first + last frame only**. |
| 2 | `FRAME_FACTOR=2` rounding | `qwen_vl_utils/vision_process.py:32,168` | Frame counts forced **even**; odd `nframes` rounds down. |
| 3 | `fps` vs `nframes` (mutually exclusive) | `chat/vllm.py:79-82` | If `fps` is set, `nframes` is **ignored**. |
| 4 | `image_pad` vs `video_pad` (frame decode path) | `protocol.py:110` (`pass_video_url`) | Different token layout / temporal handling → different logits. |
| 5 | `max_pixels` / `min_pixels` / `max_frame_num` | `chat/vllm.py:74-78` | Per-frame resolution & frame cap; changes visual detail. |
| 6 | Video reader backend (torchcodec/decord/torchvision) | `vision_process.py:380-389` | Different frame **indexing** → slightly different pixels. |
| 7 | Subtitles wording in prompt | task `utils.py` `doc_to_text` | VideoMME prompt says "and the subtitles" even when none are passed. |
| 8 | Sampling params (temperature/top_p) | `chat/vllm.py:70-71` | Defaults `temperature=0`, `top_p=0.95`. Nonzero temp = nondeterminism. |
| 9 | `max_new_tokens` / `max_model_len` truncation | `chat/vllm.py:39,69` | Too-short context truncates frames or answer. |
| 10 | Dataset subset / split / sample count | task YAML + parquet | Comparing different sample sets = not comparable. |

---

## 1. How a video QA run flows (end to end)

Command shape (VideoMME example):

```bash
python -m lmms_eval \
  --model vllm \
  --model_args model=Qwen/Qwen2.5-VL-7B-Instruct,nframes=2,max_frame_num=2,max_pixels=602112,... \
  --tasks videomme \
  --batch_size 1 --log_samples --output_path ./logs/...
```

Pipeline:

1. **Model resolution** — `--model vllm` → `AVAILABLE_CHAT_TEMPLATE_MODELS["vllm"]` = the **chat** class
   `lmms_eval/models/chat/vllm.py` (registered `vllm_chat`), a subclass of `simple/vllm.py`.
   (`lmms_eval/models/__init__.py:106,132`; selection in `get_model` / `MODEL_REGISTRY_V2`,
   default `force_simple=False` → chat class.)
2. **Task loads docs** — VideoMME task YAML + `lmms_eval/tasks/videomme/utils.py`.
   `videomme_doc_to_visual` (utils.py ~line 138) returns `[video_path]` where
   `video_path = os.path.join(cache_dir, "data", doc["videoID"] + ".mp4")`.
3. **Per-doc message build** — `chat/vllm.py:make_one_request` (line 59):
   `doc_to_messages(doc)` → `ChatMessages` → `to_openai_messages(video_kwargs=...)` (line 83).
4. **Video → content items** — `protocol.py:to_openai_messages` (line 89). For a `video`-typed
   item it either decodes frames to base64 images (default) or forwards a `video_url` (see §4).
5. **Chat template render** — vLLM applies Qwen2.5-VL's `chat_template.json`, turning each
   image/video content item into `<|vision_start|><|image_pad|><|vision_end|>` (or `<|video_pad|>`).
6. **Generate** — `client.chat(...)` in `_run_chat` (`chat/vllm.py:114-122`).
7. **Score** — task's `process_results` / metric aggregation → accuracy.

---

## 2. `nframes` → frames: the sampling pipeline

Chain: `--model_args nframes=N` → `chat/vllm.py.__init__` stores `self.nframes` (line 57) →
`make_one_request` puts it in `video_kwargs["nframes"]` (line 82) → `to_openai_messages` calls
`fetch_video({"type":"video","video":path, nframes:N, ...})` (`protocol.py:121`) →
`qwen_vl_utils/vision_process.py`.

Key facts in `qwen_vl_utils/vision_process.py`:

- `FRAME_FACTOR = 2` (line 32). `round_by_factor` (line 41) rounds counts to a multiple of 2.
- `smart_nframes` (line 144): if `nframes` in element →
  `nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)` (lines 168-169);
  asserts `FRAME_FACTOR <= nframes <= total_frames`.
- **Frame selection is evenly spaced across the whole video**, not the first N frames:
  - torchcodec `_read_video_torchcodec` (line 338): `idx = torch.linspace(start, end, nframes).round().long()` (~line 366) — seeks to those frames.
  - torchvision `_read_video_torchvision` (line 184): reads the whole video, `idx = torch.linspace(0, total_frames-1, nframes)` (~line 215).
- **`nframes=2` ⇒ first frame + last frame only.** `nframes=4` ⇒ 4 evenly spaced, etc.
- Backend order: **torchcodec > decord > torchvision** (`VIDEO_READER_BACKENDS` line 380,
  `get_video_reader_backend` line 389). In this env all three are installed and
  `FORCE_QWENVL_VIDEO_READER` is unset → **torchcodec** is used.

Library used to actually read the mp4: **torchcodec** (default here). Not decord unless forced.

---

## 3. `fps` vs `nframes` (mutually exclusive)

`chat/vllm.py:79-82`:
```python
if self.fps is not None:
    video_kwargs["fps"] = self.fps      # sample by frame rate
else:
    video_kwargs["nframes"] = self.nframes   # sample a fixed count
```
Only one is passed. **If you set `fps`, `nframes` is ignored** (and vice-versa). Default
`fps=None`, `nframes=32` (`chat/vllm.py:37-38`).

---

## 4. `image_pad` vs `video_pad` — the decode path (BIG one for results)

The placeholder tag the model sees is **not** in this repo; it comes from Qwen2.5-VL's
`chat_template.json`:
```jinja
{% if ...'image'... %}<|vision_start|><|image_pad|><|vision_end|>
{% elif ...'video'... %}<|vision_start|><|video_pad|><|vision_end|>
```
So the tag is chosen by the **`type`** of each content item handed to the template.

Decision point: `lmms_eval/protocol.py:to_openai_messages`, the `pass_video_url` flag.
- **`protocol.py:89`** — `def to_openai_messages(self, video_kwargs=None, pass_video_url: bool = False)`
- **`protocol.py:110`** — `if pass_video_url:` → append `{"type":"video_url", ...}` → server-side
  native video decode → **`<|video_pad|>`** (one video block).
- **`protocol.py:119-129`** — else (default): `fetch_video(...)` decodes frames **client-side**,
  each frame appended as `{"type":"image_url", data:base64}` → **N × `<|image_pad|>`** (one per frame).

**Who passes the flag — only ONE caller in the codebase:**
- `chat/openai.py` DOES plumb it: `__init__(pass_video_url=...)` (line 38) → `self.pass_video_url` (line 40)
  → `to_openai_messages(..., pass_video_url=self.pass_video_url)` (line 195).
- `chat/vllm.py` DOES NOT: calls at line 83 and line 176 omit it → defaults to **`False`** → **always `image_pad`**.
- `chat/async_openai.py:188` also omits it.

**Consequence for your runs:** with `--model vllm` you always get `image_pad` (frames sent as
images). The two paths produce **different model inputs** → can produce **different accuracy**:
- image path: frames are independent images, no native temporal/video position handling;
  per-frame resolution governed by `max_pixels`.
- video path: model receives a single native video, may apply temporal merging / video position ids.

**Gotcha — you cannot enable `video_pad` via `--model vllm` model_args.** `chat/vllm.py.__init__`
has no `pass_video_url` param; the kwarg falls into `**kwargs` → forwarded to `super().__init__`
→ `simple/vllm.py:232-240` passes `**kwargs` straight to vLLM's `LLM(...)` constructor, which
does **not** accept it → **TypeError at engine init (run crashes)**. Even if it didn't crash, it
never reaches line 83, so it would have no effect. To support it, mirror `chat/openai.py`:
add the param to `chat/vllm.py.__init__`, store `self.pass_video_url`, pass it at line 83.
(Not yet applied — left unchanged per user request.)

---

## 5. Full list of accuracy-affecting factors (with code anchors)

1. **Frame count (`nframes`)** — `chat/vllm.py:82`, `vision_process.py:smart_nframes`.
   More frames → more temporal coverage. `nframes=2` is extreme (first+last only).
2. **`FRAME_FACTOR=2` even-rounding** — `vision_process.py:32,41,168`. Odd requests round down
   to even; you may get fewer frames than asked.
3. **`fps` overrides `nframes`** — `chat/vllm.py:79-82`.
4. **image_pad vs video_pad** — `protocol.py:110` (§4). Different input representation.
5. **`max_pixels` / `min_image_pixels`** — `chat/vllm.py:35-36,74-76`. Per-frame resolution;
   lower `max_pixels` = coarser frames = potentially lower accuracy but less memory.
6. **`max_frame_num`** — `chat/vllm.py:32,77` (passed as `max_frames`). Caps frames on the
   `fps` path; on the pure `nframes` path the count is set by `nframes` (note both are passed).
7. **Video reader backend** — `vision_process.py:380-389`. torchcodec vs decord vs torchvision
   can pick marginally different frame indices → different pixels → small metric drift.
8. **Prompt text / subtitles** — task `utils.py` `doc_to_text`. VideoMME's template mentions
   "and the subtitles" even when subtitles are not supplied; wording affects the model.
9. **Sampling params** — `chat/vllm.py:70-71` default `temperature=0`, `top_p=0.95`. Nonzero
   temperature introduces run-to-run nondeterminism.
10. **`max_new_tokens` / `max_model_len`** — `chat/vllm.py:39,69`; `max_model_len` via model_args.
    Too small → context truncation (dropped frames) or clipped answers.
11. **`limit_mm_per_prompt`** (e.g. `{"video":1}`) — vLLM engine arg. Caps multimodal items per
    prompt; if too low for the number of image frames, inputs get dropped.
12. **`batch_size` / `max_num_seqs`** — scheduling; shouldn't change greedy outputs but can
    interact with truncation and memory-driven fallbacks.
13. **Dataset subset / split / #samples** — comparisons are only valid on the *same* rows.
    See `VIDEO_BENCHMARKS_REFERENCE.md` for per-benchmark sample counts.
14. **Model version / weights / chat template** — different Qwen2.5-VL snapshot or template
    changes tokenization of vision blocks.

---

## 6. Debugging: see the exact tokenized input (image/video tags)

**Preferred (non-invasive):** use `eval_scripts/pad_experiment/inspect_pads.py` — an offline,
CPU-only tool that renders the Qwen2.5-VL chat template both ways and prints the exact
`<|image_pad|>` / `<|video_pad|>` counts for a subset question. No edits to any repo file.

```bash
python eval_scripts/pad_experiment/inspect_pads.py --nframes 2 \
  --parquet random_debug/videomme_subset_duration.parquet
```

> An earlier debug `print(...)` block had been added directly inside `chat/vllm.py:generate_until`;
> it has since been **reverted** so the original file stays pristine. `inspect_pads.py` replaces it.

Gotchas if you ever inline a debug render again:
- A bare `breakpoint()` **hangs a background run** — use `print(..., flush=True)` instead.
- `apply_chat_template` shows `<|image_pad|>` per image item. To *faithfully* render `video_pad`
  you must feed native `{"type":"video","video":url}` items — the HF template does **not**
  recognize the OpenAI `type:"video_url"` shape (only `type:"video"` / `'video' in content`).

---

## 7. Environment specifics (this machine)

- Conda python: `/mnt/lustre/work/kuehne/kqr828/.conda/lmms_eval/bin/python`.
- Background `Bash` needs `PATH=/mnt/lustre/work/kuehne/kqr828/.conda/lmms_eval/bin:$PATH`
  prepended (login shell doesn't auto-activate conda) — else `python: command not found` (exit 127).
- Background launches also need the interactive terminal's HF env/credentials; a stripped env
  can 401 on gated datasets (see `api/task.py` `download` → `snapshot_download`).
- Video reader backend in use: **torchcodec** (`FORCE_QWENVL_VIDEO_READER` unset).

---

## 8. Related files in this repo

- `VIDEO_BENCHMARKS_REFERENCE.md` — per-benchmark sample counts & configs.
- `VIDEOMME_VLLM_ARGS.md` — VideoMME + vLLM argument notes.
- `random_debug/videomme_subset_5.parquet` — 5-row debug subset (2 videos).
- `random_debug/videomme_subset_duration.parquet` — 9 rows, 3 per short/medium/long bucket.
- `eval_scripts/pad_experiment/` — **image_pad vs video_pad experiment** (new, non-invasive files):
  `vllm_video_url.py` (new model that enables `video_pad`), `run_pad_experiment.py` (runtime-registers
  it, then runs the normal CLI), `inspect_pads.py` (offline CPU tag inspector), and `CHANGES.md`
  (full analysis of how the tag moves accuracy + how to run the comparison).

---

## 9. Changelog

- 2026-07-17 — Initial version. Captured: run flow, nframes pipeline, fps/nframes exclusivity,
  image_pad vs video_pad decision (`pass_video_url`) and why `--model vllm` can't toggle it,
  full accuracy-factor list, debug print hook, env specifics.
