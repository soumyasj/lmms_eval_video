# pad_experiment — change log & tag→accuracy analysis

All code here is **additive**: new files only, no existing repo file is modified or removed.
Goal: measure and explain how the video **tag** (`<|image_pad|>` vs `<|video_pad|>`) changes the
numbers, using the small VideoMME debug subsets.

Last updated: 2026-07-17

---

## New files added (nothing existing changed)

| File | What it is | Why it's separate |
|------|------------|-------------------|
| `vllm_video_url.py` | New model class `VLLMVideoURL(VLLM)` that flips `pass_video_url=True` → **video_pad**. Overrides only `make_one_request` + `_to_openai_messages`. | The stock `chat/vllm.py` has no way to enable `video_pad` (and passing it via model_args crashes at `LLM()` init). A subclass adds the capability without editing the original. |
| `run_pad_experiment.py` | Runner: registers `vllm_video_url` into the live `MODEL_REGISTRY_V2` at runtime, then calls the normal CLI. | Registering at runtime avoids editing `lmms_eval/models/__init__.py`. Use it exactly like `python -m lmms_eval`. |
| `inspect_pads.py` | Offline CPU tool: renders the Qwen2.5-VL chat template both ways for a subset question and prints the `<|image_pad|>` / `<|video_pad|>` counts. | Lets you *see* the tag difference with no GPU/vLLM/video-decode. |
| `run_video_qa.sh` | Configurable runner. Env-var knobs for every accuracy factor: `SAMPLING` (fps vs nframes), `PAD` (image vs video), `NFRAMES`/`FPS`, `MAX_PIXELS`, `MAX_FRAME_NUM`, `MAX_MODEL_LEN`, `GPU_MEM_UTIL`, `MAX_NUM_SEQS`, `LIMIT_MM_VIDEO`, `TASKS`, `LIMIT`, etc. Routes to stock CLI (`PAD=image`) or the new model (`PAD=video`). | One script to sweep the factors from NOTES §5 without editing anything. |
| `CHANGES.md` | This file. | Tracking + analysis. |

None of these touch `lmms_eval/models/chat/vllm.py`, `protocol.py`, `models/__init__.py`, or any
task file. The stock `--model vllm` behaves exactly as before.

---

## How to run the comparison (small dataset)

**Baseline — image_pad (stock, unchanged):**
```bash
python -m lmms_eval \
  --model vllm \
  --model_args model=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=602112,max_model_len=32768,gpu_memory_utilization=0.9,max_num_seqs=1,nframes=2,max_frame_num=2,limit_mm_per_prompt={\"video\":1} \
  --tasks videomme --batch_size 1 --log_samples \
  --output_path ./logs/videomme_image_pad
```

**Treatment — video_pad (new model):**
```bash
python eval_scripts/pad_experiment/run_pad_experiment.py \
  --model vllm_video_url \
  --model_args model=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=602112,max_model_len=32768,gpu_memory_utilization=0.9,max_num_seqs=1,nframes=2,max_frame_num=2,limit_mm_per_prompt={\"video\":1} \
  --tasks videomme --batch_size 1 --log_samples \
  --output_path ./logs/videomme_video_pad
```
Then diff the accuracy in the two `--output_path` result JSONs.

> Note for `video_pad`: vLLM decodes the video server-side. Frame sampling on this path is
> governed by vLLM's own `media_io_kwargs` / video loader, **not** by `nframes` in `video_kwargs`
> (that only controls the client-side `fetch_video` used on the image path). So the two runs may
> also differ in *how many* frames the model effectively sees — see factor (2) below.

**Offline tag check (no GPU):**
```bash
python eval_scripts/pad_experiment/inspect_pads.py --nframes 2 \
  --parquet random_debug/videomme_subset_duration.parquet
```

---

## Verified result of the offline inspector (2026-07-17)

For the first subset question, `nframes=2`:

- **image_pad path** → `<|image_pad|>` blocks: **2**, `<|video_pad|>`: 0
  `...<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>When demonstrating...`
- **video_pad path** → `<|image_pad|>`: 0, `<|video_pad|>` blocks: **1**
  `...<|vision_start|><|video_pad|><|vision_end|>When demonstrating...`

So the switch is real and observable at the prompt level.

---

## How the tag can change the accuracy (the analysis)

The tag itself is a placeholder; what differs is the **visual representation behind it**. Concretely:

1. **Number of vision blocks / token budget.**
   - image_pad: one `<|image_pad|>` block **per frame** (N blocks for N frames). Each block later
     expands (via `grid_thw`) into many patch tokens sized by `max_pixels`.
   - video_pad: a **single** `<|video_pad|>` block for the whole clip. vLLM's video processor
     decides frames, temporal merge, and token count.
   → Different total vision-token counts → different context usage and different logits.

2. **Frame sampling is controlled by different code on each path.**
   - image path: `qwen_vl_utils.fetch_video(..., nframes=N)` (client-side, torchcodec) picks
     `N` evenly-spaced frames (`nframes=2` = first+last). You control it with `nframes`/`fps`.
   - video path: vLLM loads the video itself; `nframes` in `video_kwargs` is **not** applied.
     The effective frame count comes from vLLM's video loader defaults / `media_io_kwargs`.
   → The two runs can see a **different set and count of frames**, which alone moves accuracy.

3. **Temporal position handling.**
   - image path: frames are independent images; the model may lack the native video temporal
     position ids / temporal patch merging it was trained with for video.
   - video path: native video → the model applies its trained temporal handling.
   → Tasks needing temporal reasoning (ordering, counting-over-time) can score differently.

4. **Per-frame resolution vs whole-video budget.**
   - image path: `max_pixels` caps **each** frame's resolution.
   - video path: vLLM's video pixel budget governs the clip; per-frame detail may differ.
   → Fine visual details (small text, small objects) can be resolved on one path and not the other.

5. **Encoding round-trip on the image path.**
   - image path re-encodes each decoded frame to base64 (PNG/JPEG per `LMMS_IMAGE_ENCODE_FORMAT`).
     JPEG/WEBP are lossy → tiny pixel differences vs the native decode on the video path.

**Bottom line:** switching image_pad↔video_pad is **not** a cosmetic change — it changes how many
frames the model sees, at what resolution, with what temporal structure, and how many vision
tokens are spent. Any of these can shift VideoMME accuracy, especially on a **small** subset where
a handful of questions swing the percentage a lot. Treat the two paths as different experimental
conditions and only compare runs that use the *same* path.

---

## Open items / to verify with real runs

- [ ] Run both paths on `videomme` (9-sample duration subset) and record the accuracy delta here.
- [ ] Confirm effective frame count on the video_pad path (log what vLLM actually samples).
- [ ] Check whether `limit_mm_per_prompt={"video":1}` needs raising for the image path when
      `nframes` is large (image path emits N image items, not 1 video item).

---

## Change log

- 2026-07-17 — Added `vllm_video_url.py`, `run_pad_experiment.py`, `inspect_pads.py`, this file.
  Verified runtime registration resolves the new model (`is_simple=False`) and the offline
  inspector shows 2×`<|image_pad|>` vs 1×`<|video_pad|>` for `nframes=2`. No existing files changed.
- 2026-07-17 — Reverted a leftover debug `print(...)` block that had been added directly inside
  `lmms_eval/models/chat/vllm.py:generate_until` in an earlier session, restoring that file to
  pristine (empty `git diff`). Its function is fully replaced by the new `inspect_pads.py`.
  Left untouched: a commented-out user note in `lmms_eval/protocol.py` (`# pass_video_url = True`,
  inactive) — that's your edit, not mine.
