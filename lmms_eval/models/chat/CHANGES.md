# Changes to models in this directory

## internvl_hf.py — fix video-input crash (pixel_shuffle shape error)

**File:** `internvl_hf.py`
**What changed:** Added a `video_size` constructor parameter to `InternVLHf`
and used it to force sampled video frames to be resized to that resolution
before being sent to the processor, instead of letting the processor use its
own default.

**Bug this fixes:** Every `generate_until` call on a video input (e.g. any
`videomme` run) crashed inside `transformers`, not this repo's code:

```
RuntimeError: shape '[4, 27, 13, 2048]' is invalid for input of size 2985984
  ... modeling_internvl.py, in pixel_shuffle
      vision_features = vision_features.view(...)
```

Root cause: `transformers`' `InternVLVideoProcessor` defaults video frames to
a generic 384x384 tile size. For this model family (ViT patch_size=14), that
produces a 384/14 = 27x27 patch grid. `InternVLModel.pixel_shuffle()` then
downsamples that grid by 0.5x (folding 2x2 patch blocks into the channel
dim), which requires an *even* grid — 27 is odd, so the `.view(...)` call
fails. Image inputs were unaffected because they're already resized to the
model's native (even-grid) resolution elsewhere in the pipeline; only the
video path used the mismatched default.

Because the wrapper's `generate_until` catches and logs the exception per
sample instead of raising (`except Exception as e: eval_logger.error(...)`),
this failure was silent: the eval loop kept running and reported near-0%
scores instead of erroring out. Confirmed via a full traceback (temporarily
added `traceback.print_exc()` in the except block, removed after diagnosis)
pointing at `transformers/models/internvl/modeling_internvl.py`'s
`pixel_shuffle`.

**Why this fix is correct:** Resizing video frames to the model's own native
ViT resolution — read from `self._config.vision_config.image_size[0]` after
the model loads, not hardcoded — guarantees an even patch grid for any
InternVL checkpoint this wrapper is pointed at (not just the 448px
InternVL3.5-8B default), and keeps inputs in-distribution for the model
(that's the resolution its position embeddings and weights were actually
tiled/trained at). A caller can still override it explicitly via the new
`video_size=` model arg if needed.

**Verified:** Before the fix, a 2-sample `videomme` smoke test
(`--num_frames 4 --limit 2`) produced 0 generated tokens and a `pixel_shuffle`
error on every sample. After the fix, the same run produced real generations
(non-zero tokens, no errors, e.g. `videomme_perception_score: 50` on the
2-sample subset).

**How to use:** `video_size` is optional and defaults to the model's native
resolution — no action needed for normal use. To override:
```
--model_args pretrained=OpenGVLab/InternVL3_5-8B-HF,video_size=448,...
```
Do not set it to a value where `video_size / patch_size` is odd, or the same
crash will reoccur.
