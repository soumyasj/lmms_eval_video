"""InternVLHfTiled: lmms_eval `internvl_hf` with the model card's video path.

Reference (OpenGVLab model card, `load_video`):
    frame_indices = get_index(bound, fps, max_frame, num_segments=N)   # segment midpoints
    for each frame: dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=M)
    prompt = "".join(f"Frame{i+1}: <image>\n" ...) + question

Here the same thing is done on top of transformers' InternVLProcessor:
  * frames are decoded with decord (fallback: pyav) at the reference's
    segment-midpoint indices,
  * every frame is handed to the processor as an *image*, so
    GotOcr2ImageProcessor.crop_image_to_patches does the aspect-ratio grid
    search with min_patches/max_patches and appends the thumbnail when the
    grid has more than one tile (use_thumbnail=True is its default),
  * the `<video>` placeholder the chat template emits is rewritten to
    "Frame1: <IMG_CONTEXT>\nFrame2: <IMG_CONTEXT>..." so the processor expands
    each to `<img>` + 256 x n_tiles context tokens + `</img>`.

Everything else (model loading, generation, batching) is inherited unchanged
from lmms_eval.models.chat.internvl_hf.InternVLHf, whose generate_until calls
`self.processor(images=..., videos=..., text=..., **kwargs)`; that call is
intercepted by a thin proxy around the processor.
"""

import re
from typing import List, Optional

import numpy as np
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.internvl_hf import InternVLHf


# ---------------------------------------------------------------------------
# frame sampling: identical to the model card's get_index()
# ---------------------------------------------------------------------------
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices


def load_video_frames(
    video_path: str,
    num_segments: Optional[int] = None,
    fps_target: Optional[float] = None,
    bound=None,
    backend: str = "decord",
) -> List[Image.Image]:
    """Decode `num_segments` frames (or `fps_target` frames/s) at the
    reference's segment-midpoint indices. Returns RGB PIL images."""
    if backend == "decord":
        try:
            from decord import VideoReader, cpu

            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            total = len(vr)
            fps = float(vr.get_avg_fps())
            n = _resolve_num_segments(total, fps, num_segments, fps_target)
            idx = get_index(bound, fps, total - 1, first_idx=0, num_segments=n)
            idx = np.clip(idx, 0, total - 1)
            arr = vr.get_batch(idx.tolist()).asnumpy()
            return [Image.fromarray(a).convert("RGB") for a in arr]
        except Exception as e:  # noqa: BLE001
            eval_logger.warning(f"decord failed on {video_path} ({e!r}); falling back to pyav")
    return _load_video_frames_pyav(video_path, num_segments, fps_target, bound)


def _resolve_num_segments(total: int, fps: float, num_segments: Optional[int], fps_target: Optional[float]) -> int:
    if num_segments is None:
        if fps_target is None or not fps or fps <= 0:
            raise ValueError("need num_frames or fps")
        num_segments = int(total / fps * fps_target)
    return max(1, min(int(num_segments), total))


def _load_video_frames_pyav(video_path, num_segments, fps_target, bound):
    """Sequential pyav decode keeping only the sampled indices (constant memory)."""
    import av

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        total = int(stream.frames or 0)
        if total <= 0:  # container without a frame count: count by decoding once
            total = sum(1 for _ in container.decode(stream))
            container.seek(0)
        n = _resolve_num_segments(total, fps, num_segments, fps_target)
        idx = get_index(bound, fps or 1.0, total - 1, first_idx=0, num_segments=n)
        wanted = set(int(i) for i in np.clip(idx, 0, total - 1))
        out = {}
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted:
                out[i] = frame.to_image().convert("RGB")
                if len(out) == len(wanted):
                    break
    finally:
        container.close()
    if not out:
        raise ValueError(f"pyav decoded no frames from {video_path}")
    if len(out) < len(wanted):
        # stream shorter than its declared frame count: substitute the nearest
        # decoded frame at or before the wanted index (else the first decoded)
        eval_logger.warning(f"{video_path}: declared {total} frames, decoded fewer; {len(wanted) - len(out)} sampled indices substituted")
    got = sorted(out)

    def nearest(i):
        prev = [k for k in got if k <= i]
        return out[prev[-1] if prev else got[0]]

    return [nearest(int(i)) for i in np.clip(idx, 0, total - 1)]


# ---------------------------------------------------------------------------
# processor proxy: videos -> tiled frames-as-images
# ---------------------------------------------------------------------------
class _TiledVideoProcessor:
    """Wraps an InternVLProcessor. Calls without `videos` pass straight
    through. Calls with `videos` decode the clips, splice the frames into the
    `images` list at the position of each `<video>` placeholder, rewrite the
    placeholder to per-frame image placeholders, and forward to the wrapped
    processor with the tiling kwargs (min/max_patches) -- so frames are
    processed exactly like still images."""

    def __init__(self, inner, owner: "InternVLHfTiled"):
        self._inner = inner
        self._owner = owner
        self._pattern = re.compile(f"(?P<image>{re.escape(inner.image_token)})|(?P<video>{re.escape(inner.video_token)})")

    def __getattr__(self, name):  # tokenizer, apply_chat_template, batch_decode, ...
        return getattr(self._inner, name)

    def __call__(self, images=None, videos=None, text=None, **kwargs):
        if not videos:
            return self._inner(images=images, videos=videos, text=text, **kwargs)

        # kwargs the stock wrapper sets for the *video* processor; not used here.
        num_frames = kwargs.pop("num_frames", None)
        fps = kwargs.pop("fps", None)
        kwargs.pop("do_sample_frames", None)
        kwargs.pop("size", None)
        if fps is not None:
            # fps sampling: the stock wrapper still sends its num_frames default
            # (32) alongside fps; fps is the explicit request, so it wins.
            num_frames = None
        elif num_frames is None:
            num_frames = self._owner.num_frames
        kwargs.setdefault("min_patches", self._owner.min_patches)
        kwargs.setdefault("max_patches", self._owner.max_patches)
        kwargs.setdefault("crop_to_patches", True)

        if isinstance(text, (list, tuple)):
            assert len(text) == 1, "InternVLHfTiled runs batch_size=1"
            text = text[0]
        images = list(images or [])
        videos = list(videos)

        frames_per_video = [load_video_frames(v, num_segments=num_frames, fps_target=fps, backend=self._owner.video_backend) for v in videos]

        # Walk the prompt; keep still images in place, expand each <video>.
        img_tok, vid_tok = self._inner.image_token, self._inner.video_token
        ordered_images, pieces, pos, ii, vi = [], [], 0, 0, 0
        for m in self._pattern.finditer(text):
            pieces.append(text[pos : m.start()])
            if m.lastgroup == "image":
                ordered_images.append(images[ii])
                ii += 1
                pieces.append(img_tok)
            else:
                frames = frames_per_video[vi]
                vi += 1
                ordered_images.extend(frames)
                pieces.append("\n".join(f"Frame{i + 1}: {img_tok}" for i in range(len(frames))))
            pos = m.end()
        pieces.append(text[pos:])
        if vi != len(videos):
            raise ValueError(f"prompt has {vi} {vid_tok!r} placeholders but {len(videos)} videos were given")
        new_text = "".join(pieces)

        self._owner._last_video_frames = [len(f) for f in frames_per_video]
        return self._inner(images=ordered_images, videos=None, text=new_text, **kwargs)


@register_model("internvl_hf_tiled")
class InternVLHfTiled(InternVLHf):
    """`internvl_hf` + model-card video tiling. Extra model_args:
        video_backend: decord (default) | pyav
    `min_patches` / `max_patches` (= max_num) now apply to video frames."""

    is_simple = False

    def __init__(self, *args, video_backend: str = "decord", **kwargs):
        super().__init__(*args, **kwargs)
        assert video_backend in ("decord", "pyav"), video_backend
        self.video_backend = video_backend
        self._last_video_frames = None
        self.processor = _TiledVideoProcessor(self.processor, self)
        eval_logger.info(f"InternVLHfTiled: video frames -> image tiler with min_patches={self.min_patches} max_patches={self.max_patches} (thumbnail when >1 tile), num_frames={self.num_frames} fps={self.fps} backend={video_backend}")
