"""torchcodec video loader for vLLM, registered as VLLM_VIDEO_LOADER_BACKEND=torchcodec_lmms.

Why not vLLM's default "opencv" loader:
  * it decodes the file sequentially with cap.grab() up to the last sampled
    index, i.e. the whole clip for uniform sampling -- minutes per long video;
  * it trusts CAP_PROP_FRAME_COUNT, and when the container header claims more
    frames than are decodable the last sampled index is never reached and it
    aborts the run:  "AssertionError: Expected reading 4 frames, but only
    loaded 3 frames from video."

This loader mirrors qwen_vl_utils' torchcodec reader, which the hf backend
uses (FORCE_QWENVL_VIDEO_READER=torchcodec):
    idx = torch.linspace(0, total_frames - 1, num_frames).round()
    frames = VideoDecoder(...).get_frames_at(idx)
so a vllm_fix run samples exactly the frame indices of the matching hf run.
torchcodec's default seek_mode="exact" scans the file, so `num_frames` is the
real decodable count, not the header's (the scan of a 3.4 GB LVBench file
takes ~1 s; seek_mode="approximate" would take minutes on the same file). A
decode error is still caught and the range shrunk to the last readable frame
instead of failing the run.

Local files are decoded straight from the path: vLLM's VideoMediaIO.load_file
would first read the whole file into memory (3.4 GB for the largest LVBench
clip) and hand the loader bytes; `install_media_io_patch` routes it to
`load_path` instead. Every decode is logged with its duration so a stall can
be attributed to a file.

The metadata dict follows the format of the installed vLLM's own opencv
loader, because vLLM's Qwen2.5-VL processor derives second_per_grid_ts from
it and the format changed between 0.11.0 and 0.11.1.
"""

import os
import time
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
import vllm
from loguru import logger as eval_logger
from packaging.version import Version
from vllm.multimodal.video import VIDEO_LOADER_REGISTRY, VideoLoader, VideoMediaIO

LOADER_NAME = "torchcodec_lmms"
_NEW_METADATA_FORMAT = Version(vllm.__version__) >= Version("0.11.1")


def _last_decodable_index(decoder, total_frames: int) -> int:
    """Largest index that decodes, by bisection (only used after a decode error)."""
    lo, hi = 0, total_frames - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            decoder.get_frame_at(mid)
            lo = mid
        except Exception:
            hi = mid - 1
    return lo


def _resize_like_qwen_vl_utils(frames: torch.Tensor, min_pixels: int, max_pixels: int) -> torch.Tensor:
    """(T, C, H, W) uint8 -> same, resized to the per-frame pixel budget the hf backend uses.

    vLLM's Qwen2.5-VL video processor keeps its checkpoint cap (12.8 Mpx per frame) whatever
    mm_processor_kwargs says, so without this step frames stay at native resolution: a 1080p
    clip costs ~5.2k tokens at 4 frames and anything above 1080p exceeds the 8192-token encoder
    budget, which makes vLLM's scheduler skip the request forever. Same smart_resize + bicubic
    antialiased resize as qwen_vl_utils.fetch_video, so token counts match the hf runs.
    """
    from qwen_vl_utils.vision_process import smart_resize
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    factor = 28  # Qwen2.5-VL: 14-px patches x 2x2 spatial merge (qwen_vl_utils' image_factor for this model)
    _, _, h, w = frames.shape
    rh, rw = smart_resize(h, w, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    if (rh, rw) == (h, w):
        return frames
    out = TF.resize(frames, [rh, rw], interpolation=InterpolationMode.BICUBIC, antialias=True)
    return out.round().clamp(0, 255).to(torch.uint8)


def _decode(source: Union[bytes, str], num_frames: int, label: str, min_pixels: int = None, max_pixels: int = None) -> tuple[np.ndarray, dict[str, Any]]:
    from torchcodec.decoders import VideoDecoder

    t0 = time.time()
    decoder = VideoDecoder(source, num_ffmpeg_threads=int(os.environ.get("TORCHCODEC_NUM_THREADS", 8)))
    md = decoder.metadata
    total_frames = int(md.num_frames)
    video_fps = float(md.average_fps) if md.average_fps else 0.0
    duration = total_frames / video_fps if video_fps > 0 else 0.0

    def _indices(last_index: int) -> list[int]:
        count = last_index + 1
        if num_frames == -1 or count <= num_frames:
            return list(range(count))
        # qwen_vl_utils formula (round, not floor): identical frames to the hf backend
        return torch.linspace(0, last_index, num_frames).round().long().tolist()

    idx = _indices(total_frames - 1)
    try:
        frames = decoder.get_frames_at(indices=idx).data  # (T, C, H, W) uint8
    except Exception as e:
        last = _last_decodable_index(decoder, total_frames)
        eval_logger.warning(f"torchcodec_lmms: {label}: decode failed at requested indices ({e}); header says {total_frames} frames, last decodable is {last} -> resampling")
        idx = _indices(last)
        frames = decoder.get_frames_at(indices=idx).data

    native = f"{frames.shape[3]}x{frames.shape[2]}"
    if max_pixels is not None:
        frames = _resize_like_qwen_vl_utils(frames, min_pixels or 0, max_pixels)
    frames = frames.permute(0, 2, 3, 1).contiguous().numpy()  # (T, H, W, 3), what vLLM expects
    n = len(idx)
    eval_logger.info(f"torchcodec_lmms: {label}: {n} frames {native} -> {frames.shape[2]}x{frames.shape[1]} of {total_frames} ({duration:.0f}s @ {video_fps:.2f} fps) in {time.time() - t0:.1f}s, indices {idx[:3]}..{idx[-1]}")

    if _NEW_METADATA_FORMAT:
        metadata = {
            "total_num_frames": total_frames,
            "fps": video_fps,
            "duration": duration,
            "video_backend": LOADER_NAME,
            "frames_indices": list(idx),
            "do_sample_frames": n == total_frames,
        }
    else:
        # vLLM 0.11.0 presents the sampled clip as the whole video at the sampled rate
        metadata = {
            "total_num_frames": n,
            "fps": n / duration if duration > 0 else video_fps,
            "duration": duration,
            "video_backend": LOADER_NAME,
            "frames_indices": list(range(n)),
            "do_sample_frames": n == total_frames,
        }
    return frames, metadata


@VIDEO_LOADER_REGISTRY.register(LOADER_NAME)
class TorchcodecVideoBackend(VideoLoader):
    # min_pixels / max_pixels arrive through media_io_kwargs["video"] (set by model.VLLMFix).
    @classmethod
    def load_bytes(cls, data: bytes, num_frames: int = -1, min_pixels: int = None, max_pixels: int = None, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        return _decode(data, num_frames, f"<{len(data) / 1e6:.0f} MB in memory>", min_pixels, max_pixels)

    @classmethod
    def load_path(cls, filepath: Union[str, Path], num_frames: int = -1, min_pixels: int = None, max_pixels: int = None, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        return _decode(str(filepath), num_frames, os.path.basename(str(filepath)), min_pixels, max_pixels)


_ORIG_LOAD_FILE = VideoMediaIO.load_file


def _load_file_via_path(self: VideoMediaIO, filepath: Path):
    loader = self.video_loader
    if hasattr(loader, "load_path"):
        return loader.load_path(filepath, num_frames=self.num_frames, **self.kwargs)
    return _ORIG_LOAD_FILE(self, filepath)


def install_media_io_patch() -> None:
    """Let file:// videos reach the loader as a path instead of an in-memory copy of the file."""
    if VideoMediaIO.load_file is not _load_file_via_path:
        VideoMediaIO.load_file = _load_file_via_path
