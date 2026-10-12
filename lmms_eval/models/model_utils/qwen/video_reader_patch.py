"""
qwen_vl_utils' built-in torchvision video backend (`_read_video_torchvision`) calls
torchvision.io.read_video, which decodes an entire video into a single uint8 tensor
before any frame subsampling happens -- regardless of how few frames are requested.
For long videos (VideoMME has some 70k-90k frame clips) that tensor alone can be
100-250GB, which OOMs a normal eval node no matter what --nframes/--batch_size is set to.

This patches qwen_vl_utils' VIDEO_READER_BACKENDS["torchvision"] with a seek-based
reader that only decodes the frames it actually needs (same frame-index selection
as the original, verified to match decord/torchcodec output), avoiding the full-video
decode entirely. decord/torchcodec are unaffected -- they were already seek-based.

Import this module for its side effect wherever qwen_vl_utils.process_vision_info
is used with FORCE_QWENVL_VIDEO_READER=torchvision.
"""

import torch
from loguru import logger as eval_logger

try:
    import decord
    from qwen_vl_utils.vision_process import (
        VIDEO_READER_BACKENDS,
        calculate_video_frame_range,
        smart_nframes,
    )
    from torchvision.io import VideoReader

    def _read_video_torchvision_seek(ele):
        video_path = ele["video"]
        vr = decord.VideoReader(video_path)
        total_frames, video_fps = len(vr), vr.get_avg_fps()
        start_frame, end_frame, total_frames = calculate_video_frame_range(ele, total_frames, video_fps)
        nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()

        reader = VideoReader(video_path, "video")
        frames = []
        for i in idx:
            reader.seek(i / video_fps)
            frames.append(next(reader)["data"])  # CHW
        video = torch.stack(frames)

        sample_fps = nframes / max(total_frames, 1e-6) * video_fps
        video_metadata = dict(fps=video_fps, frames_indices=idx, total_num_frames=total_frames, video_backend="torchvision")
        return video, video_metadata, sample_fps

    VIDEO_READER_BACKENDS["torchvision"] = _read_video_torchvision_seek
except ImportError as e:
    eval_logger.warning(f"Could not patch qwen_vl_utils torchvision video backend (dependency missing): {e}")
