"""VLLMFix: lmms_eval's chat `vllm` model with the .mp4 forwarded to vLLM as a video.

What changes versus lmms_eval/models/chat/vllm.py (everything else is inherited):

1. `to_openai_messages(..., pass_video_url=True)` in both request builders, so a
   video becomes {"type": "video_url", "video_url": {"url": "file:///..."}} and
   vLLM decodes it server-side (one <|video_pad|> block, temporal merge,
   timestamps) instead of N client-decoded <|image_pad|> images.

2. The vLLM engine kwargs that path depends on are filled in from the usual
   model args unless given explicitly:
     allowed_local_media_path  vLLM refuses file:// URLs without it. Default:
                               $HF_HOME (every benchmark's videos live below it).
     media_io_kwargs           {"video": {"num_frames": <nframes>}} -- vLLM's own
                               loader samples this many uniformly spaced frames
                               (default would be 32 regardless of nframes).
     mm_processor_kwargs       {"max_pixels": <max_pixels>, "min_pixels": <min_pixels>}
                               so the server-side video processor uses the same
                               per-frame pixel budget the hf runs use (the
                               client-side max_pixels of the stock path does
                               nothing on this path).

Frames are sampled by vLLM's video loader. The package sets
VLLM_VIDEO_LOADER_BACKEND to the torchcodec loader in video_loader.py, which
uses qwen_vl_utils' index formula, so the frames match the hf backend's; with
vLLM's stock "opencv" loader they can differ by a frame or two and long clips
decode much slower. `fps` sampling is not supported on this path (vLLM's loader
takes a frame count).
"""

import json
import os
from typing import Tuple

from loguru import logger as eval_logger

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.chat.vllm import VLLM as VLLMChat
from lmms_eval.protocol import ChatMessages

_DEFAULT_MEDIA_ROOT = os.path.expanduser("~/.cache/huggingface")


def _as_dict(value, name):
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"vllm_fix: {name} must be a JSON object, got {value!r}") from e
    if not isinstance(value, dict):
        raise ValueError(f"vllm_fix: {name} must be a dict, got {type(value).__name__}")
    return dict(value)


@register_model("vllm_fix")
class VLLMFix(VLLMChat):
    is_simple = False

    def __init__(self, min_pixels: int = 256 * 28 * 28, **kwargs):
        nframes = kwargs.get("nframes", 32)
        max_pixels = kwargs.get("max_pixels", 1605632)
        if kwargs.get("fps") is not None:
            raise ValueError("vllm_fix: fps sampling is not available on the server-side video path; use nframes")

        media_root = kwargs.pop("allowed_local_media_path", None) or os.environ.get("HF_HOME") or _DEFAULT_MEDIA_ROOT
        kwargs["allowed_local_media_path"] = os.path.abspath(media_root)

        # The loader (video_loader.py) does the per-frame resize itself: vLLM's video processor
        # ignores the pixel budget and would encode native-resolution frames.
        media_io = _as_dict(kwargs.pop("media_io_kwargs", None), "media_io_kwargs")
        media_io.setdefault("video", {})
        media_io["video"].setdefault("num_frames", int(nframes))
        media_io["video"].setdefault("max_pixels", int(max_pixels))
        media_io["video"].setdefault("min_pixels", int(min_pixels))
        kwargs["media_io_kwargs"] = media_io

        mm_proc = _as_dict(kwargs.pop("mm_processor_kwargs", None), "mm_processor_kwargs")
        mm_proc.setdefault("max_pixels", int(max_pixels))  # images (Video-MMMU question figures)
        mm_proc.setdefault("min_pixels", int(min_pixels))
        kwargs["mm_processor_kwargs"] = mm_proc

        # A video is ONE encoder item and must fit vLLM's encoder budget (= max_num_batched_tokens,
        # default 8192) in a single step, or the scheduler skips the request forever with
        # "Running: 1 reqs" and no progress. Budget the whole context for it unless overridden.
        max_model_len = int(kwargs.get("max_model_len", 32768))
        est_video_tokens = int(nframes) // 2 * int(max_pixels) // 784 + 64  # frame pairs x (28x28 px per token) + slack
        kwargs.setdefault("max_num_batched_tokens", max_model_len)
        if est_video_tokens > int(kwargs["max_num_batched_tokens"]):
            raise ValueError(
                f"vllm_fix: a {nframes}-frame video at max_pixels={max_pixels} needs ~{est_video_tokens} vision tokens, "
                f"more than max_num_batched_tokens={kwargs['max_num_batched_tokens']} (and max_model_len={max_model_len}); "
                "lower nframes/max_pixels or raise max_model_len"
            )

        # vLLM's default per-prompt limit is 1 video / 1 image, which is what the .mp4 tasks need.
        super().__init__(**kwargs)
        self.min_pixels = int(min_pixels)
        self.video_url_settings = {
            "allowed_local_media_path": kwargs["allowed_local_media_path"],
            "media_io_kwargs": media_io,
            "mm_processor_kwargs": mm_proc,
        }
        eval_logger.info(
            f"vllm_fix: videos forwarded as file:// URLs; vLLM samples {media_io['video']['num_frames']} frames/video "
            f"with loader {os.environ.get('VLLM_VIDEO_LOADER_BACKEND', 'opencv')}, per-frame pixel budget {mm_proc['min_pixels']}..{mm_proc['max_pixels']} "
            f"(~{est_video_tokens} vision tokens/video, encoder budget max_num_batched_tokens={kwargs['max_num_batched_tokens']}), "
            f"media root {kwargs['allowed_local_media_path']}"
        )

    def _video_kwargs(self) -> dict:
        # Only reaches the client-side path (images); kept for parity with the stock builder.
        return {"max_pixels": self.max_pixels, "min_pixels": self.min_image_pixels, "max_frames": self.max_frame_num, "nframes": self.nframes}

    def make_one_request(self, request: Instance) -> Tuple[list[dict], dict]:
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments
        raw_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        chat_messages = ChatMessages(messages=raw_messages)
        _gen = dict(gen_kwargs or {})
        _gen["max_new_tokens"] = self._select_max_new_tokens(_gen.get("max_new_tokens"))
        _gen.setdefault("temperature", 0)
        _gen.setdefault("top_p", 0.95)
        params = self._build_sampling_params_dict(_gen)
        messages = chat_messages.to_openai_messages(video_kwargs=self._video_kwargs(), pass_video_url=True)
        return messages, params

    def _to_openai_messages(self, raw_messages: list[dict]) -> list[dict]:
        chat_messages = ChatMessages(messages=raw_messages)
        return chat_messages.to_openai_messages(video_kwargs=self._video_kwargs(), pass_video_url=True)
