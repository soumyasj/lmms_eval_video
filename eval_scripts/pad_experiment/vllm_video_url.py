"""New, non-invasive vLLM chat model that forwards the video as a native URL.

Purpose
-------
The stock ``--model vllm`` (``lmms_eval/models/chat/vllm.py``) always calls
``ChatMessages.to_openai_messages(...)`` WITHOUT ``pass_video_url`` (protocol.py:83 & 176),
so it defaults to ``pass_video_url=False`` -> frames are decoded client-side and sent as many
``image_url`` items -> the model sees N x ``<|image_pad|>``.

This subclass flips that single flag to ``True`` so the video is forwarded as a ``video_url``
and decoded natively by the server -> the model sees a single ``<|video_pad|>`` block.

Nothing in the original repo is modified. This is a brand-new class registered at runtime by
``run_pad_experiment.py`` under the model id ``vllm_video_url``.

See ``CHANGES.md`` for the full rationale and how the tag choice can move the accuracy numbers.
"""

from typing import Tuple

from lmms_eval.api.instance import Instance
from lmms_eval.models.chat.vllm import VLLM as VLLMChat
from lmms_eval.protocol import ChatMessages


class VLLMVideoURL(VLLMChat):
    """Identical to the stock vLLM chat model except videos go through the ``video_url``
    (native / ``<|video_pad|>``) path instead of the frame-image (``<|image_pad|>``) path."""

    is_simple = False  # inherited, restated for clarity: uses the chat eval loop

    def _video_kwargs(self) -> dict:
        video_kwargs = {
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_image_pixels,
            "max_frames": self.max_frame_num,
        }
        if self.fps is not None:
            video_kwargs["fps"] = self.fps
        else:
            video_kwargs["nframes"] = self.nframes
        return video_kwargs

    # --- overrides: the ONLY behavioral change is pass_video_url=True ---

    def make_one_request(self, request: Instance) -> Tuple[list[dict], dict]:
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments
        raw_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        chat_messages = ChatMessages(messages=raw_messages)

        _gen = dict(gen_kwargs or {})
        _gen["max_new_tokens"] = self._select_max_new_tokens(_gen.get("max_new_tokens"))
        _gen.setdefault("temperature", 0)
        _gen.setdefault("top_p", 0.95)
        params = self._build_sampling_params_dict(_gen)

        messages = chat_messages.to_openai_messages(
            video_kwargs=self._video_kwargs(),
            pass_video_url=True,  # <-- the switch: video_url -> <|video_pad|>
        )
        return messages, params

    def _to_openai_messages(self, raw_messages: list[dict]) -> list[dict]:
        # Used by the multi-round path (generate_until_multi_round).
        chat_messages = ChatMessages(messages=raw_messages)
        return chat_messages.to_openai_messages(
            video_kwargs=self._video_kwargs(),
            pass_video_url=True,
        )
