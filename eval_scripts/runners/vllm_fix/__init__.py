"""Out-of-tree lmms_eval model plugin: `vllm_fix`.

lmms_eval's stock `--model vllm` decodes every video CLIENT-side (qwen_vl_utils)
and sends the sampled frames to vLLM as N separate images. Qwen2.5-VL then sees
N x <|image_pad|>: no 2-frame temporal merge, no video position ids, no
timestamps -- the .mp4 condition degenerates into the frames-as-images one.

`vllm_fix` forwards the .mp4 itself (file:// URL, ChatMessages.to_openai_messages
pass_video_url=True) so vLLM decodes it SERVER-side through the model's own video
processor: one <|video_pad|> block, temporal merge, timestamps -- the same input
the hf backend builds. Frame count and resolution on that path are governed by
vLLM (media_io_kwargs / mm_processor_kwargs), and the plugin derives them from
the usual nframes / max_pixels / min_pixels model args so runs stay comparable.

Successor of the older vllm_video_url.py plugin (2026-07), which only
flipped the flag and left the vLLM-side knobs (local media path, frame count,
pixel budget) to the caller.

Importing this package registers the model. Launch with
    PYTHONPATH=eval_scripts/runners python -m vllm_fix --model vllm_fix ...
(or `accelerate launch ... -m vllm_fix ...`). Nothing under lmms_eval/ is
modified; see eval_scripts/runners/run_video_qa_vllm_fix.sh.
"""

import os

from vllm_fix.video_loader import (  # registers the torchcodec loader with vLLM
    LOADER_NAME,
    install_media_io_patch,
)

from lmms_eval.models import MODEL_REGISTRY_V2
from lmms_eval.models.registry_v2 import ModelManifest

MODEL_NAME = "vllm_fix"

# vLLM's default "opencv" loader decodes sequentially and aborts on files whose header
# frame count exceeds the decodable count; use the torchcodec loader unless the caller
# picked one explicitly (export VLLM_VIDEO_LOADER_BACKEND=opencv to get the stock one).
os.environ.setdefault("VLLM_VIDEO_LOADER_BACKEND", LOADER_NAME)
# file:// videos: decode from the path instead of reading the whole file into memory first.
install_media_io_patch()

MODEL_REGISTRY_V2.register_manifest(
    ModelManifest(
        model_id=MODEL_NAME,
        chat_class_path="vllm_fix.model.VLLMFix",
    ),
    overwrite=True,
)
