"""Runner that registers the new ``vllm_video_url`` model at runtime, then hands off to the
normal lmms-eval CLI. Touches ZERO existing repo files.

Usage (same flags as ``python -m lmms_eval``, just point --model at vllm_video_url):

    python eval_scripts/pad_experiment/run_pad_experiment.py \
        --model vllm_video_url \
        --model_args model=Qwen/Qwen2.5-VL-7B-Instruct,nframes=2,max_frame_num=2,... \
        --tasks videomme \
        --batch_size 1 --log_samples --output_path ./logs/videomme_video_pad

For the image_pad baseline, just run the stock CLI unchanged:

    python -m lmms_eval --model vllm --model_args ... --tasks videomme ...

Compare the two output_paths' accuracy to see the tag effect. See CHANGES.md.
"""

import os
import sys

# Make this directory importable so the registry can load "vllm_video_url.VLLMVideoURL".
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def _register_video_url_model() -> None:
    """Add the new model to the live registry singleton (no repo files changed)."""
    from lmms_eval.models import MODEL_REGISTRY_V2
    from lmms_eval.models.registry_v2 import ModelManifest

    MODEL_REGISTRY_V2.register_manifest(
        ModelManifest(
            model_id="vllm_video_url",
            chat_class_path="vllm_video_url.VLLMVideoURL",
        ),
        overwrite=True,
    )


def main() -> None:
    _register_video_url_model()
    # Defer import until after registration; cli_evaluate parses sys.argv itself.
    from lmms_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    main()
