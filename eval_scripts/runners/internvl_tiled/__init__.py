"""Out-of-tree lmms_eval model plugin: `internvl_hf_tiled`.

InternVL (HF format) on transformers with the model card's video recipe:
each sampled frame goes through the *image* processor's dynamic tiling
(min/max_patches, i.e. `max_num`, plus the thumbnail tile), and the prompt is
built as `Frame{i}: <img>...</img>` per frame. This is what
`load_video(..., max_num=M)` + `dynamic_preprocess` do in the OpenGVLab
reference code. lmms_eval's stock `internvl_hf` instead feeds the mp4 to
transformers' InternVLVideoProcessor, which has no tiling (always one 448
tile per frame, and OPENAI_CLIP normalisation instead of ImageNet).

Importing this package registers the model. Launch with
    PYTHONPATH=eval_scripts/runners python -m internvl_tiled --model internvl_hf_tiled ...
(or `accelerate launch ... -m internvl_tiled ...`). Nothing under lmms_eval/
is modified.
"""

from lmms_eval.models import MODEL_REGISTRY_V2
from lmms_eval.models.registry_v2 import ModelManifest

MODEL_NAME = "internvl_hf_tiled"

MODEL_REGISTRY_V2.register_manifest(
    ModelManifest(
        model_id=MODEL_NAME,
        chat_class_path="internvl_tiled.model.InternVLHfTiled",
    ),
    overwrite=True,
)
