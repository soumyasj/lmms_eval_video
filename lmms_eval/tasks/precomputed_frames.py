"""Visual loader shared by every ``<task>_frames<N>`` task variant (exp7: video vs frames).

Those variants and the parquet files they read (``data/frames/<task>_frames<N>.parquet``)
are written by ``eval_scripts/make_frames_task.py``. Every row carries ``frame_paths``:
N JPEG frames sampled uniformly from the source video ahead of time. The model therefore
receives a list of images instead of the video file, which takes the video decoder and
the frame sampler of the model wrapper out of the loop -- the only thing the wrapper can
still vary is how it turns N images into tokens.

Paths in the parquet are stored relative to the repository root, so the data stays valid
when the checkout is moved or cloned elsewhere (the frames themselves have to be
regenerated or copied along). Absolute paths are used as they are.
"""

import os

from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    """PIL images of the pre-sampled frames listed in ``doc["frame_paths"]``."""
    return [Image.open(resolve(p)).convert("RGB") for p in doc["frame_paths"]]
