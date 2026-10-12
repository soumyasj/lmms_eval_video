"""Meta-VideoBench (task prefix ``metav``): helpers for the task YAMLs in this directory.

Visual and scoring functions are re-exported from the in-tree source tasks (Video-MME,
LongVideoBench, LVBench, VideoMMMU, VSI-Bench), so a Meta-VideoBench item is scored exactly
like it is in its original benchmark. Prompts are NOT computed at eval time: the text of every
item, for every prompt variant, was materialized once from the same source functions (see
``prompt_variants.py``) into ``data/metav_prompts.json`` and ``doc_to_text_<source>`` only
looks it up. The variant is picked by ``METAV_PROMPT_VARIANT`` (env) or ``prompt_variant`` in
the task's lmms_eval_specific_kwargs (default: ``default``). ``METAV_POST_PROMPT`` (env)
instead renders the prompt at eval time with that string as the post prompt of every source
(see ``prompt_variants.POST_PROMPT_KEYS``); ``METAV_OPTION_STYLE`` (env, a key of
``prompt_variants.OPTION_STYLES`` or ``PROMPT_LAYOUTS``) does the same for the option labels
("A. text", "(A) text", ...). The other addition is a ``process_docs`` filter per source task
that keeps just the selected question ids listed in ``data/metav_ids.json``.

The old ``META_VIDEOBENCH_*`` environment variable names are still honoured (mapped onto
``METAV_*`` at import time).
"""

import json
import os
import re
from functools import partial

import datasets

# accept the pre-rename environment variables as aliases of METAV_*
for _k in [k for k in os.environ if k.startswith("META_VIDEOBENCH_")]:
    os.environ.setdefault("METAV_" + _k[len("META_VIDEOBENCH_") :], os.environ[_k])

from lmms_eval.tasks.longvideobench.utils import (  # noqa: F401
    longvideobench_aggregate_results,
    longvideobench_doc_to_visual_v,
    longvideobench_process_results,
)
from lmms_eval.tasks.lvbench.utils import (  # noqa: F401
    lvbench_doc_to_visual,
    lvbench_process_results,
)

# ---- re-exports (names referenced by the YAMLs) ---------------------------------
from lmms_eval.tasks.videomme.utils import (  # noqa: F401
    videomme_aggregate_results,
    videomme_doc_to_visual,
    videomme_process_results,
)
from lmms_eval.tasks.videommmu.utils import (  # noqa: F401
    videommmu_aggregate_results,
    videommmu_doc_to_answer,
    videommmu_doc_to_visual,
    videommmu_process_results,
)
from lmms_eval.tasks.vsibench.utils import (  # noqa: F401
    MCA_QUESTION_TYPES,
    METRICS_FOR_MCA,
    METRICS_FOR_NA,
    NA_QUESTION_TYPES,
    vsibench_doc_to_visual,
    vsibench_process_results,
)

# ---- response cleaning ---------------------------------------------------------
# Some models wrap the answer in their own markers, which every source's parser then fails to read:
# GLM emits "<|begin_of_box|>C<|end_of_box|>", and a reasoning model prefixes "<think> ... </think>".
# VSI-Bench is the worst affected: it reads the first whitespace token as a number, so even a bare
# "<|begin_of_box|>210<|end_of_box|>" scores zero. Strip the markers (keeping what is inside) and drop
# the reasoning block before the source's own parser sees the text. Models that emit neither are
# unaffected, so this does not change any existing score.
_BOX_RE = re.compile(r"<\|(?:begin|end)_of_box\|>")
_THINK_RE = re.compile(r"(?s)^.*</think>")


def clean_response(text):
    if not isinstance(text, str):
        return text
    out = _THINK_RE.sub("", text, count=1)
    return _BOX_RE.sub("", out).strip()


def _clean_results(fn):
    """Wrap a source's process_results so it sees cleaned model output."""

    def inner(doc, results, *a, **kw):
        return fn(doc, [clean_response(r) for r in results], *a, **kw)

    inner.__name__ = getattr(fn, "__name__", "process_results")
    inner.__doc__ = f"clean_response() then {getattr(fn, '__name__', 'the source parser')}"
    return inner


# VSI-Bench numeric items: the metric reads the FIRST whitespace token as the number, so a model that
# writes "The answer is 210." scores zero however right it is. Only when that standard parse fails, pull
# the number out of the sentence: prefer one introduced by an answer word, else the first number present.
# A parse that already succeeds is never touched, so no existing score can change.
_ANS_NUM_RE = re.compile(r"(?:answer|result|approximately|about|is|are|=)\D{0,15}?(-?\d+(?:\.\d+)?)", re.I)
_ANY_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")

# Chain-of-thought post prompts ("... end your reply with Answer: ") put the final answer after the last
# "Answer:", with the reasoning -- and usually other numbers -- before it. Search that tail first so the
# fallback does not pick a number out of the reasoning; if it holds no number, fall back to the whole text.
_ANSWER_TAIL_RE = re.compile(r"(?:^|\n|\s)answer\s*[:\-]\s*", re.I)


def _first_number(text):
    """The number the answer most likely refers to, or None. Never raises."""
    m = _ANS_NUM_RE.search(text) or _ANY_NUM_RE.search(text)
    if m is None:
        return None
    return m.group(1) if m.lastindex else m.group(0)


def _vsi_numeric_fallback(doc, text):
    from lmms_eval.tasks.vsibench.utils import (
        NA_QUESTION_TYPES,
        fuzzy_matching,
        to_float,
    )

    if doc.get("question_type") not in NA_QUESTION_TYPES or not isinstance(text, str):
        return text
    if to_float(fuzzy_matching(text)) is not None:
        return text  # the source parser can already read it
    tails = _ANSWER_TAIL_RE.split(text)
    if len(tails) > 1:  # CoT reply: prefer what follows the last "Answer:"
        n = _first_number(tails[-1])
        if n is not None:
            return n
    n = _first_number(text)
    return n if n is not None else text


def _clean_results_vsi(fn):
    def inner(doc, results, *a, **kw):
        return fn(doc, [_vsi_numeric_fallback(doc, clean_response(r)) for r in results], *a, **kw)

    inner.__name__ = getattr(fn, "__name__", "process_results")
    inner.__doc__ = "clean_response() + numeric fallback, then the VSI-Bench parser"
    return inner


longvideobench_process_results = _clean_results(longvideobench_process_results)
lvbench_process_results = _clean_results(lvbench_process_results)
videomme_process_results = _clean_results(videomme_process_results)
videommmu_process_results = _clean_results(videommmu_process_results)
vsibench_process_results = _clean_results_vsi(vsibench_process_results)


def frames_doc_to_visual(doc):
    """exp7 frames variants: doc["frame_paths"] holds N pre-sampled JPEG paths
    (eval_scripts/make_frames_task.py); the model gets PIL images instead of the video."""
    from PIL import Image

    return [Image.open(p).convert("RGB") for p in doc["frame_paths"]]


def vsibench_aggregate_overall(results):
    """Same score as lmms_eval.tasks.vsibench.utils.vsibench_aggregate_overall
    (mean over the 8 VSI-Bench categories, the three object_rel_direction levels
    averaged into one), but tolerant of question types that are absent from the
    scored docs: the in-tree version KeyErrors as soon as one of the
    easy/medium/hard rel-direction levels is missing, which happens on partial
    runs (--limit, a crashed shard). Meta-VideoBench (metav) contains all
    ten question types, so on a full run the two functions agree exactly."""
    import pandas as pd

    df = pd.DataFrame(results)
    output = {}
    for qtype, idx in df.groupby("question_type").groups.items():
        sub = df.iloc[idx]
        metrics = METRICS_FOR_MCA if qtype in MCA_QUESTION_TYPES else METRICS_FOR_NA if qtype in NA_QUESTION_TYPES else None
        if metrics is None:
            raise ValueError(f"Unknown question type: {qtype}")
        for metric in metrics:
            output[f"{qtype}_{metric}"] = sub[metric].mean()
    rel = [output.pop(k) for k in ("object_rel_direction_easy_accuracy", "object_rel_direction_medium_accuracy", "object_rel_direction_hard_accuracy") if k in output]
    if rel:
        output["object_rel_direction_accuracy"] = sum(rel) / len(rel)
    return round(sum(output.values()) / len(output), 6) if output else 0.0


_IDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "metav_ids.json")
_IDS = json.load(open(_IDS_PATH))  # source task -> dataset-native ids of the 1000 Meta-VideoBench items

# dataset-native id column used to filter each source task
_ID_FIELD = {
    "videomme": "question_id",
    "longvideobench_val_v": "id",
    "lvbench": "uid",
    "video_mmmu_perception": "id",
    "video_mmmu_comprehension": "id",
    "video_mmmu_adaptation": "id",
    "vsibench": "id",
}


def _filter(dataset: datasets.Dataset, task: str) -> datasets.Dataset:
    field = _ID_FIELD[task]
    keep = set(str(x) for x in _IDS[task])
    out = dataset.filter(lambda d: str(d[field]) in keep)
    if len(out) != len(keep):
        raise RuntimeError(f"metav/{task}: expected {len(keep)} docs, got {len(out)}; dataset version mismatch?")
    return out


# one process_docs per source task; referenced from the YAMLs
for _task in _ID_FIELD:
    globals()[f"process_docs_{_task}"] = partial(_filter, task=_task)


# ---- prompts: materialized text per item and variant -----------------------------
_PROMPTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "metav_prompts.json")
_PROMPTS = None


def _prompts():
    global _PROMPTS
    if _PROMPTS is None:
        if not os.path.exists(_PROMPTS_PATH):
            raise FileNotFoundError(f"{_PROMPTS_PATH} missing; it ships with the repository (lmms_eval/tasks/metav/data)")
        _PROMPTS = json.load(open(_PROMPTS_PATH))
    return _PROMPTS


_PV = None


def _prompt_variants():
    """prompt_variants.py next to this file (lmms_eval loads utils.py by path, so the dir is not on sys.path)."""
    global _PV
    if _PV is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("metav_prompt_variants", os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_variants.py"))
        _PV = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_PV)
    return _PV


# exp7 frame separators (METAV_FRAME_SEP, frames tasks only): text placed after each frame.
# The model must run with interleave_visuals=True (qwen2_5_vl), which puts frame N at "<image N>".
FRAME_SEPS = {
    "newline": "<image {n}>\n",  # <frame 1>\n<frame 2>\n... question
    "label": "Frame {n}: <image {n}>\n",  # Frame 1: <frame 1>\nFrame 2: <frame 2>\n... question
}
_IMAGE_REF_RE = re.compile(r"<image (\d+)>")


# exp8 layouts (METAV_LAYOUT): order of the visual block and the prompt text in the user turn.
# "v" = all visuals of the item (the N .jpg frames of a frames task, the single video otherwise), "t" = the
# prompt text. The wrapper's default order is "vt" (visuals, then text). The model must run with
# interleave_visuals=True (qwen2_5_vl), which places visual i at every "<image i>" marker, so the same
# frames can be inserted twice. Nothing is put between the blocks: the visual tokens of vtv / tvt are
# exactly those of the plain run, repeated or moved.
LAYOUTS = {
    "vtv": "vtv",  # <visuals> prompt <visuals>
    "tvt": "tvt",  # prompt <visuals> prompt
}


def _doc_to_text(doc, lmms_eval_specific_kwargs=None, task=None):
    text = _base_text(doc, lmms_eval_specific_kwargs, task)
    sep = os.environ.get("METAV_FRAME_SEP")
    layout = os.environ.get("METAV_LAYOUT")
    if sep and layout:
        raise ValueError("metav: METAV_FRAME_SEP and METAV_LAYOUT are separate experiments; set only one")
    if layout:
        return _layout_text(text, doc, layout)
    if not sep or "frame_paths" not in doc:
        return text
    # "<image N>" already in a question (VideoMMMU's figure reference) would be taken for frame N
    text = _IMAGE_REF_RE.sub(r"image \1", text)
    return "".join(FRAME_SEPS[sep].format(n=i + 1) for i in range(len(doc["frame_paths"]))) + text


def _layout_text(text, doc, layout):
    if layout not in LAYOUTS:
        raise KeyError(f"metav: unknown METAV_LAYOUT {layout!r}; one of {sorted(LAYOUTS)}")
    if "frame_paths" in doc:
        n_visuals = len(doc["frame_paths"])
    else:
        n_visuals = 1  # every source's doc_to_visual returns [video_path]
        if layout.count("v") > 1:
            # the qwen2_5_vl wrapper subsamples only video_inputs[0] to max_num_frames; a second copy of the
            # .mp4 would keep qwen_vl_utils' own frame sampling, i.e. not the same tokens. Use the frames tasks.
            raise ValueError(f"metav: layout {layout!r} repeats the visuals and needs a frames task (metav_frames<N>), not the .mp4 task")
    text = _IMAGE_REF_RE.sub(r"image \1", text)  # a literal "<image N>" in a question would be taken for a marker
    visuals = "".join(f"<image {i + 1}>" for i in range(n_visuals))
    return "".join(visuals if part == "v" else text for part in LAYOUTS[layout])


def _base_text(doc, lmms_eval_specific_kwargs=None, task=None):
    if os.environ.get("METAV_BUILDING_PROMPTS"):  # set by the prompt builder that produced data/metav_prompts.json: the task loader probes doc_to_text before the file exists
        return ""
    variant = os.environ.get("METAV_PROMPT_VARIANT") or (lmms_eval_specific_kwargs or {}).get("prompt_variant") or "default"
    post_prompt = os.environ.get("METAV_POST_PROMPT")
    option_style = os.environ.get("METAV_OPTION_STYLE") or None
    if post_prompt is not None or option_style:  # one post prompt / option style for every source: rendered on the fly, not looked up
        return _prompt_variants().render(variant, task, doc, post_prompt=post_prompt, option_style=option_style)
    prompts = _prompts()
    if variant not in prompts["variants"]:
        raise KeyError(f"metav prompt variant {variant!r} not in {prompts['variants']}; the shipped data/metav_prompts.json does not contain it")
    key = f"{task}::{doc[_ID_FIELD[task]]}"
    try:
        return prompts["items"][key]["prompts"][variant]
    except KeyError:
        raise KeyError(f"metav: no {variant!r} prompt for {key}; the shipped data/metav_prompts.json does not contain it")


# one doc_to_text per source task; referenced from the YAMLs
for _task in _ID_FIELD:
    globals()[f"doc_to_text_{_task}"] = partial(_doc_to_text, task=_task)
