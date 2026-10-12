"""Prompt variants of Meta-VideoBench: which source prompt function and which
prompt kwargs produce each named variant.

This is the single place that knows how a Meta-VideoBench item is turned into
prompt text. The prompt builder ran it once over every item and
stores the resulting strings in ``data/metav_prompts.json``; at eval
time ``utils.doc_to_text_<source>`` only looks the string up, it never calls
the source functions.

Variants
--------
``default``   the ``default`` lmms_eval_specific_kwargs of every in-tree source task.
``qwen3_vl``  what a run with ``--model qwen3_vl`` sees: videomme has a
              model-specific prompt (VLMEvalKit style); the other sources have
              none, so lmms_eval falls back to ``default`` and so do we.

A one-off post prompt for every source needs no variant: set
``METAV_POST_PROMPT="..."`` (env) and the prompt is rendered at eval time
with that string in place of each source's post prompt (see POST_PROMPT_KEYS).
Likewise ``METAV_OPTION_STYLE=<name in OPTION_STYLES>`` re-labels the option
lines of every source in one style ("A. text", "(A) text", ...) instead of each source's
own; only the option block changes, preambles and post prompts stay as they are.
The names in PROMPT_LAYOUTS (prefix, option_letter, start, middle, observation) go through the same
variable but rebuild the whole prompt: question, options and instruction in a fixed order.

To add a prompt to try: add a key to VARIANTS with, per source, the doc_to_text
function and kwargs (or ``None`` to fall back to ``default``), rerun
rebuild ``data/metav_prompts.json``, then select it with
``METAV_PROMPT_VARIANT=<name>`` or ``prompt_variant: <name>`` in the YAML.
"""

import importlib
import re

SOURCES = ("videomme", "longvideobench_val_v", "lvbench", "vsibench", "video_mmmu_perception", "video_mmmu_comprehension", "video_mmmu_adaptation")

_VMMMU_KWARGS = {
    "pre_prompt": "You should watch and learn the video content. Then apply what you learned to ",
    "perception_and_comprehension_prompt": "\nPlease ignore the Quiz question in last frame of the video.",
    "mcq_prompt": "answer the following multi-choice question. The image for this question is at the end of the video.\n",
    "open_ended_prompt": "answer the following open-ended question. The image for this question is at the end of the video.\n",
    "mcq_post_prompt": "",
}

# variant -> source -> (doc_to_text function under lmms_eval.tasks, lmms_eval_specific_kwargs) | None (= use "default")
VARIANTS = {
    "default": {
        "videomme": ("videomme.utils.videomme_doc_to_text", {"pre_prompt": "", "post_prompt": "\nAnswer with the option's letter from the given choices directly."}),
        "longvideobench_val_v": ("longvideobench.utils.longvideobench_doc_to_text", {"pre_prompt": "", "post_prompt": "Answer with the option's letter from the given choices directly.\n"}),
        "lvbench": ("lvbench.utils.lvbench_doc_to_text", {"pre_prompt": "", "post_prompt": "\nAnswer the question with the option letter"}),
        "vsibench": (
            "vsibench.utils.vsibench_doc_to_text",
            {"pre_prompt": "", "mca_post_prompt": "Answer with the option's letter from the given choices directly.", "na_post_prompt": "Please answer the question using a single word or phrase."},
        ),
        "video_mmmu_perception": ("videommmu.utils.videommmu_doc_to_text_perception_comprehension", _VMMMU_KWARGS),
        "video_mmmu_comprehension": ("videommmu.utils.videommmu_doc_to_text_perception_comprehension", _VMMMU_KWARGS),
        "video_mmmu_adaptation": ("videommmu.utils.videommmu_doc_to_text_adaptation", _VMMMU_KWARGS),
    },
    "qwen3_vl": {
        "videomme": ("videomme.utils.videomme_doc_to_text", {"format": "qwen3_vl", "pre_prompt": "Question: ", "post_prompt": "Answer with the option letter only."}),
        "longvideobench_val_v": None,
        "lvbench": None,
        "vsibench": None,
        "video_mmmu_perception": None,
        "video_mmmu_comprehension": None,
        "video_mmmu_adaptation": None,
    },
}


# the kwarg(s) of each source function that hold the answer-format instruction after the question
POST_PROMPT_KEYS = {
    "videomme": ("post_prompt",),
    "longvideobench_val_v": ("post_prompt",),
    "lvbench": ("post_prompt",),
    "vsibench": ("mca_post_prompt", "na_post_prompt"),
    "video_mmmu_perception": ("mcq_post_prompt",),
    "video_mmmu_comprehension": ("mcq_post_prompt",),
    "video_mmmu_adaptation": ("mcq_post_prompt",),
}


# option label styles: name -> format of one option line given its letter and text
OPTION_STYLES = {
    "dot": "{L}. {text}",
    "paren": "({L}) {text}",
    "rparen": "{L}) {text}",
    "colon": "{L}: {text}",
    "bracket": "[{L}] {text}",
}
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# a label the sources themselves use in front of an option text: "A.", "A)", "A:", "(A)", "[A]"
_LABEL_RE = re.compile(r"^\s*(?:\(([A-Z])\)|\[([A-Z])\]|([A-Z])[.):])\s*")


def _strip_label(line: str) -> str:
    return _LABEL_RE.sub("", line, count=1)


def option_block(source: str, doc: dict) -> tuple[str, list[str]]:
    """(the option block exactly as the source's default prompt function renders it, the plain
    option texts without labels). Empty block when ``doc`` has no options (VSI-Bench numeric)."""
    if source == "videomme":
        opts = list(doc["options"])
        return "\n".join(opts), [_strip_label(o) for o in opts]
    if source == "longvideobench_val_v":
        texts = [doc.get(f"option{i}") for i in range(5)]
        texts = [t for t in texts if t != "N/A" and t is not None]
        return "\n".join(f"{_LETTERS[i]}. {t}" for i, t in enumerate(texts)), texts
    if source == "lvbench":  # labels are baked into the question text as "(A) ..." lines
        lines = [l for l in doc["question"].split("\n") if re.match(r"^\([A-Z]\) ", l)]
        return "\n".join(lines), [_strip_label(l) for l in lines]
    if source.startswith("video_mmmu"):
        opts = list(doc["options"])
        if opts and all(o.startswith(f"{_LETTERS[i]}.") for i, o in enumerate(opts)):
            return "\n".join(opts), [_strip_label(o) for o in opts]
        return "\n".join(f"{_LETTERS[i]}. {o}" for i, o in enumerate(opts)), opts
    if source == "vsibench":
        opts = doc.get("options")
        if opts is None or len(opts) == 0:
            return "", []
        opts = list(opts)
        return "\n".join(opts), [_strip_label(o) for o in opts]
    raise KeyError(source)


def restyle_options(text: str, source: str, doc: dict, option_style: str) -> str:
    """``text`` (a rendered prompt) with its option block re-labelled in ``option_style``."""
    fmt = OPTION_STYLES[option_style]
    old, texts = option_block(source, doc)
    if not old:
        return text
    if old not in text:  # e.g. a VideoMMMU open-ended item that carries an unused options list: nothing to restyle
        return text
    new = "\n".join(fmt.format(L=_LETTERS[i], text=t) for i, t in enumerate(texts))
    return text.replace(old, new, 1)


# prompt layouts: the whole prompt rebuilt from question, options and one instruction line, the same
# on every source (source preambles such as "These are frames of a video." are dropped).
# {q} question, {o} options as "A. text" lines, {oo} options as "Option A: text" lines, {i} instruction.
LAYOUT_INSTRUCTION = "Answer with the option letter from the given choices directly."
OBSERVATION_INSTRUCTION = (
    "Carefully observe the video, focusing on the order and causes of events, the movement and details of objects, "
    "as well as the actions and poses of persons. Based on these observations, choose the option letter that best answers the question.\n"
    "Based on your observations, select"
)
PROMPT_LAYOUTS = {
    "prefix": "Question: {q}\nOptions:\n{o}\n{i}",  # "Question:" / "Options:" prefixes
    "option_letter": "{q}\n{oo}\n{i}",  # "Option A: text" labels
    "start": "{i}\n{q}\n{o}",  # instruction before the question
    "middle": "{q}\n{i}\n{o}",  # instruction between question and options
    "observation": "{q}\n{o}\n" + OBSERVATION_INSTRUCTION,  # Observation-Driven Analysis; fixed text, POST_PROMPT ignored
}


def question_text(source: str, doc: dict) -> str:
    """The bare question of ``doc``, without options (lvbench bakes its "(A) ..." lines into the question)."""
    if source == "lvbench":
        return "\n".join(l for l in doc["question"].split("\n") if not re.match(r"^\([A-Z]\) ", l)).strip()
    return str(doc["question"]).strip()


def layout_prompt(source: str, doc: dict, layout: str, instruction: str | None = None) -> str | None:
    """Prompt of ``doc`` in ``layout`` (a key of PROMPT_LAYOUTS); None for items without options
    (VSI-Bench numeric, VideoMMMU open-ended), which keep their source prompt."""
    if source.startswith("video_mmmu") and doc.get("question_type") != "multiple-choice":
        return None
    _, texts = option_block(source, doc)
    if not texts:
        return None
    return PROMPT_LAYOUTS[layout].format(
        q=question_text(source, doc),
        o="\n".join(f"{_LETTERS[i]}. {t}" for i, t in enumerate(texts)),
        oo="\n".join(f"Option {_LETTERS[i]}: {t}" for i, t in enumerate(texts)),
        i=(instruction if instruction is not None else LAYOUT_INSTRUCTION).strip(),
    )


def render(variant: str, source: str, doc: dict, post_prompt: str | None = None, option_style: str | None = None) -> str:
    """Prompt text of ``doc`` (a source-dataset row) under ``variant``.

    ``post_prompt`` replaces the source's own post prompt (POST_PROMPT_KEYS) with one
    string shared by every source; this is what METAV_POST_PROMPT does at eval time.
    ``option_style`` (a key of OPTION_STYLES) re-labels the option lines of every source in
    that style; this is what METAV_OPTION_STYLE does at eval time. A key of
    PROMPT_LAYOUTS instead rebuilds the whole prompt in that layout (``post_prompt``, if given,
    is its instruction line); items without options keep the source prompt.
    """
    if option_style in PROMPT_LAYOUTS:
        text = layout_prompt(source, doc, option_style, post_prompt)
        if text is not None:
            return text
        option_style = None
    fn_path, kwargs = VARIANTS[variant][source] or VARIANTS["default"][source]
    kwargs = dict(kwargs)
    if post_prompt is not None:
        if source == "lvbench" and not post_prompt.startswith("\n"):
            post_prompt = "\n" + post_prompt  # lvbench's function joins question and post prompt without a separator
        kwargs.update({k: post_prompt for k in POST_PROMPT_KEYS[source]})
    mod, _, fn = fn_path.rpartition(".")
    text = getattr(importlib.import_module(f"lmms_eval.tasks.{mod}"), fn)(doc, kwargs)
    if option_style:
        text = restyle_options(text, source, doc, option_style)
    return text
