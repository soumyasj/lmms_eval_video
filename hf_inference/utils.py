"""Utilities for the standalone Qwen2.5-VL VideoMME eval.

Everything that is NOT model loading / inference lives here: prompt styles,
prompt building, CLI parsing, seeding, video-path resolution, answer
extraction, and results aggregation / reporting.

The model-specific code (loading + generation) lives in `qwenvl.py`.
"""

import argparse
import importlib.util
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch

# Repo root (two levels up from this file: hf_inference/ -> repo).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reuse lmms-eval's robust MCQ answer extractor. Load the module directly by
# file path so we don't trigger the heavy `lmms_eval.tasks` package __init__.
_MCQ_PATH = os.path.join(REPO_ROOT, "lmms_eval", "tasks", "_task_utils", "mcq_extract.py")
_spec = importlib.util.spec_from_file_location("mcq_extract", _MCQ_PATH)
_mcq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mcq)
extract_mcq_answer = _mcq.extract_mcq_answer

# Two selectable VideoMME prompt styles (see --prompt-style):
#
#   "lmms"       Mirrors exactly what `lmms_eval` feeds `qwen2_5_vl`:
#                  - text: tasks/videomme/utils.py `videomme_doc_to_text` with
#                    the `default` lmms_eval_specific_kwargs (post_prompt below).
#                  - system prompt: models/simple/qwen2_5_vl.py default.
#
#   "standalone" The original frames-only prompt used by this script (no
#                subtitles mention, "The best answer is:" post_prompt, no
#                system message).
PROMPT_STYLES = {
    "lmms": {
        "option_prompt": (
            "Select the best answer to the following multiple-choice question "
            "based on the video and the subtitles. Respond with only the letter "
            "(A, B, C, or D) of the correct option."
        ),
        "post_prompt": "\nAnswer with the option's letter from the given choices directly.",
        "system_prompt": "You are a helpful assistant.",
    },
    "standalone": {
        "option_prompt": (
            "Select the best answer to the following multiple-choice question "
            "based on the video. Respond with only the letter (A, B, C, or D) "
            "of the correct option."
        ),
        "post_prompt": "The best answer is:",
        "system_prompt": None,
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic for reproducible runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_prompt(doc, style):
    """Assemble the multiple-choice text prompt for the given prompt style."""
    question = doc["question"]
    options = "\n".join(doc["options"])
    cfg = PROMPT_STYLES[style]
    return f"{cfg['option_prompt']}\n{question}\n{options}\n{cfg['post_prompt']}"


def system_prompt_for(style):
    """Return the system prompt for a style, or None if it uses no system turn."""
    return PROMPT_STYLES[style]["system_prompt"]


def resolve_video_path(video_root, video_id):
    """Find <video_root>/<video_id>.{mp4,MP4,mkv}; return None if none exist."""
    for ext in (".mp4", ".MP4", ".mkv"):
        candidate = os.path.join(video_root, video_id + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def score_answer(raw_pred, gt_ans):
    """Extract the predicted letter and compare to the gold answer.

    Returns (pred_letter, is_correct).
    """
    pred_ans = extract_mcq_answer(raw_pred, choices=["A", "B", "C", "D"])
    is_correct = pred_ans.lower() == gt_ans.lower()
    return pred_ans, is_correct


def build_summary(args, num_questions, evaluated, correct, by_group):
    """Build the JSON-serializable run summary dict."""
    overall_acc = correct / evaluated if evaluated else 0.0
    return {
        "model": args.model,
        "prompt_style": args.prompt_style,
        "sampling": (
            {"nframes": args.nframes} if args.nframes is not None else {"fps": args.fps}
        ),
        "max_pixels": args.max_pixels,
        "num_questions": num_questions,
        "evaluated": evaluated,
        "correct": correct,
        "overall_accuracy": overall_acc,
        "breakdown": {
            k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0.0}
            for k, v in sorted(by_group.items())
        },
    }


def new_group_counter():
    """Fresh accumulator for per-category correct/total counts."""
    return defaultdict(lambda: {"correct": 0, "total": 0})


def update_groups(by_group, doc, is_correct):
    """Tally a result into the duration / domain / task_type buckets."""
    for key in (
        f"duration/{doc.get('duration')}",
        f"domain/{doc.get('domain')}",
        f"task/{doc.get('task_type')}",
    ):
        by_group[key]["total"] += 1
        by_group[key]["correct"] += int(is_correct)


def print_summary(summary):
    print("\n===== Summary =====")
    print(
        f"Overall accuracy: {summary['overall_accuracy']:.4f} "
        f"({summary['correct']}/{summary['evaluated']})"
    )
    for k, v in summary["breakdown"].items():
        print(f"  {k:40s} {v['accuracy']:.4f} ({v['correct']}/{v['total']})")


def save_results(output, summary, records):
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(f"\nWrote results to {output}")


def load_docs(json_path, limit=None):
    with open(json_path, "r") as f:
        docs = json.load(f)
    if limit is not None:
        docs = docs[:limit]
    return docs


def parse_args():
    default_video_root = os.path.join(
        os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/")),
        "videomme",
        "data",
    )
    parser = argparse.ArgumentParser(description="Qwen2.5-VL VideoMME evaluation")
    parser.add_argument(
        "--json",
        default=os.path.join(REPO_ROOT, "random_debug", "videomme_subset_200.json"),
        help="Path to the VideoMME subset JSON file.",
    )
    parser.add_argument(
        "--video-root",
        default=default_video_root,
        help="Directory containing <videoID>.mp4 files.",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--prompt-style",
        choices=list(PROMPT_STYLES.keys()),
        default="lmms",
        help="'lmms' mirrors the lmms_eval videomme/qwen2_5_vl prompt (default); "
        "'standalone' uses this script's original frames-only prompt.",
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument(
        "--nframes",
        type=int,
        default=None,
        help="Fixed number of frames to sample per video. Mutually exclusive "
        "with --fps; when set, --fps is ignored.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=360 * 420,
        help="Max pixels per frame passed to the vision processor.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        help="Use flash_attention_2 (requires flash-attn installed).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only evaluate the first N questions."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write per-question predictions + summary (JSON).",
    )
    return parser.parse_args()
