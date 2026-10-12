#!/usr/bin/env python
"""Meta-VideoBench (metav) score from an lmms-eval results.json.

metav score = macro average over the five source benchmarks (Video-MME, LongVideoBench,
LVBench, VideoMMMU, VSI-Bench); VideoMMMU itself is the mean of its three tracks. Every
per-task metric is mapped to [0, 100]. Works for metav runs and for their metav_frames<N>
variants (the source suffix of the subtask name is what is matched).

usage: python eval_scripts/score_metav.py [--write] <path/to/*_results.json> [...]
  --write  also store the scores in the file, under the group row (metav[_frames<N>])
           that lmms_eval leaves empty because the subtasks have different metric names:
           {"metav_score": <macro>, "<source>_score": ..., "prompt_variant": ..., "post_prompt": ...,
            "option_style": ...}. The prompt fields are read from METAV_PROMPT_VARIANT /
           METAV_POST_PROMPT / METAV_OPTION_STYLE in the environment (as the run had them).
"""

import json
import os
import sys

TASK2SOURCE = {
    "videomme": ("videomme", "videomme_perception_score,none", 1.0),
    "longvideobench_val_v": ("longvideobench", "lvb_acc,none", 100.0),
    "lvbench": ("lvbench", "lvbench_score,none", 100.0),
    "video_mmmu_perception": ("video_mmmu", "mmmu_acc,none", 100.0),
    "video_mmmu_comprehension": ("video_mmmu", "mmmu_acc,none", 100.0),
    "video_mmmu_adaptation": ("video_mmmu", "mmmu_acc,none", 100.0),
    "vsibench": ("vsibench", "vsibench_overall,none", 100.0),
}
GROUPS = ("metav",) + tuple(f"metav_frames{n}" for n in (4, 8, 16, 32, 64, 128))


def score(results_json):
    """(macro score, {source: score}) of one results.json; sources are matched by task-name suffix."""
    res = json.load(open(results_json))["results"]
    per_source = {}
    for task, vals in res.items():
        for suffix, (source, metric, scale) in TASK2SOURCE.items():
            if task.endswith(suffix) and metric in vals:
                per_source.setdefault(source, []).append(vals[metric] * scale)
    per_source = {k: sum(v) / len(v) for k, v in per_source.items()}
    meta = sum(per_source.values()) / len(per_source) if per_source else float("nan")
    return meta, per_source


def prompt_info():
    return {
        "prompt_variant": os.environ.get("METAV_PROMPT_VARIANT") or "default (per source, or the model-specific entry of the task yaml)",
        "post_prompt": os.environ.get("METAV_POST_PROMPT") or "default (per source)",
        "option_style": os.environ.get("METAV_OPTION_STYLE") or "default (per source)",
    }


def write(results_json, meta, per_source):
    d = json.load(open(results_json))
    group = next((k for k in d["results"] if k in GROUPS), "metav")
    d["results"][group] = {"alias": group, "metav_score": round(meta, 4), **{f"{k}_score": round(v, 4) for k, v in sorted(per_source.items())}, **prompt_info()}
    json.dump(d, open(results_json, "w"), indent=4, default=str)


if __name__ == "__main__":
    args = sys.argv[1:]
    do_write = "--write" in args
    paths = [a for a in args if a != "--write"]
    if not paths:
        sys.exit(__doc__)
    for p in paths:
        meta, per = score(p)
        print(p)
        for k, v in sorted(per.items()):
            print(f"  {k:16s} {v:6.2f}")
        print(f"  {'metav':16s} {meta:6.2f}  (macro avg over {len(per)} sources)")
        if do_write:
            write(p, meta, per)
