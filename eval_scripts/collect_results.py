#!/usr/bin/env python
"""Collect every finished sweep run into one table.

    python eval_scripts/collect_results.py                      # logs/sweep_params logs/sweep_frames logs/sweep_vllm_fix
    python eval_scripts/collect_results.py logs/sweep_params --csv runs.csv --md runs.md

Walks the given roots for the config.json that eval_scripts/lib.sh writes next to every run,
pairs it with the *_results.json lmms_eval wrote below it and extracts one score per run:
  * a plain task: its first non-stderr metric (e.g. videomme_perception_score);
  * a group: the group row if lmms_eval aggregated one (mvbench), else the mean over its subtasks;
  * metav_*: the Meta-VideoBench macro score (eval_scripts/score_metav.py), plus the source scores.
Prints one table per (sweep, task, model, experiment); --csv / --md write the same to files.
Runs without a results file are listed as missing so a partial sweep is visible.
"""

import argparse
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_metav import score as metav_score  # noqa: E402

DEFAULT_ROOTS = ["logs/sweep_params", "logs/sweep_frames", "logs/sweep_vllm_fix"]
KNOBS = ["backend", "nframes", "fps", "max_frames", "max_pixels", "max_patches", "batch_size", "video_reader", "frame_sampler", "temperature", "batch_invariant", "seed"]


def primary_metric(results, task):
    """(metric name, value) for `task` in a results.json 'results' dict."""
    if task.startswith("metav"):
        return "metav_score", None  # filled by the caller
    vals = results.get(task)
    if isinstance(vals, dict):
        for k, v in vals.items():
            if k.endswith(",none") and "stderr" not in k and isinstance(v, (int, float)):
                return k[: -len(",none")], float(v)
    # group without an aggregated row: mean of the subtasks' first metric
    subs = []
    for t, vals in results.items():
        if t == task or not isinstance(vals, dict):
            continue
        for k, v in vals.items():
            if k.endswith(",none") and "stderr" not in k and isinstance(v, (int, float)):
                subs.append(float(v))
                break
    return ("mean_of_subtasks", sum(subs) / len(subs)) if subs else ("", None)


def collect(roots):
    rows = []
    for root in roots:
        for cfg_path in sorted(glob.glob(os.path.join(root, "**", "config.json"), recursive=True)):
            run_dir = os.path.dirname(cfg_path)
            cfg = json.load(open(cfg_path))
            row = {"sweep": os.path.relpath(root), "run_dir": run_dir, "run": os.path.basename(run_dir), "model": cfg.get("model", ""), "family": cfg.get("family", ""), "exp": cfg.get("exp", ""), "task": cfg.get("task", "")}
            row.update({k: cfg.get(k, "") for k in KNOBS})
            res_files = sorted(glob.glob(os.path.join(run_dir, "*", "*_results.json")))
            if not res_files:
                row.update({"metric": "", "value": "", "status": "missing"})
                rows.append(row)
                continue
            res = json.load(open(res_files[-1]))["results"]
            metric, value = primary_metric(res, row["task"])
            if metric == "metav_score":
                value, per_source = metav_score(res_files[-1])
                row.update({f"{k}_score": round(v, 3) for k, v in per_source.items()})
            row.update({"metric": metric, "value": round(value, 4) if value is not None else "", "status": "ok" if value is not None else "no metric"})
            rows.append(row)
    return rows


def tables(rows):
    out = []
    groups = {}
    for r in rows:
        groups.setdefault((r["sweep"], r["task"].split("_frames")[0] if not r["task"].startswith("metav") else r["task"], r["model"], r["exp"]), []).append(r)
    for (sweep, task, model, exp), rs in sorted(groups.items()):
        out.append(f"\n## {sweep} | {task} | {model} | {exp}\n")
        cols = [k for k in KNOBS if any(r.get(k) not in ("", None) for r in rs)]
        out.append("| run | " + " | ".join(cols) + " | metric | value |")
        out.append("|---|" + "---|" * len(cols) + "---|---|")
        for r in sorted(rs, key=lambda r: r["run"]):
            out.append(f"| {r['run']} | " + " | ".join(str(r.get(c, "")) for c in cols) + f" | {r['metric']} | {r['value'] if r['status'] == 'ok' else r['status']} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=DEFAULT_ROOTS)
    ap.add_argument("--csv")
    ap.add_argument("--md")
    a = ap.parse_args()
    rows = collect([r for r in a.roots if os.path.isdir(r)])
    if not rows:
        sys.exit("no runs found (no config.json under " + ", ".join(a.roots) + ")")
    text = tables(rows)
    print(text)
    if a.csv:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("sweep", "task", "model", "exp", "run"), k))
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.csv} ({len(rows)} runs)")
    if a.md:
        open(a.md, "w").write(text + "\n")
        print(f"wrote {a.md}")


if __name__ == "__main__":
    main()
