#!/usr/bin/env python
"""Precomputed-frame variants of a video task (exp7: video vs frames).

For a task T -- or a group such as ``mvbench`` or ``metav`` -- this script

  1. loads T exactly as lmms_eval evaluates it (same YAML, same process_docs, same
     doc_to_visual, same HF cache), so the rows and the video files are the ones a raw-video
     run sees;
  2. extracts MASTER (default 128) uniformly spaced JPEG frames from every distinct video into
         data/frames/videos/<video key>/frame_000.jpg ... frame_127.jpg
     (shared between tasks that use the same file, e.g. videomme and metav_videomme);
  3. writes, for every N in --nframes, ``data/frames/<T>_frames<N>.parquet``: the rows of T plus a
     ``frame_paths`` column holding N of the master frames (uniform subsample);
  4. writes the task YAML ``lmms_eval/tasks/<dir of T>/<T>_frames<N>.yaml``. It *includes* T's own
     YAML and only swaps the dataset for the parquet and doc_to_visual for
     ``lmms_eval.tasks.precomputed_frames.doc_to_visual``, so prompt, metric and generation
     kwargs are identical to T by construction. For a group G a ``<G>_frames<N>`` group is
     written as well.

Frame selection: master indices = np.linspace(0, total-1, 128, dtype=int) (unique, last frame
kept, padded for clips shorter than 128 frames), then N of the 128 with the same linspace rule.
This is the uniform rule of the hf wrappers and the rule the earlier exp7 data were built with.
A video that is missing or unreadable is reported and its rows are dropped from the parquet.

Usage (from the repo root, inside the lmms_eval conda env):
    python eval_scripts/make_frames_task.py --task videomme
    python eval_scripts/make_frames_task.py --task metav --workers 16
    python eval_scripts/make_frames_task.py --task mvbench,vsibench --nframes 4,8,16,32,64,128
    python eval_scripts/make_frames_task.py --task vsibench --limit 3      # smoke test -> data/frames/smoke/, no YAML
Afterwards ``--tasks videomme_frames32`` works like any other task, and sweep_params.sh (exp7)
and sweep_frames.sh (exp7) pick the variants up automatically.

Extraction is CPU-only. Long benchmarks (Video-MME, LongVideoBench, LVBench) take hours on one
node; run it as a batch job with --workers set to the allocated cores. Re-running is
incremental: videos whose 128 frames already exist are skipped.
"""

import argparse
import multiprocessing as mp
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)  # the checkout's lmms_eval, never a site-packages copy
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("LMMS_EVAL_DATASETS_CACHE", os.path.join(os.environ["HF_HOME"], "datasets"))

import numpy as np
from PIL import Image
from tqdm import tqdm

DEFAULT_NFRAMES = "4,8,16,32,64,128"
IMAGE_EXT = (".jpg", ".jpeg", ".png")


# ---- frame selection ---------------------------------------------------------------------
def master_indices(total: int, n: int) -> np.ndarray:
    """n frame indices out of `total`: uniform, unique, last frame kept, padded for short clips."""
    if total <= 0:
        raise ValueError("video has no frames")
    idx = np.unique(np.linspace(0, total - 1, n, dtype=int))
    if total - 1 not in idx:
        idx = np.append(idx, total - 1)
    if len(idx) < n:
        idx = np.sort(np.concatenate([idx, np.full(n - len(idx), idx[-1])]))
    return idx[:n]


def subsample(n_master: int, n: int) -> np.ndarray:
    return np.unique(np.linspace(0, n_master - 1, n, dtype=int))


# ---- extraction (worker process) ----------------------------------------------------------
def extract_job(job):
    key, kind, src, out_dir, master, chunk, quality = job
    paths = [os.path.join(out_dir, f"frame_{i:03d}.jpg") for i in range(master)]
    if all(os.path.exists(p) for p in paths):
        return key, True, "cached"
    try:
        os.makedirs(out_dir, exist_ok=True)
        if kind == "video":
            import decord

            vr = decord.VideoReader(src, num_threads=1)
            idx = master_indices(len(vr), master)
            for s in range(0, master, chunk):  # small batches: 128 frames of an hour-long 1080p clip is ~800 MB
                frames = vr.get_batch(idx[s : s + chunk].tolist()).asnumpy()
                for o, fr in enumerate(frames):
                    Image.fromarray(fr).save(paths[s + o], "JPEG", quality=quality)
                del frames
        elif kind == "frame_dir":  # benchmarks that ship a folder of frames instead of a video
            files = sorted(f for f in os.listdir(src) if f.lower().endswith(IMAGE_EXT))
            idx = master_indices(len(files), master)
            for i, j in enumerate(idx):
                Image.open(os.path.join(src, files[j])).convert("RGB").save(paths[i], "JPEG", quality=quality)
        else:
            raise ValueError(f"unknown source kind {kind}")
        return key, True, "ok"
    except Exception as e:  # one broken file must not sink the whole task
        return key, False, f"{type(e).__name__}: {e}"


def save_images(images, out_dir, master, quality):
    """doc_to_visual already returned PIL frames (e.g. MVBench episodic_reasoning): subsample them in-process."""
    paths = [os.path.join(out_dir, f"frame_{i:03d}.jpg") for i in range(master)]
    if all(os.path.exists(p) for p in paths):
        return True, "cached"
    try:
        os.makedirs(out_dir, exist_ok=True)
        for i, j in enumerate(master_indices(len(images), master)):
            images[j].convert("RGB").save(paths[i], "JPEG", quality=quality)
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ---- task plumbing ------------------------------------------------------------------------
def leaf_tasks(task_dict):
    """[(name, ConfigurableTask)] below a get_task_dict() result (groups are nested dicts)."""
    out = []
    for k, v in task_dict.items():
        if isinstance(v, dict):
            out.extend(leaf_tasks(v))
        elif v is not None:
            out.append((str(getattr(k, "group", k)), v))
    return out


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", s).strip("_")


def video_key(path: str) -> str:
    """Directory name for the frames of one video: its path below HF_HOME (or the absolute path)."""
    hf = os.path.abspath(os.path.expanduser(os.environ["HF_HOME"]))
    ap = os.path.abspath(path)
    rel = os.path.relpath(ap, hf) if ap.startswith(hf + os.sep) else ap.lstrip(os.sep)
    if os.path.isfile(ap):
        rel = os.path.splitext(rel)[0]
    return sanitize(rel)


def classify(visuals, doc, i, task_name):
    """What doc_to_visual returned -> (key, kind, source). kind: video | frame_dir | images."""
    if isinstance(visuals, (str, os.PathLike)):
        visuals = [visuals]
    if not isinstance(visuals, (list, tuple)) or not visuals:
        raise ValueError(f"doc_to_visual returned {type(visuals).__name__}, not a list")
    paths = [os.fspath(v) for v in visuals if isinstance(v, (str, os.PathLike))]
    if paths:
        src = paths[0]  # every video task here returns [video_path]; extra entries (subtitles, images) are not frames
        if os.path.isdir(src):
            return video_key(src), "frame_dir", src
        if os.path.isfile(src):
            return video_key(src), "video", src
        raise FileNotFoundError(src)
    if all(isinstance(v, Image.Image) for v in visuals):
        ident = next((str(doc[c]) for c in ("video", "video_path", "videoID", "video_id", "id") if c in doc and doc[c] is not None), f"doc{i}")
        return sanitize(f"{task_name}__{ident}"), "images", list(visuals)
    raise ValueError("doc_to_visual returned neither file paths nor PIL images")


def process_leaf(name, task, args):
    split = task.config.test_split if task.has_test_docs() else task.config.validation_split
    docs = task.test_docs() if task.has_test_docs() else task.validation_docs()
    if args.limit:
        docs = docs.select(range(min(args.limit, len(docs))))
    print(f"\n### {name}: {len(docs)} rows in split '{split}'")

    specs, keys = {}, []
    for i, doc in enumerate(tqdm(docs, desc=f"{name}: resolving videos", unit="doc")):
        try:
            key, kind, src = classify(task.doc_to_visual(doc), doc, i, name)
        except KeyboardInterrupt:
            raise
        except BaseException as e:  # BaseException: videomme's doc_to_visual calls sys.exit() on a missing file
            tqdm.write(f"  WARNING: {name} row {i}: {e}")
            keys.append(None)
            continue
        keys.append(key)
        specs.setdefault(key, (kind, src))
    print(f"  {len(specs)} distinct videos, {sum(k is None for k in keys)} rows without a readable source")
    if args.dry_run:
        return split, {}, sum(k is None for k in keys)

    frames_root = os.path.join(args.out_root, "videos")
    status, jobs = {}, []
    for key, (kind, src) in specs.items():
        out_dir = os.path.join(frames_root, key)
        if kind == "images":
            status[key] = save_images(src, out_dir, args.master_frames, args.jpeg_quality)
        else:
            jobs.append((key, kind, src, out_dir, args.master_frames, args.decode_chunk, args.jpeg_quality))
    if jobs:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for key, ok, msg in tqdm(pool.imap_unordered(extract_job, jobs), total=len(jobs), desc=f"{name}: extracting", unit="video"):
                status[key] = (ok, msg)
                if not ok:
                    tqdm.write(f"  WARNING: {key}: {msg}")
    ok_keys = {k for k, (ok, _) in status.items() if ok}
    keep = [i for i, k in enumerate(keys) if k in ok_keys]
    dropped = len(keys) - len(keep)
    if dropped:
        print(f"  WARNING: {name}: {dropped} of {len(keys)} rows dropped (video missing or unreadable)")
    if not keep:
        raise SystemExit(f"{name}: no video could be read; nothing written")

    parquets = {}
    kept = docs.select(keep)
    if "frame_paths" in kept.column_names:
        kept = kept.remove_columns("frame_paths")
    for n in args.nframes:
        sel = subsample(args.master_frames, n)
        lists = [[os.path.join(frames_root, keys[i], f"frame_{j:03d}.jpg") for j in sel] for i in keep]
        ds = kept.add_column("frame_paths", lists)
        out = os.path.join(args.out_root, f"{name}_frames{n}.parquet")
        ds.to_parquet(out)
        parquets[n] = out
        print(f"  wrote {out} ({len(ds)} rows, {len(sel)} frames each)")
    return split, parquets, dropped


def write_task_yaml(name, yaml_path, split, n, parquet):
    out = os.path.join(os.path.dirname(yaml_path), f"{name}_frames{n}.yaml")
    with open(out, "w") as f:
        f.write(
            f"# {name} with {n} precomputed frames per video (exp7: video vs frames).\n"
            f"# Generated by eval_scripts/make_frames_task.py from {os.path.basename(yaml_path)}; regenerate instead of editing.\n"
            f"include: {os.path.basename(yaml_path)}\n"
            f"task: {name}_frames{n}\n"
            "dataset_path: parquet\n"
            "dataset_name: null\n"
            "dataset_kwargs:\n"
            "  data_files:\n"
            f"    {split}: {parquet}\n"
            "doc_to_visual: !function lmms_eval.tasks.precomputed_frames.doc_to_visual\n"
        )
    return out


def write_group_yaml(group, yaml_path, n, members):
    out = os.path.join(os.path.dirname(yaml_path), f"{group}_frames{n}.yaml")
    with open(out, "w") as f:
        f.write(f"# {group} with {n} precomputed frames per video; generated by eval_scripts/make_frames_task.py\n")
        f.write(f"group: {group}_frames{n}\ntask:\n" + "".join(f"- {m}\n" for m in members))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="task or group name(s), comma-separated (videomme, mvbench, metav, ...)")
    ap.add_argument("--nframes", default=DEFAULT_NFRAMES, help=f"comma-separated frame counts to write (default {DEFAULT_NFRAMES})")
    ap.add_argument("--master_frames", type=int, default=128, help="frames extracted per video; every N is a subsample of these")
    ap.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)), help="decoder processes")
    ap.add_argument("--decode_chunk", type=int, default=8, help="frames decoded per get_batch call (memory cap)")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    ap.add_argument("--out_root", default="data/frames", help="where parquets and frames go (relative to the repo root)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N rows only, into <out_root>/smoke/, no YAML")
    ap.add_argument("--no_yaml", action="store_true", help="write parquets only")
    ap.add_argument("--dry_run", action="store_true", help="resolve and count the videos, extract nothing")
    args = ap.parse_args()
    args.nframes = sorted({int(x) for x in args.nframes.split(",") if x})
    if not args.nframes or max(args.nframes) > args.master_frames:
        sys.exit(f"--nframes must be non-empty and <= --master_frames ({args.master_frames})")
    write_yaml = not (args.no_yaml or args.limit or args.dry_run)
    if args.limit:
        args.out_root = os.path.join(args.out_root, "smoke")
        print(f"--limit set: writing to {args.out_root}, no task YAML")
    os.makedirs(args.out_root, exist_ok=True)

    from lmms_eval.tasks import TaskManager, get_task_dict

    tm = TaskManager()
    summary = []
    for task_name in [t for t in args.task.split(",") if t]:
        if task_name not in tm.task_index:
            sys.exit(f"unknown task {task_name!r} (python -m lmms_eval --tasks list)")
        entry = tm.task_index[task_name]
        leaves = leaf_tasks(get_task_dict([task_name], tm))
        members = {n: [] for n in args.nframes}
        for name, t in leaves:
            split, parquets, dropped = process_leaf(name, t, args)
            for n, pq in parquets.items():
                members[n].append(f"{name}_frames{n}")
                if write_yaml:
                    summary.append(write_task_yaml(name, tm.task_index[name]["yaml_path"], split, n, pq))
        if write_yaml and entry["type"] == "group":
            for n in args.nframes:
                summary.append(write_group_yaml(task_name, entry["yaml_path"], n, members[n]))
    if summary:
        print("\nwrote task YAMLs:")
        for p in summary:
            print("  " + os.path.relpath(p, REPO_ROOT))
        print("\nrun e.g.:  --tasks " + os.path.basename(summary[-1])[: -len(".yaml")])


if __name__ == "__main__":
    main()
