#!/usr/bin/env python
"""Run a standardized video evaluation from ONE YAML file.

    python eval_scripts/run_eval.py configs/example.yaml            # run
    python eval_scripts/run_eval.py configs/example.yaml --dry_run  # print the resolved runs and commands
    python eval_scripts/run_eval.py configs/example.yaml --set nframes=32 --set tasks=videomme

The YAML holds every setting that changes what the model sees and how it is decoded (see
configs/example.yaml for all keys and their defaults). Four keys may be lists and are toggled
into a grid of runs: `tasks`, `model`, `nframes`, `backend`; `vllm_fix` may be a list too.
Everything else is fixed for the whole grid.

For every run this script
  1. resolves the model alias -> HF id -> runner family (same tables as eval_scripts/lib.sh),
  2. checks that the family's wrapper can honour every requested setting (and refuses otherwise:
     a run must never silently use an uncontrolled setting),
  3. writes <output_root>/<name>/<task>/<model>/<run tag>/config.json (read by
     eval_scripts/collect_results.py) and evaluation_card.md (the full resolved configuration:
     what exactly is fed to the model, plus the exact command),
  4. calls the family's runner in eval_scripts/runners/, which builds the `python -m lmms_eval`
     command. A run dir that already holds a *_results.json is skipped (resume: true).
"""

import argparse
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("pyyaml is required: pip install pyyaml")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RUNNERS = os.path.join("eval_scripts", "runners")

# ---------------------------------------------------------------------------------------------
# keys, defaults, allowed values
# ---------------------------------------------------------------------------------------------
TOGGLES = ("tasks", "model", "nframes", "backend", "vllm_fix")
DEFAULTS = {
    # --- toggles (scalar or list) ---
    "tasks": "metav",
    "model": "qwen25vl_7b",
    "nframes": 32,
    "backend": "hf",  # hf | vllm | hf_tiled (InternVL only)
    "vllm_fix": False,  # vllm backend only: forward the .mp4 to vLLM instead of N images (Qwen2.5-VL only)
    # --- fixed for the grid ---
    "name": None,  # run-set name; default: the YAML file name
    "family": None,  # qwen25vl | qwen3vl | qwen35 | internvl | glm | gemma; inferred from the model id
    "input": "video",  # video (.mp4, the wrapper samples frames) | frames (<task>_frames<N>: pre-extracted JPEGs)
    "frame_sampling": "uniform",  # uniform (exactly nframes) | fps (fps rate, capped at max_frames)
    "fps": 1,
    "max_frames": None,  # fps mode cap; default: nframes
    "frame_sampler": "uniform",  # which positions: uniform | random (Qwen2.5-VL hf only; needs seed)
    "video_reader": "torchcodec",  # decord | torchvision | torchcodec
    "max_pixels": None,  # per-frame pixel budget; default: the family's protocol value
    "min_pixels": None,
    "max_patches": 1,  # InternVL: 448x448 tiles per frame
    "min_patches": 1,
    "batch_size": 1,
    "batch_invariant": False,  # vllm only: VLLM_BATCH_INVARIANT=1 (needs vllm >= 0.11.1)
    "temperature": None,  # None = the task yaml's generation_kwargs (0 for every MCQ task here)
    "max_new_tokens": None,  # None = the task yaml's generation_kwargs
    "enable_thinking": False,  # Qwen3.5 / Qwen3-VL thinking checkpoints
    "attn": None,  # sdpa | flash_attention_2 | eager (hf)
    "seed": 22,
    "limit": None,  # int or 0-1 fraction; None = the whole task
    "num_gpus": 1,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 32768,
    "max_num_seqs": 8,
    "env": None,  # bin dir of the conda/venv to run in; None = the current environment
    "output_root": "logs/runs",
    "resume": True,
    "include_path": None,
    "extra_model_args": None,  # appended verbatim to --model_args (Qwen2.5-VL hf only)
    "prompt": {},  # metav only: variant | post_prompt | option_style | layout (METAV_* env vars)
}
CHOICES = {
    "backend": {"hf", "vllm", "hf_tiled"},
    "input": {"video", "frames"},
    "frame_sampling": {"uniform", "fps"},
    "frame_sampler": {"uniform", "random"},
    "video_reader": {"decord", "torchvision", "torchcodec", "pyav"},
}
PROMPT_KEYS = {"variant": "METAV_PROMPT_VARIANT", "post_prompt": "METAV_POST_PROMPT", "option_style": "METAV_OPTION_STYLE", "layout": "METAV_LAYOUT"}
FAMILIES = ("qwen25vl", "qwen3vl", "qwen35", "internvl", "glm", "gemma")

# ---------------------------------------------------------------------------------------------
# model / family tables (kept in sync with eval_scripts/lib.sh)
# ---------------------------------------------------------------------------------------------


def resolve_model(m):
    """alias -> HF id (anything that is not an alias is returned unchanged)."""
    low = m.lower()
    size = m.split("_", 1)[1] if "_" in m else ""
    if re.fullmatch(r"qwen25vl_\d+b", low):
        return f"Qwen/Qwen2.5-VL-{size[:-1]}B-Instruct"
    if re.fullmatch(r"qwen3vl_\d+b", low):
        return f"Qwen/Qwen3-VL-{size[:-1]}B-Instruct"
    if low.startswith("qwen35_"):
        return f"Qwen/Qwen3.5-{size.upper().replace('_', '-')}"
    if low.startswith("internvl35_"):
        return f"OpenGVLab/InternVL3_5-{size.upper().replace('_', '-')}-HF"
    if low.startswith("internvl3_"):
        return f"OpenGVLab/InternVL3-{size.upper().replace('_', '-')}-hf"
    return {"glm46v_flash": "zai-org/GLM-4.6V-Flash", "glm46v": "zai-org/GLM-4.6V", "glm45v": "zai-org/GLM-4.5V", "glm41v_9b": "zai-org/GLM-4.1V-9B-Thinking"}.get(low) or (
        f"google/gemma-3-{size[:-1]}b-it" if re.fullmatch(r"gemma3_\d+b", low) else m
    )


def detect_family(hf_id):
    low = hf_id.lower()
    for pat, fam in [("qwen2.5-vl|qwen2_5_vl|qwen2.5vl|video-r1", "qwen25vl"), ("qwen3-vl|qwen3_vl|qwen3vl", "qwen3vl"), ("qwen3.5|qwen3_5", "qwen35"), ("internvl", "internvl"), ("glm", "glm"), ("gemma", "gemma")]:
        if re.search(pat, low):
            return fam
    return None


FAM = {
    #            runner                                   default max_pixels  tokens per frame
    "qwen25vl": ("run_video_qa_hf_multigpu.sh", 602112, "max_pixels / 784 per frame, frames merged in pairs (temporal patch 2)"),
    "qwen3vl": ("run_video_qa_qwen3vl_multigpu.sh", 524288, "max_pixels / 1024 per frame, frames merged in pairs; '<t seconds>' timestamp text before every pair"),
    "qwen35": ("run_video_qa_qwen3_5_multigpu.sh", 131072, "max_pixels / 1024 per frame, frames merged in pairs; timestamp text as Qwen3-VL"),
    "internvl": ("run_video_qa_internvl_protocol.sh", None, "256 tokens per 448x448 tile; max_patches tiles per frame"),
    "glm": ("run_video_qa_glm_protocol.sh", 401408, "max_pixels / 784 per frame (vllm)"),
    "gemma": ("run_video_qa_gemma_protocol.sh", 1605632, "256 tokens per frame (SigLIP 896x896), max_pixels has no effect"),
}
LMMS_MODEL = {("qwen25vl", "hf"): "qwen2_5_vl", ("qwen3vl", "hf"): "qwen3_vl", ("qwen35", "hf"): "qwen3_5", ("internvl", "hf"): "internvl_hf", ("internvl", "hf_tiled"): "internvl_hf_tiled", ("glm", "hf"): "glm4v", ("gemma", "hf"): "gemma3"}


def fam_has(fam, cap):
    """what a family's wrappers can vary; mirrors fam_has in lib.sh."""
    table = {
        "qwen25vl": {"vllm", "hf_video", "fps", "video_reader", "frame_sampler", "hf_batch", "max_pixels"},
        "qwen3vl": {"vllm", "hf_video", "fps", "video_reader", "hf_batch", "max_pixels"},
        "qwen35": {"vllm", "hf_video", "fps", "video_reader", "hf_batch", "max_pixels"},
        "internvl": {"vllm", "hf_video"},
        "glm": {"vllm", "hf_batch", "max_pixels"},
        "gemma": {"hf_batch"},
    }
    return cap in table[fam]


# ---------------------------------------------------------------------------------------------
_TASK_NAMES = None


def has_task(name):
    """is the task registered in-tree? (task/group names at the top level of lmms_eval/tasks/**/*.yaml)"""
    global _TASK_NAMES
    if _TASK_NAMES is None:
        _TASK_NAMES = set()
        for dp, _, fs in os.walk(os.path.join(ROOT, "lmms_eval", "tasks")):
            for f in fs:
                if f.endswith(".yaml"):
                    for line in open(os.path.join(dp, f), errors="ignore"):
                        m = re.match(r"^(task|group):\s*[\"']?([A-Za-z0-9_.-]+)", line)
                        if m:
                            _TASK_NAMES.add(m.group(2))
    return name in _TASK_NAMES


def as_list(v):
    return list(v) if isinstance(v, (list, tuple)) else [v]


def load_config(path, overrides):
    cfg = yaml.safe_load(open(path)) or {}
    if not isinstance(cfg, dict):
        sys.exit(f"{path}: the config must be a mapping of key: value")
    for kv in overrides:
        k, _, v = kv.partition("=")
        cfg[k] = yaml.safe_load(v)
    unknown = sorted(set(cfg) - set(DEFAULTS))
    if unknown:
        sys.exit(f"{path}: unknown key(s) {unknown}; valid keys: {', '.join(sorted(DEFAULTS))}")
    full = dict(DEFAULTS)
    full.update({k: v for k, v in cfg.items() if v is not None})
    for k, allowed in CHOICES.items():
        for v in as_list(full[k]):
            if v not in allowed:
                sys.exit(f"{path}: {k}={v!r}; allowed: {sorted(allowed)}")
    bad = sorted(set(full["prompt"] or {}) - set(PROMPT_KEYS))
    if bad:
        sys.exit(f"{path}: prompt.{bad[0]} is not a prompt key; allowed: {sorted(PROMPT_KEYS)}")
    for k in set(DEFAULTS) - set(TOGGLES):
        if isinstance(full[k], list):
            sys.exit(f"{path}: only {TOGGLES} may be lists; {k} is a list")
    full["name"] = full["name"] or os.path.splitext(os.path.basename(path))[0]
    return full


# ---------------------------------------------------------------------------------------------
class Run:
    """one resolved (task, model, nframes, backend, vllm_fix) point of the grid."""

    def __init__(self, cfg, task, model, nframes, backend, vllm_fix):
        self.cfg, self.nframes, self.backend, self.vllm_fix = cfg, int(nframes), backend, bool(vllm_fix)
        self.model = resolve_model(str(model))
        self.fam = cfg["family"] or detect_family(self.model)
        if self.fam not in FAMILIES:
            sys.exit(f"cannot infer the runner family of '{self.model}'; set family: one of {FAMILIES}")
        self.task = f"{task}_frames{self.nframes}" if cfg["input"] == "frames" else task
        self.max_pixels = cfg["max_pixels"] or FAM[self.fam][1]
        self.bi = bool(cfg["batch_invariant"]) and backend == "vllm"  # batch invariance is a vLLM mode; ignored on hf
        self.problems = []
        self._check()

    # ---- capability checks: refuse settings the wrapper would silently ignore ----------------
    def _check(self):
        c, f, b = self.cfg, self.fam, self.backend
        if not has_task(self.task):
            self.problems.append(f"task '{self.task}' is not registered under lmms_eval/tasks" + (" (generate it: python eval_scripts/make_frames_task.py --task " + self.task.split("_frames")[0] + ")" if "_frames" in self.task else ""))
        if b == "vllm" and not fam_has(f, "vllm"):
            self.problems.append(f"{f} has no vllm backend")
        if b == "hf_tiled" and f != "internvl":
            self.problems.append("backend hf_tiled exists only for InternVL")
        if b != "vllm" and c["input"] == "video" and not fam_has(f, "hf_video"):
            self.problems.append(f"the {f} hf wrapper takes frame lists only: use input: frames (or backend: vllm)")
        if self.vllm_fix and (b != "vllm" or f != "qwen25vl"):
            self.problems.append("vllm_fix needs backend: vllm and a Qwen2.5-VL model (the plugin sets Qwen-VL processor kwargs)")
        if c["frame_sampling"] == "fps":
            if b == "vllm" or self.vllm_fix:
                self.problems.append("fps sampling is not available on vllm (it samples a frame count); use frame_sampling: uniform")
            elif not fam_has(f, "fps"):
                self.problems.append(f"{f}: no rate-based (fps) sampling on its hf path")
        if c["frame_sampler"] == "random" and not (fam_has(f, "frame_sampler") and b == "hf"):
            self.problems.append("frame_sampler: random exists only in the Qwen2.5-VL hf wrapper")
        if c["batch_size"] != 1 and b == "hf" and not fam_has(f, "hf_batch"):
            self.problems.append(f"{f} hf wrapper asserts batch_size == 1")
        if self.bi and f == "qwen35":
            self.problems.append("Qwen3.5 cannot run batch-invariant (Gated DeltaNet layers are unsupported)")
        if c["video_reader"] == "pyav" and b != "hf_tiled":
            self.problems.append("video_reader pyav exists only on the InternVL hf_tiled backend")
        if b == "hf_tiled" and c["video_reader"] not in ("decord", "pyav"):
            self.problems.append("the InternVL hf_tiled backend decodes with decord or pyav; set video_reader accordingly")
        if c["max_pixels"] and not fam_has(f, "max_pixels"):
            self.problems.append(f"{f}: the per-frame pixel budget is not a knob (InternVL: max_patches; Gemma: fixed 256 tokens/frame)")
        if f != "internvl" and (c["max_patches"] != 1 or c["min_patches"] != 1):
            self.problems.append("max_patches / min_patches are InternVL settings")
        if b == "hf" and f == "internvl" and c["max_patches"] != 1 and c["input"] == "video":
            self.problems.append("InternVL's stock video processor ignores max_patches on raw video; use backend: hf_tiled for tiles > 1")
        if c["prompt"] and not self.task.startswith("metav"):
            self.problems.append("prompt.* settings apply to the metav tasks only (materialized prompts); other tasks use their yaml prompt")
        if c["extra_model_args"] and not (f == "qwen25vl" and b == "hf"):
            self.problems.append("extra_model_args is forwarded by the Qwen2.5-VL hf runner only")

    # ---- names ------------------------------------------------------------------------------
    @property
    def model_tag(self):
        return self.model.split("/")[-1]

    @property
    def lmms_model(self):
        return "vllm_fix" if self.vllm_fix else "vllm" if self.backend == "vllm" else LMMS_MODEL[(self.fam, self.backend)]

    @property
    def sample_desc(self):
        c = self.cfg
        if c["frame_sampling"] == "fps":
            return f"{c['fps']} fps, capped at {c['max_frames'] or self.nframes} frames"
        return f"{self.nframes} frames, uniformly spaced" + (", random positions" if c["frame_sampler"] == "random" else "")

    @property
    def tag(self):
        c = self.cfg
        s = f"fps{str(c['fps']).replace('.', 'p')}_cap{c['max_frames'] or self.nframes}" if c["frame_sampling"] == "fps" else f"nf{self.nframes}"
        be = "vllmfix" if self.vllm_fix else self.backend.replace("_", "")
        parts = [s, be + ("_bi" if self.bi else ""), f"bs{c['batch_size']}"]
        if c["input"] == "frames":
            parts.insert(1, "frames")
        if c["frame_sampler"] == "random":
            parts.append("fsrandom")
        if c["temperature"] is not None:
            parts.append("t" + str(c["temperature"]).replace(".", "p"))
        parts.append(f"seed{c['seed']}")
        return "_".join(parts)

    @property
    def run_dir(self):
        return os.path.join(self.cfg["output_root"], self.cfg["name"], self.task, self.model_tag, self.tag)

    # ---- runner command ---------------------------------------------------------------------
    def command(self):
        c, f, b = self.cfg, self.fam, self.backend
        runner = os.path.join(RUNNERS, "run_video_qa_vllm_fix.sh" if self.vllm_fix else FAM[f][0])
        a = [
            "bash",
            runner,
            "--model",
            self.model,
            "--tasks",
            self.task,
            "--batch_size",
            str(c["batch_size"]),
            "--seed",
            str(c["seed"]),
            "--output_root",
            os.path.dirname(self.run_dir),
            "--run_tag",
            self.tag,
            "--num_gpus",
            str(c["num_gpus"]),
        ]
        # frames
        if c["frame_sampling"] == "fps":
            a += ["--fps", str(c["fps"]), "--max_frames", str(c["max_frames"] or self.nframes)]
        else:
            a += ["--nframes", str(self.nframes)]
        # backend + what the backend decodes
        if f == "gemma" or self.vllm_fix:
            pass  # these runners have one backend
        else:
            a += ["--backend", b]
        if (fam_has(f, "video_reader") and not self.vllm_fix) or b == "vllm" or b == "hf_tiled":
            a += ["--video_reader", c["video_reader"]]
        if fam_has(f, "max_pixels") or self.vllm_fix or b == "vllm":
            a += ["--max_pixels", str(self.max_pixels)]
            if c["min_pixels"] and f != "glm":
                a += ["--min_pixels", str(c["min_pixels"])]
        if f == "internvl":
            a += ["--max_patches", str(c["max_patches"]), "--min_patches", str(c["min_patches"])]
        if c["frame_sampler"] == "random":
            a += ["--frame_sampler", "random"]
        if f == "qwen35" or (f == "qwen3vl" and c["enable_thinking"]):
            a += ["--enable_thinking", "true" if c["enable_thinking"] else "false"]
        if b == "vllm" or self.vllm_fix:
            a += [
                "--batch_invariant",
                "1" if self.bi else "0",
                "--tensor_parallel_size",
                str(c["tensor_parallel_size"]),
                "--gpu_memory_utilization",
                str(c["gpu_memory_utilization"]),
                "--max_model_len",
                str(c["max_model_len"]),
                "--max_num_seqs",
                str(c["max_num_seqs"]),
            ]
        for key, flag in [("temperature", "--temperature"), ("max_new_tokens", "--max_new_tokens"), ("limit", "--limit"), ("attn", "--attn"), ("include_path", "--include_path")]:
            if c[key] is not None and not (key == "attn" and (b == "vllm" or self.vllm_fix)):
                a += [flag, str(c[key])]
        return a

    def env(self):
        e = {}
        if self.cfg["env"]:
            e["LMMS_EVAL_ENV_BIN"] = self.cfg["env"]
        else:  # the runners call `python`; make sure it is this interpreter
            e["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
        if self.cfg["extra_model_args"]:
            e["EXTRA_MODEL_ARGS"] = self.cfg["extra_model_args"]
        for k, var in PROMPT_KEYS.items():
            if (self.cfg["prompt"] or {}).get(k) is not None:
                e[var] = str(self.cfg["prompt"][k])
        return e

    # ---- the evaluation card --------------------------------------------------------------
    def card(self, cmd):
        c = self.cfg
        rows = [
            ("Scope", "Model", f"{self.model} (family {self.fam}, lmms_eval model `{self.lmms_model}`)"),
            ("", "Task", self.task),
            (
                "Runtime",
                "Inference backend",
                (
                    "vLLM, video forwarded as .mp4 (vllm_fix plugin)"
                    if self.vllm_fix
                    else "vLLM, frames sent as images" if self.backend == "vllm" else "HF transformers" + (" (InternVL image tiler per frame)" if self.backend == "hf_tiled" else "")
                ),
            ),
            ("", "Batch size", str(c["batch_size"])),
            ("", "Batch invariant", str(self.bi) + ("" if self.backend == "vllm" else " (vLLM mode; n/a on hf)")),
            ("", "GPUs", f"{c['num_gpus']} (tensor parallel {c['tensor_parallel_size']})" if self.backend == "vllm" else str(c["num_gpus"])),
            ("", "Environment", c["env"] or "current environment"),
            ("Input", "Input modality", ".mp4 video" if c["input"] == "video" else "pre-extracted JPEG frames"),
            ("", "Frames", self.sample_desc),
            ("", "Video decoder", c["video_reader"] if (fam_has(self.fam, "video_reader") or self.backend in ("vllm", "hf_tiled")) and not self.vllm_fix else "the wrapper's own" + (" (torchcodec, server side)" if self.vllm_fix else "")),
            (
                "",
                "Frame resolution",
                (
                    (f"max_pixels {self.max_pixels}" + (f", min_pixels {c['min_pixels']}" if c["min_pixels"] else ""))
                    if fam_has(self.fam, "max_pixels")
                    else (f"{c['max_patches']} tile(s) of 448x448 per frame" if self.fam == "internvl" else "fixed by the wrapper")
                ),
            ),
            ("", "Tokens per frame", FAM[self.fam][2]),
            ("Decoding", "Temperature", str(c["temperature"]) if c["temperature"] is not None else "task yaml (0 for the MCQ tasks)"),
            ("", "Max new tokens", str(c["max_new_tokens"]) if c["max_new_tokens"] is not None else "task yaml"),
            ("", "Thinking", str(bool(c["enable_thinking"])) if self.fam in ("qwen35", "qwen3vl", "glm") else "n/a"),
            ("", "Prompt", ", ".join(f"{k}={v}" for k, v in (c["prompt"] or {}).items()) or "task default"),
            ("", "Seed", str(c["seed"])),
            ("Command", "", " ".join(shlex.quote(x) for x in cmd)),
        ]
        out = [f"# Evaluation card: {c['name']} / {self.task} / {self.model_tag} / {self.tag}", "", "| | setting | value |", "|---|---|---|"]
        out += [f"| {('**' + g + '**') if g else ''} | {k} | {v} |" for g, k, v in rows]
        return "\n".join(out) + "\n"

    def config_json(self, cmd):
        c = self.cfg
        d = {
            "model": self.model,
            "family": self.fam,
            "runner": os.path.basename(cmd[1]),
            "env": c["env"] or "",
            "exp": c["name"],
            "task": self.task,
            "backend": "vllm_fix" if self.vllm_fix else self.backend,
            "nframes": self.nframes,
            "batch_size": c["batch_size"],
            "seed": c["seed"],
            "video_reader": c["video_reader"],
            "frame_sampler": c["frame_sampler"],
            "batch_invariant": int(self.bi),
            "input": c["input"],
        }
        if c["frame_sampling"] == "fps":
            d.update(fps=c["fps"], max_frames=c["max_frames"] or self.nframes)
        if fam_has(self.fam, "max_pixels"):
            d["max_pixels"] = self.max_pixels
        if self.fam == "internvl":
            d["max_patches"] = c["max_patches"]
        if c["temperature"] is not None:
            d["temperature"] = c["temperature"]
        d["cmd"] = " ".join(shlex.quote(x) for x in cmd)
        d["env_vars"] = self.env()
        return d


def results_exist(run_dir):
    import glob

    return bool(glob.glob(os.path.join(run_dir, "*", "*_results.json")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="YAML file; see configs/example.yaml")
    ap.add_argument("--dry_run", action="store_true", help="print the resolved runs, cards and commands; run nothing")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key (YAML syntax for the value)")
    a = ap.parse_args()
    os.chdir(ROOT)
    cfg = load_config(a.config, a.set)

    grid = []
    for task, model, nf, be, fix in itertools.product(as_list(cfg["tasks"]), as_list(cfg["model"]), as_list(cfg["nframes"]), as_list(cfg["backend"]), as_list(cfg["vllm_fix"])):
        point = (task, model, nf, be, bool(fix) if be == "vllm" else False)  # vllm_fix only means something on vllm
        if point not in grid:
            grid.append(point)
    runs = [Run(cfg, *point) for point in grid]
    bad = [r for r in runs if r.problems]
    if bad:
        for r in bad:
            print(f"refused: {r.task} / {r.model} / {r.tag}", file=sys.stderr)
            for p in r.problems:
                print(f"    - {p}", file=sys.stderr)
        sys.exit(1)

    print(f"{cfg['name']}: {len(runs)} run(s) -> {cfg['output_root']}/{cfg['name']}/")
    n_run = n_skip = n_fail = 0
    for r in runs:
        cmd = r.command()
        if cfg["resume"] and results_exist(r.run_dir):
            print(f"--- SKIP (done)  {r.run_dir}")
            n_skip += 1
            continue
        env = dict(os.environ, **r.env())
        print(f"=== RUN  {r.run_dir}")
        if a.dry_run:
            print(r.card(cmd))
            dry = cmd + ["--dry_run"] if r.vllm_fix else cmd
            subprocess.call(dry, env=dict(env, DRY_RUN="1"))  # the runner prints the final python -m lmms_eval command
            continue
        os.makedirs(r.run_dir, exist_ok=True)
        json.dump(r.config_json(cmd), open(os.path.join(r.run_dir, "config.json"), "w"), indent=2)
        open(os.path.join(r.run_dir, "evaluation_card.md"), "w").write(r.card(cmd) + f"\nstarted {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        n_run += 1
        rc = subprocess.call(cmd, env=env)
        if rc == 0 and results_exist(r.run_dir):
            print(f"=== DONE {r.run_dir}")
        else:
            print(f"!!! FAILED rc={rc} {r.run_dir} (no results file; rerun the same config to retry)", file=sys.stderr)
            n_fail += 1
    if not a.dry_run:
        print(f"\nruns started: {n_run}, skipped (already done): {n_skip}, failed: {n_fail}")
        sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
