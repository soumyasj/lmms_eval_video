# Models

How each model family is run: the lmms_eval wrapper behind it, the arguments the runner builds from a config, how a
video becomes tokens, what can and cannot be varied, and the issues this fork fixes or works around. The config keys
referred to are those of `eval_scripts/run_eval.py` ([configs/example.yaml](./configs/example.yaml)).

## Families at a glance

| family | checkpoints (aliases) | hf wrapper | vLLM | runner | environment |
|---|---|---|---|---|---|
| `qwen25vl` | `Qwen/Qwen2.5-VL-{3,7,32,72}B-Instruct` (`qwen25vl_7b`), fine-tunes such as `Video-R1/Video-R1-7B` (`family: qwen25vl`) | `qwen2_5_vl` | yes, + `vllm_fix` | `run_video_qa_hf_multigpu.sh` | A |
| `qwen3vl` | `Qwen/Qwen3-VL-{2,4,8,32}B-Instruct` (`qwen3vl_8b`) | `qwen3_vl` | yes | `run_video_qa_qwen3vl_multigpu.sh` | A |
| `qwen35` | `Qwen/Qwen3.5-{0.8B,2B,4B,9B,...}` (`qwen35_9b`) | `qwen3_5` | yes (no batch invariance) | `run_video_qa_qwen3_5_multigpu.sh` | B |
| `internvl` | `OpenGVLab/InternVL3_5-{1,2,4,8,...}B-HF` (`internvl35_8b`), `OpenGVLab/InternVL3-*-hf` (`internvl3_2b`) | `internvl_hf`, `internvl_hf_tiled` | yes | `run_video_qa_internvl_protocol.sh` | B |
| `glm` | `zai-org/GLM-4.6V-Flash` (`glm46v_flash`), `GLM-4.6V`, `GLM-4.5V`, `GLM-4.1V-9B-Thinking` | `glm4v` (frames only) | yes | `run_video_qa_glm_protocol.sh` | B |
| `gemma` | `google/gemma-3-{4,12,27}b-it` (`gemma3_12b`) | `gemma3` (frames only) | no | `run_video_qa_gemma_protocol.sh` | A |

### Environments

| | transformers | vLLM | torch | used for |
|---|---|---|---|---|
| A | 4.57 | 0.11.0 (0.11.1 for `batch_invariant`) | 2.9 | Qwen2.5-VL, Qwen3-VL, Gemma 3 |
| B | 5.16 | 0.28 | 2.13 | Qwen3.5, InternVL3.5, GLM-V |

`eval_scripts/env/requirements-lock.txt` pins environment A as used for the paper. Point `env:` in a config (or
`LMMS_EVAL_ENV_BIN`) at the `bin` directory of the environment a family needs; the sweep scripts take `ENV_A`,
`ENV_A_BI` and `ENV_B`.

## What the model receives

Every task hands the wrapper the path of an `.mp4`. From there the families differ.

### Qwen2.5-VL (`qwen25vl`)

- **Frames.** `qwen_vl_utils` decodes the clip with the reader named by `video_reader` (`FORCE_QWENVL_VIDEO_READER`):
  `nframes` frames at `linspace(0, total-1, n)` (uniform), or `fps` frames per second capped at `max_frames`.
  `frame_sampler: random` instead draws `nframes` distinct positions out of a pool of decoded frames, seeded per video
  (`blake2b(seed:video_path)`), so a video gets the same frames at a given seed regardless of batching or sharding.
- **Tokens.** Frames are resized by `smart_resize` into `[min_pixels, max_pixels]` on a 28-pixel grid; one token
  covers 28x28 pixels after the 2x2 spatial merge, and the temporal patch merges **frame pairs**. Tokens per prompt
  are about `nframes / 2 x max_pixels / 784`: 32 frames at 602,112 pixels = about 12,300 vision tokens, inside one
  `<|vision_start|><|video_pad|>...<|vision_end|>` block with video position ids.
- **hf model_args** (`--model qwen2_5_vl`): `pretrained=<id>,max_pixels=<P>,min_pixels=<p>,max_num_frames=<N>`
  (or `fps=<f>,max_num_frames=<cap>`), `device_map=auto`, `frame_sampler=uniform|random`, `frame_sampler_seed=<seed>`,
  optional `attn_implementation`, `extra_model_args` appended verbatim (e.g. `interleave_visuals=True` for the
  Meta-Video layout experiments).
- **vLLM model_args** (`--model vllm`): `model=<id>,max_frame_num=<N>,nframes=<N>,max_pixels=<P>,data_parallel_size,
  tensor_parallel_size,gpu_memory_utilization,max_model_len,max_num_seqs` (+ `mm_encoder_attn_backend=TORCH_SDPA` when
  batch invariant). **The stock vLLM wrapper decodes the video client-side and sends the N frames as N separate images**:
  the model sees N x `<|image_pad|>`, no temporal merge, no timestamps. That is what the 4 to 6 point hf/vLLM gap is.
- **`vllm_fix`** (`--model vllm_fix`, plugin `eval_scripts/runners/vllm_fix/`): forwards the `.mp4` as a `file://` URL,
  vLLM decodes it server-side with the model's own processor (torchcodec, the same index formula as hf) and builds one
  `<|video_pad|>` block. Extra args: `allowed_local_media_path`, `media_io_kwargs.video.num_frames` (from `nframes`),
  `mm_processor_kwargs.max_pixels/min_pixels`. No fps sampling on this path.
- **Prompt block.** Tasks with a `qwen2_5_vl:` block use it under hf; under vLLM every task uses `default:`.

### Qwen3-VL (`qwen3vl`)

- **Frames** as Qwen2.5-VL (`qwen_vl_utils`, uniform or fps), no random sampler (the wrapper rejects unknown kwargs).
- **Tokens.** 16-pixel patches with 2x2 merge: one token = 32x32 = 1,024 pixels, so a Qwen2.5-VL `max_pixels` is not
  comparable (protocol default 524,288 = 512 tokens per frame). Frame pairs are merged, and the processor writes a
  literal **`<t seconds>` timestamp string before every pair**, computed from the frame indices and the native fps,
  so the decoder and the sampling mode change the prompt text, not only the pixels. Without video metadata the
  processor silently assumes 24 fps.
- **hf model_args** (`--model qwen3_vl`): `pretrained,max_pixels,min_pixels,max_num_frames|fps+max_num_frames,device_map`,
  optional `total_pixels`, `attn_implementation`, `system_prompt`, `enable_thinking` (thinking checkpoints).
- **vLLM model_args**: as Qwen2.5-VL plus `mm_processor_kwargs={"size":{"shortest_edge":<min>,"longest_edge":<max>}}`
  and `mm_processor_cache_gb=0` (the engine asserted `Expected a cached item for mm_hash` at 64+ frames). Frames are
  sent as images, as above.
- **Prompt block.** Video-MME has a `qwen3_vl:` block (`Question:` prefix); most tasks do not and fall back to
  `default:`, so published Qwen3-VL numbers are not reproduced on those. Under vLLM the block is never used.

### Qwen3.5 (`qwen35`)

- Inherits the Qwen3-VL path (timestamp text, 1,024 pixels per token, uniform sampling; protocol default
  `max_pixels` 131,072 = 128 tokens per frame, the wrapper default).
- **Thinking is on by default** (`enable_thinking=True`, 1,024 tokens, temperature 0.7 in the wrapper). The runner
  passes `enable_thinking=false` unless the config enables it; on vLLM, which cannot forward `chat_template_kwargs`,
  the runner writes a copy of the checkpoint's chat template with `enable_thinking` preset to false
  (`eval_scripts/runners/chat_templates/generated/`) and passes it as `chat_template=`.
- Needs transformers >= 5 (`Qwen3_5ForConditionalGeneration`) and a vLLM that knows the class (0.28). Cannot run
  batch-invariant (Gated DeltaNet layers are unsupported); the config refuses it.
- Registered as `qwen3_5`: no task has a `qwen3_5:` prompt block, so hf and vLLM see the same `default:` prompt.

### InternVL3 / InternVL3.5 (`internvl`)

- **Frames.** hf: transformers' `InternVLVideoProcessor`, `num_frames` exact (clamped to the shortest clip of the
  batch) or fps; **every frame is resized to one 448x448 tile = 256 tokens**, whatever `max_patches` says (the video
  processor ignores it). `video_size` can change the square size. Batch size must be 1 on hf.
- **`hf_tiled`** (`internvl_hf_tiled`, `eval_scripts/runners/internvl_tiled/`): the model card's `load_video(max_num=M)`
  recipe. Frames at segment midpoints (decord or pyav, `video_reader`), each frame through the **image** tiler with
  `min_patches` / `max_patches` 448x448 tiles plus a thumbnail when more than one, ImageNet normalisation, and a
  `Frame{i}: <img>...</img>` prompt. The only hf path where the tile budget changes a video run.
- **vLLM**: frames sent as images (as the Qwen vLLM path); tiles pinned with
  `mm_processor_kwargs={"max_patches":M,"min_patches":m}` and the resize sized to M tiles.
- Context: InternVL3.5-8B-HF has 40,960 text positions, so 128 frames must stay at 1 tile (32,768 tokens).
- **hf model_args** (`--model internvl_hf`): `pretrained,num_frames=<N>|fps,min_patches,max_patches,device_map`,
  optional `attn_implementation`, `video_size`. No task has an internvl prompt block.
- Fixed in this fork: the stock video processor resized frames to 384x384, which is not divisible by the 2x2 pixel
  shuffle, so **every video sample raised and the run reported ~0 %** instead of failing; the wrapper now defaults
  `video_size` to the model's native 448.

### GLM-4.6V / 4.5V (`glm`)

- hf (`glm4v`): accepts frame lists only (no `nframes` / `fps` / `max_pixels`), so a raw video task is refused on hf;
  use `input: frames`. vLLM: frames sent as images, 14-pixel patch with 2x2 merge = 784 pixels per token (protocol
  default `max_pixels` 401,408 = 512 tokens per frame), context 131,072.
- Thinking: the chat template thinks by default; the vLLM runner passes `chat_templates/glm4v_nothink.jinja`
  (the verbatim template with `enable_thinking` preset to false). MoE checkpoints (GLM-4.6V, 4.5V) depend on vLLM's MoE
  kernels and are unverified; GLM-4.6V-Flash is dense and runs batch-invariant.

### Gemma 3 (`gemma`)

- hf only (`gemma3`); frame lists only: for a raw `.mp4` the wrapper hands the path to the chat template without a
  frame budget, so the config refuses `input: video`. Each frame is one SigLIP 896x896 image = **256 tokens whatever
  `max_pixels` says**; 128 frames = 32k tokens inside the 128k context. Batching works. No task has a gemma prompt block.

## What can be varied, per family

| setting | qwen25vl | qwen3vl | qwen35 | internvl | glm | gemma |
|---|---|---|---|---|---|---|
| raw `.mp4` on hf with a frame budget | yes | yes | yes | yes | no (frames) | no (frames) |
| `nframes` | yes | yes | yes | yes | yes (vLLM) | yes (frames) |
| `fps` + `max_frames` | hf | hf | hf | no | no | no |
| `video_reader` | hf, vLLM | hf, vLLM | hf, vLLM | vLLM; `hf_tiled`: decord / pyav | vLLM | no |
| `frame_sampler: random` | hf | no | no | no | no | no |
| `max_pixels` | yes | yes | yes | no (`max_patches`) | vLLM | no effect |
| `batch_size` > 1 on hf | yes | yes | yes | no | yes | yes |
| `batch_invariant` (vLLM) | yes | yes | no | yes | yes (dense) | no vLLM |
| `enable_thinking` | - | thinking checkpoints | yes | - | via chat template | - |
| `vllm_fix` | yes | no | no | no | no | no |

## Batch invariance on vLLM

`batch_invariant: true` sets `VLLM_BATCH_INVARIANT=1` (vLLM >= 0.11.1; the runner fails loudly on an older vLLM
instead of mislabeling a run). It forces the FLASH_ATTN backend globally, whose bundled kernel lacks the Qwen-VL vision
tower's head dimension (80); the runners therefore pin the vision encoder to SDPA (`mm_encoder_attn_backend=TORCH_SDPA`)
and keep the language model on the invariant path. Within one backend the batch size is then score-neutral (std
0.15 points across batch sizes 1 to 32 in the paper).

## What this fork changes

Changes in `lmms_eval/` (base: upstream `3b614543`), each found because a sweep produced a number that could not be right.

| where | change | why |
|---|---|---|
| `models/model_utils/qwen/video_reader_patch.py` (new) | seek-based torchvision reader | `torchvision.io.read_video` decoded the whole clip into memory (100 to 250 GB on Video-MME) whatever `nframes` said |
| `models/chat/qwen2_5_vl.py`, `qwen3_vl.py`, `vllm.py` | clamp `nframes` to the clip length, retry stepping down by 2 | `qwen_vl_utils` raises on clips shorter than the request (MVBench has clips under 64 frames) |
| `models/chat/qwen2_5_vl.py` | forward the frame cap in fps mode | `max_num_frames` had no effect at 1 fps; hour-long videos expanded without bound |
| `models/chat/internvl_hf.py` | `video_size` defaults to the native 448; clamp `num_frames` to the shortest clip | 384x384 frames broke the pixel shuffle; every sample raised and the run scored ~0 % |
| `models/simple/qwen2_5_vl.py`, `models/chat/qwen2_5_vl.py` | `frame_sampler=uniform\|random`, `frame_sampler_seed`, `frame_pool_factor` | seeded random frame positions for the frame-selection experiment |
| `__main__.py`, `loggers/evaluation_tracker.py` | `--log_samples_suffix` also prefixes the result files | result files named after the run configuration |
| `api/task.py` | a valid `create_link` symlink counts as cached | the Hub snapshot was re-resolved on every run (rate limited) |
| `tasks/metav/`, `tasks/precomputed_frames.py`, `tasks/tempcompass/utils.py` | see [DATASETS.md](./DATASETS.md#task-level-changes) | |

Workarounds that live in the runners rather than in `lmms_eval/`: vLLM ignored `--nframes` (both `max_frame_num` and
`nframes` are passed); the batch-invariant vision-tower pin above; `mm_processor_cache_gb=0` for Qwen3-VL on vLLM;
no-thinking chat templates for Qwen3.5 and GLM; InternVL tiles via `mm_processor_kwargs` (vLLM) and `internvl_hf_tiled`
(hf); GLM and Gemma refuse raw-video tasks on hf; every script exports `HF_HOME` and keeps `LMMS_EVAL_DATASETS_CACHE`
inside it (a node-local `/tmp` arrow cache that gets reaped made `datasets` fail to mmap a build it believed existed).

## Known limitations

- **Prompt blocks are keyed by the registered model name.** `--model qwen3_vl` gets a task's `qwen3_vl:` block,
  `--model vllm` gets `default:`, so an hf-vs-vLLM comparison on such a task also changes the prompt. Meta-Video uses
  materialized prompts selected by `prompt.variant` instead.
- `max_new_tokens` in a task YAML is a floor, not a cap, on the vLLM path (the wrapper takes the maximum).
- Multi-GPU in the runners is data parallelism (one replica per GPU); a model that does not fit on one GPU needs
  `backend: vllm` with `tensor_parallel_size > 1`.
