"""Offline (CPU-only) demonstration of how the video *tag* changes the tokenized prompt.

It renders the Qwen2.5-VL chat template two ways for a sample question and prints the exact
vision placeholders each produces:

  * image_pad path  (what `--model vllm` does): 1 image item per frame  ->  N x <|image_pad|>
  * video_pad path  (what `--model vllm_video_url` does): 1 video item  ->  1 x <|video_pad|>

At the *chat-template* (text) level, each image item emits one <|image_pad|> and each video item
emits one <|video_pad|>; the later processor step expands these into the real per-patch pad tokens
using grid_thw (which is where the token *counts*, and thus the model's effective input, differ).

No GPU, no vLLM, no video decode needed -- only the cached tokenizer/chat-template.

Usage:
    python eval_scripts/pad_experiment/inspect_pads.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct --nframes 2 \
        --parquet random_debug/videomme_subset_duration.parquet
"""

import argparse


def _load_question(parquet_path: str) -> str:
    """Pull one representative question from the debug subset; fall back to a stub."""
    try:
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        row = df.iloc[0]
        q = str(row.get("question", "What happens in the video?"))
        opts = row.get("options", None)
        if opts is not None:
            opts_str = "\n".join(list(opts)) if not isinstance(opts, str) else str(opts)
            return f"{q}\n{opts_str}"
        return q
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not read {parquet_path} ({exc}); using stub question.")
        return "What happens in the video?\nA. ...\nB. ...\nC. ...\nD. ..."


def _build_messages(question: str, nframes: int):
    """Native HF-form messages. Item *type* is what the chat template branches on, so no real
    pixels/video are needed to reveal the tags."""
    image_content = [{"type": "image"} for _ in range(nframes)]
    image_content.append({"type": "text", "text": question})
    image_messages = [{"role": "user", "content": image_content}]

    video_messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
    return image_messages, video_messages


def _render(model: str, messages) -> str:
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        return proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:  # noqa: BLE001
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--nframes", type=int, default=2)
    ap.add_argument("--parquet", default="random_debug/videomme_subset_duration.parquet")
    args = ap.parse_args()

    question = _load_question(args.parquet)
    image_messages, video_messages = _build_messages(question, args.nframes)

    img_prompt = _render(args.model, image_messages)
    vid_prompt = _render(args.model, video_messages)

    def report(label, prompt):
        n_img = prompt.count("<|image_pad|>")
        n_vid = prompt.count("<|video_pad|>")
        print(f"\n===== {label} =====")
        print(f"<|image_pad|> blocks: {n_img}   <|video_pad|> blocks: {n_vid}")
        print("-" * 60)
        print(prompt)
        print("=" * 60)

    report(f"IMAGE_PAD path (--model vllm, nframes={args.nframes})", img_prompt)
    report("VIDEO_PAD path (--model vllm_video_url)", vid_prompt)

    print(
        "\nTakeaway: image path scales vision blocks with nframes; video path is a single "
        "native video block. Different block structure -> different tokenization -> can shift accuracy."
    )


if __name__ == "__main__":
    main()
