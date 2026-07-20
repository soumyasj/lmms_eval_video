"""Standalone Qwen2.5-VL evaluation on a VideoMME subset JSON.

Model loading + inference + the eval loop. All prompt/config/scoring helpers
live in `utils.py`.

Example:
    python hf_inference/qwenvl.py \
        --json random_debug/videomme_subset_200.json \
        --output random_debug/qwenvl_videomme_results.json
"""

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import utils


def load_model(args):
    kwargs = {"torch_dtype": "auto", "device_map": "auto"}
    if args.flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
        kwargs["torch_dtype"] = torch.bfloat16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    return model, processor


@torch.no_grad()
def run_inference(model, processor, video_path, prompt, args):
    video_content = {
        "type": "video",
        "video": video_path,
        "max_pixels": args.max_pixels,
    }
    # `qwen_vl_utils` accepts either `nframes` (fixed count) or `fps`, not both.
    if args.nframes is not None:
        video_content["nframes"] = args.nframes
    else:
        video_content["fps"] = args.fps

    messages = []
    system_prompt = utils.system_prompt_for(args.prompt_style)
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": prompt},
            ],
        }
    )

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


def main():
    args = utils.parse_args()
    utils.set_seed(args.seed)

    docs = utils.load_docs(args.json, args.limit)

    sampling_desc = (
        f"nframes={args.nframes}" if args.nframes is not None else f"fps={args.fps}"
    )
    print(f"Loaded {len(docs)} questions from {args.json}")
    print(f"Prompt style: {args.prompt_style}")
    print(f"Video sampling: {sampling_desc}, max_pixels={args.max_pixels}")
    print(f"Loading model {args.model} ...")
    model, processor = load_model(args)

    records = []
    correct = 0
    evaluated = 0
    by_group = utils.new_group_counter()

    for i, doc in enumerate(docs):
        video_path = utils.resolve_video_path(args.video_root, doc["videoID"])
        if video_path is None:
            print(f"[{i + 1}/{len(docs)}] MISSING video for {doc['videoID']}, skipping")
            records.append(
                {
                    "question_id": doc.get("question_id"),
                    "videoID": doc["videoID"],
                    "error": "missing_video",
                }
            )
            continue

        prompt = utils.build_prompt(doc, args.prompt_style)
        try:
            raw_pred = run_inference(model, processor, video_path, prompt, args)
        except Exception as e:  # keep going on per-video failures
            print(f"[{i + 1}/{len(docs)}] ERROR on {doc['videoID']}: {e}")
            records.append(
                {
                    "question_id": doc.get("question_id"),
                    "videoID": doc["videoID"],
                    "error": str(e),
                }
            )
            continue

        gt_ans = doc["answer"]
        pred_ans, is_correct = utils.score_answer(raw_pred, gt_ans)

        evaluated += 1
        correct += int(is_correct)
        utils.update_groups(by_group, doc, is_correct)

        records.append(
            {
                "question_id": doc.get("question_id"),
                "videoID": doc["videoID"],
                "duration": doc.get("duration"),
                "domain": doc.get("domain"),
                "task_type": doc.get("task_type"),
                "raw_pred": raw_pred,
                "pred_answer": pred_ans,
                "answer": gt_ans,
                "correct": is_correct,
            }
        )

        running_acc = correct / evaluated if evaluated else 0.0
        print(
            f"[{i + 1}/{len(docs)}] {doc.get('question_id')} "
            f"pred={pred_ans} gt={gt_ans} {'✓' if is_correct else '✗'} "
            f"(running acc {running_acc:.3f})"
        )

    summary = utils.build_summary(args, len(docs), evaluated, correct, by_group)
    utils.print_summary(summary)

    if args.output:
        utils.save_results(args.output, summary, records)


if __name__ == "__main__":
    main()
