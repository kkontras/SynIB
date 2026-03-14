"""
MUStARD verification script — generation-based diagnostics.

Loads a QwenVL_MUStARD_Prompt model and a handful of MUStARD samples,
then runs three generation prompts per sample to verify that the model
is correctly processing both video frames and text:

  Prompt 1 – Video description: "Describe what you see in these frames…"
  Prompt 2 – Text + context description: "What is the literal meaning / tone?"
  Prompt 3 – Sarcasm reasoning (chain-of-thought): "Step 1 … Step 4 …"

For each sample the CLS-based binary prediction (from the trained head) is
also shown alongside the generation-based answer.

Usage (standalone):
    PYTHONPATH=src python -m synib.mydatasets.MUStARD.mustard_verify \
        --data_root  path/to/mustard_raw \
        --model_name Qwen/Qwen3-VL-2B-Instruct \
        --num_samples 5 \
        --split test \
        [--ckpt path/to/checkpoint.pt]   # optional: load fine-tuned weights
        [--save_base_dir path/to/hf_cache]
        [--output path/to/verify_output.json]

Or import and call verify() directly from another script.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import torch
from easydict import EasyDict as edict
from torchvision.transforms.functional import to_pil_image

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builders (standalone — no model needed for construction)
# ---------------------------------------------------------------------------

def _build_video_description_messages(processor, speaker: str, n_frames: int) -> dict:
    """Prompt 1: describe what is visible in the video frames."""
    text = (
        "Look at these video frames from a TV show.\n"
        "Describe what you see: the speaker's facial expression, "
        "body language, and any other visual cues relevant to their emotional state."
    )
    content = [{"type": "image"} for _ in range(n_frames)]
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def _build_text_description_messages(
    processor, utterance: str, speaker: str, context: List[str]
) -> str:
    """Prompt 2: describe the tone and literal meaning of the text."""
    ctx_str = "\n".join(context) if context else "(none)"
    text = (
        f"Read the following dialogue excerpt from a TV show.\n\n"
        f"Prior conversation:\n{ctx_str}\n\n"
        f"Utterance by {speaker}: \"{utterance}\"\n\n"
        "1. Briefly summarise what is being discussed in the prior conversation.\n"
        "2. What is the literal meaning of the utterance by the speaker?\n"
        "3. Given the context of the prior conversation, does the utterance seem sincere or ironic/sarcastic? Explain why."
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _build_sarcasm_reasoning_messages(
    processor, utterance: str, speaker: str, context: List[str], n_frames: int
) -> str:
    """Prompt 3: chain-of-thought sarcasm reasoning."""
    ctx_str = "\n".join(context) if context else "(none)"
    text = (
        f"Speaker: {speaker}\n"
        f"Prior conversation:\n{ctx_str}\n\n"
        f"Utterance: \"{utterance}\"\n\n"
        "Step 1: What is being discussed in the prior conversation, and what does the speaker say in response?\n"
        "Step 2: What does the speaker's face and body language show?\n"
        "Step 3: Is there a mismatch between the utterance (given the conversation context) and how the speaker looks?\n"
        "Step 4: Based on this, is the speaker being sarcastic? "
        "Answer Yes or No, with your reasoning."
    )
    content = [{"type": "image"} for _ in range(n_frames)]
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------

def verify(
    model,
    dataset,
    num_samples: int = 10,
    max_new_tokens: int = 512,
    output_path: Optional[str] = None,
    print_prompts: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run all three generation prompts on the first *num_samples* items of
    *dataset* and return a list of result dicts.

    Args:
        model:         An instantiated QwenVL_MUStARD_Prompt (or similar).
                       Must have .processor and .generate_answer().
        dataset:       A MUStARD_RawDataset instance.
        num_samples:   How many samples to verify.
        max_new_tokens: Token budget for each generation.
        output_path:   If provided, save results as JSON to this path.

    Returns:
        List of dicts with keys:
            id, utterance, speaker, context, label,
            video_description, text_description, sarcasm_reasoning,
            cls_pred, cls_prob_sarcastic
    """
    model.eval()
    processor = model.processor
    results = []

    indices = range(min(num_samples, len(dataset)))

    for idx in indices:
        item = dataset[idx]
        data = item["data"]
        clip_id = item["id"]
        label = int(item["label"].item())

        utterance = data["text"]
        speaker = data["speaker"]
        context = data["context"]
        video_frames = data["video_frames"]  # (F, 3, H, W)

        n_frames = video_frames.shape[0]
        model_device = model.backbone.device

        # Convert frames to PIL
        pil_images = [
            to_pil_image(video_frames[f].clamp(0.0, 1.0))
            for f in range(n_frames)
        ]

        log.info("[%d/%d] clip=%s  label=%d  frames=%d", idx + 1, len(indices), clip_id, label, n_frames)
        print(f"\n{'='*70}")
        print(f"Sample {idx+1}/{len(indices)}  |  clip_id={clip_id}  |  label={'SARCASTIC' if label else 'NOT sarcastic'}")
        print(f"Speaker: {speaker}")
        print(f"Context: {context}")
        print(f"Utterance: {utterance}")
        print(f"Frames: {n_frames}")

        # ------------------------------------------------------------------
        # CLS-based prediction (forward pass through combined model)
        # ------------------------------------------------------------------
        cls_pred = None
        cls_prob_sarcastic = None
        try:
            with torch.no_grad():
                # Build a single-sample batch
                batch_x = {
                    "text": [utterance],
                    "speaker": [speaker],
                    "context": [context],
                    "video_frames": video_frames.unsqueeze(0),  # (1, F, 3, H, W)
                }
                out = model(batch_x, label=None)
            logits = out["preds"]["combined"]  # (1, 2)
            probs = torch.softmax(logits.float(), dim=-1)[0]
            cls_pred = int(probs.argmax().item())
            cls_prob_sarcastic = float(probs[1].item())
        except Exception as e:
            log.warning("CLS forward failed for clip=%s: %s", clip_id, e)

        print(f"\n[CLS prediction]  pred={'SARCASTIC' if cls_pred else 'NOT sarcastic'}  "
              f"p(sarcastic)={cls_prob_sarcastic:.3f}" if cls_pred is not None
              else "\n[CLS prediction]  FAILED")

        # ------------------------------------------------------------------
        # Prompt 1 — Video description
        # ------------------------------------------------------------------
        video_description = None
        try:
            prompt1 = _build_video_description_messages(processor, speaker, n_frames)
            proc1 = processor(
                text=[prompt1],
                images=pil_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            proc1 = {k: (v.to(model_device) if torch.is_tensor(v) else v) for k, v in proc1.items()}
            texts1 = model.generate_answer(
                proc1, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=0.0, top_p=1.0, min_new_tokens=10, strip_prompt=True,
            )
            video_description = texts1[0] if texts1 else ""
        except Exception as e:
            log.warning("Prompt 1 failed for clip=%s: %s", clip_id, e)
            video_description = f"[ERROR: {e}]"

        print(f"\n[Prompt 1 — Video description]\n{video_description}")

        # ------------------------------------------------------------------
        # Prompt 2 — Text + context description
        # ------------------------------------------------------------------
        text_description = None
        try:
            prompt2 = _build_text_description_messages(processor, utterance, speaker, context)
            if print_prompts:
                print(f"\n[Raw Prompt 2]\n{prompt2}\n")
            proc2 = processor(
                text=[prompt2],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            proc2 = {k: (v.to(model_device) if torch.is_tensor(v) else v) for k, v in proc2.items()}
            texts2 = model.generate_answer(
                proc2, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=0.0, top_p=1.0, min_new_tokens=10, strip_prompt=True,
            )
            text_description = texts2[0] if texts2 else ""
        except Exception as e:
            log.warning("Prompt 2 failed for clip=%s: %s", clip_id, e)
            text_description = f"[ERROR: {e}]"

        print(f"\n[Prompt 2 — Text/context description]\n{text_description}")

        # ------------------------------------------------------------------
        # Prompt 3 — Sarcasm reasoning (chain-of-thought)
        # ------------------------------------------------------------------
        sarcasm_reasoning = None
        try:
            prompt3 = _build_sarcasm_reasoning_messages(
                processor, utterance, speaker, context, n_frames
            )
            if print_prompts:
                print(f"\n[Raw Prompt 3]\n{prompt3}\n")
            proc3 = processor(
                text=[prompt3],
                images=pil_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            proc3 = {k: (v.to(model_device) if torch.is_tensor(v) else v) for k, v in proc3.items()}
            texts3 = model.generate_answer(
                proc3, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=0.0, top_p=1.0, min_new_tokens=20, strip_prompt=True,
            )
            sarcasm_reasoning = texts3[0] if texts3 else ""
        except Exception as e:
            log.warning("Prompt 3 failed for clip=%s: %s", clip_id, e)
            sarcasm_reasoning = f"[ERROR: {e}]"

        print(f"\n[Prompt 3 — Sarcasm reasoning]\n{sarcasm_reasoning}")

        results.append({
            "id": clip_id,
            "utterance": utterance,
            "speaker": speaker,
            "context": context,
            "label": label,
            "video_description": video_description,
            "text_description": text_description,
            "sarcasm_reasoning": sarcasm_reasoning,
            "cls_pred": cls_pred,
            "cls_prob_sarcastic": cls_prob_sarcastic,
        })

    # Optionally save to JSON
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="MUStARD generation-based verification script"
    )
    p.add_argument(
        "--data_root",
        default=os.environ.get(
            "SYNIB_MUSTARD_DATA_ROOT",
            "src/synib/mydatasets/MUStARD/prepared/mustard_raw",
        ),
        help="Path to the mustard_raw/ directory (contains metadata.json and videos/).",
    )
    p.add_argument(
        "--model_name",
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="HuggingFace model ID or local path.",
    )
    p.add_argument(
        "--save_base_dir",
        default=None,
        help="HuggingFace cache directory (also used to locate checkpoints).",
    )
    p.add_argument(
        "--ckpt",
        default=None,
        help="Optional path to a fine-tuned checkpoint (.pt). "
             "Loads state_dict['model'] if available.",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to verify.",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to draw samples from.",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frame sampling rate (frames per second).",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max tokens to generate per prompt.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Path to save verification results as JSON.",
    )
    p.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of output classes (should be 2 for sarcasm).",
    )
    p.add_argument(
        "--bf16",
        action="store_true",
        help="Load backbone in bfloat16 (default: float16).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--print-prompts",
        action="store_true",
        help="Print the raw rendered prompt string before each generation call.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ── 1. Dataset ─────────────────────────────────────────────────────────
    from synib.mydatasets.MUStARD.MUStARD_Dataset import MUStARD_RawDataset

    cfg_dataset = edict({
        "dataset": {
            "data_roots": args.data_root,
            "fps": args.fps,
            "image_size": 224,
            "val_split_rate": 0.15,
            "test_split_rate": 0.15,
        },
        "training_params": {"batch_size": 1, "seed": args.seed},
    })
    dataset = MUStARD_RawDataset(config=cfg_dataset, split=args.split)
    log.info("Loaded %d samples from split=%s", len(dataset), args.split)

    # ── 2. Model ────────────────────────────────────────────────────────────
    from synib.models.vlm.qwen_mustard_prompt import QwenVL_MUStARD_Prompt
    from synib.models.vlm.qwen_base_models import LinearHead_Qwen

    model_args = edict({
        "model_name": args.model_name,
        "save_base_dir": args.save_base_dir,
        "num_classes": args.num_classes,
        "cls_finetune": False,
        "bf16": args.bf16,
        "max_new_tokens": args.max_new_tokens,
        "generate_reasoning": False,  # we do generation manually below
        "lora_config": {"use_lora": False},
    })

    # Build enc_0 with a dummy d_model; will be resized after backbone init
    # LinearHead_Qwen is initialized with (d_model, num_classes)
    # We pass a placeholder — the model __init__ sets self.d_model from backbone
    enc_0_placeholder = LinearHead_Qwen(edict({"num_classes": args.num_classes, "d_model": 2048}))
    model = QwenVL_MUStARD_Prompt(args=model_args, encs=[enc_0_placeholder])

    # Resize enc_0 to actual d_model if needed
    if model.d_model != 2048:
        log.info("Resizing enc_0: 2048 → %d", model.d_model)
        model.enc_0 = LinearHead_Qwen(edict({"num_classes": args.num_classes, "d_model": model.d_model}))
        for p in model.enc_0.parameters():
            p.requires_grad = True

    # Move enc_0 to backbone device (backbone loads onto GPU; head is created on CPU)
    backbone_device = next(model.backbone.parameters()).device
    model.enc_0.to(backbone_device)
    log.info("enc_0 moved to %s", backbone_device)

    # ── 3. Optional checkpoint ──────────────────────────────────────────────
    if args.ckpt:
        log.info("Loading checkpoint: %s", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            log.warning("Missing keys: %s", missing)
        if unexpected:
            log.warning("Unexpected keys: %s", unexpected)

    # ── 4. Run verification ─────────────────────────────────────────────────
    results = verify(
        model=model,
        dataset=dataset,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        output_path=args.output,
        print_prompts=args.print_prompts,
    )

    print(f"\n{'='*70}")
    print(f"Verification complete. Processed {len(results)} samples.")

    # Summary: how many predictions match label?
    matched = sum(
        1 for r in results
        if r["cls_pred"] is not None and r["cls_pred"] == r["label"]
    )
    valid = sum(1 for r in results if r["cls_pred"] is not None)
    if valid:
        print(f"CLS accuracy on verified samples: {matched}/{valid} = {matched/valid:.1%}")

    return results


if __name__ == "__main__":
    main()
