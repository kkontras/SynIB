
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScienceQA → Qwen3-VL offline token cache builder.

What this script does (per example):
  - Builds the exact prompt you use (image token + "\n" + [hint] + "\n\n" + [QA] + "\n\n" + [instruction] + "\n\n" + "<CLS>")
  - Runs AutoProcessor once to get:
      input_ids, attention_mask, pixel_values, image_grid_thw
  - Builds per-token masks for:
      image / hint / question / choices / instruction / cls / other
    and a token_type_ids uint8 array encoding the same categories.
  - (Optional) also caches "vision tokens" as float embeddings by running the model's visual module:
      model.model.visual(pixel_values, grid_thw=image_grid_thw)
    NOTE: these are NOT integer IDs; they are embeddings (tensor or tuple/dict of tensors).

Outputs:
  out_dir/<split>/
    meta.json
    manifest.jsonl
    shard_00000.pt
    shard_00001.pt
    ...

Each shard is a list[dict] with keys:
  id, label,
  input_ids, attention_mask,
  token_type_ids, masks (dict of bool tensors),
  pixel_values, image_grid_thw (if present),
  image_embeds (optional; tensor OR dict[str,tensor] OR list[tensor])

Example:
  python cache_scienceqa_qwen3vl.py \
    --data_root /path/to/hf_cache \
    --out_dir /path/to/cache \
    --model_name Qwen/Qwen3-VL-2B-Instruct \
    --split train \
    --batch_size 16 \
    --num_workers 8 \
    --cache_image_embeds
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, AutoConfig

# Try importing Qwen3VL model class (preferred). If your transformers doesn't have it,
# fall back to AutoModelForCausalLM and access .model.visual if present.
try:
    from transformers import Qwen3VLForConditionalGeneration  # type: ignore
    HAVE_QWEN3VL_CLASS = True
except Exception:
    from transformers import AutoModelForCausalLM
    HAVE_QWEN3VL_CLASS = False


LETTERS_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# =========================
# Filtering / loading
# =========================
def load_scienceqa_filtered(
    data_root: str,
    split: str,
    require_image: bool = True,
    require_outside_knowledge: bool = True,
    drop_near_blank: bool = True,
    blank_std_thresh: float = 0.01,
):
    ds = load_dataset("derek-thomas/ScienceQA", split=split, cache_dir=data_root)

    keep_indices: List[int] = []
    for i, ex in enumerate(ds):
        pil_img = ex.get("image", None)
        has_img = pil_img is not None

        lecture = (ex.get("lecture") or "").strip()
        hint = (ex.get("hint") or "").strip()
        has_outside = (len(lecture) > 0) or (len(hint) > 0)

        if require_image and not has_img:
            continue
        if require_outside_knowledge and not has_outside:
            continue

        if drop_near_blank and has_img:
            t = to_tensor(pil_img)
            if t.std().item() < blank_std_thresh:
                continue

        keep_indices.append(i)

    print(
        f"[ScienceQA] Split={split}: kept {len(keep_indices)} / {len(ds)} "
        f"(image={require_image}, outside_k={require_outside_knowledge}, "
        f"drop_near_blank={drop_near_blank})"
    )
    return ds, keep_indices


# =========================
# Text builders
# =========================
def build_scienceqa_hint_text(example: Dict[str, Any]) -> str:
    lecture = (example.get("lecture") or "").strip()
    hint = (example.get("hint") or "").strip()

    parts = []
    if lecture:
        parts.append("Science passage:\n" + lecture)
    if hint:
        parts.append("Hint:\n" + hint)
    return "\n\n".join(parts)


def build_question_text(question: str) -> str:
    q = (question or "").strip()
    return ("Question:\n" + q) if q else ""


def build_choices_text(choices: List[str], letters: List[str]) -> str:
    assert len(choices) == len(letters)
    lines = [f"({L}) {c}" for L, c in zip(letters, choices)]
    return "Choices:\n" + "\n".join(lines)


def build_instruction_text(letters: List[str]) -> str:
    if not letters:
        return ""
    letters_str = ", ".join(f"({L})" for L in letters)
    return f"Answer with only one of: {letters_str}."


def build_full_prompt(
    *,
    image_token_str: str,
    hint_text: str,
    question_text: str,
    choices_text: str,
    instr_text: str,
    cls_text: str = "<CLS>",
) -> str:
    """
    Mirrors your model logic:
      parts = [hint?, qa?, instr?, "<CLS>"]
      prompts = "\n\n".join(parts)
      prompts_with_image = image_token_str + "\n" + prompts
    Where qa is question + "\n\n" + choices (if both exist).
    """
    hint_text = (hint_text or "").strip()
    question_text = (question_text or "").strip()
    choices_text = (choices_text or "").strip()
    instr_text = (instr_text or "").strip()

    qa_parts = []
    if question_text:
        qa_parts.append(question_text)
    if choices_text:
        if qa_parts:
            qa_parts.append("")  # produces the \n\n between q and choices
        qa_parts.append(choices_text)
    qa_block = "\n\n".join([p for p in qa_parts if p != ""])

    parts = []
    if hint_text:
        parts.append(hint_text)
    if qa_block:
        parts.append(qa_block)
    if instr_text:
        parts.append(instr_text)
    parts.append(cls_text)

    prompt = "\n\n".join(parts)
    return image_token_str + "\n" + prompt


# =========================
# Token types / masks
# =========================
TT_PAD = 0
TT_IMAGE = 1
TT_HINT = 2
TT_QUESTION = 3
TT_CHOICES = 4
TT_INSTR = 5
TT_CLS = 6
TT_OTHER = 7


def _tokenize_no_special(tok, s: str) -> List[int]:
    if not s:
        return []
    return tok(s, add_special_tokens=False).input_ids


def build_token_type_ids_best_effort(
    *,
    tok,
    input_ids_1d: torch.Tensor,
    attention_mask_1d: torch.Tensor,
    image_token_id: int,
    cls_token_id: int,
    hint_text: str,
    question_text: str,
    choices_text: str,
    instr_text: str,
) -> torch.Tensor:
    """
    Best-effort span alignment:
      - hard marks image tokens by ID
      - hard marks CLS token by ID
      - attempts to lay out tokenized text blocks starting at first non-image token
        in the unpadded region.

    Works well for this kind of deterministic concatenation prompt.
    """
    # Keep only the unpadded region (since we slice inputs to true_len before calling this)
    ids = input_ids_1d.tolist()
    L = len(ids)

    ttid = torch.full((L,), TT_OTHER, dtype=torch.uint8)

    # mark image tokens robustly
    for i, tid in enumerate(ids):
        if tid == int(image_token_id):
            ttid[i] = TT_IMAGE

    # find first index after initial image tokens (common in Qwen-VL prompts)
    j = 0
    while j < L and ids[j] == int(image_token_id):
        j += 1
    # after image tokens, there is typically a "\n" then text; we start best-effort at j
    pos = j

    sep = "\n\n"
    hint_text_s = (hint_text or "").strip()
    q_s = (question_text or "").strip()
    c_s = (choices_text or "").strip()
    instr_s = (instr_text or "").strip()

    def place_block(tt: int, text: str):
        nonlocal pos
        toks = _tokenize_no_special(tok, text)
        for _ in toks:
            if pos >= L:
                break
            if ttid[pos] != TT_IMAGE:
                ttid[pos] = tt
            pos += 1

    def place_sep():
        nonlocal pos
        toks = _tokenize_no_special(tok, sep)
        for _ in toks:
            if pos >= L:
                break
            if ttid[pos] != TT_IMAGE:
                ttid[pos] = TT_OTHER
            pos += 1

    # Major blocks: hint, (question+choices), instr, cls
    any_prior = False
    if hint_text_s:
        place_block(TT_HINT, hint_text_s)
        any_prior = True

    if q_s or c_s:
        if any_prior:
            place_sep()
        if q_s:
            place_block(TT_QUESTION, q_s)
        if q_s and c_s:
            place_sep()
        if c_s:
            place_block(TT_CHOICES, c_s)
        any_prior = True

    if instr_s:
        if any_prior:
            place_sep()
        place_block(TT_INSTR, instr_s)
        any_prior = True

    # sep before CLS if any previous text existed (your join adds \n\n between parts)
    if any_prior:
        place_sep()

    # CLS token is in the ids; we mark by ID below (more robust than tokenizing "<CLS>")
    for i, tid in enumerate(ids):
        if tid == int(cls_token_id):
            ttid[i] = TT_CLS

    return ttid


# =========================
# Dataset returning raw pieces (PIL + strings)
# =========================
class ScienceQA_Raw(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        require_image: bool = True,
        require_outside_knowledge: bool = True,
        drop_near_blank: bool = True,
        blank_std_thresh: float = 0.01,
        max_samples: int = 0,
    ):
        super().__init__()
        self.split = split
        self.raw_ds, self.keep_indices = load_scienceqa_filtered(
            data_root=data_root,
            split=split,
            require_image=require_image,
            require_outside_knowledge=require_outside_knowledge,
            drop_near_blank=drop_near_blank,
            blank_std_thresh=blank_std_thresh,
        )

        if max_samples and max_samples > 0:
            self.keep_indices = self.keep_indices[:max_samples]

    def __len__(self) -> int:
        return len(self.keep_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ridx = self.keep_indices[idx]
        ex = self.raw_ds[ridx]

        img: Image.Image = ex["image"].convert("RGB")

        hint_text = build_scienceqa_hint_text(ex)

        choices = list(ex.get("choices", []))
        n = len(choices)
        letters = list(LETTERS_POOL[:n])

        question_text = build_question_text(ex.get("question"))
        choices_text = build_choices_text(choices, letters)
        instr_text = build_instruction_text(letters)

        label = int(ex["answer"])
        sample_id = f"{self.split}_{ridx}"

        return {
            "id": sample_id,
            "image": img,
            "hint_text": hint_text,
            "question_text": question_text,
            "choices_text": choices_text,
            "instr_text": instr_text,
            "letters": letters,
            "choices": choices,
            "label": label,
        }


def raw_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "ids": [b["id"] for b in batch],
        "images": [b["image"] for b in batch],
        "hint_texts": [b["hint_text"] for b in batch],
        "question_texts": [b["question_text"] for b in batch],
        "choices_texts": [b["choices_text"] for b in batch],
        "instr_texts": [b["instr_text"] for b in batch],
        "letters": [b["letters"] for b in batch],
        "choices": [b["choices"] for b in batch],
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


# =========================
# Optional: vision embeddings ("vision tokens") caching
# =========================
def _to_cpu_serializable(obj: Any) -> Any:
    """Convert tensors to CPU tensors recursively for saving."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().contiguous()
    if isinstance(obj, dict):
        return {k: _to_cpu_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_cpu_serializable(v) for v in obj]
    return obj


@torch.no_grad()
def compute_qwen3vl_visual_outputs(
    model,
    pixel_values: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor],
) -> Any:
    """
    Attempts to run the vision module to produce "vision tokens" (embeddings).

    Expected API in many Qwen-VL implementations:
      model.model.visual(pixel_values, grid_thw=image_grid_thw)

    Some versions may use different kwarg name; we try a couple.
    Return can be:
      - Tensor (B, N, D) or similar
      - Tuple/List of tensors
      - Dict of tensors (e.g., deepstack/base)
    """
    model.eval()

    if not hasattr(model, "model") or not hasattr(model.model, "visual"):
        raise RuntimeError("Model does not expose model.visual; cannot compute vision embeddings.")

    visual = model.model.visual

    # Try common signatures
    try:
        out = visual(pixel_values, grid_thw=image_grid_thw)
        return out
    except TypeError:
        pass

    try:
        out = visual(pixel_values=pixel_values, grid_thw=image_grid_thw)
        return out
    except TypeError:
        pass

    try:
        out = visual(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        return out
    except TypeError as e:
        raise RuntimeError(f"Could not call model.model.visual with known signatures: {e}")


# =========================
# Cache builder
# =========================
@torch.no_grad()
def build_and_save_cache(
    *,
    data_root: str,
    out_dir: str,
    model_name: str,
    split: str,
    batch_size: int,
    num_workers: int,
    shard_size: int,
    max_samples: int,
    require_image: bool,
    require_outside_knowledge: bool,
    drop_near_blank: bool,
    blank_std_thresh: float,
    cache_image_embeds: bool,
    device: str,
    dtype: str,
):
    os.makedirs(out_dir, exist_ok=True)
    split_out = os.path.join(out_dir, split)
    os.makedirs(split_out, exist_ok=True)

    processor = AutoProcessor.from_pretrained(model_name, cache_dir=data_root)
    tok = processor.tokenizer
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Ensure CLS exists
    tok.add_special_tokens({"additional_special_tokens": ["<CLS>"]})
    cls_token_id = int(tok.convert_tokens_to_ids("<CLS>"))

    # Get image token id/str from config
    cfg = AutoConfig.from_pretrained(model_name, cache_dir=data_root)
    if not hasattr(cfg, "image_token_id"):
        raise RuntimeError("Config has no image_token_id; cannot build image token masks.")
    image_token_id = int(getattr(cfg, "image_token_id"))
    image_token_str = tok.convert_ids_to_tokens(image_token_id)

    # Optional model for visual outputs caching
    model = None
    if cache_image_embeds:
        torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
        if HAVE_QWEN3VL_CLASS:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device,
                cache_dir=data_root,
            )
        else:
            # fallback (may or may not work depending on your transformers build)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device,
                cache_dir=data_root,
                trust_remote_code=True,
            )

    ds = ScienceQA_Raw(
        data_root=data_root,
        split=split,
        require_image=require_image,
        require_outside_knowledge=require_outside_knowledge,
        drop_near_blank=drop_near_blank,
        blank_std_thresh=blank_std_thresh,
        max_samples=max_samples,
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=raw_collate,
        pin_memory=False,
    )

    manifest_path = os.path.join(split_out, "manifest.jsonl")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    meta_path = os.path.join(split_out, "meta.json")

    shard_idx = 0
    shard_items: List[Dict[str, Any]] = []

    def flush_shard():
        nonlocal shard_idx, shard_items
        if not shard_items:
            return
        shard_file = os.path.join(split_out, f"shard_{shard_idx:05d}.pt")
        torch.save(shard_items, shard_file)
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"shard": os.path.basename(shard_file), "num_items": len(shard_items)}) + "\n")
        shard_items = []
        shard_idx += 1

    pbar = tqdm(dl, desc=f"[cache] {split}", total=len(dl))

    for batch in pbar:
        ids = batch["ids"]
        images = batch["images"]
        hint_texts = batch["hint_texts"]
        question_texts = batch["question_texts"]
        choices_texts = batch["choices_texts"]
        instr_texts = batch["instr_texts"]
        labels = batch["labels"]

        # Build full prompts
        full_texts: List[str] = []
        for ht, qt, ct, it in zip(hint_texts, question_texts, choices_texts, instr_texts):
            full_texts.append(
                build_full_prompt(
                    image_token_str=image_token_str,
                    hint_text=ht,
                    question_text=qt,
                    choices_text=ct,
                    instr_text=it,
                    cls_text="<CLS>",
                )
            )

        # Processor outputs (language tokens + image inputs)
        proc = processor(
            text=full_texts,
            images=images,
            padding=True,      # pad for batching; we will store trimmed per-sample
            truncation=True,
            return_tensors="pt",
        )

        input_ids = proc["input_ids"]               # (B, T)
        attention_mask = proc["attention_mask"]     # (B, T)
        pixel_values = proc["pixel_values"]         # (B, C, H, W)
        image_grid_thw = proc.get("image_grid_thw", None)  # (B, 3) typically

        # Optional: compute vision embeddings for the entire batch
        visual_out = None
        if cache_image_embeds:
            assert model is not None
            pv = pixel_values.to(model.device, non_blocking=True)
            thw = image_grid_thw.to(model.device, non_blocking=True) if image_grid_thw is not None else None
            visual_out = compute_qwen3vl_visual_outputs(model, pv, thw)
            visual_out = _to_cpu_serializable(visual_out)

        B = input_ids.size(0)
        for b in range(B):
            true_len = int(attention_mask[b].sum().item())
            # left padding => keep the last true_len tokens
            ids_1d = input_ids[b, -true_len:].contiguous()
            attn_1d = attention_mask[b, -true_len:].contiguous()

            ttid = build_token_type_ids_best_effort(
                tok=tok,
                input_ids_1d=ids_1d,
                attention_mask_1d=attn_1d,
                image_token_id=image_token_id,
                cls_token_id=cls_token_id,
                hint_text=hint_texts[b],
                question_text=question_texts[b],
                choices_text=choices_texts[b],
                instr_text=instr_texts[b],
            )

            masks = {
                "image": (ttid == TT_IMAGE),
                "hint": (ttid == TT_HINT),
                "question": (ttid == TT_QUESTION),
                "choices": (ttid == TT_CHOICES),
                "instr": (ttid == TT_INSTR),
                "cls": (ttid == TT_CLS),
                "other": (ttid == TT_OTHER),
            }

            item: Dict[str, Any] = {
                "id": ids[b],
                "label": labels[b].clone(),
                "input_ids": ids_1d.cpu(),
                "attention_mask": attn_1d.cpu(),
                "token_type_ids": ttid.cpu(),
                "masks": {k: v.cpu() for k, v in masks.items()},
                "pixel_values": pixel_values[b].cpu().contiguous(),
            }
            if image_grid_thw is not None:
                item["image_grid_thw"] = image_grid_thw[b].cpu().contiguous()

            if cache_image_embeds:
                # visual_out can be Tensor(B,...) or dict/list
                if isinstance(visual_out, torch.Tensor):
                    # assume batch dimension first
                    item["image_embeds"] = visual_out[b].contiguous()
                elif isinstance(visual_out, dict):
                    item["image_embeds"] = {k: (v[b].contiguous() if isinstance(v, torch.Tensor) and v.size(0) == B else v)
                                            for k, v in visual_out.items()}
                elif isinstance(visual_out, list):
                    # list of tensors (some with batch dim)
                    embeds_list = []
                    for v in visual_out:
                        if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.size(0) == B:
                            embeds_list.append(v[b].contiguous())
                        else:
                            embeds_list.append(v)
                    item["image_embeds"] = embeds_list
                else:
                    item["image_embeds"] = visual_out  # rare, but keep it

            shard_items.append(item)

            if len(shard_items) >= shard_size:
                flush_shard()

    flush_shard()

    meta = {
        "model_name": model_name,
        "split": split,
        "cls_token_id": cls_token_id,
        "image_token_id": image_token_id,
        "image_token_str": image_token_str,
        "token_type_map": {
            "PAD": TT_PAD,
            "IMAGE": TT_IMAGE,
            "HINT": TT_HINT,
            "QUESTION": TT_QUESTION,
            "CHOICES": TT_CHOICES,
            "INSTR": TT_INSTR,
            "CLS": TT_CLS,
            "OTHER": TT_OTHER,
        },
        "cache_image_embeds": bool(cache_image_embeds),
        "dtype": dtype,
        "padding_side": tok.padding_side,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Wrote cache to: {split_out}")
    print(f"[OK] Manifest: {manifest_path}")
    print(f"[OK] Meta: {meta_path}")


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="HF cache directory")
    ap.add_argument("--out_dir", type=str, required=True, help="Output cache directory")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")

    ap.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--shard_size", type=int, default=4096)
    ap.add_argument("--max_samples", type=int, default=0, help="0 means all filtered samples")

    ap.add_argument("--require_image", action="store_true", default=True)
    ap.add_argument("--require_outside_knowledge", action="store_true", default=True)
    ap.add_argument("--drop_near_blank", action="store_true", default=True)
    ap.add_argument("--blank_std_thresh", type=float, default=0.01)

    ap.add_argument("--cache_image_embeds", action="store_true",
                    help="Also cache vision 'tokens' (embeddings) via model.model.visual(...)")
    ap.add_argument("--device", type=str, default="cuda:0", help='device_map value, e.g. "cuda:0"')
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])

    args = ap.parse_args()

    build_and_save_cache(
        data_root=args.data_root,
        out_dir=args.out_dir,
        model_name=args.model_name,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shard_size=args.shard_size,
        max_samples=args.max_samples,
        require_image=args.require_image,
        require_outside_knowledge=args.require_outside_knowledge,
        drop_near_blank=args.drop_near_blank,
        blank_std_thresh=args.blank_std_thresh,
        cache_image_embeds=args.cache_image_embeds,
        device=args.device,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
#
# CUDA_VISIBLE_DEVICES=4 python ScienceQA_Codebook.py --data_root "/scratch/kkontras/ESNLI" --out_dir "/scratch/kkontras/ScienceQA/cache_tokens8B" --model_name "Qwen/Qwen3-VL-8B-Instruct" --split test
# CUDA_VISIBLE_DEVICES=4 python ScienceQA_Codebook.py --data_root "/scratch/kkontras/ScienceQA" --out_dir "/scratch/kkontras/ScienceQA/cache_tokens8B" --model_name "Qwen/Qwen3-VL-8B-Instruct" --split validation
# CUDA_VISIBLE_DEVICES=4 python ScienceQA_Codebook.py --data_root "/scratch/kkontras/ScienceQA" --out_dir "/scratch/kkontras/ScienceQA/cache_tokens8B" --model_name "Qwen/Qwen3-VL-8B-Instruct"
#
# CUDA_VISIBLE_DEVICES=4 python ScienceQA_Codebook.py --data_root "/scratch/kkontras/ScienceQA" --out_dir "/scratch/kkontras/ScienceQA/cache_tokens32B" --model_name "Qwen/Qwen3-VL-32B-Instruct" --split test
# CUDA_VISIBLE_DEVICES=4 python ScienceQA_Codebook.py --data_root "/scratch/kkontras/ScienceQA" --out_dir "/scratch/kkontras/ScienceQA/cache_tokens32B" --model_name "Qwen/Qwen3-VL-32B-Instruct" --split validation
# CUDA_VISIBLE_DEVICES=4 python ScienceQA_Codebook.py --data_root "/scratch/kkontras/ScienceQA" --out_dir "/scratch/kkontras/ScienceQA/cache_tokens32B" --model_name "Qwen/Qwen3-VL-32B-Instruct"

