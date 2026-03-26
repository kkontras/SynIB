#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TQA (TextbookQA) → Qwen3-VL offline token cache builder.

Dataset structure (after extraction):
  tqa/{train,val,test}/
    tqa_v1_train.json   (list of lessons)
    tqa_v1_val.json
    tqa_v2_test.json
    question_images/    (diagram question images)
    abc_question_images/
    textbook_images/
    teaching_images/

Each lesson JSON structure:
  {
    "lessonName": "...",
    "globalID": "...",
    "topics": {
      "T_xxxx": {
        "topicName": "...",
        "content": {"text": "...", "figures": [...], ...}
      }, ...
    },
    "questions": {
      "diagramQuestions": {
        "DQ_xxxx": {
          "questionType": "Diagram Multiple Choice",
          "beingAsked": {"processedText": "...", "rawText": "..."},
          "answerChoices": {
            "a": {"processedText": "...", "rawText": "..."},
            "b": ..., "c": ..., "d": ...
          },
          "correctAnswer": {"processedText": "b", "rawText": "..."},
          "imagePath": "question_images/xxx.png",
          "globalID": "DQ_xxxx"
        }, ...
      },
      "nonDiagramQuestions": { ... }
    }
  }

Context is built by concatenating all topics[*].content.text.

Outputs:
  out_dir/<split>/
    meta.json
    manifest.jsonl
    shard_00000.pt  ...

Each shard item dict keys:
  id, label,
  input_ids, attention_mask,
  token_type_ids, masks (dict of bool tensors),
  pixel_values, image_grid_thw (if present),
  image_embeds (optional)

Example:
  python TQA_Codebook.py \\
    --data_root /path/to/TQA \\
    --out_dir /path/to/TQA/cache_qwen3_vl_2b \\
    --split train \\
    --batch_size 4 \\
    --cache_image_embeds
"""

import os
import json
import glob
import argparse
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, AutoConfig

try:
    from transformers import Qwen3VLForConditionalGeneration  # type: ignore
    HAVE_QWEN3VL_CLASS = True
except Exception:
    from transformers import AutoModelForCausalLM
    HAVE_QWEN3VL_CLASS = False


LETTERS_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Map from split name to JSON filename
SPLIT_JSON = {
    "train": "tqa_v1_train.json",
    "val":   "tqa_v1_val.json",
    "test":  "tqa_v2_test.json",
}

# =========================
# Token type constants
# =========================
TT_PAD = 0
TT_IMAGE = 1
TT_HINT = 2      # context / passage text
TT_QUESTION = 3
TT_CHOICES = 4
TT_INSTR = 5
TT_CLS = 6
TT_OTHER = 7


# =========================
# TQA data loading
# =========================

def _build_context_from_lesson(lesson: Dict[str, Any]) -> str:
    """Concatenate all topic text blocks for a lesson."""
    topics = lesson.get("topics", {})
    parts = []
    for t in topics.values():
        txt = (t.get("content", {}).get("text") or "").strip()
        if txt:
            parts.append(txt)
    return " ".join(parts)


def load_tqa_items(
    data_root: str,
    split: str,
    *,
    require_image: bool = True,
    require_context: bool = True,
    drop_near_blank: bool = True,
    blank_std_thresh: float = 0.01,
    max_samples: int = 0,
) -> List[Dict[str, Any]]:
    """
    Load and flatten TQA data into a list of question items.

    Returns list of dicts with keys:
      id, lesson_name, context_text, question_text, choices, letters,
      answer_idx, image_path
    """
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"TQA split directory not found: {split_dir}")

    json_name = SPLIT_JSON.get(split)
    if json_name is None:
        raise ValueError(f"Unknown split '{split}'. Expected one of: {list(SPLIT_JSON.keys())}")

    json_path = os.path.join(split_dir, json_name)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"TQA JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        lessons = json.load(f)

    items: List[Dict[str, Any]] = []

    for lesson in lessons:
        lesson_name = lesson.get("lessonName", "")
        context_text = _build_context_from_lesson(lesson)

        if require_context and not context_text.strip():
            continue

        dq = lesson.get("questions", {}).get("diagramQuestions", {})
        if not dq:
            continue

        for qid, qdata in dq.items():
            # Question text
            question_text = (
                (qdata.get("beingAsked") or {}).get("processedText") or ""
            ).strip()
            if not question_text:
                continue

            # Answer choices — keys are lowercase letters a/b/c/d
            answer_choices_raw = qdata.get("answerChoices", {})
            if not answer_choices_raw:
                continue

            sorted_keys = sorted(answer_choices_raw.keys())  # ['a','b','c','d']
            letters = list(LETTERS_POOL[: len(sorted_keys)])  # ['A','B','C','D']
            key_to_letter = {k: l for k, l in zip(sorted_keys, letters)}

            choices_text_list = []
            for k in sorted_keys:
                ch = answer_choices_raw[k]
                if isinstance(ch, dict):
                    txt = (ch.get("processedText") or ch.get("rawText") or "").strip()
                else:
                    txt = str(ch).strip()
                choices_text_list.append(txt)

            # Correct answer — processedText is the lowercase key (e.g. "b")
            correct_raw = (
                (qdata.get("correctAnswer") or {}).get("processedText") or ""
            ).strip().lower()
            answer_letter = key_to_letter.get(correct_raw)
            answer_idx = letters.index(answer_letter) if answer_letter in letters else -1

            # Image path — relative to split_dir
            img_rel = qdata.get("imagePath", "")
            img_abs = os.path.join(split_dir, img_rel) if img_rel else ""

            if require_image and not img_abs:
                continue
            if require_image and img_abs and not os.path.isfile(img_abs):
                continue

            # Near-blank filter
            if drop_near_blank and img_abs and os.path.isfile(img_abs):
                try:
                    pil = Image.open(img_abs).convert("RGB")
                    t = to_tensor(pil)
                    if t.std().item() < blank_std_thresh:
                        continue
                except Exception:
                    continue

            items.append({
                "id": f"{split}_{qid}",
                "lesson_name": lesson_name,
                "context_text": context_text,
                "question_text": question_text,
                "choices": choices_text_list,
                "letters": letters,
                "answer_idx": answer_idx,
                "image_path": img_abs,
            })

            if max_samples and len(items) >= max_samples:
                break
        if max_samples and len(items) >= max_samples:
            break

    print(
        f"[TQA] split={split}: loaded {len(items)} diagram questions "
        f"(require_image={require_image}, require_context={require_context})"
    )
    return items


# =========================
# Text builders
# =========================

def build_tqa_context_text(context: str) -> str:
    ctx = (context or "").strip()
    return ("Context:\n" + ctx) if ctx else ""


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
    context_text: str,
    question_text: str,
    choices_text: str,
    instr_text: str,
    cls_text: str = "<CLS>",
) -> str:
    context_text = (context_text or "").strip()
    question_text = (question_text or "").strip()
    choices_text = (choices_text or "").strip()
    instr_text = (instr_text or "").strip()

    qa_parts = []
    if question_text:
        qa_parts.append(question_text)
    if choices_text:
        if qa_parts:
            qa_parts.append("")
        qa_parts.append(choices_text)
    qa_block = "\n\n".join([p for p in qa_parts if p != ""])

    parts = []
    if context_text:
        parts.append(context_text)
    if qa_block:
        parts.append(qa_block)
    if instr_text:
        parts.append(instr_text)
    parts.append(cls_text)

    prompt = "\n\n".join(parts)
    return image_token_str + "\n" + prompt


# =========================
# Token type IDs
# =========================

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
    context_text: str,
    question_text: str,
    choices_text: str,
    instr_text: str,
) -> torch.Tensor:
    ids = input_ids_1d.tolist()
    L = len(ids)

    ttid = torch.full((L,), TT_OTHER, dtype=torch.uint8)

    for i, tid in enumerate(ids):
        if tid == int(image_token_id):
            ttid[i] = TT_IMAGE

    j = 0
    while j < L and ids[j] == int(image_token_id):
        j += 1
    pos = j

    sep = "\n\n"
    ctx_s = (context_text or "").strip()
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

    any_prior = False
    if ctx_s:
        place_block(TT_HINT, ctx_s)
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

    if any_prior:
        place_sep()

    for i, tid in enumerate(ids):
        if tid == int(cls_token_id):
            ttid[i] = TT_CLS

    return ttid


# =========================
# Raw dataset for DataLoader
# =========================

class TQA_Raw(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        require_image: bool = True,
        require_context: bool = True,
        drop_near_blank: bool = True,
        blank_std_thresh: float = 0.01,
        max_samples: int = 0,
    ):
        super().__init__()
        self.split = split
        self.split_dir = os.path.join(data_root, split)
        self.items = load_tqa_items(
            data_root=data_root,
            split=split,
            require_image=require_image,
            require_context=require_context,
            drop_near_blank=drop_near_blank,
            blank_std_thresh=blank_std_thresh,
            max_samples=max_samples,
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]

        img_path = item["image_path"]
        if img_path and os.path.isfile(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        else:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        context_text = build_tqa_context_text(item["context_text"])
        question_text = build_question_text(item["question_text"])
        choices_text = build_choices_text(item["choices"], item["letters"])
        instr_text = build_instruction_text(item["letters"])

        return {
            "id": item["id"],
            "image": img,
            "context_text": context_text,
            "question_text": question_text,
            "choices_text": choices_text,
            "instr_text": instr_text,
            "letters": item["letters"],
            "label": item["answer_idx"],
        }


def raw_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "ids": [b["id"] for b in batch],
        "images": [b["image"] for b in batch],
        "context_texts": [b["context_text"] for b in batch],
        "question_texts": [b["question_text"] for b in batch],
        "choices_texts": [b["choices_text"] for b in batch],
        "instr_texts": [b["instr_text"] for b in batch],
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


# =========================
# Vision embedding caching
# =========================

def _to_cpu_serializable(obj: Any) -> Any:
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
    model.eval()

    if not hasattr(model, "model") or not hasattr(model.model, "visual"):
        raise RuntimeError("Model does not expose model.visual; cannot compute vision embeddings.")

    visual = model.model.visual

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
    require_context: bool,
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

    tok.add_special_tokens({"additional_special_tokens": ["<CLS>"]})
    cls_token_id = int(tok.convert_tokens_to_ids("<CLS>"))

    cfg = AutoConfig.from_pretrained(model_name, cache_dir=data_root)
    if not hasattr(cfg, "image_token_id"):
        raise RuntimeError("Config has no image_token_id; cannot build image token masks.")
    image_token_id = int(getattr(cfg, "image_token_id"))
    image_token_str = tok.convert_ids_to_tokens(image_token_id)

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
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device,
                cache_dir=data_root,
                trust_remote_code=True,
            )

    ds = TQA_Raw(
        data_root=data_root,
        split=split,
        require_image=require_image,
        require_context=require_context,
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

    pbar = tqdm(dl, desc=f"[tqa cache] {split}", total=len(dl))

    for batch in pbar:
        ids = batch["ids"]
        images = batch["images"]
        context_texts = batch["context_texts"]
        question_texts = batch["question_texts"]
        choices_texts = batch["choices_texts"]
        instr_texts = batch["instr_texts"]
        labels = batch["labels"]

        full_texts: List[str] = []
        for ct, qt, cht, it in zip(context_texts, question_texts, choices_texts, instr_texts):
            full_texts.append(
                build_full_prompt(
                    image_token_str=image_token_str,
                    context_text=ct,
                    question_text=qt,
                    choices_text=cht,
                    instr_text=it,
                    cls_text="<CLS>",
                )
            )

        proc = processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = proc["input_ids"]
        attention_mask = proc["attention_mask"]
        pixel_values = proc["pixel_values"]
        image_grid_thw = proc.get("image_grid_thw", None)

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
            ids_1d = input_ids[b, -true_len:].contiguous()
            attn_1d = attention_mask[b, -true_len:].contiguous()

            ttid = build_token_type_ids_best_effort(
                tok=tok,
                input_ids_1d=ids_1d,
                attention_mask_1d=attn_1d,
                image_token_id=image_token_id,
                cls_token_id=cls_token_id,
                context_text=context_texts[b],
                question_text=question_texts[b],
                choices_text=choices_texts[b],
                instr_text=instr_texts[b],
            )

            masks = {
                "image": (ttid == TT_IMAGE),
                "hint":  (ttid == TT_HINT),
                "question": (ttid == TT_QUESTION),
                "choices": (ttid == TT_CHOICES),
                "instr": (ttid == TT_INSTR),
                "cls":   (ttid == TT_CLS),
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
                if isinstance(visual_out, torch.Tensor):
                    item["image_embeds"] = visual_out[b].contiguous()
                elif isinstance(visual_out, dict):
                    item["image_embeds"] = {
                        k: (v[b].contiguous() if isinstance(v, torch.Tensor) and v.size(0) == B else v)
                        for k, v in visual_out.items()
                    }
                elif isinstance(visual_out, list):
                    embeds_list = []
                    for v in visual_out:
                        if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.size(0) == B:
                            embeds_list.append(v[b].contiguous())
                        else:
                            embeds_list.append(v)
                    item["image_embeds"] = embeds_list
                else:
                    item["image_embeds"] = visual_out

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
    ap.add_argument("--data_root", type=str, required=True,
                    help="Root of extracted TQA directory (contains train/, val/, test/)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output cache directory")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")

    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--shard_size", type=int, default=4096)
    ap.add_argument("--max_samples", type=int, default=0, help="0 means all samples")

    ap.add_argument("--require_image", action="store_true", default=True)
    ap.add_argument("--no_require_image", dest="require_image", action="store_false")
    ap.add_argument("--require_context", action="store_true", default=True)
    ap.add_argument("--no_require_context", dest="require_context", action="store_false")
    ap.add_argument("--drop_near_blank", action="store_true", default=True)
    ap.add_argument("--blank_std_thresh", type=float, default=0.01)

    ap.add_argument("--cache_image_embeds", action="store_true",
                    help="Also cache vision embeddings via model.model.visual(...)")
    ap.add_argument("--device", type=str, default="cuda:0")
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
        require_context=args.require_context,
        drop_near_blank=args.drop_near_blank,
        blank_std_thresh=args.blank_std_thresh,
        cache_image_embeds=args.cache_image_embeds,
        device=args.device,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
#
# CUDA_VISIBLE_DEVICES=0 python TQA_Codebook.py --data_root /path/to/TQA --out_dir /path/to/TQA/cache_qwen3_vl_2b --split train --batch_size 4 --cache_image_embeds
# CUDA_VISIBLE_DEVICES=0 python TQA_Codebook.py --data_root /path/to/TQA --out_dir /path/to/TQA/cache_qwen3_vl_2b --split val   --batch_size 4 --cache_image_embeds
# CUDA_VISIBLE_DEVICES=0 python TQA_Codebook.py --data_root /path/to/TQA --out_dir /path/to/TQA/cache_qwen3_vl_2b --split test  --batch_size 4 --cache_image_embeds
