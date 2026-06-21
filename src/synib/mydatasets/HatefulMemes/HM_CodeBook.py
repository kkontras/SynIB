"""
Cache builder for Hateful Memes.

Precomputes frozen text + vision encoder features once per split and writes them
as sharded .pt files (list of per-example dicts) + a manifest.

Per-example record schema:
    {
        "id":         str,
        "label":      int (-1 if missing, e.g. test_unseen),
        "text":       str,
        "img_tokens": FloatTensor (N_img, D_v) fp16,
        "img_pool":   FloatTensor (D_v,)       fp16,
        "txt_tokens": FloatTensor (N_txt, D_t) fp16,
        "txt_pool":   FloatTensor (D_t,)       fp16,
        "txt_mask":   BoolTensor  (N_txt,),
    }

Usage:
    python -m synib.mydatasets.HatefulMemes.HM_CodeBook \
        --data_root /data/HatefulMemes \
        --image_subdir img_clean \
        --out_dir /data/HatefulMemes/cache_clip_b16_deberta_base \
        --split train \
        --vision_ckpt openai/clip-vit-base-patch16 \
        --text_ckpt microsoft/deberta-v3-base \
        --text_kind encoder
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .hm_utils import iter_hm_jsonl, record_label, resolve_image_path, verify_hm_layout


log = logging.getLogger("HM_CodeBook")


def _build_vision(ckpt: str, device: torch.device, dtype: torch.dtype):
    """Returns (model, processor, native_dim, token_count)."""
    from transformers import CLIPImageProcessor, CLIPVisionModel

    model = CLIPVisionModel.from_pretrained(ckpt).to(device=device, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    proc = CLIPImageProcessor.from_pretrained(ckpt)
    return model, proc


def _build_text(ckpt: str, kind: str, device: torch.device, dtype: torch.dtype):
    """kind in {"encoder", "decoder"}."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(ckpt).to(device=device, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def _encode_text_batch(model, tokenizer, texts: List[str], kind: str,
                       device: torch.device, max_length: int = 64):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=False, return_dict=True)
    hidden = out.last_hidden_state  # (B, T, D)
    mask = attention_mask.bool()

    if kind == "encoder":
        # CLS token is at position 0 for DeBERTa/BERT.
        pooled = hidden[:, 0, :]
    else:
        # decoder-only LM: last non-pad token per row.
        lengths = attention_mask.sum(dim=1)
        idx = (lengths - 1).clamp(min=0)
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]

    return hidden, pooled, mask


def _encode_vision_batch(model, proc, images: List[Image.Image], device: torch.device):
    pix = proc(images=images, return_tensors="pt")["pixel_values"].to(device=device,
                                                                      dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        out = model(pixel_values=pix, return_dict=True)
    hidden = out.last_hidden_state  # (B, N_img, D)
    pooled = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else hidden[:, 0, :]
    return hidden, pooled


def _write_shard(records, out_dir: str, shard_idx: int) -> dict:
    shard_name = f"shard_{shard_idx:05d}.pt"
    shard_path = os.path.join(out_dir, shard_name)
    torch.save(records, shard_path)
    return {"shard": shard_name, "num_items": len(records)}


def build_cache(
    data_root: str,
    out_dir: str,
    split: str,
    vision_ckpt: str,
    text_ckpt: str,
    *,
    text_kind: str = "encoder",
    image_subdir: str = "img_clean",
    image_size: int = 224,
    batch_size: int = 16,
    shard_size: int = 2000,
    max_text_length: int = 64,
    device: Optional[str] = None,
    fp_dtype: str = "float16",
):
    paths = verify_hm_layout(data_root, image_subdir=image_subdir)
    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[fp_dtype]

    log.info(f"Loading vision encoder {vision_ckpt}")
    v_model, v_proc = _build_vision(vision_ckpt, dev, dtype)
    log.info(f"Loading text encoder {text_ckpt} (kind={text_kind})")
    t_model, t_tok = _build_text(text_ckpt, text_kind, dev, dtype)

    tf = transforms.Compose([transforms.Resize((image_size, image_size))])

    shard_records: list = []
    shard_idx = 0
    manifest_entries: list = []
    buffer_images: list = []
    buffer_meta: list = []  # (id, text, label)

    def flush_buffer():
        if not buffer_meta:
            return
        texts = [m[1] for m in buffer_meta]
        # vision
        v_hidden, v_pool = _encode_vision_batch(v_model, v_proc, buffer_images, dev)
        # text
        t_hidden, t_pool, t_mask = _encode_text_batch(
            t_model, t_tok, texts, text_kind, dev, max_length=max_text_length
        )

        for i, (rid, text, lab) in enumerate(buffer_meta):
            # keep only valid token positions for text
            keep = t_mask[i]
            t_tokens = t_hidden[i][keep].to(dtype).cpu()
            t_pooled = t_pool[i].to(dtype).cpu()
            shard_records.append({
                "id": str(rid),
                "label": int(lab),
                "text": text,
                "img_tokens": v_hidden[i].to(dtype).cpu(),
                "img_pool": v_pool[i].to(dtype).cpu(),
                "txt_tokens": t_tokens,
                "txt_pool": t_pooled,
                "txt_mask": torch.ones(t_tokens.shape[0], dtype=torch.bool),
            })

        buffer_images.clear()
        buffer_meta.clear()

    for rec in tqdm(iter_hm_jsonl(data_root, split), desc=f"hm[{split}]"):
        lab = record_label(rec)
        lab = -1 if lab is None else int(lab)
        img_path = resolve_image_path(rec, paths["image_dir"])
        if not os.path.isfile(img_path):
            log.warning(f"Missing image {img_path!r}, skipping record {rec.get('id')}")
            continue
        with Image.open(img_path) as im:
            im = tf(im.convert("RGB"))
        text = str(rec.get("text", "")).strip()
        buffer_images.append(im)
        buffer_meta.append((rec.get("id"), text, lab))

        if len(buffer_meta) >= batch_size:
            flush_buffer()

        if len(shard_records) >= shard_size:
            manifest_entries.append(_write_shard(shard_records, split_dir, shard_idx))
            shard_records = []
            shard_idx += 1

    if buffer_meta:
        flush_buffer()

    if shard_records:
        manifest_entries.append(_write_shard(shard_records, split_dir, shard_idx))

    manifest_path = os.path.join(split_dir, "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    meta_path = os.path.join(split_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "vision_ckpt": vision_ckpt,
            "text_ckpt": text_ckpt,
            "text_kind": text_kind,
            "image_subdir": image_subdir,
            "image_size": image_size,
            "max_text_length": max_text_length,
            "num_shards": len(manifest_entries),
            "num_items": sum(e["num_items"] for e in manifest_entries),
        }, f, indent=2)

    log.info(f"Wrote {len(manifest_entries)} shards ({sum(e['num_items'] for e in manifest_entries)} items) to {split_dir}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build HM encoder-feature cache.")
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split", required=True, choices=("train", "dev_seen", "dev_unseen", "test_seen", "test_unseen"))
    p.add_argument("--vision_ckpt", required=True)
    p.add_argument("--text_ckpt", required=True)
    p.add_argument("--text_kind", default="encoder", choices=("encoder", "decoder"))
    p.add_argument("--image_subdir", default="img_clean")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--shard_size", type=int, default=2000)
    p.add_argument("--max_text_length", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--fp_dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    p.add_argument("--verbose", action="store_true")
    return p


def main():
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    build_cache(
        data_root=args.data_root,
        out_dir=args.out_dir,
        split=args.split,
        vision_ckpt=args.vision_ckpt,
        text_ckpt=args.text_ckpt,
        text_kind=args.text_kind,
        image_subdir=args.image_subdir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        max_text_length=args.max_text_length,
        device=args.device,
        fp_dtype=args.fp_dtype,
    )


if __name__ == "__main__":
    main()
