#!/usr/bin/env python3
"""
Build a shared-sample Qwen3-VL cache for raw FactorCL VT datasets.

Expected raw root:
  <dataset>_raw/
    metadata.jsonl
    folds.json              # optional
    media/                  # optional, for relative video_path references

metadata.jsonl rows:
  {
    "id": "sample-id",
    "text": "utterance text",
    "label": 0,
    "video_path": "media/foo.mp4",
    "context": ["optional", "context"],
    "folds": {"0": "train", "1": "val", "2": "test"}   # optional
  }

folds.json schema:
  {
    "0": {"train": ["id1"], "val": ["id2"], "test": ["id3"]},
    "1": ...
  }
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from synib.mydatasets.MUStARD.MUStARD_CodeBook import (
    _is_cuda_invalid_argument_error,
    _normalize_pos_to_B_3_1_T,
    _stack_deep_levels_per_sample,
    _unpack_image_feature_outputs,
    verify_shard,
)

TT_PAD = 0
TT_IMAGE = 1
TT_TEXT = 2
TT_CLS = 3
TT_OTHER = 4

TASK_PROMPTS = {
    "mosi": "Is the speaker's sentiment positive or negative? <CLS>",
    "mosei": "Is the speaker's sentiment positive or negative? <CLS>",
    "ur_funny": "Is this utterance intended to be funny? <CLS>",
    "mustard": 'Is the speaker being sarcastic? <CLS>',
}


def _load_metadata(raw_root: Path) -> List[Dict[str, Any]]:
    metadata_path = raw_root / "metadata.jsonl"
    if metadata_path.exists():
        rows = []
        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    legacy_metadata = raw_root / "metadata.json"
    if legacy_metadata.exists():
        with legacy_metadata.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        norm = []
        for row in rows:
            item = dict(row)
            item.setdefault("text", item.get("utterance", ""))
            if "label" not in item:
                if "sarcasm" in item:
                    item["label"] = int(bool(item["sarcasm"]))
                elif "sentiment" in item:
                    item["label"] = int(item["sentiment"])
            if "video_path" not in item:
                item["video_path"] = f"videos/{item['id']}.mp4"
            norm.append(item)
        return norm

    raise FileNotFoundError(f"Need metadata.jsonl or metadata.json under {raw_root}")


def _load_fold_mapping(raw_root: Path, rows: List[Dict[str, Any]], folds: List[int]) -> Dict[int, Dict[str, List[str]]]:
    folds_path = raw_root / "folds.json"
    if folds_path.exists():
        payload = json.loads(folds_path.read_text(encoding="utf-8"))
        out: Dict[int, Dict[str, List[str]]] = {}
        for fold in folds:
            fold_payload = payload.get(str(fold), {})
            out[fold] = {
                "train": list(fold_payload.get("train", [])),
                "val": list(fold_payload.get("val", fold_payload.get("validation", []))),
                "test": list(fold_payload.get("test", [])),
            }
        return out

    out = {int(f): {"train": [], "val": [], "test": []} for f in folds}
    default_split = None
    for row in rows:
        row_id = str(row["id"])
        if "folds" in row:
            for fold_str, split in row["folds"].items():
                fold = int(fold_str)
                if fold in out:
                    split_name = "val" if str(split).lower() in {"val", "validation", "dev"} else str(split).lower()
                    if split_name not in out[fold]:
                        continue
                    out[fold][split_name].append(row_id)
            continue

        split = row.get("split", default_split or "train")
        split_name = "val" if str(split).lower() in {"val", "validation", "dev"} else str(split).lower()
        for fold in folds:
            if split_name in out[fold]:
                out[fold][split_name].append(row_id)
    return out


def _extract_frames(video_path: Path, fps: float, image_size: int) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return torch.zeros(1, 3, image_size, image_size)

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    duration_secs = total_frames / native_fps
    n_target = max(1, int(duration_secs * fps))
    target_indices = [
        min(int(i * native_fps / fps), total_frames - 1)
        for i in range(n_target)
    ]
    target_set = set(target_indices)
    stop_at = max(target_set)

    frame_map: Dict[int, torch.Tensor] = {}
    frame_idx = 0
    last_tensor: Optional[torch.Tensor] = None
    while frame_idx <= stop_at:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target_set:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, (image_size, image_size))
            t = torch.from_numpy(resized.copy()).permute(2, 0, 1).float() / 255.0
            frame_map[frame_idx] = t
            last_tensor = t
        frame_idx += 1
    cap.release()

    zero = torch.zeros(3, image_size, image_size)
    fallback = last_tensor if last_tensor is not None else zero
    frames = [frame_map.get(fi, fallback) for fi in target_indices]
    return torch.stack(frames, dim=0)


def _build_prompt(
    *,
    processor,
    text: str,
    dataset_name: str,
    context: Optional[List[str]],
    n_frames: int,
) -> str:
    parts: List[str] = []
    if context:
        parts.append("Context:")
        parts.extend([str(c) for c in context if str(c).strip()])
    parts.append(f'Utterance: "{text}"')
    parts.append(TASK_PROMPTS[dataset_name])
    prompt_text = "\n".join(parts)

    content = [{"type": "image"} for _ in range(n_frames)]
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _build_token_type_ids(input_ids_1d: torch.Tensor, *, image_token_id: int, cls_token_id: int) -> torch.Tensor:
    ids = input_ids_1d.reshape(-1).to(torch.long)
    out = torch.full_like(ids, TT_TEXT)
    out[ids == image_token_id] = TT_IMAGE
    out[ids == cls_token_id] = TT_CLS
    out[ids == 0] = TT_PAD
    return out.unsqueeze(0)


@torch.no_grad()
def build_shared_cache(
    *,
    dataset_name: str,
    raw_root: str,
    out_dir: str,
    model_name: str,
    folds: List[int],
    fps: float,
    image_size: int,
    shard_size: int,
    device: str,
    dtype: str,
    local_files_only: bool,
) -> None:
    raw_root_p = Path(raw_root).expanduser().resolve()
    out_dir_p = Path(out_dir).expanduser().resolve()
    samples_dir = out_dir_p / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    hf_cache = out_dir_p / "hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)

    rows = _load_metadata(raw_root_p)
    rows_by_id = {str(r["id"]): r for r in rows}
    fold_mapping = _load_fold_mapping(raw_root_p, rows, folds)

    processor = AutoProcessor.from_pretrained(
        model_name, cache_dir=str(hf_cache), local_files_only=bool(local_files_only)
    )
    tok = processor.tokenizer
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": ["<CLS>"]})
    cls_token_id = int(tok.convert_tokens_to_ids("<CLS>"))

    cfg = AutoConfig.from_pretrained(
        model_name, cache_dir=str(hf_cache), local_files_only=bool(local_files_only)
    )
    image_token_id = int(getattr(cfg, "image_token_id"))

    model_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    load_kwargs = {
        "trust_remote_code": True,
        "cache_dir": str(hf_cache),
        "torch_dtype": model_dtype,
        "local_files_only": bool(local_files_only),
    }
    if str(device).startswith("cuda"):
        load_kwargs["device_map"] = {"": device}
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
    if not str(device).startswith("cuda"):
        model = model.to(device)
    model.eval()

    sample_manifest_path = out_dir_p / "samples_manifest.jsonl"
    seen_existing = set()
    if sample_manifest_path.exists():
        with sample_manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    seen_existing.add(json.loads(line)["id"])

    unique_ids = sorted({sid for split_map in fold_mapping.values() for ids in split_map.values() for sid in ids})
    shard_items: List[Dict[str, Any]] = []
    shard_idx = len(sorted(glob.glob(str(samples_dir / "shard_*.pt"))))

    def flush_shard() -> None:
        nonlocal shard_items, shard_idx
        if not shard_items:
            return
        shard_path = samples_dir / f"shard_{shard_idx:05d}.pt"
        torch.save(shard_items, shard_path)
        with sample_manifest_path.open("a", encoding="utf-8") as f:
            for item in shard_items:
                f.write(json.dumps({"id": item["id"], "shard": shard_path.name}) + "\n")
        verify_shard(str(shard_path), n_show=1)
        shard_items = []
        shard_idx += 1

    start_time = time.time()
    for sample_id in tqdm(unique_ids, desc=f"[factorcl-cache] {dataset_name}"):
        if sample_id in seen_existing:
            continue
        row = rows_by_id[sample_id]
        rel_video_path = row.get("video_path")
        if not rel_video_path:
            raise KeyError(f"Sample {sample_id} missing video_path")
        video_path = (raw_root_p / rel_video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Sample {sample_id} video not found: {video_path}")

        frames = _extract_frames(video_path, fps=fps, image_size=image_size)
        pil_images = [to_pil_image(frames[i].clamp(0.0, 1.0)) for i in range(int(frames.shape[0]))]
        prompt = _build_prompt(
            processor=processor,
            text=str(row.get("text", row.get("utterance", ""))),
            dataset_name=dataset_name,
            context=row.get("context", []),
            n_frames=len(pil_images),
        )
        proc = processor(
            text=[prompt],
            images=pil_images if pil_images else None,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)
        pixel_values = proc["pixel_values"].to(device)
        image_grid_thw_cpu = proc["image_grid_thw"]
        image_grid_thw_dev = image_grid_thw_cpu.to(device)

        token_embeds = model.model.get_input_embeddings()(input_ids)
        image_grid_thw_for_ops = image_grid_thw_dev
        try:
            image_feat_out = model.get_image_features(pixel_values, image_grid_thw_for_ops)
        except RuntimeError as e:
            if _is_cuda_invalid_argument_error(e):
                image_grid_thw_for_ops = image_grid_thw_cpu
                image_feat_out = model.get_image_features(pixel_values, image_grid_thw_for_ops)
            else:
                raise

        image_embeds_list, deep_stack_viz_list = _unpack_image_feature_outputs(image_feat_out)
        image_embeds_cat = torch.cat(image_embeds_list, dim=0).to(token_embeds.device, token_embeds.dtype)
        placeholder_mask, _ = model.model.get_placeholder_mask(
            input_ids, inputs_embeds=token_embeds, image_features=image_embeds_cat
        )
        placeholder_mask_2d = placeholder_mask[..., 0] if placeholder_mask.dim() == 3 else placeholder_mask
        token_embeds = token_embeds.masked_scatter(placeholder_mask, image_embeds_cat)

        try:
            position_ids, _ = model.model.get_rope_index(
                input_ids, image_grid_thw_for_ops, None, attention_mask=attention_mask
            )
        except RuntimeError as e:
            if _is_cuda_invalid_argument_error(e) and image_grid_thw_for_ops is image_grid_thw_dev:
                position_ids, _ = model.model.get_rope_index(
                    input_ids, image_grid_thw_cpu, None, attention_mask=attention_mask
                )
            else:
                raise

        input_ids_cpu = input_ids.detach().cpu()
        attention_cpu = attention_mask.detach().cpu()
        token_embeds_cpu = token_embeds.detach().cpu()
        placeholder_mask_2d_cpu = placeholder_mask_2d.detach().cpu()
        pos_b = _normalize_pos_to_B_3_1_T(position_ids.detach().cpu(), 1)
        deep_per_sample = _stack_deep_levels_per_sample(deep_stack_viz_list, 1)

        keep = attention_cpu[0].bool()
        input_ids_keep = input_ids_cpu[0][keep].contiguous().unsqueeze(0)
        attention_keep = attention_cpu[0][keep].contiguous().unsqueeze(0)
        input_embeds_keep = token_embeds_cpu[0][keep].contiguous().unsqueeze(0)
        visual_pos_masks = placeholder_mask_2d_cpu[0][keep].contiguous().unsqueeze(0).bool()
        pos_keep = pos_b[0][:, :, keep].contiguous()
        deepstack_visual_embeds = deep_per_sample[0]
        token_type_ids = _build_token_type_ids(
            input_ids_keep, image_token_id=image_token_id, cls_token_id=cls_token_id
        )

        item = {
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "id": sample_id,
            "prompt": prompt,
            "token_type_ids": token_type_ids.cpu(),
            "masks": {
                "image": (token_type_ids == TT_IMAGE).cpu(),
                "hint": (token_type_ids == TT_TEXT).cpu(),
                "cls": (token_type_ids == TT_CLS).cpu(),
                "other": (token_type_ids == TT_OTHER).cpu(),
            },
            "input_ids": input_ids_keep,
            "attention_mask": attention_keep,
            "position_ids": pos_keep,
            "input_embeds": input_embeds_keep,
            "visual_pos_masks": visual_pos_masks,
            "deepstack_visual_embeds": deepstack_visual_embeds,
        }
        shard_items.append(item)
        seen_existing.add(sample_id)
        if len(shard_items) >= shard_size:
            flush_shard()

    flush_shard()

    for fold in folds:
        fold_dir = out_dir_p / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            manifest_path = fold_dir / f"{split}_manifest.jsonl"
            with manifest_path.open("w", encoding="utf-8") as f:
                for sample_id in fold_mapping.get(fold, {}).get(split, []):
                    f.write(json.dumps({"id": sample_id}) + "\n")

    meta = {
        "dataset": dataset_name,
        "model_name": model_name,
        "folds": folds,
        "fps": fps,
        "image_size": image_size,
        "dtype": dtype,
        "cache_mode": "shared_samples_with_fold_manifests",
        "elapsed_s": int(time.time() - start_time),
    }
    with (out_dir_p / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a shared-sample Qwen3-VL cache for raw FactorCL VT datasets.")
    ap.add_argument("--dataset", type=str, required=True, choices=["mosi", "mosei", "ur_funny", "mustard"])
    ap.add_argument("--raw_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--shard_size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()

    build_shared_cache(
        dataset_name=args.dataset,
        raw_root=args.raw_root,
        out_dir=args.out_dir,
        model_name=args.model_name,
        folds=list(args.folds),
        fps=args.fps,
        image_size=args.image_size,
        shard_size=args.shard_size,
        device=args.device,
        dtype=args.dtype,
        local_files_only=bool(args.local_files_only),
    )


if __name__ == "__main__":
    main()
