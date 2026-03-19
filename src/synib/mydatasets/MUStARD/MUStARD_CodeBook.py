#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MUStARD_CodeBook.py

Cache builder for MUStARD sarcasm detection.

Produces per-sample .pt shards with:
  input_ids              : (1, L) int64
  attention_mask         : (1, L) int64
  position_ids           : (3, 1, L) int64
  input_embeds           : (1, L, 2048) float16
  visual_pos_masks       : (1, L) bool
  deepstack_visual_embeds: (K, 64, 2048) float16
  label                  : LongTensor scalar (0 = not sarcastic, 1 = sarcastic)
  masks                  : dict of bool tensors {image, hint, cls, other}
"""
print("[BOOTSTRAP] MUStARD_CodeBook starting import sequence...", flush=True)

import os
import json
import glob as _glob
import argparse
import time
from typing import Any, Dict, List, Optional, Tuple


def _bootstrap_import(label: str, stmt: str) -> None:
    t0 = time.time()
    print(f"[BOOTSTRAP] importing {label} ...", flush=True)
    exec(stmt, globals())
    dt = time.time() - t0
    print(f"[BOOTSTRAP] imported {label} ({dt:.2f}s)", flush=True)


os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_bootstrap_import("numpy", "import numpy as np")
_bootstrap_import("torch", "import torch")
_bootstrap_import("torch dataloader", "from torch.utils.data import Dataset, DataLoader")
_bootstrap_import("torchvision", "from torchvision.transforms.functional import to_pil_image")
_bootstrap_import("PIL", "from PIL import Image")
_bootstrap_import("cv2", "import cv2")
_bootstrap_import("sklearn", "from sklearn.model_selection import train_test_split")
_bootstrap_import(
    "transformers symbols",
    "from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, AutoConfig",
)
_bootstrap_import("tqdm", "from tqdm import tqdm")
print("[BOOTSTRAP] MUStARD_CodeBook imports ready.", flush=True)


# ---------------------------------------------------------------------------
# Token type IDs  (must match ESNLI_CodeBook_v3 / ESNLI_CB)
# ---------------------------------------------------------------------------
TT_PAD   = 0
TT_IMAGE = 1
TT_TEXT  = 2
TT_CLS   = 3
TT_OTHER = 4


# ---------------------------------------------------------------------------
# MUStARD dataset – inline split logic from MUStARD_Dataset.py
# ---------------------------------------------------------------------------

class MUStARD_ForCache(Dataset):
    """
    Minimal dataset that replicates MUStARD_RawDataset split logic:
      stratified train / val / test from metadata.json
      seed=109, val_rate=0.15, test_rate=0.15
    """

    def __init__(
        self,
        data_roots: str,
        split: str,
        fps: float = 1.0,
        image_size: int = 224,
        seed: int = 109,
        val_split_rate: float = 0.15,
        test_split_rate: float = 0.15,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.split = split
        self.fps = fps
        self.image_size = image_size

        # Locate metadata.json and videos/
        if os.path.isfile(data_roots):
            metadata_path = data_roots
            self.videos_dir = os.path.join(os.path.dirname(metadata_path), "videos")
        else:
            metadata_path = os.path.join(data_roots, "metadata.json")
            self.videos_dir = os.path.join(data_roots, "videos")

        with open(metadata_path) as f:
            all_records = json.load(f)

        # Stratified train / val / test split
        labels = [int(r["sarcasm"]) for r in all_records]
        train_val, test_records = train_test_split(
            all_records,
            test_size=test_split_rate,
            random_state=seed,
            stratify=labels,
        )
        tv_labels = [int(r["sarcasm"]) for r in train_val]
        val_frac = val_split_rate / (1.0 - test_split_rate)
        train_records, val_records = train_test_split(
            train_val,
            test_size=val_frac,
            random_state=seed,
            stratify=tv_labels,
        )

        if split == "train":
            self.records = train_records
        elif split in ("val", "validation", "dev"):
            self.records = val_records
        elif split == "test":
            self.records = test_records
        else:
            raise ValueError(f"Unknown split: {split!r}. Use 'train', 'val', or 'test'.")

        if max_samples is not None and max_samples > 0:
            self.records = self.records[:max_samples]

        print(
            f"[MUStARD_ForCache] split={split} num_samples={len(self.records)} "
            f"fps={fps} image_size={image_size}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.records)

    def _extract_frames(self, video_path: str) -> "torch.Tensor":
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] Could not open video: {video_path} — returning zero frame.", flush=True)
            return torch.zeros(1, 3, self.image_size, self.image_size)

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        duration_secs = total_frames / native_fps
        n_target = max(1, int(duration_secs * self.fps))

        target_indices = [
            min(int(i * native_fps / self.fps), total_frames - 1)
            for i in range(n_target)
        ]
        target_set = set(target_indices)
        stop_at = max(target_set)

        frame_map: Dict[int, "torch.Tensor"] = {}
        frame_idx = 0
        last_tensor: Optional["torch.Tensor"] = None

        while frame_idx <= stop_at:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in target_set:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(frame_rgb, (self.image_size, self.image_size))
                t = torch.from_numpy(resized.copy()).permute(2, 0, 1).float() / 255.0
                frame_map[frame_idx] = t
                last_tensor = t
            frame_idx += 1

        cap.release()

        zero = torch.zeros(3, self.image_size, self.image_size)
        fallback = last_tensor if last_tensor is not None else zero
        frames = [frame_map.get(fi, fallback) for fi in target_indices]
        return torch.stack(frames, dim=0)  # (n_target, 3, H, W)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        video_path = os.path.join(self.videos_dir, f"{rec['id']}.mp4")
        video_frames = self._extract_frames(video_path)

        return {
            "text": rec["utterance"],
            "speaker": rec["speaker"],
            "context": list(rec.get("context", [])),
            "video_frames": video_frames,   # (F, 3, H, W)
            "label": int(rec["sarcasm"]),
            "id": rec["id"],
        }


# ---------------------------------------------------------------------------
# Prompt building  (mirrors qwen_mustard_prompt._build_sarcasm_messages)
# ---------------------------------------------------------------------------

def _build_sarcasm_prompt(
    processor,
    text: str,
    speaker: str,
    context: List[str],
    n_frames: int,
    max_context_turns: int = 3,
) -> str:
    """Return a chat-template prompt string for a single MUStARD sample."""
    ctx_turns = list(context)[-max_context_turns:] if context else []

    parts = [f"Speaker: {speaker}"]
    if ctx_turns:
        parts.append("Context:")
        parts.extend(ctx_turns)
    parts.append(f'\nUtterance: "{text}"')
    parts.append("\nIs the speaker being sarcastic? <CLS>")
    prompt_text = "\n".join(parts)

    content = [{"type": "image"} for _ in range(n_frames)]
    content.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Helpers copied verbatim from ESNLI_CodeBook_v3.py
# ---------------------------------------------------------------------------

def _find_subseq(haystack, needle, start=0, end=None):
    if end is None:
        end = len(haystack)
    n = len(needle)
    if n == 0:
        return -1
    last = end - n
    for i in range(start, last + 1):
        if haystack[i:i+n] == needle:
            return i
    return -1


def _normalize_pos_to_B_3_1_T(position_ids: "torch.Tensor", B: int) -> "torch.Tensor":
    pos = position_ids

    if pos.dim() == 3 and tuple(pos.shape[:2]) == (3, 1):
        if B != 1:
            raise RuntimeError(f"Got position_ids (3,1,T) but B={B}.")
        return pos.unsqueeze(0)  # (1,3,1,T)

    if pos.dim() == 4 and int(pos.shape[0]) == B and tuple(pos.shape[1:3]) == (3, 1):
        return pos  # (B,3,1,T)

    if pos.dim() == 3 and int(pos.shape[0]) == B and int(pos.shape[1]) == 3:
        return pos.unsqueeze(2)  # (B,3,1,T)

    if pos.dim() == 3 and int(pos.shape[0]) == 3 and int(pos.shape[1]) == B:
        return pos.permute(1, 0, 2).unsqueeze(2)  # (B,3,1,T)

    if pos.dim() == 4 and int(pos.shape[0]) == 3 and int(pos.shape[1]) == B:
        return pos.permute(1, 0, 2, 3)  # (B,3,1,T)

    raise RuntimeError(f"Unrecognized position_ids shape={tuple(pos.shape)} for B={B}")


def _stack_deep_levels_per_sample(deep_stack_viz_list: Any, B: int) -> List["torch.Tensor"]:
    if not isinstance(deep_stack_viz_list, (list, tuple)):
        return [torch.empty((0, 64, 2048), dtype=torch.float16) for _ in range(B)]

    levels = [t for t in deep_stack_viz_list if torch.is_tensor(t)]
    if len(levels) == 0:
        return [torch.empty((0, 64, 2048), dtype=torch.float16) for _ in range(B)]

    per_sample_levels: List[List["torch.Tensor"]] = [[] for _ in range(B)]

    for lvl in levels:
        t = lvl.detach().cpu()

        # Case 1: 2D (total_tokens, 2048)
        if t.dim() == 2 and int(t.shape[1]) == 2048:
            total = int(t.shape[0])
            if B == 1:
                # All tokens belong to the single sample (any N, e.g. 64, 256, ...)
                per_sample_levels[0].append(t)
            elif total % B == 0:
                n_per = total // B
                tb = t.view(B, n_per, 2048)
                for i in range(B):
                    per_sample_levels[i].append(tb[i])
            else:
                raise RuntimeError(
                    f"deep level shape {tuple(t.shape)} total tokens {total} not divisible by B={B}"
                )
            continue

        # Case 2: 3D (B, N, 2048) for any N
        if t.dim() == 3 and int(t.shape[0]) == B and int(t.shape[2]) == 2048:
            for i in range(B):
                per_sample_levels[i].append(t[i])
            continue

        raise RuntimeError(
            f"Unexpected deep level shape {tuple(t.shape)}. B={B}"
        )

    out: List["torch.Tensor"] = []
    for i in range(B):
        if len(per_sample_levels[i]) == 0:
            out.append(torch.empty((0, 64, 2048), dtype=torch.float16))
        else:
            out.append(torch.cat([x.unsqueeze(0) for x in per_sample_levels[i]], dim=0))
    return out


def _unpack_image_feature_outputs(feat_out: Any) -> Tuple[List["torch.Tensor"], Any]:
    if isinstance(feat_out, tuple):
        if len(feat_out) == 0:
            raise RuntimeError("get_image_features returned an empty tuple.")
        image_part = feat_out[0]
        deep_part = feat_out[1] if len(feat_out) > 1 else []
    else:
        image_part = feat_out
        deep_part = []
        if hasattr(feat_out, "pooler_output") or hasattr(feat_out, "last_hidden_state"):
            pooler = getattr(feat_out, "pooler_output", None)
            hidden = getattr(feat_out, "last_hidden_state", None)
            deep = getattr(feat_out, "deepstack_features", None)
            image_part = pooler if pooler is not None else hidden
            deep_part = deep if deep is not None else []
        elif isinstance(feat_out, dict):
            pooler = feat_out.get("pooler_output", None)
            hidden = feat_out.get("last_hidden_state", None)
            deep = feat_out.get("deepstack_features", None)
            image_part = pooler if pooler is not None else (hidden if hidden is not None else feat_out)
            deep_part = deep if deep is not None else []

    if torch.is_tensor(image_part):
        image_embeds_list = [image_part]
    elif isinstance(image_part, (list, tuple)):
        image_embeds_list = [t for t in image_part if torch.is_tensor(t)]
    else:
        raise RuntimeError(f"Unexpected image feature type: {type(image_part)}")

    if len(image_embeds_list) == 0:
        raise RuntimeError("No tensor image embeddings produced by get_image_features.")

    return image_embeds_list, deep_part


def _is_cuda_invalid_argument_error(err: BaseException) -> bool:
    msg = str(err).lower()
    return ("cuda" in msg) and ("invalid argument" in msg)


def _tensor_stats_line(name: str, t: Any) -> str:
    if not torch.is_tensor(t):
        return f"{name:22s}: {type(t)}"
    if t.numel() == 0:
        return f"{name:22s}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} numel=0"
    x = t.detach()
    xf = x.float()
    mn = xf.min().item()
    mx = xf.max().item()
    mean = xf.mean().item()
    std = xf.std(unbiased=False).item()
    nonzero = (x != 0).float().mean().item() * 100.0
    return (
        f"{name:22s}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
        f"min={mn:.6g} max={mx:.6g} mean={mean:.6g} std={std:.6g} nonzero={nonzero:.2f}%"
    )


def verify_shard(shard_path: str, n_show: int = 1) -> None:
    print(f"\n[verify] loading shard: {shard_path}")
    items = torch.load(shard_path, map_location="cpu")
    if not isinstance(items, (list, tuple)):
        raise TypeError(f"Shard is not list/tuple: {type(items)}")
    print(f"[verify] items: {len(items)}")
    n_show = min(int(n_show), len(items))
    for i in range(n_show):
        ex = items[i]
        print(f"\n[verify] example {i} id={ex.get('id', None)}")
        for k in ["input_ids", "attention_mask", "position_ids",
                  "input_embeds", "visual_pos_masks", "deepstack_visual_embeds"]:
            print(_tensor_stats_line(k, ex.get(k, None)))


def _tokenize_no_special(tok, s: str) -> List[int]:
    if not s:
        return []
    return tok(s, add_special_tokens=False).input_ids


def build_token_type_ids(
    *,
    tok,
    input_ids_1d: "torch.Tensor",
    image_token_id: int,
    cls_token_id: int,
    hint_text: str,
    search_after_images: bool = True,
    verbose: bool = False,
) -> "torch.Tensor":
    ids = input_ids_1d.reshape(-1).tolist()
    L = len(ids)

    ttid = torch.full((L,), TT_OTHER, dtype=torch.uint8)

    img_id = int(image_token_id)
    for i, tid in enumerate(ids):
        if tid == img_id:
            ttid[i] = TT_IMAGE

    start = 0
    if search_after_images:
        while start < L and ids[start] == img_id:
            start += 1

    def label_block(start, tt, text, name):
        text = (text or "").strip()
        if not text:
            return start

        candidates = [text, text + "\n"]
        for _n in range(2, 9):
            candidates.append(text + "\n" * _n)

        i = -1
        needle = []
        for cand in candidates:
            cand_ids = _tokenize_no_special(tok, cand)
            if len(cand_ids) == 0:
                continue
            hit = _find_subseq(ids, cand_ids, start=start, end=L)
            if hit >= 0:
                i = hit
                needle = cand_ids
                break

        if i < 0:
            if verbose:
                print(f"[MISS] {name}: couldn't find tokens for text={text[:80]!r}")
            return start

        for k in range(i, min(i + len(needle), L)):
            if ttid[k] not in (TT_IMAGE, TT_CLS):
                ttid[k] = tt

        start = i + len(needle)
        return start

    start = label_block(start, TT_TEXT, hint_text, "hint")

    cls_id = int(cls_token_id)
    for i, tid in enumerate(ids):
        if tid == cls_id:
            ttid[i] = TT_CLS

    return ttid


# ---------------------------------------------------------------------------
# Main cache builder
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_and_save_cache(
    *,
    data_roots: str,
    out_dir: str,
    model_name: str,
    split: str,
    fps: float,
    val_split_rate: float,
    test_split_rate: float,
    seed: int,
    max_context_turns: int,
    batch_size: int,
    num_workers: int,
    shard_size: int,
    max_samples: int,
    device: str,
    dtype: str,
    heartbeat_every: int,
    local_files_only: bool,
):
    def tmux_log(msg: str) -> None:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

    heartbeat_every = max(1, int(heartbeat_every))
    os.makedirs(out_dir, exist_ok=True)
    split_out = os.path.join(out_dir, split)
    os.makedirs(split_out, exist_ok=True)
    hf_cache = os.path.join(out_dir, "hf_cache")
    os.makedirs(hf_cache, exist_ok=True)

    tmux_log(f"START split={split} data_roots={data_roots} out_dir={split_out}")
    tmux_log(f"Loading processor: {model_name}")
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=hf_cache,
        local_files_only=bool(local_files_only),
    )
    tok = processor.tokenizer
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tok.add_special_tokens({"additional_special_tokens": ["<CLS>"]})
    cls_token_id = int(tok.convert_tokens_to_ids("<CLS>"))

    tmux_log(f"Loading model config: {model_name}")
    cfg = AutoConfig.from_pretrained(
        model_name,
        cache_dir=hf_cache,
        local_files_only=bool(local_files_only),
    )
    if not hasattr(cfg, "image_token_id"):
        raise RuntimeError("Config has no image_token_id.")
    image_token_id = int(getattr(cfg, "image_token_id"))
    image_token_str = tok.convert_ids_to_tokens(image_token_id)

    tmux_log(f"Loading model weights: {model_name} (device={device}, dtype={dtype})")
    if dtype == "fp16":
        model_dtype = torch.float16
    elif dtype == "bf16":
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    load_kwargs = {
        "trust_remote_code": True,
        "cache_dir": hf_cache,
        "torch_dtype": model_dtype,
        "local_files_only": bool(local_files_only),
    }
    if str(device).startswith("cuda"):
        load_kwargs["device_map"] = {"": device}

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
    if not str(device).startswith("cuda"):
        model = model.to(device)
    model.eval()
    tmux_log("Model loaded and set to eval().")

    tmux_log(f"Building dataset split={split} ...")
    ds = MUStARD_ForCache(
        data_roots=data_roots,
        split=split,
        fps=fps,
        seed=seed,
        val_split_rate=val_split_rate,
        test_split_rate=test_split_rate,
        max_samples=max_samples if max_samples > 0 else None,
    )

    # batch_size=1 because samples have variable frame counts
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x,
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
            return None
        shard_file = os.path.join(split_out, f"shard_{shard_idx:05d}.pt")
        torch.save(shard_items, shard_file)
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"shard": os.path.basename(shard_file), "num_items": len(shard_items)}) + "\n")
        tmux_log(f"Flushed shard={os.path.basename(shard_file)} num_items={len(shard_items)}")
        shard_items = []
        shard_idx += 1
        return shard_file

    tmux_log(f"Dataloader ready: batches={len(dl)}")
    pbar = tqdm(dl, desc=f"[cache] {split}", total=len(dl))
    t0 = time.time()
    items_seen = 0
    last_beat = t0
    batch_idx = 0

    for raw_batch in pbar:
        batch_idx += 1
        # raw_batch is a list of 1 dict (batch_size=1, collate_fn=lambda x: x)
        sample = raw_batch[0]
        video_frames = sample["video_frames"]  # (F, 3, H, W)
        n_frames = int(video_frames.shape[0])
        label_int = int(sample["label"])
        text = sample["text"]
        speaker = sample["speaker"]
        context = sample["context"]
        sample_id = sample["id"]

        # Convert frames to PIL images
        pil_images = [
            to_pil_image(video_frames[f].clamp(0.0, 1.0))
            for f in range(n_frames)
        ]

        # Build prompt
        prompt = _build_sarcasm_prompt(
            processor=processor,
            text=text,
            speaker=speaker,
            context=context,
            n_frames=n_frames,
            max_context_turns=max_context_turns,
        )

        # Processor
        proc = processor(
            text=[prompt],
            images=pil_images if pil_images else None,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = proc["input_ids"].to(device)           # (1, T)
        attention_mask = proc["attention_mask"].to(device) # (1, T)
        pixel_values = proc["pixel_values"].to(device)
        image_grid_thw_cpu = proc["image_grid_thw"]        # (N_frames, 3)
        image_grid_thw_dev = image_grid_thw_cpu.to(device)

        with torch.no_grad():
            token_embeds = model.model.get_input_embeddings()(input_ids)  # (1,T,2048)

            image_grid_thw_for_ops = image_grid_thw_dev
            try:
                image_feat_out = model.get_image_features(pixel_values, image_grid_thw_for_ops)
            except RuntimeError as e:
                if _is_cuda_invalid_argument_error(e):
                    tmux_log("WARN get_image_features failed with CUDA grid_thw; retrying with CPU.")
                    image_grid_thw_for_ops = image_grid_thw_cpu
                    image_feat_out = model.get_image_features(pixel_values, image_grid_thw_for_ops)
                else:
                    raise
            image_embeds_list, deep_stack_viz_list = _unpack_image_feature_outputs(image_feat_out)

            image_embeds_cat = torch.cat(image_embeds_list, dim=0).to(token_embeds.device, token_embeds.dtype)

            placeholder_mask, _ = model.model.get_placeholder_mask(
                input_ids,
                inputs_embeds=token_embeds,
                image_features=image_embeds_cat,
            )
            placeholder_mask_2d = placeholder_mask[..., 0] if placeholder_mask.dim() == 3 else placeholder_mask  # (1,T)

            token_embeds = token_embeds.masked_scatter(placeholder_mask, image_embeds_cat)

            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            try:
                position_ids, _ = model.model.get_rope_index(
                    input_ids,
                    image_grid_thw_for_ops,
                    None,
                    attention_mask=attention_mask_tensor,
                )
            except RuntimeError as e:
                if _is_cuda_invalid_argument_error(e) and image_grid_thw_for_ops is image_grid_thw_dev:
                    tmux_log("WARN get_rope_index failed with CUDA grid_thw; retrying with CPU.")
                    image_grid_thw_for_ops = image_grid_thw_cpu
                    position_ids, _ = model.model.get_rope_index(
                        input_ids,
                        image_grid_thw_for_ops,
                        None,
                        attention_mask=attention_mask_tensor,
                    )
                else:
                    raise

        B = 1
        input_ids_cpu = input_ids.detach().cpu()
        attention_cpu = attention_mask.detach().cpu()
        placeholder_mask_2d_cpu = placeholder_mask_2d.detach().cpu()
        token_embeds_cpu = token_embeds.detach().cpu()
        position_ids_cpu = position_ids.detach().cpu()
        pos_b = _normalize_pos_to_B_3_1_T(position_ids_cpu, B)
        deep_per_sample = _stack_deep_levels_per_sample(deep_stack_viz_list, B)

        keep = attention_cpu[0].bool()
        if keep.sum().item() <= 0:
            raise RuntimeError(f"attention_mask had no True tokens for sample id={sample_id}")

        input_ids_keep = input_ids_cpu[0][keep].contiguous().unsqueeze(0)                 # (1,L)
        attention_keep = attention_cpu[0][keep].contiguous().unsqueeze(0)                 # (1,L)
        input_embeds_keep = token_embeds_cpu[0][keep].contiguous().unsqueeze(0)           # (1,L,2048)
        visual_pos_masks = placeholder_mask_2d_cpu[0][keep].contiguous().unsqueeze(0).bool()  # (1,L)
        pos_keep = pos_b[0][:, :, keep].contiguous()                                      # (3,1,L)
        deepstack_visual_embeds = deep_per_sample[0]                                       # (K,64,2048)

        ttid = build_token_type_ids(
            tok=tok,
            input_ids_1d=input_ids_keep,
            image_token_id=image_token_id,
            cls_token_id=cls_token_id,
            hint_text=text,
        )

        masks = {
            "image": (ttid == TT_IMAGE),
            "hint":  (ttid == TT_TEXT),
            "cls":   (ttid == TT_CLS),
            "other": (ttid == TT_OTHER),
        }

        item: Dict[str, Any] = {
            "label": torch.tensor(label_int, dtype=torch.long),
            "id": sample_id,
            "token_type_ids": ttid.cpu(),
            "masks": {k: v.cpu() for k, v in masks.items()},
            "input_ids": input_ids_keep,
            "attention_mask": attention_keep,
            "position_ids": pos_keep,
            "input_embeds": input_embeds_keep,
            "visual_pos_masks": visual_pos_masks,
            "deepstack_visual_embeds": deepstack_visual_embeds,
        }

        shard_items.append(item)
        items_seen += 1

        if len(shard_items) >= shard_size:
            flush_shard()

        if (batch_idx % heartbeat_every) == 0 or (time.time() - last_beat) > 120:
            elapsed = time.time() - t0
            ips = items_seen / max(elapsed, 1e-6)
            cuda_msg = ""
            if torch.cuda.is_available():
                dev = torch.cuda.current_device()
                mem_gb = torch.cuda.memory_allocated(dev) / (1024 ** 3)
                cuda_msg = f" cuda_mem_gb={mem_gb:.2f}"
            tmux_log(
                f"Heartbeat split={split} batch={batch_idx}/{len(dl)} "
                f"items={items_seen} shards_written={shard_idx} "
                f"rate={ips:.2f} items/s elapsed_s={int(elapsed)}{cuda_msg}"
            )
            last_beat = time.time()

    shard_path = flush_shard()
    if shard_path is None:
        written = sorted(_glob.glob(os.path.join(split_out, "shard_*.pt")))
        shard_path = written[-1] if written else None
    if shard_path is not None:
        verify_shard(shard_path, n_show=1)

    meta = {
        "model_name": model_name,
        "split": split,
        "cls_token_id": cls_token_id,
        "image_token_id": image_token_id,
        "image_token_str": image_token_str,
        "token_type_map": {
            "PAD": TT_PAD,
            "IMAGE": TT_IMAGE,
            "TEXT": TT_TEXT,
            "CLS": TT_CLS,
            "OTHER": TT_OTHER,
        },
        "dtype": dtype,
        "fps": fps,
        "seed": seed,
        "val_split_rate": val_split_rate,
        "test_split_rate": test_split_rate,
        "padding_side": tok.padding_side,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    tmux_log(f"DONE split={split} total_items={items_seen} shards={shard_idx}")
    print(f"[OK] Wrote cache to: {split_out}")
    print(f"[OK] Manifest: {manifest_path}")
    print(f"[OK] Meta: {meta_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build MUStARD Qwen3-VL embedding cache.")
    ap.add_argument("--data_roots", type=str, required=True,
                    help="Path to mustard_raw/ directory or metadata.json")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output cache directory")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--fps", type=float, default=1.0,
                    help="Frames per second to sample from video")
    ap.add_argument("--val_split_rate", type=float, default=0.15)
    ap.add_argument("--test_split_rate", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=109,
                    help="Random seed for stratified split")
    ap.add_argument("--max_context_turns", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=1,
                    help="Kept for CLI compatibility; internally always 1 (variable frame counts)")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--shard_size", type=int, default=512,
                    help="Max items per shard .pt file")
    ap.add_argument("--max_samples", type=int, default=0,
                    help="0 = all samples")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--heartbeat_every", type=int, default=50,
                    help="Print heartbeat every N batches")
    args = ap.parse_args()

    build_and_save_cache(
        data_roots=args.data_roots,
        out_dir=args.out_dir,
        model_name=args.model_name,
        split=args.split,
        fps=args.fps,
        val_split_rate=args.val_split_rate,
        test_split_rate=args.test_split_rate,
        seed=args.seed,
        max_context_turns=args.max_context_turns,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shard_size=args.shard_size,
        max_samples=args.max_samples,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        heartbeat_every=args.heartbeat_every,
    )


if __name__ == "__main__":
    main()
