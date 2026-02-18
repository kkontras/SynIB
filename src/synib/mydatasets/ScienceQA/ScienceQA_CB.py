import os
import json
import random
import multiprocessing
from typing import Any, Dict, List, Optional, Tuple

import einops
import torch
from torch.utils.data import Dataset, DataLoader


# =========================
# Debug / checks
# =========================

def _require_tensor(ex: dict, key: str) -> torch.Tensor:
    if key not in ex:
        raise KeyError(f"Missing key in shard item: {key}")
    t = ex[key]
    if not torch.is_tensor(t):
        raise TypeError(f"Key {key} must be a torch.Tensor, got {type(t)}")
    return t


def _as_scalar_int(x: Any, default: int = 0) -> int:
    if x is None:
        return int(default)
    if torch.is_tensor(x):
        return int(x.detach().cpu().reshape(-1)[0].item()) if x.numel() else int(default)
    try:
        return int(x)
    except Exception:
        return int(default)


def _load_manifest_shards(split_dir: str) -> List[str]:
    manifest_path = os.path.join(split_dir, "manifest.jsonl")
    if not os.path.isfile(manifest_path):
        return []
    recs = [json.loads(l) for l in open(manifest_path, "r", encoding="utf-8")]
    shard_paths = [os.path.join(split_dir, r["shard"]) for r in recs]
    return shard_paths


def _load_split_items(
    split_dir: str,
    *,
    shard_index: Optional[int] = None,
    shard_path: Optional[str] = None,
    max_items: Optional[int] = None,
) -> List[Dict[str, Any]]:
    # Option A: explicit shard_path
    if shard_path is not None:
        if not os.path.isfile(shard_path):
            raise FileNotFoundError(f"Shard not found: {shard_path}")
        items = torch.load(shard_path, map_location="cpu")
        items = list(items)
        return items[: int(max_items)] if max_items is not None else items

    # Option B: manifest shards
    shard_paths = _load_manifest_shards(split_dir)
    if shard_paths:
        if shard_index is not None:
            si = int(shard_index)
            if si < 0 or si >= len(shard_paths):
                raise IndexError(f"shard_index {si} out of range [0, {len(shard_paths)-1}]")
            items = torch.load(shard_paths[si], map_location="cpu")
            items = list(items)
            return items[: int(max_items)] if max_items is not None else items

        # load all shards
        out: List[Dict[str, Any]] = []
        for sp in shard_paths:
            items = torch.load(sp, map_location="cpu")
            out.extend(list(items))
            if max_items is not None and len(out) >= int(max_items):
                out = out[: int(max_items)]
                break
        return out

    # Option C: single data.pt
    data_pt = os.path.join(split_dir, "data.pt")
    if os.path.isfile(data_pt):
        items = torch.load(data_pt, map_location="cpu")
        items = list(items)
        return items[: int(max_items)] if max_items is not None else items

    raise FileNotFoundError(f"Need manifest.jsonl shards or data.pt in {split_dir}")


# =========================
# Dataset (no memmap)
# =========================

class ScienceQA_MemmapDataset(Dataset):
    """
    Name kept as ScienceQA_MemmapDataset for drop-in compatibility, but this version
    is a SIMPLE torch.load cache reader (no memmap).
    """
    def __init__(
        self,
        cache_root: str,
        split: str,
        shard_index: Optional[int] = None,
        shard_path: Optional[str] = None,
        max_items: Optional[int] = None,
        deep_dim: int = 2048,
    ):
        super().__init__()
        self.split_dir = os.path.join(cache_root, split)
        self.items = _load_split_items(
            self.split_dir,
            shard_index=shard_index,
            shard_path=shard_path,
            max_items=max_items,
        )
        if not isinstance(self.items, (list, tuple)):
            raise TypeError(f"Split cache must contain list/tuple, got {type(self.items)}")
        self.items = list(self.items)
        self.deep_dim = int(deep_dim)

        # detect optional fields
        self.has_vision = any(("vision_embeds" in ex and ex["vision_embeds"] is not None) for ex in self.items[: min(len(self.items), 256)])
        self.has_deep = any(("deepstack_visual_embeds" in ex and ex["deepstack_visual_embeds"] is not None) for ex in self.items[: min(len(self.items), 256)])

        print(f"[ScienceQA SimpleDataset] split={split} N={len(self.items)} dir={self.split_dir} vision={self.has_vision} deep={self.has_deep}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.items[idx]
        if not isinstance(ex, dict):
            raise TypeError(f"Example[{idx}] is not a dict: {type(ex)}")

        input_ids = _require_tensor(ex, "input_ids").to(torch.long).reshape(-1)
        attention_mask = _require_tensor(ex, "attention_mask").to(torch.long).reshape(-1)
        position_ids = _require_tensor(ex, "position_ids")
        visual_pos_masks = _require_tensor(ex, "visual_pos_masks")
        hint_mask = _require_tensor(ex["masks"], "hint")
        lab = _as_scalar_int(ex.get("label", 0), default=0)


        out: Dict[str, Any] = {
            "id": ex.get("id", idx),
            "prompt": ex.get("prompt", ""),
            "label": torch.tensor(lab, dtype=torch.long),

            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "visual_pos_masks": visual_pos_masks,
            "hint_mask": hint_mask,
            "input_embeds": _require_tensor(ex, "input_embeds").to(torch.float16),
            "deepstack_visual_embeds": _require_tensor(ex, "deepstack_visual_embeds").to(torch.float16),
        }


        return out


# =========================
# Collate (keep SAME name)
# =========================

def _left_pad_1d(seqs: List[torch.Tensor], pad_val: int, dtype: torch.dtype) -> torch.Tensor:
    max_len = max(int(s.numel()) for s in seqs) if len(seqs) > 0 else 0
    out = torch.full((len(seqs), max_len), pad_val, dtype=dtype)
    for i, s in enumerate(seqs):
        s = s.reshape(-1)
        L = int(s.numel())
        if L > 0:
            out[i, -L:] = s.to(dtype)
    return out


def _left_pad_bool(seqs: List[torch.Tensor]) -> torch.Tensor:
    max_len = max(int(s.numel()) for s in seqs) if len(seqs) > 0 else 0
    out = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        s = s.reshape(-1)
        L = int(s.numel())
        if L > 0:
            out[i, -L:] = s.bool()
    return out

def _right_pad_1d(seqs: List[torch.Tensor], pad_val: int, dtype: torch.dtype) -> torch.Tensor:
    max_len = max(int(s.numel()) for s in seqs) if len(seqs) > 0 else 0
    out = torch.full((len(seqs), max_len), pad_val, dtype=dtype)
    for i, s in enumerate(seqs):
        s = s.reshape(-1)
        L = int(s.numel())
        if L > 0:
            out[i, :L] = s.to(dtype)
    return out


def _right_pad_bool(seqs: List[torch.Tensor]) -> torch.Tensor:
    max_len = max(int(s.numel()) for s in seqs) if len(seqs) > 0 else 0
    out = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        s = s.reshape(-1)
        L = int(s.numel())
        if L > 0:
            out[i, :L] = s.bool()
    return out


def _pad_2d_by_rows(seqs: List[torch.Tensor], pad_val: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads a list of per-example vision tensors into:
      padded: (B, Nmax, D)
      mask:   (B, Nmax)  True where real rows exist

    Accepts per-example shapes:
      - (N, D)
      - (1, N, D)  (will squeeze the leading 1)
      - (D,)       (treated as (1, D))
      - empty -> (0, D)
    """
    B = len(seqs)
    if B == 0:
        return torch.empty((0, 0, 0)), torch.empty((0, 0), dtype=torch.bool)

    # normalize each x to (N, D)
    norm: List[torch.Tensor] = []
    D = None
    ref_dtype = torch.float16

    for x in seqs:
        if not torch.is_tensor(x) or x.numel() == 0:
            norm.append(torch.empty((0, 0), dtype=torch.float16))
            continue

        if x.dim() == 3:
            # expected (1, N, D) or (K, N, D) -> take first if K==1 else flatten K into N
            if x.shape[0] == 1:
                x2 = x[0]            # (N, D)
            else:
                x2 = x.reshape(-1, x.shape[-1])  # (K*N, D)
        elif x.dim() == 2:
            x2 = x
        elif x.dim() == 1:
            x2 = x.view(1, -1)
        else:
            x2 = x.reshape(-1, x.shape[-1])

        x2 = x2.detach().cpu()
        norm.append(x2)

        if x2.numel() > 0 and x2.dim() == 2:
            D = int(x2.shape[1]) if D is None else D
            ref_dtype = x2.dtype

    if D is None:
        # everything was empty
        return torch.empty((B, 0, 0)), torch.empty((B, 0), dtype=torch.bool)

    # compute Nmax
    Nmax = 0
    for x2 in norm:
        if torch.is_tensor(x2) and x2.dim() == 2:
            Nmax = max(Nmax, int(x2.shape[0]))

    padded = torch.full((B, Nmax, D), float(pad_val), dtype=ref_dtype)
    mask = torch.zeros((B, Nmax), dtype=torch.bool)

    for i, x2 in enumerate(norm):
        if (not torch.is_tensor(x2)) or x2.numel() == 0:
            continue

        # fix D mismatch if any (crop or pad last dim)
        if int(x2.shape[1]) > D:
            x2 = x2[:, :D]
        elif int(x2.shape[1]) < D:
            pad = torch.zeros((int(x2.shape[0]), D - int(x2.shape[1])), dtype=x2.dtype)
            x2 = torch.cat([x2, pad], dim=1)

        n = int(x2.shape[0])
        padded[i, :n, :] = x2.to(ref_dtype)
        mask[i, :n] = True

    return padded, mask



def _pad_deep_3d(seqs: List[torch.Tensor], deep_dim: int, pad_val: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads [T_i, N_i, D] -> [B, Tmax, Nmax, D] and mask [B, Tmax, Nmax]
    """
    B = len(seqs)
    if B == 0:
        return (
            torch.empty((0, 0, 0, deep_dim), dtype=torch.float16),
            torch.empty((0, 0, 0), dtype=torch.bool),
        )

    Tmax = 0
    Nmax = 0
    ref_dtype = torch.float16
    for x in seqs:
        if torch.is_tensor(x) and x.numel() > 0:
            if x.dim() == 4:
                x = x[0]
            if x.dim() == 3:
                Tmax = max(Tmax, int(x.shape[0]))
                Nmax = max(Nmax, int(x.shape[1]))
                ref_dtype = x.dtype

    padded = torch.full((B, Tmax, Nmax, deep_dim), float(pad_val), dtype=ref_dtype)
    mask = torch.zeros((B, Tmax, Nmax), dtype=torch.bool)

    for i, x in enumerate(seqs):
        if (not torch.is_tensor(x)) or x.numel() == 0:
            continue
        if x.dim() == 4:
            x = x[0]
        if x.dim() != 3:
            continue

        T, N, D = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        if D > deep_dim:
            x = x[..., :deep_dim]
        elif D < deep_dim:
            pad = torch.zeros((T, N, deep_dim - D), dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)

        padded[i, :T, :N, :] = x
        mask[i, :T, :N] = True

    return padded, mask


def scienceqa_memmap_collate(
    batch: List[Dict[str, Any]],
    pad_token_id: int = 0,
    padding_side: str = "right",   # <-- important
) -> Dict[str, Any]:
    ids = [b.get("id", None) for b in batch]
    prompts = [b.get("prompt", "") for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)

    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    pad_1d  = _left_pad_1d  if padding_side == "left"  else _right_pad_1d
    pad_bool = _left_pad_bool if padding_side == "left" else _right_pad_bool

    input_ids = pad_1d([b["input_ids"] for b in batch], pad_val=int(pad_token_id), dtype=torch.long)
    attention_mask = pad_1d([b["attention_mask"] for b in batch], pad_val=0, dtype=torch.long)

    # position_ids
    pos_list = [b["position_ids"] for b in batch]
    Lmax = int(input_ids.shape[1])

    if torch.is_tensor(pos_list[0]) and pos_list[0].dim() == 3:
        # expect (3, 1, L)
        pos_out = torch.zeros((len(batch), 3, 1, Lmax), dtype=torch.long)
        for i, p in enumerate(pos_list):
            if p.dim() != 3 or p.shape[0] != 3 or p.shape[1] != 1:
                raise ValueError(f"position_ids mix: expected (3,1,L), got {tuple(p.shape)}")
            Li = int(p.shape[-1])
            if padding_side == "left":
                pos_out[i, :, :, -Li:] = p.to(torch.long)
            else:
                pos_out[i, :, :, :Li] = p.to(torch.long)
        position_ids = pos_out
    else:
        # (L,) per example
        # NOTE: if you ever want to LEFT pad while keeping processor-style absolute positions,
        # you may need to offset position_ids by pad_len. See note below.
        position_ids = pad_1d([p.reshape(-1) for p in pos_list], pad_val=0, dtype=torch.long)

    # keep your model-expected layout
    position_ids = einops.rearrange(position_ids, "b c i j -> c b (i j)", i=1) if position_ids.dim() == 4 else position_ids

    image_mask = pad_bool([b["visual_pos_masks"] for b in batch])
    hint_mask  = pad_bool([b["hint_mask"] for b in batch])

    data: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "visual_pos_masks": image_mask,
        "hint_mask": hint_mask,
    }

    if "input_embeds" in batch[0]:
        vis_list = [b.get("input_embeds") for b in batch]
        vis_pad, _ = _pad_2d_by_rows(vis_list, pad_val=0.0)
        data["input_embeds"] = vis_pad

    if "deepstack_visual_embeds" in batch[0]:
        deep_list = [b.get("deepstack_visual_embeds", torch.empty((0, 0, 2048), dtype=torch.float16)) for b in batch]
        deep_dim = int(deep_list[0].shape[-1]) if (torch.is_tensor(deep_list[0]) and deep_list[0].numel() > 0) else 2048
        deep_pad, deep_mask = _pad_deep_3d(deep_list, deep_dim=deep_dim, pad_val=0.0)
        deep_pad = einops.rearrange(deep_pad, "a b c d -> b (a c) d")
        data["deepstack_visual_embeds"] = deep_pad
        data["deep_mask"] = deep_mask

    return {"ids": ids, "prompts": prompts, "label": labels, "data": data}


# =========================
# Dataloader wrapper (same name)
# =========================

class ScienceQA_MemmapDataloader:
    """
    Name kept for drop-in compatibility.
    This reads cached torch shards directly (no memmap).
    """
    def __init__(
        self,
        config,
        *,
        split: str = "validation",
        shard_index: Optional[int] = None,
        shard_path: Optional[str] = None,
        max_items: Optional[int] = None,
        deep_dim: int = 2048,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        pad_token_id: int = 0,
        shuffle: bool = False,
        print_first_batch_stats: bool = False,
    ):
        cache_root = config.dataset.cache_root
        batch_size = int(config.training_params.batch_size)
        test_batch_size = int(config.training_params.test_batch_size)

        # if config has pad_token_id, prefer it
        pad_token_id = int(getattr(getattr(config, "model", None), "pad_token_id", pad_token_id))

        train_ds = ScienceQA_MemmapDataset(
            cache_root=cache_root,
            split="train",
            shard_index=shard_index,
            shard_path=shard_path,
            max_items=max_items,
            deep_dim=deep_dim,
        )
        val_ds = ScienceQA_MemmapDataset(
            cache_root=cache_root,
            split="validation",
            shard_index=shard_index,
            shard_path=shard_path,
            max_items=max_items,
            deep_dim=deep_dim,
        )
        test_ds = ScienceQA_MemmapDataset(
            cache_root=cache_root,
            split="test",
            shard_index=shard_index,
            shard_path=shard_path,
            max_items=max_items,
            deep_dim=deep_dim,
        )


        self.collate_fn = lambda batch: scienceqa_memmap_collate(batch, pad_token_id=pad_token_id)
        num_workers = 0
        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=bool(shuffle),
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            collate_fn=self.collate_fn,
            prefetch_factor=int(prefetch_factor) if int(num_workers) > 0 else None,
            persistent_workers=bool(persistent_workers) if int(num_workers) > 0 else False,
        )
        self.valid_loader = DataLoader(
            val_ds,
            batch_size=test_batch_size,
            shuffle=bool(shuffle),
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            collate_fn=self.collate_fn,
            prefetch_factor=int(prefetch_factor) if int(num_workers) > 0 else None,
            persistent_workers=bool(persistent_workers) if int(num_workers) > 0 else False,
        )
        self.test_loader = DataLoader(
            test_ds,
            batch_size=test_batch_size,
            shuffle=bool(shuffle),
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            collate_fn=self.collate_fn,
            prefetch_factor=int(prefetch_factor) if int(num_workers) > 0 else None,
            persistent_workers=bool(persistent_workers) if int(num_workers) > 0 else False,
        )


        if print_first_batch_stats:
            batch = next(iter(self.train_loader))
            ex = batch["data"]
            print("\n[scienceqa-cache-reader] first batch loaded (padded from cache):")
            for k in ex.keys():
                t = ex[k]
                if torch.is_tensor(t):
                    print(f"{k:24s}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} numel={t.numel()}")
                else:
                    print(f"{k:24s}: {type(t)}")
