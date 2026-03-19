"""
MUStARD_CB.py

Fast cached dataloader for MUStARD sarcasm detection.

Reads pre-built .pt shards from MUStARD_CodeBook.py.
Drop-in replacement for MUStARD_Raw_Dataloader in cached-training configs.

Split directories:
  <cache_root>/train/
  <cache_root>/val/
  <cache_root>/test/
"""

import os
import json
from bisect import bisect_right
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import einops
import torch
from torch.utils.data import Dataset, DataLoader

__all__ = ["MUStARD_ShardedLazyDataset", "MUStARD_MemmapDataloader"]


# ---------------------------------------------------------------------------
# Debug helpers  (copied verbatim from ESNLI_CB.py)
# ---------------------------------------------------------------------------

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


def _load_manifest(split_dir: str) -> List[Dict[str, Any]]:
    mp = os.path.join(split_dir, "manifest.jsonl")
    if not os.path.isfile(mp):
        raise FileNotFoundError(f"Missing manifest.jsonl in {split_dir}")
    return [json.loads(l) for l in open(mp, "r", encoding="utf-8")]


# ---------------------------------------------------------------------------
# Padding helpers  (copied verbatim from ESNLI_CB.py)
# ---------------------------------------------------------------------------

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
    """
    B = len(seqs)
    if B == 0:
        return torch.empty((0, 0, 0)), torch.empty((0, 0), dtype=torch.bool)

    norm: List[torch.Tensor] = []
    D = None
    ref_dtype = torch.float16

    for x in seqs:
        if not torch.is_tensor(x) or x.numel() == 0:
            norm.append(torch.empty((0, 0), dtype=torch.float16))
            continue

        if x.dim() == 3:
            if x.shape[0] == 1:
                x2 = x[0]
            else:
                x2 = x.reshape(-1, x.shape[-1])
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
        return torch.empty((B, 0, 0)), torch.empty((B, 0), dtype=torch.bool)

    Nmax = 0
    for x2 in norm:
        if torch.is_tensor(x2) and x2.dim() == 2:
            Nmax = max(Nmax, int(x2.shape[0]))

    padded = torch.full((B, Nmax, D), float(pad_val), dtype=ref_dtype)
    mask = torch.zeros((B, Nmax), dtype=torch.bool)

    for i, x2 in enumerate(norm):
        if (not torch.is_tensor(x2)) or x2.numel() == 0:
            continue

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


# ---------------------------------------------------------------------------
# Collate  (copied verbatim from ESNLI_CB.py)
# ---------------------------------------------------------------------------

def memmap_collate(
    batch: List[Dict[str, Any]],
    pad_token_id: int = 0,
    padding_side: str = "right",
) -> Dict[str, Any]:
    ids = [b.get("id", None) for b in batch]
    prompts = [b.get("prompt", "") for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)

    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    pad_1d   = _left_pad_1d   if padding_side == "left" else _right_pad_1d
    pad_bool = _left_pad_bool if padding_side == "left" else _right_pad_bool

    input_ids      = pad_1d([b["input_ids"]      for b in batch], pad_val=int(pad_token_id), dtype=torch.long)
    attention_mask = pad_1d([b["attention_mask"]  for b in batch], pad_val=0,                dtype=torch.long)

    pos_list = [b["position_ids"] for b in batch]
    Lmax = int(input_ids.shape[1])

    if torch.is_tensor(pos_list[0]) and pos_list[0].dim() == 3:
        pos_out = torch.zeros((len(batch), 3, 1, Lmax), dtype=torch.long)
        for i, p in enumerate(pos_list):
            if p.dim() != 3 or p.shape[0] != 3 or p.shape[1] != 1:
                raise ValueError(f"position_ids expected (3,1,L), got {tuple(p.shape)}")
            Li = int(p.shape[-1])
            if padding_side == "left":
                pos_out[i, :, :, -Li:] = p.to(torch.long)
            else:
                pos_out[i, :, :, :Li] = p.to(torch.long)
        position_ids = pos_out
    else:
        position_ids = pad_1d([p.reshape(-1) for p in pos_list], pad_val=0, dtype=torch.long)

    position_ids = einops.rearrange(position_ids, "b c i j -> c b (i j)", i=1) if position_ids.dim() == 4 else position_ids

    image_mask = pad_bool([b["visual_pos_masks"] for b in batch])
    hint_mask  = pad_bool([b["hint_mask"]        for b in batch])

    data: Dict[str, Any] = {
        "input_ids":       input_ids,
        "attention_mask":  attention_mask,
        "position_ids":    position_ids,
        "visual_pos_masks": image_mask,
        "hint_mask":       hint_mask,
    }

    if "input_embeds" in batch[0]:
        vis_list = [b.get("input_embeds") for b in batch]
        vis_pad, _ = _pad_2d_by_rows(vis_list, pad_val=0.0)
        data["input_embeds"] = vis_pad

    if "deepstack_visual_embeds" in batch[0]:
        deep_list = [b.get("deepstack_visual_embeds", torch.empty((0, 0, 2048), dtype=torch.float16)) for b in batch]
        # Each deep_list[b] has shape (K, N_b, D) where N_b = actual visual tokens for sample b.
        # MUStARD clips have variable frame counts → variable N_b.
        # We must concatenate actual tokens (NOT pad) so that deep_stack[k].shape[0]
        # equals visual_pos_masks.sum() (total actual visual positions across the batch).
        valid = [d for d in deep_list if torch.is_tensor(d) and d.numel() > 0 and d.dim() == 3]
        if valid:
            K = max(int(d.shape[0]) for d in valid)
            D = int(valid[0].shape[-1])
            levels = []
            for k in range(K):
                parts = [d[k] for d in deep_list
                         if torch.is_tensor(d) and d.dim() == 3 and int(d.shape[0]) > k]
                levels.append(torch.cat(parts, dim=0) if parts else torch.empty((0, D), dtype=torch.float16))
            data["deepstack_visual_embeds"] = torch.stack(levels, dim=0)  # (K, sum_N, D)
        else:
            data["deepstack_visual_embeds"] = torch.empty((0, 0, 2048), dtype=torch.float16)

    return {"ids": ids, "prompts": prompts, "label": labels, "data": data}


# ---------------------------------------------------------------------------
# Lazy sharded dataset  (adapted from ESNLI_ShardedLazyDataset)
# ---------------------------------------------------------------------------

class MUStARD_ShardedLazyDataset(Dataset):
    def __init__(
        self,
        cache_root: str,
        split: str,
        *,
        max_items: Optional[int] = None,
        deep_dim: int = 2048,
        shard_cache_size: int = 2,
        **kwargs
    ):
        super().__init__()
        self.split_dir = os.path.join(cache_root, split)
        self.recs = _load_manifest(self.split_dir)

        self.shard_paths: List[str] = [os.path.join(self.split_dir, r["shard"]) for r in self.recs]
        self.shard_counts: List[int] = [int(r["num_items"]) for r in self.recs]

        self.cum: List[int] = []
        s = 0
        for n in self.shard_counts:
            s += n
            self.cum.append(s)

        self.N = self.cum[-1] if self.cum else 0
        if max_items is not None:
            self.N = min(self.N, int(max_items))

        self.deep_dim = int(deep_dim)
        self.shard_cache_size = int(shard_cache_size)
        self._shard_cache: "OrderedDict[int, List[Dict[str, Any]]]" = OrderedDict()

        print(
            f"[MUStARD_ShardedLazyDataset] split={split} N={self.N} "
            f"shards={len(self.shard_paths)} cache_shards={self.shard_cache_size}"
        )

    def __len__(self) -> int:
        return self.N

    def _locate(self, idx: int) -> Tuple[int, int]:
        shard_id = bisect_right(self.cum, idx)
        prev = 0 if shard_id == 0 else self.cum[shard_id - 1]
        local = idx - prev
        return shard_id, local

    def _get_shard(self, shard_id: int) -> List[Dict[str, Any]]:
        if shard_id in self._shard_cache:
            self._shard_cache.move_to_end(shard_id)
            return self._shard_cache[shard_id]

        sp = self.shard_paths[shard_id]
        items = torch.load(sp, map_location="cpu")
        items = list(items)

        self._shard_cache[shard_id] = items
        self._shard_cache.move_to_end(shard_id)

        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)

        return items

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_id, local = self._locate(int(idx))
        shard = self._get_shard(shard_id)
        ex = shard[local]

        if not isinstance(ex, dict):
            raise TypeError(f"Example[{idx}] is not a dict: {type(ex)}")

        input_ids      = _require_tensor(ex, "input_ids").to(torch.long).reshape(-1)
        attention_mask = _require_tensor(ex, "attention_mask").to(torch.long).reshape(-1)
        position_ids   = _require_tensor(ex, "position_ids")
        visual_pos_masks = _require_tensor(ex, "visual_pos_masks")

        hint_mask = None
        masks = ex.get("masks", None)
        if isinstance(masks, dict):
            if "hint" in masks and torch.is_tensor(masks["hint"]):
                hint_mask = masks["hint"]
            elif "text" in masks and torch.is_tensor(masks["text"]):
                hint_mask = masks["text"]

        if hint_mask is None:
            hint_mask = attention_mask.bool() & (~visual_pos_masks.bool())

        hint_mask = hint_mask.to(torch.bool).reshape(-1)
        lab = _as_scalar_int(ex.get("label", 0), default=0)

        return {
            "id":           ex.get("id", idx),
            "prompt":       ex.get("prompt", ""),
            "label":        torch.tensor(lab, dtype=torch.long),
            "input_ids":    input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "visual_pos_masks": visual_pos_masks,
            "hint_mask":    hint_mask,
            "input_embeds": _require_tensor(ex, "input_embeds").to(torch.float16),
            "deepstack_visual_embeds": _require_tensor(ex, "deepstack_visual_embeds").to(torch.float16),
        }


# ---------------------------------------------------------------------------
# Dataloader wrapper
# ---------------------------------------------------------------------------

class MUStARD_MemmapDataloader:
    """
    Reads cached MUStARD torch shards directly.
    Uses "val" (not "validation") for the validation split directory.
    """

    def __init__(
        self,
        config,
        *,
        split: str = "val",
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
        cache_root      = config.dataset.cache_root
        batch_size      = int(config.training_params.batch_size)
        test_batch_size = int(config.training_params.test_batch_size)

        pad_token_id = int(getattr(getattr(config, "model", None), "pad_token_id", pad_token_id))

        train_max_items = getattr(config.dataset, "train_max_items", None)
        if train_max_items is not None:
            train_max_items = int(train_max_items)

        train_ds = MUStARD_ShardedLazyDataset(
            cache_root=cache_root,
            split="train",
            shard_path=shard_path,
            max_items=train_max_items,
            deep_dim=deep_dim,
        )
        val_ds = MUStARD_ShardedLazyDataset(
            cache_root=cache_root,
            split="val",           # MUStARD uses "val", not "validation"
            shard_path=shard_path,
            max_items=max_items,
            deep_dim=deep_dim,
        )
        test_ds = MUStARD_ShardedLazyDataset(
            cache_root=cache_root,
            split="test",
            shard_path=shard_path,
            max_items=max_items,
            deep_dim=deep_dim,
        )

        self.collate_fn = lambda batch: memmap_collate(batch, pad_token_id=pad_token_id)
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
            print("\n[mustard-cache-reader] first batch loaded (padded from cache):")
            for k in ex.keys():
                t = ex[k]
                if torch.is_tensor(t):
                    print(f"{k:24s}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}")
                else:
                    print(f"{k:24s}: {type(t)}")
