#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TQA cached-token dataloader.

Reads the shard cache built by TQA_Codebook.py and exposes
TQA_MemmapDataset / TQA_MemmapDataloader with the same interface
and batch format as ScienceQA_MemmapDataloader.
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import einops
import torch
from torch.utils.data import Dataset, DataLoader

# Re-use collate utilities from ScienceQA_CB (identical shard format)
from synib.mydatasets.ScienceQA.ScienceQA_CB import (
    scienceqa_memmap_collate,
    _load_manifest_shards,
    _load_split_items,
    _require_tensor,
    _as_scalar_int,
)


# =========================
# Dataset
# =========================

class TQA_MemmapDataset(Dataset):
    """
    Reads TQA shard cache (built by TQA_Codebook.py).
    Drop-in replacement for ScienceQA_MemmapDataset.
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

        self.has_vision = any(
            ("vision_embeds" in ex and ex["vision_embeds"] is not None)
            for ex in self.items[: min(len(self.items), 256)]
        )
        self.has_deep = any(
            ("deepstack_visual_embeds" in ex and ex["deepstack_visual_embeds"] is not None)
            for ex in self.items[: min(len(self.items), 256)]
        )

        print(
            f"[TQA Dataset] split={split} N={len(self.items)} "
            f"dir={self.split_dir} vision={self.has_vision} deep={self.has_deep}"
        )

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
# Dataloader wrapper
# =========================

class TQA_MemmapDataloader:
    """
    Reads TQA cached torch shards and wraps them in train/valid/test DataLoaders.
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
        cache_root = config.dataset.cache_root
        batch_size = int(config.training_params.batch_size)
        test_batch_size = int(config.training_params.test_batch_size)

        pad_token_id = int(getattr(getattr(config, "model", None), "pad_token_id", pad_token_id))

        train_ds = TQA_MemmapDataset(
            cache_root=cache_root, split="train",
            shard_index=shard_index, shard_path=shard_path,
            max_items=max_items, deep_dim=deep_dim,
        )
        val_ds = TQA_MemmapDataset(
            cache_root=cache_root, split="val",
            shard_index=shard_index, shard_path=shard_path,
            max_items=max_items, deep_dim=deep_dim,
        )
        test_ds = TQA_MemmapDataset(
            cache_root=cache_root, split="test",
            shard_index=shard_index, shard_path=shard_path,
            max_items=max_items, deep_dim=deep_dim,
        )

        self.collate_fn = lambda batch: scienceqa_memmap_collate(batch, pad_token_id=pad_token_id)
        num_workers = 0

        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=bool(shuffle),
            num_workers=num_workers, pin_memory=bool(pin_memory),
            collate_fn=self.collate_fn,
            prefetch_factor=int(prefetch_factor) if num_workers > 0 else None,
            persistent_workers=bool(persistent_workers) if num_workers > 0 else False,
        )
        self.valid_loader = DataLoader(
            val_ds, batch_size=test_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=bool(pin_memory),
            collate_fn=self.collate_fn,
            prefetch_factor=int(prefetch_factor) if num_workers > 0 else None,
            persistent_workers=bool(persistent_workers) if num_workers > 0 else False,
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=test_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=bool(pin_memory),
            collate_fn=self.collate_fn,
            prefetch_factor=int(prefetch_factor) if num_workers > 0 else None,
            persistent_workers=bool(persistent_workers) if num_workers > 0 else False,
        )

        if print_first_batch_stats:
            batch = next(iter(self.train_loader))
            ex = batch["data"]
            print("\n[tqa-cache-reader] first batch loaded (padded from cache):")
            for k in ex.keys():
                t = ex[k]
                if torch.is_tensor(t):
                    print(f"{k:24s}: shape={tuple(t.shape)} dtype={t.dtype} numel={t.numel()}")
                else:
                    print(f"{k:24s}: {type(t)}")
