#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TQA cached-token dataloader.

Reads the shard cache built by TQA_Codebook.py and exposes
TQA_MemmapDataset / TQA_MemmapDataloader with the same interface
and batch format as ScienceQA_MemmapDataloader.
"""

import bisect
import gc
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

def _validate_cached_example_schema(ex: Dict[str, Any], *, shard_path: str, local_idx: int) -> None:
    if not isinstance(ex, dict):
        raise TypeError(f"Cached shard item must be a dict, got {type(ex)} in {shard_path} item {local_idx}")

    required = (
        "input_ids",
        "attention_mask",
        "position_ids",
        "input_embeds",
        "visual_pos_masks",
        "deepstack_visual_embeds",
    )
    missing = [k for k in required if k not in ex]
    if missing:
        present = ", ".join(sorted(ex.keys()))
        needed = ", ".join(required)
        missing_s = ", ".join(missing)
        raise RuntimeError(
            "Incompatible TQA cache shard format. "
            f"Shard {shard_path} item {local_idx} is missing [{missing_s}]. "
            f"Present keys: [{present}]. "
            f"TQA cached training expects the same rich per-sample schema used by MUStARD/ESNLI "
            f"with keys [{needed}]. Regenerate the TQA cache with a compatible codebook."
        )


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
        self.deep_dim = int(deep_dim)
        self._loaded_shard_key: Optional[str] = None
        self._loaded_items: Optional[List[Dict[str, Any]]] = None

        if shard_path is not None:
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"Shard not found: {shard_path}")
            self.shard_paths = [shard_path]
            shard_counts = [len(torch.load(shard_path, map_location="cpu", weights_only=False))]
        else:
            manifest_path = os.path.join(self.split_dir, "manifest.jsonl")
            if os.path.isfile(manifest_path):
                recs = [json.loads(l) for l in open(manifest_path, "r", encoding="utf-8")]
                if shard_index is not None:
                    si = int(shard_index)
                    if si < 0 or si >= len(recs):
                        raise IndexError(f"shard_index {si} out of range [0, {len(recs)-1}]")
                    recs = [recs[si]]
                self.shard_paths = [os.path.join(self.split_dir, r["shard"]) for r in recs]
                shard_counts = [int(r["num_items"]) for r in recs]
            else:
                data_pt = os.path.join(self.split_dir, "data.pt")
                if not os.path.isfile(data_pt):
                    raise FileNotFoundError(f"Need manifest.jsonl shards or data.pt in {self.split_dir}")
                self.shard_paths = [data_pt]
                shard_counts = [len(torch.load(data_pt, map_location="cpu", weights_only=False))]

        if max_items is not None:
            remaining = int(max_items)
            limited_counts = []
            limited_paths = []
            for sp, count in zip(self.shard_paths, shard_counts):
                if remaining <= 0:
                    break
                take = min(int(count), remaining)
                limited_paths.append(sp)
                limited_counts.append(take)
                remaining -= take
            self.shard_paths = limited_paths
            self.shard_counts = limited_counts
        else:
            self.shard_counts = [int(c) for c in shard_counts]

        self.cum_counts: List[int] = []
        running = 0
        for count in self.shard_counts:
            running += int(count)
            self.cum_counts.append(running)
        self.total_items = running

        self.has_vision, self.has_deep = self._detect_cached_modalities()

        print(
            f"[TQA Dataset] split={split} N={self.total_items} "
            f"dir={self.split_dir} vision={self.has_vision} deep={self.has_deep}"
        )

    def _detect_cached_modalities(self) -> Tuple[bool, bool]:
        for shard_idx, shard_path in enumerate(self.shard_paths):
            if int(self.shard_counts[shard_idx]) <= 0:
                continue
            items = torch.load(shard_path, map_location="cpu", weights_only=False)
            if not isinstance(items, (list, tuple)) or len(items) == 0:
                continue
            ex = items[0]
            if not isinstance(ex, dict):
                continue

            has_vision = (
                "visual_pos_masks" in ex and
                torch.is_tensor(ex["visual_pos_masks"]) and
                ex["visual_pos_masks"].numel() > 0
            )
            has_deep = (
                "deepstack_visual_embeds" in ex and
                torch.is_tensor(ex["deepstack_visual_embeds"]) and
                ex["deepstack_visual_embeds"].numel() > 0
            )
            return bool(has_vision), bool(has_deep)

        return False, False

    def __len__(self) -> int:
        return self.total_items

    def _load_shard_items(self, shard_idx: int) -> List[Dict[str, Any]]:
        shard_key = self.shard_paths[shard_idx]
        if self._loaded_shard_key != shard_key or self._loaded_items is None:
            items = torch.load(shard_key, map_location="cpu", weights_only=False)
            if not isinstance(items, (list, tuple)):
                raise TypeError(f"Split cache shard must contain list/tuple, got {type(items)}")
            items = list(items)
            shard_limit = int(self.shard_counts[shard_idx])
            if len(items) > shard_limit:
                items = items[:shard_limit]
            if items:
                _validate_cached_example_schema(items[0], shard_path=shard_key, local_idx=0)
            self._loaded_shard_key = shard_key
            self._loaded_items = items
        return self._loaded_items

    def clear_cache(self) -> None:
        self._loaded_shard_key = None
        self._loaded_items = None
        gc.collect()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx += self.total_items
        if idx < 0 or idx >= self.total_items:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_items}")

        shard_idx = bisect.bisect_right(self.cum_counts, idx)
        shard_start = 0 if shard_idx == 0 else self.cum_counts[shard_idx - 1]
        local_idx = idx - shard_start
        ex = self._load_shard_items(shard_idx)[local_idx]
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


class _SplitAwareDataLoader(DataLoader):
    def __init__(self, *args, on_iter_start=None, **kwargs):
        self._on_iter_start = on_iter_start
        super().__init__(*args, **kwargs)

    def __iter__(self):
        if callable(self._on_iter_start):
            self._on_iter_start()
        return super().__iter__()


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

        def _clear_other_caches(active_ds: TQA_MemmapDataset) -> None:
            for ds in (train_ds, val_ds, test_ds):
                if ds is not active_ds:
                    ds.clear_cache()

        self.train_loader = _SplitAwareDataLoader(
            train_ds, batch_size=batch_size, shuffle=bool(shuffle),
            num_workers=num_workers, pin_memory=bool(pin_memory),
            collate_fn=self.collate_fn,
            prefetch_factor=int(prefetch_factor) if num_workers > 0 else None,
            persistent_workers=bool(persistent_workers) if num_workers > 0 else False,
            on_iter_start=lambda: _clear_other_caches(train_ds),
        )
        self.valid_loader = _SplitAwareDataLoader(
            val_ds, batch_size=test_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=bool(pin_memory),
            collate_fn=self.collate_fn,
            prefetch_factor=int(prefetch_factor) if num_workers > 0 else None,
            persistent_workers=bool(persistent_workers) if num_workers > 0 else False,
            on_iter_start=lambda: _clear_other_caches(val_ds),
        )
        self.test_loader = _SplitAwareDataLoader(
            test_ds, batch_size=test_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=bool(pin_memory),
            collate_fn=self.collate_fn,
            prefetch_factor=int(prefetch_factor) if num_workers > 0 else None,
            persistent_workers=bool(persistent_workers) if num_workers > 0 else False,
            on_iter_start=lambda: _clear_other_caches(test_ds),
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
