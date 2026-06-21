"""
Cached Hateful Memes Dataset + Dataloader (reads pre-built encoder features).

Records produced by HM_CodeBook have per-example keys:
    id, label, text, img_tokens, img_pool, txt_tokens, txt_pool, txt_mask

Collate emits the contract the SynIB FusionIBModel_Mask expects downstream:
    batch = {
        "id": [str, ...],
        "label": LongTensor (B,),
        "data": {
            0: {"tokens": (B, Ntxt_max, D_t), "mask": (B, Ntxt_max), "pool": (B, D_t)},
            1: {"tokens": (B, Nimg, D_v),     "mask": (B, Nimg),     "pool": (B, D_v)},
        },
    }
"""

from __future__ import annotations

import json
import os
from bisect import bisect_right
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


def _load_manifest(split_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(split_dir, "manifest.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing manifest.jsonl at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class HM_ShardedLazyDataset(Dataset):
    def __init__(
        self,
        cache_root: str,
        split: str,
        *,
        max_items: Optional[int] = None,
        shard_cache_size: int = 2,
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

        self.shard_cache_size = int(shard_cache_size)
        self._cache: "OrderedDict[int, List[Dict[str, Any]]]" = OrderedDict()

    def __len__(self) -> int:
        return self.N

    def _locate(self, idx: int) -> Tuple[int, int]:
        sid = bisect_right(self.cum, idx)
        prev = 0 if sid == 0 else self.cum[sid - 1]
        return sid, idx - prev

    def _get_shard(self, sid: int) -> List[Dict[str, Any]]:
        if sid in self._cache:
            self._cache.move_to_end(sid)
            return self._cache[sid]
        items = torch.load(self.shard_paths[sid], map_location="cpu")
        items = list(items)
        self._cache[sid] = items
        self._cache.move_to_end(sid)
        while len(self._cache) > self.shard_cache_size:
            self._cache.popitem(last=False)
        return items

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid, local = self._locate(int(idx))
        ex = self._get_shard(sid)[local]
        return {
            "id": str(ex.get("id", idx)),
            "text": ex.get("text", ""),
            "label": torch.tensor(int(ex.get("label", -1)), dtype=torch.long),
            "img_tokens": ex["img_tokens"].to(torch.float32),
            "img_pool": ex["img_pool"].to(torch.float32),
            "txt_tokens": ex["txt_tokens"].to(torch.float32),
            "txt_pool": ex["txt_pool"].to(torch.float32),
            "txt_mask": ex["txt_mask"].to(torch.bool),
        }


def _pad_tokens(seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    if not seqs:
        return torch.empty((0, 0, 0)), torch.empty((0, 0), dtype=torch.bool)
    lens = [int(s.shape[0]) for s in seqs]
    Lmax = max(lens)
    D = int(seqs[0].shape[1])
    dtype = seqs[0].dtype
    padded = torch.zeros((len(seqs), Lmax, D), dtype=dtype)
    mask = torch.zeros((len(seqs), Lmax), dtype=torch.bool)
    for i, s in enumerate(seqs):
        L = int(s.shape[0])
        if L == 0:
            continue
        padded[i, :L, :] = s
        mask[i, :L] = True
    return padded, mask


def collate_hm_cached(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    ids = [b["id"] for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)
    texts = [b["text"] for b in batch]

    img_tokens = torch.stack([b["img_tokens"] for b in batch], dim=0)
    img_pool = torch.stack([b["img_pool"] for b in batch], dim=0)
    img_mask = torch.ones(img_tokens.shape[:2], dtype=torch.bool)

    txt_tokens, txt_mask = _pad_tokens([b["txt_tokens"] for b in batch])
    txt_pool = torch.stack([b["txt_pool"] for b in batch], dim=0)

    return {
        "id": ids,
        "text": texts,
        "label": labels,
        "data": {
            0: {"tokens": txt_tokens, "mask": txt_mask, "pool": txt_pool},
            1: {"tokens": img_tokens, "mask": img_mask, "pool": img_pool},
        },
    }


class HM_MemmapDataloader:
    """Dataloader for the cached HM features.

    Pulls from splits: train / dev_unseen / test_seen by default
    (test_seen has public labels; dev_unseen drives early stopping per the plan).
    """

    def __init__(
        self,
        config,
        *,
        train_split: Optional[str] = None,
        valid_split: Optional[str] = None,
        test_split: Optional[str] = None,
        max_items: Optional[int] = None,
        train_max_items: Optional[int] = None,
        pin_memory: bool = False,
        shuffle: bool = True,
    ):
        cache_root = str(config.dataset.cache_root)
        batch_size = int(config.training_params.batch_size)
        test_batch_size = int(getattr(config.training_params, "test_batch_size", batch_size))

        train_split = train_split or getattr(config.dataset, "train_split", "train")
        valid_split = valid_split or getattr(config.dataset, "valid_split", "dev_unseen")
        test_split = test_split or getattr(config.dataset, "test_split", "test_seen")
        train_max_items = train_max_items or getattr(config.dataset, "train_max_items", None)
        train_max_items = int(train_max_items) if train_max_items is not None else None

        train_ds = HM_ShardedLazyDataset(cache_root, train_split, max_items=train_max_items)
        valid_ds = HM_ShardedLazyDataset(cache_root, valid_split, max_items=max_items)
        test_ds = HM_ShardedLazyDataset(cache_root, test_split, max_items=max_items)

        self.collate_fn = collate_hm_cached
        workers = int(getattr(config.training_params, "data_loader_workers", 0))

        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn,
            drop_last=True,
        )
        self.valid_loader = DataLoader(
            valid_ds,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn,
            drop_last=False,
        )
        self.test_loader = DataLoader(
            test_ds,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn,
            drop_last=False,
        )
