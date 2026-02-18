
import os
import json
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader


def _stats(name: str, t: Any) -> str:
    if not torch.is_tensor(t):
        return f"{name:24s}: {type(t)}"
    if t.numel() == 0:
        return f"{name:24s}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} numel=0"
    x = t.detach()
    xf = x.float()
    mn = xf.min().item()
    mx = xf.max().item()
    mean = xf.mean().item()
    std = xf.std(unbiased=False).item()
    nonzero = (x != 0).float().mean().item() * 100.0
    return (f"{name:24s}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
            f"min={mn:.6g} max={mx:.6g} mean={mean:.6g} std={std:.6g} nonzero={nonzero:.2f}%")
def _require_tensor(ex: dict, key: str) -> torch.Tensor:
    if key not in ex:
        raise KeyError(f"Missing key in shard item: {key}")
    t = ex[key]
    if not torch.is_tensor(t):
        raise TypeError(f"Key {key} must be a torch.Tensor, got {type(t)}")
    return t
def _check_shapes(ex: dict) -> None:
    input_ids = _require_tensor(ex, "input_ids")
    attention_mask = _require_tensor(ex, "attention_mask")
    position_ids = _require_tensor(ex, "position_ids")
    input_embeds = _require_tensor(ex, "input_embeds")
    visual_pos_masks = _require_tensor(ex, "visual_pos_masks")
    deep = _require_tensor(ex, "deepstack_visual_embeds")

    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids expected (1,L), got {tuple(input_ids.shape)}")
    if attention_mask.shape != input_ids.shape:
        raise ValueError(f"attention_mask expected {tuple(input_ids.shape)}, got {tuple(attention_mask.shape)}")
    if input_embeds.dim() != 3 or input_embeds.shape[0] != 1 or input_embeds.shape[1] != input_ids.shape[1]:
        raise ValueError(f"input_embeds expected (1,L,2048), got {tuple(input_embeds.shape)} vs L={input_ids.shape[1]}")
    if input_embeds.shape[-1] != 2048:
        raise ValueError(f"input_embeds last dim must be 2048, got {input_embeds.shape[-1]}")
    if visual_pos_masks.shape != input_ids.shape:
        raise ValueError(f"visual_pos_masks expected {tuple(input_ids.shape)}, got {tuple(visual_pos_masks.shape)}")
    if position_ids.dim() != 3 or tuple(position_ids.shape[:2]) != (3, 1) or position_ids.shape[-1] != input_ids.shape[1]:
        raise ValueError(f"position_ids expected (3,1,L), got {tuple(position_ids.shape)} vs L={input_ids.shape[1]}")
    if deep.dim() != 3 or deep.shape[1:] != (64, 2048):
        raise ValueError(f"deepstack_visual_embeds expected (K,64,2048), got {tuple(deep.shape)}")

def _pad_1d(x: torch.Tensor, L: int, pad_value: int, padding_side: str) -> torch.Tensor:
    x = x.reshape(-1)
    n = int(x.numel())
    if n == L:
        return x
    if n > L:
        return x[-L:] if padding_side == "left" else x[:L]
    out = torch.full((L,), pad_value, dtype=x.dtype)
    if padding_side == "left":
        out[-n:] = x
    else:
        out[:n] = x
    return out
def _pad_2d_time(x: torch.Tensor, L: int, pad_value: float, padding_side: str) -> torch.Tensor:
    # x: (T,D)
    if x.dim() != 2:
        raise ValueError(f"_pad_2d_time expects 2D (T,D), got {tuple(x.shape)}")
    T, D = int(x.shape[0]), int(x.shape[1])
    if T == L:
        return x
    if T > L:
        return x[-L:, :] if padding_side == "left" else x[:L, :]
    out = torch.full((L, D), float(pad_value), dtype=x.dtype)
    if padding_side == "left":
        out[-T:, :] = x
    else:
        out[:T, :] = x
    return out
def _pad_pos_3_1_T(pos: torch.Tensor, L: int, padding_side: str) -> torch.Tensor:
    # pos: (3,1,T)
    if pos.dim() != 3 or int(pos.shape[0]) != 3 or int(pos.shape[1]) != 1:
        raise ValueError(f"position_ids expected (3,1,T), got {tuple(pos.shape)}")
    T = int(pos.shape[-1])
    if T == L:
        return pos
    if T > L:
        return pos[:, :, -L:] if padding_side == "left" else pos[:, :, :L]
    out = torch.zeros((3, 1, L), dtype=pos.dtype)
    if padding_side == "left":
        out[:, :, -T:] = pos
    else:
        out[:, :, :T] = pos
    return out
def _pad_deepstack_K_64_2048(x: torch.Tensor, Kmax: int, padding_side: str) -> torch.Tensor:
    # x: (K,64,2048) -> (Kmax,64,2048)
    if Kmax == 0:
        return torch.zeros((0, 64, 2048), dtype=x.dtype if torch.is_tensor(x) else torch.float16)

    if (not torch.is_tensor(x)) or x.numel() == 0:
        return torch.zeros((Kmax, 64, 2048), dtype=torch.float16)

    if x.dim() != 3 or x.shape[1:] != (64, 2048):
        raise ValueError(f"deepstack_visual_embeds expected (K,64,2048), got {tuple(x.shape)}")

    K = int(x.shape[0])
    if K == Kmax:
        return x
    if K > Kmax:
        return x[-Kmax:] if padding_side == "left" else x[:Kmax]

    out = torch.zeros((Kmax, 64, 2048), dtype=x.dtype)
    if padding_side == "left":
        out[-K:] = x
    else:
        out[:K] = x
    return out
def _pad_deepstack_stack64(x: torch.Tensor, Kmax: int, padding_side: str) -> torch.Tensor:
    """
    Stack the 64s across deepstack levels.

    Input:
      x: (K, 64, 2048)

    Output (padded/truncated to Kmax levels):
      (Kmax*64, 2048)

    Padding:
      - If x has fewer than Kmax levels -> pad with zeros (as extra 64-blocks)
      - If x has more than Kmax levels -> truncate levels (from left/right depending on padding_side)
    """
    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    # target rows after stacking 64s
    target_rows = int(Kmax) * 64

    # handle empty target
    if target_rows == 0:
        dtype = x.dtype if torch.is_tensor(x) else torch.float16
        return torch.zeros((0, 2048), dtype=dtype)

    # missing/empty deepstack -> zeros
    if (not torch.is_tensor(x)) or x.numel() == 0:
        return torch.zeros((target_rows, 2048), dtype=torch.float16)

    if x.dim() != 3 or tuple(x.shape[1:]) != (64, 2048):
        raise ValueError(f"deepstack_visual_embeds expected (K,64,2048), got {tuple(x.shape)}")

    K = int(x.shape[0])

    # truncate/pad in K first
    if K == Kmax:
        x2 = x
    elif K > Kmax:
        x2 = x[-Kmax:] if padding_side == "left" else x[:Kmax]
    else:
        pad_levels = torch.zeros((Kmax - K, 64, 2048), dtype=x.dtype)
        x2 = torch.cat([pad_levels, x], dim=0) if padding_side == "left" else torch.cat([x, pad_levels], dim=0)

    # now stack 64s: (Kmax,64,2048) -> (Kmax*64, 2048)
    return x2.reshape(Kmax * 64, 2048).contiguous()


class ESNLI_MemmapDataset(Dataset):
    def __init__(
        self,
        cache_root: str,
        split: str,
        shard_index: int = 0,
        shard_path: Optional[str] = None,
        max_items: Optional[int] = None,
    ):
        super().__init__()
        split_dir = os.path.join(cache_root, split)

        if shard_path is None:
            manifest_path = os.path.join(split_dir, "manifest.jsonl")
            if not os.path.isfile(manifest_path):
                raise FileNotFoundError(f"Missing manifest.jsonl at {manifest_path}")
            recs = [json.loads(l) for l in open(manifest_path, "r", encoding="utf-8")]
            if len(recs) == 0:
                raise RuntimeError(f"Empty manifest: {manifest_path}")
            shard_index = int(shard_index)
            if shard_index < 0 or shard_index >= len(recs):
                raise IndexError(f"shard_index {shard_index} out of range [0, {len(recs)-1}]")
            shard_path = os.path.join(split_dir, recs[shard_index]["shard"])

        if not os.path.isfile(shard_path):
            raise FileNotFoundError(f"Shard not found: {shard_path}")

        items = torch.load(shard_path, map_location="cpu")
        if not isinstance(items, (list, tuple)):
            raise TypeError(f"Expected shard to contain list/tuple, got {type(items)}")

        if max_items is not None:
            items = list(items)[: int(max_items)]
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.items[idx]
        if not isinstance(ex, dict):
            raise TypeError(f"Example[{idx}] is not a dict: {type(ex)}")

        lab = ex.get("label", 0)
        if torch.is_tensor(lab):
            lab = int(lab.detach().cpu().reshape(-1)[0].item()) if lab.numel() else 0
        try:
            lab = int(lab)
        except Exception:
            lab = 0

        out = {
            "id": ex.get("id", idx),
            "prompt": ex.get("prompt", ""),
            "label": torch.tensor(lab, dtype=torch.long),

            "input_ids": _require_tensor(ex, "input_ids").to(torch.long),
            "attention_mask": _require_tensor(ex, "attention_mask").to(torch.long),
            "position_ids": _require_tensor(ex, "position_ids").to(torch.long),
            "input_embeds": _require_tensor(ex, "input_embeds").to(torch.float16),
            "visual_pos_masks": _require_tensor(ex, "visual_pos_masks").to(torch.bool),
            "deepstack_visual_embeds": _require_tensor(ex, "deepstack_visual_embeds").to(torch.float16),
        }

        _check_shapes(out)
        return out

def make_collate_from_cache( *, padding_side: str = "right", pad_token_id: int = 0,):
    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(batch)
        ids_ = [b["id"] for b in batch]
        prompts_ = [b.get("prompt", "") for b in batch]
        labels = torch.stack([b["label"] for b in batch], dim=0)

        # squeeze cached (1,L)->(L) etc; collect lengths and K
        ids_list, attn_list, pos_list, emb_list, vmask_list, deep_list = [], [], [], [], [], []
        lengths, Ks = [], []

        for b in batch:
            ids = b["input_ids"]
            am = b["attention_mask"]
            pos = b["position_ids"]
            emb = b["input_embeds"]
            vm = b["visual_pos_masks"]
            ds = b["deepstack_visual_embeds"]

            # squeeze sample batch dim
            ids = ids[0] if (ids.dim() == 2 and ids.shape[0] == 1) else ids
            am  = am[0]  if (am.dim() == 2 and am.shape[0] == 1) else am
            emb = emb[0] if (emb.dim() == 3 and emb.shape[0] == 1) else emb  # (L,2048)
            vm  = vm[0]  if (vm.dim() == 2 and vm.shape[0] == 1) else vm      # (L,)
            # pos stays (3,1,L)

            L = int(ids.numel())
            K = int(ds.shape[0]) if (torch.is_tensor(ds) and ds.dim() == 3) else 0
            lengths.append(L)
            Ks.append(K)

            ids_list.append(ids)
            attn_list.append(am)
            pos_list.append(pos)
            emb_list.append(emb)
            vmask_list.append(vm)
            deep_list.append(ds)

        Lmax = max(lengths) if lengths else 0
        Kmax = max(Ks) if Ks else 0

        input_ids = torch.stack([_pad_1d(x, Lmax, pad_value=pad_token_id, padding_side=padding_side) for x in ids_list], dim=0)
        attention_mask = torch.stack([_pad_1d(x, Lmax, pad_value=0, padding_side=padding_side) for x in attn_list], dim=0)

        # (B,3,1,Lmax)
        position_ids = torch.stack([_pad_pos_3_1_T(p, Lmax, padding_side=padding_side) for p in pos_list], dim=0)

        # (B,Lmax,2048)
        input_embeds = torch.stack([_pad_2d_time(e, Lmax, pad_value=0.0, padding_side=padding_side) for e in emb_list], dim=0)

        # (B,Lmax) bool
        visual_pos_masks = torch.stack([_pad_1d(v.to(torch.bool), Lmax, pad_value=0, padding_side=padding_side).to(torch.bool) for v in vmask_list], dim=0)

        # (B,Kmax,64,2048)
        deepstack_visual_embeds = torch.stack(
            [_pad_deepstack_stack64(d, Kmax, padding_side=padding_side) for d in deep_list],
            dim=0,
        )  # (B, Kmax*64, 2048)

        return {
            "ids": ids_,
            "prompts": prompts_,
            "label": labels,
            "data": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "input_embeds": input_embeds,
                "visual_pos_masks": visual_pos_masks,
                "deepstack_visual_embeds": deepstack_visual_embeds,
            },
        }

    return collate

class ESNLI_MemmapDataloader:
    def __init__(
        self,
        config,
        *,
        split: str = "validation",
        shard_index: int = 0,
        shard_path: Optional[str] = None,
        max_items: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        padding_side: str = "right",
        pad_token_id: int = 0,
        print_first_batch_stats: bool = True,
    ):
        cache_root = config.dataset.cache_root
        batch_size = int(config.training_params.batch_size)

        ds = ESNLI_MemmapDataset(
            cache_root=cache_root,
            split=split,
            shard_index=shard_index,
            shard_path=shard_path,
            max_items=max_items,
        )

        collate_fn = make_collate_from_cache(
            padding_side=padding_side,
            pad_token_id=pad_token_id,
        )

        self.loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            collate_fn=collate_fn,
        )

        self.train_loader = self.loader
        self.valid_loader = self.loader
        self.test_loader = self.loader

        if print_first_batch_stats:
            batch = next(iter(self.loader))
            ex = batch["data"]
            print("\n[cache-reader] first batch loaded (padded from cache):")
            for k in ["input_ids", "attention_mask", "position_ids", "input_embeds", "visual_pos_masks", "deepstack_visual_embeds"]:
                print(_stats(k, ex[k]))
















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

import os, json
from bisect import bisect_right
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

def _load_manifest(split_dir: str) -> List[Dict[str, Any]]:
    mp = os.path.join(split_dir, "manifest.jsonl")
    if not os.path.isfile(mp):
        raise FileNotFoundError(f"Missing manifest.jsonl in {split_dir}")
    return [json.loads(l) for l in open(mp, "r", encoding="utf-8")]

class ESNLI_ShardedLazyDataset(Dataset):
    def __init__(
        self,
        cache_root: str,
        split: str,
        *,
        max_items: Optional[int] = None,
        deep_dim: int = 2048,
        shard_cache_size: int = 2,   # keep only 1–4 shards in RAM
        **kwargs
    ):
        super().__init__()
        self.split_dir = os.path.join(cache_root, split)
        self.recs = _load_manifest(self.split_dir)

        # absolute shard paths + counts
        self.shard_paths: List[str] = [os.path.join(self.split_dir, r["shard"]) for r in self.recs]
        self.shard_counts: List[int] = [int(r["num_items"]) for r in self.recs]

        # prefix sums to map global idx -> shard/local
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

        print(f"[LazyShardDataset] split={split} N={self.N} shards={len(self.shard_paths)} cache_shards={self.shard_cache_size}")

    def __len__(self) -> int:
        return self.N

    def _locate(self, idx: int) -> Tuple[int, int]:
        # shard_id = first cum > idx
        shard_id = bisect_right(self.cum, idx)
        prev = 0 if shard_id == 0 else self.cum[shard_id - 1]
        local = idx - prev
        return shard_id, local

    def _get_shard(self, shard_id: int) -> List[Dict[str, Any]]:
        # LRU cache
        if shard_id in self._shard_cache:
            self._shard_cache.move_to_end(shard_id)
            return self._shard_cache[shard_id]

        sp = self.shard_paths[shard_id]
        items = torch.load(sp, map_location="cpu")
        items = list(items)

        self._shard_cache[shard_id] = items
        self._shard_cache.move_to_end(shard_id)

        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)  # evict LRU

        return items

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_id, local = self._locate(int(idx))
        shard = self._get_shard(shard_id)
        ex = shard[local]

        # ---- your existing per-example logic (unchanged) ----
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


def memmap_collate(
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

class ESNLI_MemmapDataloader:
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

        train_ds = ESNLI_ShardedLazyDataset(
            cache_root=cache_root,
            split="train",
            shard_path=shard_path,
            max_items=max_items,
            deep_dim=deep_dim,
        )
        val_ds = ESNLI_ShardedLazyDataset(
            cache_root=cache_root,
            split="validation",
            shard_path=shard_path,
            max_items=max_items,
            deep_dim=deep_dim,
        )
        test_ds = ESNLI_ShardedLazyDataset(
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
            print("\n[scienceqa-cache-reader] first batch loaded (padded from cache):")
            for k in ex.keys():
                t = ex[k]
                if torch.is_tensor(t):
                    print(f"{k:24s}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} numel={t.numel()}")
                else:
                    print(f"{k:24s}: {type(t)}")


