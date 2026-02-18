"""
ScienceQA memmap cache (simple, readable)

What this file gives you:
1) A ONE-TIME converter that turns your existing torch shards (list[dict]) into a
   memmap-friendly format: flat binary arrays + a small .npz index.
2) A Dataset + Dataloader that reads those memmaps with O(1) slicing and minimal Python overhead.

Why this is faster than torch.load(list-of-dicts):
- torch.load recreates many Python objects every run
- memmap lets the OS page in only the bytes you touch + avoids deserialization

Directory layout (per split):
cache_root/
  train/
    manifest.jsonl or data.pt or shards...
  train_memmap/            <-- created by convert
    index.npz
    meta.json
    input_ids.bin
    attention_mask.bin
    position_ids.bin
    visual_pos_masks.bin
    hint_mask.bin
    input_embeds.bin
    deepstack_visual_embeds.bin

Notes / Simplifications:
- This version stores only tensors + labels. If you want "prompt" strings, store a prompts.jsonl
  alongside the memmaps and load by idx (kept out to keep this simple and fast).
- position_ids is stored flattened. Your collate already handles both (L,) and (3,1,L);
  we keep it simple: store exactly whatever flatten gives you, and collate pads it.

Dependencies: numpy, torch, einops
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import einops
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Small utilities
# ============================================================

def _as_scalar_int(x: Any, default: int = 0) -> int:
    if x is None:
        return int(default)
    if torch.is_tensor(x):
        return int(x.detach().cpu().reshape(-1)[0].item()) if x.numel() else int(default)
    try:
        return int(x)
    except Exception:
        return int(default)


def _normalize_input_embeds(x: torch.Tensor) -> torch.Tensor:
    """
    Accepts per-example shapes:
      - (N, D)
      - (1, N, D) -> squeeze
      - (D,) -> (1, D)
      - (K, N, D) -> flatten K into N
    Returns (N, D) float16 CPU contiguous.
    """
    if x is None or (not torch.is_tensor(x)) or x.numel() == 0:
        return torch.empty((0, 0), dtype=torch.float16)

    if x.dim() == 3:
        if x.shape[0] == 1:
            x = x[0]
        else:
            x = x.reshape(-1, x.shape[-1])
    elif x.dim() == 2:
        pass
    elif x.dim() == 1:
        x = x.view(1, -1)
    else:
        x = x.reshape(-1, x.shape[-1])

    return x.detach().to(dtype=torch.float16, device="cpu").contiguous()


def _normalize_deep(x: torch.Tensor, deep_dim: int) -> Tuple[torch.Tensor, int, int]:
    """
    Expects deepstack_visual_embeds roughly as:
      - (T, N, D) or (1, T, N, D)
    Returns:
      deep_rows: (T*N, deep_dim) float16 CPU contiguous
      T, N
    """
    if x is None or (not torch.is_tensor(x)) or x.numel() == 0:
        return torch.empty((0, deep_dim), dtype=torch.float16), 0, 0

    if x.dim() == 4:
        x = x[0]
    if x.dim() != 3:
        return torch.empty((0, deep_dim), dtype=torch.float16), 0, 0

    T, N, D = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    x = x.detach().to(dtype=torch.float16, device="cpu").contiguous()

    # crop/pad last dim to deep_dim
    if D > deep_dim:
        x = x[..., :deep_dim]
    elif D < deep_dim:
        pad = torch.zeros((T, N, deep_dim - D), dtype=torch.float16)
        x = torch.cat([x, pad], dim=-1)

    x = x.reshape(T * N, deep_dim).contiguous()
    return x, T, N


def _load_manifest_shards(split_dir: str) -> List[str]:
    manifest_path = os.path.join(split_dir, "manifest.jsonl")
    if not os.path.isfile(manifest_path):
        return []
    recs = [json.loads(l) for l in open(manifest_path, "r", encoding="utf-8")]
    return [os.path.join(split_dir, r["shard"]) for r in recs]


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
        items = list(torch.load(shard_path, map_location="cpu"))
        return items[: int(max_items)] if max_items is not None else items

    # Option B: manifest shards
    shard_paths = _load_manifest_shards(split_dir)
    if shard_paths:
        if shard_index is not None:
            si = int(shard_index)
            if si < 0 or si >= len(shard_paths):
                raise IndexError(f"shard_index {si} out of range [0, {len(shard_paths)-1}]")
            items = list(torch.load(shard_paths[si], map_location="cpu"))
            return items[: int(max_items)] if max_items is not None else items

        out: List[Dict[str, Any]] = []
        for sp in shard_paths:
            out.extend(list(torch.load(sp, map_location="cpu")))
            if max_items is not None and len(out) >= int(max_items):
                out = out[: int(max_items)]
                break
        return out

    # Option C: single data.pt
    data_pt = os.path.join(split_dir, "data.pt")
    if os.path.isfile(data_pt):
        items = list(torch.load(data_pt, map_location="cpu"))
        return items[: int(max_items)] if max_items is not None else items

    raise FileNotFoundError(f"Need manifest.jsonl shards or data.pt in {split_dir}")


# ============================================================
# 1) One-time conversion: list-of-dicts -> memmaps
# ============================================================

def convert_split_to_memmap(
    cache_root: str,
    split: str,
    *,
    out_split_name: Optional[str] = None,
    shard_index: Optional[int] = None,
    shard_path: Optional[str] = None,
    max_items: Optional[int] = None,
    deep_dim: int = 2048,
) -> str:
    """
    Reads the existing torch cached split (list[dict]) and writes a memmap version.
    Returns the memmap directory path.

    By default writes to: <cache_root>/<split>_memmap/
    """
    split_dir = os.path.join(cache_root, split)
    items = _load_split_items(split_dir, shard_index=shard_index, shard_path=shard_path, max_items=max_items)

    if out_split_name is None:
        out_split_name = f"{split}_memmap"
    out_dir = os.path.join(cache_root, out_split_name)
    os.makedirs(out_dir, exist_ok=True)

    N = len(items)
    if N == 0:
        raise ValueError("No items loaded; nothing to convert.")

    # ---- PASS 1: compute sizes (lengths/rows) and infer Dv
    len_input = np.zeros(N, dtype=np.int32)
    len_attn  = np.zeros(N, dtype=np.int32)
    len_pos   = np.zeros(N, dtype=np.int32)
    len_vpm   = np.zeros(N, dtype=np.int32)
    len_hint  = np.zeros(N, dtype=np.int32)

    rows_vis  = np.zeros(N, dtype=np.int32)   # rows for input_embeds (N_i)
    rows_deep = np.zeros(N, dtype=np.int64)   # rows for deep (T_i * N_i)
    deep_T    = np.zeros(N, dtype=np.int32)
    deep_N    = np.zeros(N, dtype=np.int32)

    labels = np.zeros(N, dtype=np.int64)

    Dv = 0  # input_embeds last dim
    for i, ex in enumerate(items):
        input_ids = ex["input_ids"].to(torch.long).reshape(-1)
        attention_mask = ex["attention_mask"].to(torch.long).reshape(-1)
        pos_f = ex["position_ids"].reshape(-1).to(torch.long)

        vpm = ex["visual_pos_masks"].reshape(-1)
        hint = ex["masks"]["hint"].reshape(-1)

        len_input[i] = input_ids.numel()
        len_attn[i]  = attention_mask.numel()
        len_pos[i]   = pos_f.numel()
        len_vpm[i]   = vpm.numel()
        len_hint[i]  = hint.numel()
        labels[i]    = _as_scalar_int(ex.get("label", 0), default=0)

        vis = ex.get("input_embeds", None)
        if torch.is_tensor(vis) and vis.numel() > 0:
            vis2 = _normalize_input_embeds(vis)
            if Dv == 0 and vis2.numel() > 0:
                Dv = int(vis2.shape[1])
            rows_vis[i] = int(vis2.shape[0])

        deep = ex.get("deepstack_visual_embeds", None)
        if torch.is_tensor(deep) and deep.numel() > 0:
            deep_rows, T, Nn = _normalize_deep(deep, deep_dim=deep_dim)
            deep_T[i] = T
            deep_N[i] = Nn
            rows_deep[i] = int(deep_rows.shape[0])

    # offsets = cumulative sums
    def _offsets(lengths: np.ndarray) -> np.ndarray:
        off = np.zeros_like(lengths, dtype=np.int64)
        if len(lengths) > 1:
            off[1:] = np.cumsum(lengths[:-1], dtype=np.int64)
        return off

    off_input = _offsets(len_input)
    off_attn  = _offsets(len_attn)
    off_pos   = _offsets(len_pos)
    off_vpm   = _offsets(len_vpm)
    off_hint  = _offsets(len_hint)
    off_vis   = _offsets(rows_vis)
    off_deep  = _offsets(rows_deep.astype(np.int64))

    total_input = int(off_input[-1] + len_input[-1])
    total_attn  = int(off_attn[-1]  + len_attn[-1])
    total_pos   = int(off_pos[-1]   + len_pos[-1])
    total_vpm   = int(off_vpm[-1]   + len_vpm[-1])
    total_hint  = int(off_hint[-1]  + len_hint[-1])
    total_vis_rows  = int(off_vis[-1]  + rows_vis[-1]) if Dv > 0 else 0
    total_deep_rows = int(off_deep[-1] + rows_deep[-1])

    # ---- Allocate memmaps (simple dtypes)
    # You can change int32->int64 if your token ids exceed int32 range (almost never).
    mm_input = np.memmap(os.path.join(out_dir, "input_ids.bin"), mode="w+", dtype=np.int32, shape=(total_input,))
    mm_attn  = np.memmap(os.path.join(out_dir, "attention_mask.bin"), mode="w+", dtype=np.int8,  shape=(total_attn,))
    mm_pos   = np.memmap(os.path.join(out_dir, "position_ids.bin"), mode="w+", dtype=np.int32, shape=(total_pos,))
    mm_vpm   = np.memmap(os.path.join(out_dir, "visual_pos_masks.bin"), mode="w+", dtype=np.uint8, shape=(total_vpm,))
    mm_hint  = np.memmap(os.path.join(out_dir, "hint_mask.bin"), mode="w+", dtype=np.uint8, shape=(total_hint,))

    mm_vis = None
    if Dv > 0:
        mm_vis = np.memmap(os.path.join(out_dir, "input_embeds.bin"), mode="w+", dtype=np.float16, shape=(total_vis_rows, Dv))

    mm_deep = np.memmap(os.path.join(out_dir, "deepstack_visual_embeds.bin"), mode="w+", dtype=np.float16, shape=(total_deep_rows, deep_dim))

    # ---- PASS 2: write data
    for i, ex in enumerate(items):
        # 1D
        input_ids = ex["input_ids"].to(torch.long).reshape(-1).to(torch.int32).cpu().numpy()
        attention_mask = ex["attention_mask"].to(torch.long).reshape(-1).to(torch.int8).cpu().numpy()
        pos_f = ex["position_ids"].reshape(-1).to(torch.long).to(torch.int32).cpu().numpy()
        vpm = ex["visual_pos_masks"].reshape(-1).to(torch.uint8).cpu().numpy()
        hint = ex["masks"]["hint"].reshape(-1).to(torch.uint8).cpu().numpy()

        a0, a1 = int(off_input[i]), int(off_input[i] + len_input[i])
        b0, b1 = int(off_attn[i]),  int(off_attn[i]  + len_attn[i])
        c0, c1 = int(off_pos[i]),   int(off_pos[i]   + len_pos[i])
        d0, d1 = int(off_vpm[i]),   int(off_vpm[i]   + len_vpm[i])
        e0, e1 = int(off_hint[i]),  int(off_hint[i]  + len_hint[i])

        mm_input[a0:a1] = input_ids
        mm_attn[b0:b1]  = attention_mask
        mm_pos[c0:c1]   = pos_f
        mm_vpm[d0:d1]   = vpm
        mm_hint[e0:e1]  = hint

        # input_embeds
        if mm_vis is not None and rows_vis[i] > 0:
            vis2 = _normalize_input_embeds(ex["input_embeds"])
            # ensure correct D
            if int(vis2.shape[1]) > Dv:
                vis2 = vis2[:, :Dv]
            elif int(vis2.shape[1]) < Dv:
                pad = torch.zeros((int(vis2.shape[0]), Dv - int(vis2.shape[1])), dtype=torch.float16)
                vis2 = torch.cat([vis2, pad], dim=1)
            r0, r1 = int(off_vis[i]), int(off_vis[i] + rows_vis[i])
            mm_vis[r0:r1, :] = vis2.numpy()

        # deepstack
        if rows_deep[i] > 0:
            deep_rows, _, _ = _normalize_deep(ex["deepstack_visual_embeds"], deep_dim=deep_dim)
            r0, r1 = int(off_deep[i]), int(off_deep[i] + rows_deep[i])
            mm_deep[r0:r1, :] = deep_rows.numpy()

    # flush to disk
    mm_input.flush(); mm_attn.flush(); mm_pos.flush(); mm_vpm.flush(); mm_hint.flush()
    if mm_vis is not None:
        mm_vis.flush()
    mm_deep.flush()

    # store index
    np.savez(
        os.path.join(out_dir, "index.npz"),
        len_input=len_input, off_input=off_input,
        len_attn=len_attn, off_attn=off_attn,
        len_pos=len_pos, off_pos=off_pos,
        len_vpm=len_vpm, off_vpm=off_vpm,
        len_hint=len_hint, off_hint=off_hint,
        rows_vis=rows_vis, off_vis=off_vis, Dv=np.int32(Dv),
        rows_deep=rows_deep, off_deep=off_deep,
        deep_T=deep_T, deep_N=deep_N,
        labels=labels,
        deep_dim=np.int32(deep_dim),
    )
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"N": int(N), "Dv": int(Dv), "deep_dim": int(deep_dim)}, f)

    print(f"[convert_split_to_memmap] wrote: {out_dir}  N={N}  Dv={Dv}  deep_dim={deep_dim}")
    return out_dir


# ============================================================
# 2) Dataset: read memmaps with slicing
# ============================================================

class ScienceQA_MemmapDataset(Dataset):
    """
    Drop-in name: ScienceQA_MemmapDataset
    This version actually uses memmaps.
    """
    def __init__(self, memmap_dir: str):
        super().__init__()
        self.memmap_dir = memmap_dir

        idx = np.load(os.path.join(memmap_dir, "index.npz"))
        self.len_input = idx["len_input"]; self.off_input = idx["off_input"]
        self.len_attn  = idx["len_attn"];  self.off_attn  = idx["off_attn"]
        self.len_pos   = idx["len_pos"];   self.off_pos   = idx["off_pos"]
        self.len_vpm   = idx["len_vpm"];   self.off_vpm   = idx["off_vpm"]
        self.len_hint  = idx["len_hint"];  self.off_hint  = idx["off_hint"]

        self.rows_vis = idx["rows_vis"]; self.off_vis = idx["off_vis"]; self.Dv = int(idx["Dv"])
        self.rows_deep = idx["rows_deep"]; self.off_deep = idx["off_deep"]
        self.deep_T = idx["deep_T"]; self.deep_N = idx["deep_N"]
        self.deep_dim = int(idx["deep_dim"])

        self.labels = torch.from_numpy(idx["labels"]).long()

        # open memmaps
        self.mm_input = np.memmap(os.path.join(memmap_dir, "input_ids.bin"), mode="r", dtype=np.int32)
        self.mm_attn  = np.memmap(os.path.join(memmap_dir, "attention_mask.bin"), mode="r", dtype=np.int8)
        self.mm_pos   = np.memmap(os.path.join(memmap_dir, "position_ids.bin"), mode="r", dtype=np.int32)
        self.mm_vpm   = np.memmap(os.path.join(memmap_dir, "visual_pos_masks.bin"), mode="r", dtype=np.uint8)
        self.mm_hint  = np.memmap(os.path.join(memmap_dir, "hint_mask.bin"), mode="r", dtype=np.uint8)

        self.mm_vis = None
        if self.Dv > 0:
            vis_path = os.path.join(memmap_dir, "input_embeds.bin")
            # infer total rows from file size
            nbytes = os.path.getsize(vis_path)
            total_rows = nbytes // (2 * self.Dv)  # float16 = 2 bytes
            self.mm_vis = np.memmap(vis_path, mode="r", dtype=np.float16, shape=(total_rows, self.Dv))

        deep_path = os.path.join(memmap_dir, "deepstack_visual_embeds.bin")
        nbytes = os.path.getsize(deep_path)
        total_rows = nbytes // (2 * self.deep_dim)
        self.mm_deep = np.memmap(deep_path, mode="r", dtype=np.float16, shape=(total_rows, self.deep_dim))

        print(f"[ScienceQA MemmapDataset] dir={memmap_dir} N={len(self)} Dv={self.Dv} deep_dim={self.deep_dim}")

    def __len__(self) -> int:
        return int(self.len_input.shape[0])

    def __getitem__(self, i: int) -> Dict[str, Any]:
        # 1D slices
        a0, a1 = int(self.off_input[i]), int(self.off_input[i] + self.len_input[i])
        b0, b1 = int(self.off_attn[i]),  int(self.off_attn[i]  + self.len_attn[i])
        c0, c1 = int(self.off_pos[i]),   int(self.off_pos[i]   + self.len_pos[i])
        d0, d1 = int(self.off_vpm[i]),   int(self.off_vpm[i]   + self.len_vpm[i])
        e0, e1 = int(self.off_hint[i]),  int(self.off_hint[i]  + self.len_hint[i])

        input_ids = torch.from_numpy(self.mm_input[a0:a1]).long()
        attention_mask = torch.from_numpy(self.mm_attn[b0:b1]).long()
        position_ids = torch.from_numpy(self.mm_pos[c0:c1]).long()
        visual_pos_masks = torch.from_numpy(self.mm_vpm[d0:d1]).bool()
        hint_mask = torch.from_numpy(self.mm_hint[e0:e1]).bool()

        # input_embeds: (rows, Dv)
        if self.mm_vis is not None and int(self.rows_vis[i]) > 0:
            r0, r1 = int(self.off_vis[i]), int(self.off_vis[i] + self.rows_vis[i])
            input_embeds = torch.from_numpy(self.mm_vis[r0:r1, :]).to(torch.float16)
        else:
            input_embeds = torch.empty((0, 0), dtype=torch.float16)

        # deep: stored as (T*N, deep_dim) then reshaped to (T, N, deep_dim)
        if int(self.rows_deep[i]) > 0:
            r0, r1 = int(self.off_deep[i]), int(self.off_deep[i] + self.rows_deep[i])
            deep_rows = torch.from_numpy(self.mm_deep[r0:r1, :]).to(torch.float16)
            T, Nn = int(self.deep_T[i]), int(self.deep_N[i])
            deepstack_visual_embeds = deep_rows.view(T, Nn, self.deep_dim) if (T > 0 and Nn > 0) else torch.empty((0, 0, self.deep_dim), dtype=torch.float16)
        else:
            deepstack_visual_embeds = torch.empty((0, 0, self.deep_dim), dtype=torch.float16)

        return {
            "id": i,
            "prompt": "",  # intentionally omitted for simplicity
            "label": self.labels[i],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "visual_pos_masks": visual_pos_masks,
            "hint_mask": hint_mask,
            "input_embeds": input_embeds,
            "deepstack_visual_embeds": deepstack_visual_embeds,
        }


# ============================================================
# 3) Collate (mostly your original, but simpler / no extra cpu copies)
# ============================================================

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


def _pad_2d_by_rows(seqs: List[torch.Tensor], pad_val: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads a list of (N_i, D) into:
      padded: (B, Nmax, D)
      mask:   (B, Nmax)
    """
    B = len(seqs)
    if B == 0:
        return torch.empty((0, 0, 0)), torch.empty((0, 0), dtype=torch.bool)

    # infer D
    D = 0
    ref_dtype = torch.float16
    for x in seqs:
        if torch.is_tensor(x) and x.numel() > 0:
            x2 = x
            if x2.dim() == 3:
                x2 = x2[0] if x2.shape[0] == 1 else x2.reshape(-1, x2.shape[-1])
            elif x2.dim() == 1:
                x2 = x2.view(1, -1)
            elif x2.dim() != 2:
                x2 = x2.reshape(-1, x2.shape[-1])
            D = int(x2.shape[1])
            ref_dtype = x2.dtype
            break

    if D == 0:
        return torch.empty((B, 0, 0)), torch.empty((B, 0), dtype=torch.bool)

    Nmax = max(int(x.shape[0]) for x in seqs if torch.is_tensor(x) and x.dim() == 2) if any(
        torch.is_tensor(x) and x.dim() == 2 for x in seqs
    ) else 0

    padded = torch.full((B, Nmax, D), float(pad_val), dtype=ref_dtype)
    mask = torch.zeros((B, Nmax), dtype=torch.bool)

    for i, x in enumerate(seqs):
        if (not torch.is_tensor(x)) or x.numel() == 0:
            continue
        if x.dim() == 3:
            x = x[0] if x.shape[0] == 1 else x.reshape(-1, x.shape[-1])
        elif x.dim() == 1:
            x = x.view(1, -1)
        elif x.dim() != 2:
            x = x.reshape(-1, x.shape[-1])

        n = int(x.shape[0])
        padded[i, :n, :] = x.to(ref_dtype)
        mask[i, :n] = True

    return padded, mask


def _pad_deep_3d(seqs: List[torch.Tensor], deep_dim: int, pad_val: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads list of (T_i, N_i, D) into:
      padded: (B, Tmax, Nmax, D)
      mask:   (B, Tmax, Nmax)
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
        if torch.is_tensor(x) and x.numel() > 0 and x.dim() == 3:
            Tmax = max(Tmax, int(x.shape[0]))
            Nmax = max(Nmax, int(x.shape[1]))
            ref_dtype = x.dtype

    padded = torch.full((B, Tmax, Nmax, deep_dim), float(pad_val), dtype=ref_dtype)
    mask = torch.zeros((B, Tmax, Nmax), dtype=torch.bool)

    for i, x in enumerate(seqs):
        if (not torch.is_tensor(x)) or x.numel() == 0 or x.dim() != 3:
            continue
        T, Nn, D = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])

        # crop/pad last dim
        if D > deep_dim:
            x = x[..., :deep_dim]
        elif D < deep_dim:
            pad = torch.zeros((T, Nn, deep_dim - D), dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)

        padded[i, :T, :Nn, :] = x.to(ref_dtype)
        mask[i, :T, :Nn] = True

    return padded, mask


def scienceqa_memmap_collate(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, Any]:
    ids = [b.get("id", None) for b in batch]
    prompts = [b.get("prompt", "") for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)

    input_ids = _left_pad_1d([b["input_ids"] for b in batch], pad_val=int(pad_token_id), dtype=torch.long)
    attention_mask = _left_pad_1d([b["attention_mask"] for b in batch], pad_val=0, dtype=torch.long)

    # position_ids: support (flattened L,) OR flattened (3*L,)
    pos_list = [b["position_ids"] for b in batch]
    position_ids = _left_pad_1d([p.reshape(-1) for p in pos_list], pad_val=0, dtype=torch.long)

    # Your original did: einops.rearrange(position_ids, "b c i j-> c b (i j)", i=1)
    # But that assumed (3,1,L). Here we keep it simple:
    # If you need the old format, convert at model input time based on expected shape.
    # Keeping position_ids as (B, Lmax) is often fine.
    # If you *must* have (3, B, L), you can store pos separately in converter.
    # We'll keep your original rearrange only if it looks like 3*L.
    # (Simple heuristic)
    if position_ids.shape[1] % 3 == 0:
        # interpret as (B, 3*L) -> (3, B, L)
        L = position_ids.shape[1] // 3
        position_ids = position_ids.view(len(batch), 3, L).permute(1, 0, 2).contiguous()

    image_mask = _left_pad_bool([b["visual_pos_masks"] for b in batch])
    hint_mask = _left_pad_bool([b["hint_mask"] for b in batch])

    data: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "visual_pos_masks": image_mask,
        "hint_mask": hint_mask,
    }

    # input_embeds: (B, Nmax, D)
    vis_list = [b.get("input_embeds") for b in batch]
    vis_pad, _ = _pad_2d_by_rows(vis_list, pad_val=0.0)
    data["input_embeds"] = vis_pad

    # deepstack: pad then flatten like your original: (B, Tmax, Nmax, D) -> (Tmax, B*Nmax, D)
    deep_list = [b.get("deepstack_visual_embeds") for b in batch]
    deep_dim = int(deep_list[0].shape[-1]) if (torch.is_tensor(deep_list[0]) and deep_list[0].numel() > 0) else 2048
    deep_pad, deep_mask = _pad_deep_3d(deep_list, deep_dim=deep_dim, pad_val=0.0)
    deep_pad = einops.rearrange(deep_pad, "b t n d -> t (b n) d")
    data["deepstack_visual_embeds"] = deep_pad
    data["deep_mask"] = deep_mask

    return {"ids": ids, "prompts": prompts, "label": labels, "data": data}


# ============================================================
# 4) Dataloader wrapper (drop-in name)
# ============================================================

class ScienceQA_MemmapTrueDataloader:
    """
    Drop-in wrapper that returns the same attributes you used:
      self.loader, self.train_loader, self.valid_loader, self.test_loader
    """
    def __init__(
        self,
        config,
        *,
        split: str = "validation",
        memmap_split_name: Optional[str] = None,
        pad_token_id: int = 0,
        shuffle: bool = False,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
    ):
        cache_root = config.dataset.cache_root
        batch_size = int(config.training_params.batch_size)
        pad_token_id = int(getattr(getattr(config, "model", None), "pad_token_id", pad_token_id))

        if memmap_split_name is None:
            memmap_split_name = f"{split}_memmap"
        memmap_dir = os.path.join(cache_root, memmap_split_name)

        if not os.path.isdir(memmap_dir) or not os.path.isfile(os.path.join(memmap_dir, "index.npz")):
            raise FileNotFoundError(
                f"Memmap split not found at {memmap_dir}. "
                f"Run convert_split_to_memmap(cache_root, split) once to create it."
            )

        ds = ScienceQA_MemmapDataset(memmap_dir=memmap_dir)
        self.collate_fn = lambda batch: scienceqa_memmap_collate(batch, pad_token_id=pad_token_id)

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=bool(shuffle),
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            collate_fn=self.collate_fn,
            prefetch_factor=int(prefetch_factor) if int(num_workers) > 0 else None,
            persistent_workers=bool(persistent_workers) if int(num_workers) > 0 else False,
        )

        self.loader = loader
        self.train_loader = loader
        self.valid_loader = loader
        self.test_loader = loader


# ============================================================
# Example usage (manual)
# ============================================================
if __name__ == "__main__":
    # Example one-time conversion:
    # convert_split_to_memmap("/path/to/cache_root", "validation", deep_dim=2048)

    # Example dataset read test:
    # ds = ScienceQA_MemmapDataset("/path/to/cache_root/validation_memmap")
    # x = ds[0]
    # print({k: (tuple(v.shape), v.dtype) if torch.is_tensor(v) else type(v) for k, v in x.items()})
    pass
