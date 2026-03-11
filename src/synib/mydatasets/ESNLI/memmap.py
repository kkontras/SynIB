import os
import json
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Helpers you already had
# ============================================================

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


def _pad_deepstack_stack64(x: torch.Tensor, Kmax: int, padding_side: str) -> torch.Tensor:
    """
    Stack 64s across deepstack levels.

    Input:  x = (K, 64, 2048)
    Output: (Kmax*64, 2048)

    (This is what you requested: "stack the 64s")
    """
    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    target_rows = int(Kmax) * 64
    if target_rows == 0:
        dtype = x.dtype if torch.is_tensor(x) else torch.float16
        return torch.zeros((0, 2048), dtype=dtype)

    if (not torch.is_tensor(x)) or x.numel() == 0:
        return torch.zeros((target_rows, 2048), dtype=torch.float16)

    if x.dim() != 3 or tuple(x.shape[1:]) != (64, 2048):
        raise ValueError(f"deepstack_visual_embeds expected (K,64,2048), got {tuple(x.shape)}")

    K = int(x.shape[0])

    # pad/truncate in K first
    if K == Kmax:
        x2 = x
    elif K > Kmax:
        x2 = x[-Kmax:] if padding_side == "left" else x[:Kmax]
    else:
        pad_levels = torch.zeros((Kmax - K, 64, 2048), dtype=x.dtype)
        x2 = torch.cat([pad_levels, x], dim=0) if padding_side == "left" else torch.cat([x, pad_levels], dim=0)

    return x2.reshape(Kmax * 64, 2048).contiguous()


# ============================================================
# Part 1: Build memmap from your existing shard cache
# ============================================================

def _read_manifest(split_dir: str) -> List[Dict[str, Any]]:
    manifest_path = os.path.join(split_dir, "manifest.jsonl")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing manifest.jsonl at {manifest_path}")
    recs = [json.loads(l) for l in open(manifest_path, "r", encoding="utf-8")]
    if len(recs) == 0:
        raise RuntimeError(f"Empty manifest: {manifest_path}")
    return recs


def build_memmap_from_shards(
    *,
    cache_root: str,
    split: str,
    out_dir: Optional[str] = None,
    input_ids_dtype: str = "int32",
    embed_dtype: str = "float16",
) -> str:
    """
    Converts your existing torch shards (list-of-dicts) into fast memmaps.

    Returns:
      out_dir used.
    """
    split_dir = os.path.join(cache_root, split)
    recs = _read_manifest(split_dir)

    if out_dir is None:
        out_dir = os.path.join(cache_root, f"{split}_mm")
    os.makedirs(out_dir, exist_ok=True)

    # If already built, just return.
    meta_path = os.path.join(out_dir, "meta.json")
    done_flag = os.path.join(out_dir, ".DONE")
    if os.path.isfile(meta_path) and os.path.isfile(done_flag):
        return out_dir

    if input_ids_dtype not in ("int32", "int64"):
        input_ids_dtype = "int32"
    if embed_dtype not in ("float16", "float32"):
        embed_dtype = "float16"

    ids_np_dtype = np.int32 if input_ids_dtype == "int32" else np.int64
    emb_np_dtype = np.float16 if embed_dtype == "float16" else np.float32

    # ---------- Pass 1: count totals ----------
    N = 0
    total_tokens = 0
    total_deep_blocks = 0  # each block = one (64,2048) level

    for r in recs:
        shard_path = os.path.join(split_dir, r["shard"])
        items = torch.load(shard_path, map_location="cpu")
        for ex in items:
            inp = ex["input_ids"]
            if inp.dim() == 2 and inp.shape[0] == 1:
                L = int(inp.shape[1])
            else:
                L = int(inp.numel())
            total_tokens += L
            N += 1

            deep = ex.get("deepstack_visual_embeds", None)
            if torch.is_tensor(deep) and deep.dim() == 3 and deep.shape[1:] == (64, 2048):
                total_deep_blocks += int(deep.shape[0])

    if N <= 0 or total_tokens <= 0:
        raise RuntimeError("No items/tokens found to build memmap.")

    # ---------- index arrays ----------
    offsets = np.zeros((N,), dtype=np.int64)
    lengths = np.zeros((N,), dtype=np.int32)
    labels = np.zeros((N,), dtype=np.int64)

    deep_offsets = np.zeros((N,), dtype=np.int64)
    deep_k = np.zeros((N,), dtype=np.int32)

    # ---------- ids/prompts jsonl ----------
    ids_path = os.path.join(out_dir, "ids.jsonl")
    prompts_path = os.path.join(out_dir, "prompts.jsonl")
    ids_f = open(ids_path + ".tmp", "w", encoding="utf-8")
    prompts_f = open(prompts_path + ".tmp", "w", encoding="utf-8")

    # ---------- memmaps ----------
    input_ids_mm = np.memmap(os.path.join(out_dir, "input_ids.bin"), mode="w+", dtype=ids_np_dtype, shape=(total_tokens,))
    attn_mm = np.memmap(os.path.join(out_dir, "attention.bin"), mode="w+", dtype=np.uint8, shape=(total_tokens,))
    vmask_mm = np.memmap(os.path.join(out_dir, "visual_pos_masks.bin"), mode="w+", dtype=np.uint8, shape=(total_tokens,))
    pos_mm = np.memmap(os.path.join(out_dir, "position_ids.bin"), mode="w+", dtype=np.int32, shape=(3, total_tokens))
    emb_mm = np.memmap(os.path.join(out_dir, "input_embeds.bin"), mode="w+", dtype=emb_np_dtype, shape=(total_tokens, 2048))

    deep_mm = np.memmap(
        os.path.join(out_dir, "deep.bin"),
        mode="w+",
        dtype=np.float16,  # deepstack stored fp16 for bandwidth
        shape=(max(int(total_deep_blocks), 1), 64, 2048),
    )

    # ---------- Pass 2: fill ----------
    tok_cur = 0
    deep_cur = 0
    i = 0

    for r in recs:
        shard_path = os.path.join(split_dir, r["shard"])
        items = torch.load(shard_path, map_location="cpu")

        for ex in items:
            # ---- metadata ----
            ids_f.write(json.dumps(ex.get("id", i)) + "\n")
            prompts_f.write(json.dumps(ex.get("prompt", "")) + "\n")

            lab = ex.get("label", 0)
            if torch.is_tensor(lab):
                lab = int(lab.detach().cpu().reshape(-1)[0].item()) if lab.numel() else 0
            labels[i] = int(lab)

            # ---- tokens ----
            inp = ex["input_ids"]
            att = ex["attention_mask"]
            vm = ex["visual_pos_masks"]
            pos = ex["position_ids"]
            emb = ex["input_embeds"]

            # normalize shapes to match your canonical ones
            if inp.dim() == 2 and inp.shape[0] == 1:
                inp1 = inp[0]
            else:
                inp1 = inp.reshape(-1)

            if att.dim() == 2 and att.shape[0] == 1:
                att1 = att[0]
            else:
                att1 = att.reshape(-1)

            if vm.dim() == 2 and vm.shape[0] == 1:
                vm1 = vm[0]
            else:
                vm1 = vm.reshape(-1).to(torch.bool)

            # position_ids: (3,1,L) -> (3,L)
            if pos.dim() != 3 or tuple(pos.shape[:2]) != (3, 1):
                raise RuntimeError(f"Bad position_ids in cache item: shape={tuple(pos.shape)} expected (3,1,L)")
            pos3 = pos[:, 0, :]  # (3,L)

            # embeds: (1,L,2048) -> (L,2048)
            if emb.dim() == 3 and emb.shape[0] == 1:
                emb2 = emb[0]
            else:
                raise RuntimeError(f"Bad input_embeds in cache item: shape={tuple(emb.shape)} expected (1,L,2048)")

            L = int(inp1.numel())
            offsets[i] = tok_cur
            lengths[i] = L

            # write contiguous token slices
            input_ids_mm[tok_cur:tok_cur + L] = inp1.numpy().astype(ids_np_dtype, copy=False)
            attn_mm[tok_cur:tok_cur + L] = att1.numpy().astype(np.uint8, copy=False)
            vmask_mm[tok_cur:tok_cur + L] = vm1.numpy().astype(np.uint8, copy=False)
            pos_mm[:, tok_cur:tok_cur + L] = pos3.numpy().astype(np.int32, copy=False)
            emb_mm[tok_cur:tok_cur + L, :] = emb2.numpy().astype(emb_np_dtype, copy=False)

            tok_cur += L

            # ---- deepstack blocks ----
            deep = ex.get("deepstack_visual_embeds", None)
            if torch.is_tensor(deep) and deep.numel() > 0:
                if deep.dim() != 3 or deep.shape[1:] != (64, 2048):
                    raise RuntimeError(f"Bad deepstack_visual_embeds: shape={tuple(deep.shape)} expected (K,64,2048)")
                K = int(deep.shape[0])
                deep_offsets[i] = deep_cur
                deep_k[i] = K
                deep_mm[deep_cur:deep_cur + K, :, :] = deep.numpy().astype(np.float16, copy=False)
                deep_cur += K
            else:
                deep_offsets[i] = deep_cur
                deep_k[i] = 0

            i += 1

    # flush to disk
    input_ids_mm.flush()
    attn_mm.flush()
    vmask_mm.flush()
    pos_mm.flush()
    emb_mm.flush()
    deep_mm.flush()

    ids_f.close()
    prompts_f.close()
    os.replace(ids_path + ".tmp", ids_path)
    os.replace(prompts_path + ".tmp", prompts_path)

    np.save(os.path.join(out_dir, "offsets.npy"), offsets)
    np.save(os.path.join(out_dir, "lengths.npy"), lengths)
    np.save(os.path.join(out_dir, "labels.npy"), labels)
    np.save(os.path.join(out_dir, "deep_offsets.npy"), deep_offsets)
    np.save(os.path.join(out_dir, "deep_k.npy"), deep_k)

    meta = {
        "version": 1,
        "N": int(N),
        "total_tokens": int(total_tokens),
        "total_deep_blocks": int(total_deep_blocks),
        "input_ids_dtype": input_ids_dtype,
        "embed_dtype": embed_dtype,
        "paths": {
            "offsets": "offsets.npy",
            "lengths": "lengths.npy",
            "labels": "labels.npy",
            "ids": "ids.jsonl",
            "prompts": "prompts.jsonl",
            "input_ids": "input_ids.bin",
            "attention": "attention.bin",
            "visual_pos_masks": "visual_pos_masks.bin",
            "position_ids": "position_ids.bin",
            "input_embeds": "input_embeds.bin",
            "deep_offsets": "deep_offsets.npy",
            "deep_k": "deep_k.npy",
            "deep": "deep.bin",
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(done_flag, "w", encoding="utf-8") as f:
        f.write("ok\n")

    return out_dir


# ============================================================
# Part 2: Memmap Dataset (fast runtime)
# ============================================================

class ESNLI_MemmapDataset(Dataset):
    """
    If use_memmap=True:
      - reads from {cache_root}/{split}_mm/ memmaps (fast)
    Else:
      - falls back to your original shard torch.load (slow)
    """

    def __init__(
        self,
        cache_root: str,
        split: str,
        *,
        use_memmap: bool = True,
        memmap_dir: Optional[str] = None,
        # shard fallback args (only used when use_memmap=False)
        shard_index: int = 0,
        shard_path: Optional[str] = None,
        max_items: Optional[int] = None,
    ):
        super().__init__()
        self.cache_root = cache_root
        self.split = split
        self.use_memmap = bool(use_memmap)

        if self.use_memmap:
            if memmap_dir is None:
                memmap_dir = os.path.join(cache_root, f"{split}_mm")
            meta_path = os.path.join(memmap_dir, "meta.json")
            if not os.path.isfile(meta_path):
                raise FileNotFoundError(
                    f"Memmap not found at {memmap_dir}. "
                    f"Build it first (build_memmap_from_shards)."
                )
            self.memmap_dir = memmap_dir
            meta = json.load(open(meta_path, "r", encoding="utf-8"))

            # indices
            self.offsets = np.load(os.path.join(memmap_dir, meta["paths"]["offsets"]))
            self.lengths = np.load(os.path.join(memmap_dir, meta["paths"]["lengths"]))
            self.labels = np.load(os.path.join(memmap_dir, meta["paths"]["labels"]))
            self.deep_offsets = np.load(os.path.join(memmap_dir, meta["paths"]["deep_offsets"]))
            self.deep_k = np.load(os.path.join(memmap_dir, meta["paths"]["deep_k"]))

            # text metadata
            self.ids_path = os.path.join(memmap_dir, meta["paths"]["ids"])
            self.prompts_path = os.path.join(memmap_dir, meta["paths"]["prompts"])

            # data memmaps
            ids_dtype = np.int32 if meta["input_ids_dtype"] == "int32" else np.int64
            emb_dtype = np.float16 if meta["embed_dtype"] == "float16" else np.float32

            self.input_ids_mm = np.memmap(os.path.join(memmap_dir, meta["paths"]["input_ids"]), mode="r", dtype=ids_dtype)
            self.attn_mm = np.memmap(os.path.join(memmap_dir, meta["paths"]["attention"]), mode="r", dtype=np.uint8)
            self.vmask_mm = np.memmap(os.path.join(memmap_dir, meta["paths"]["visual_pos_masks"]), mode="r", dtype=np.uint8)

            # position_ids stored as (3,total_tokens)
            self.pos_mm = np.memmap(os.path.join(memmap_dir, meta["paths"]["position_ids"]), mode="r", dtype=np.int32)
            self.pos_mm = self.pos_mm.reshape(3, -1)

            # embeds stored as (total_tokens,2048)
            self.emb_mm = np.memmap(os.path.join(memmap_dir, meta["paths"]["input_embeds"]), mode="r", dtype=emb_dtype)
            self.emb_mm = self.emb_mm.reshape(-1, 2048)

            # deep stored as (total_deep_blocks,64,2048)
            self.deep_mm = np.memmap(os.path.join(memmap_dir, meta["paths"]["deep"]), mode="r", dtype=np.float16)
            # shape is known from file size; we can infer blocks count:
            blocks = self.deep_mm.size // (64 * 2048)
            self.deep_mm = self.deep_mm.reshape(blocks, 64, 2048)

            # (optional) cache ids/prompts in memory for speed
            # If you have huge N and don't want RAM usage, set to False and stream lines (slower).
            self._cache_text = True
            if self._cache_text:
                self._ids = [json.loads(l) for l in open(self.ids_path, "r", encoding="utf-8")]
                self._prompts = [json.loads(l) for l in open(self.prompts_path, "r", encoding="utf-8")]
            else:
                self._ids = None
                self._prompts = None

            if max_items is not None:
                self.N = min(int(max_items), int(self.lengths.shape[0]))
            else:
                self.N = int(self.lengths.shape[0])

        else:
            # ---------- your original shard-based behavior ----------
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
            self.N = len(self.items)

    def __len__(self) -> int:
        return self.N

    def _get_text(self, idx: int) -> (Any, str):
        if getattr(self, "_ids", None) is not None:
            return self._ids[idx], self._prompts[idx]
        # slow fallback: stream line idx (not implemented here)
        return idx, ""

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not self.use_memmap:
            ex = self.items[idx]
            # keep your shard behavior identical:
            lab = ex.get("label", 0)
            if torch.is_tensor(lab):
                lab = int(lab.detach().cpu().reshape(-1)[0].item()) if lab.numel() else 0
            return {
                "id": ex.get("id", idx),
                "prompt": ex.get("prompt", ""),
                "label": torch.tensor(int(lab), dtype=torch.long),
                "input_ids": ex["input_ids"].to(torch.long),
                "attention_mask": ex["attention_mask"].to(torch.long),
                "position_ids": ex["position_ids"].to(torch.long),
                "input_embeds": ex["input_embeds"].to(torch.float16),
                "visual_pos_masks": ex["visual_pos_masks"].to(torch.bool),
                "deepstack_visual_embeds": ex["deepstack_visual_embeds"].to(torch.float16),
            }

        # -------- memmap path (fast) --------
        start = int(self.offsets[idx])
        L = int(self.lengths[idx])
        end = start + L

        ex_id, prompt = self._get_text(idx)
        lab = int(self.labels[idx])

        # (1,L)
        input_ids = torch.from_numpy(np.array(self.input_ids_mm[start:end], copy=False)).to(torch.long).unsqueeze(0)
        attention_mask = torch.from_numpy(np.array(self.attn_mm[start:end], copy=False)).to(torch.long).unsqueeze(0)
        visual_pos_masks = torch.from_numpy(np.array(self.vmask_mm[start:end], copy=False)).to(torch.bool).unsqueeze(0)

        # position_ids: stored (3,total_tokens) -> (3,1,L)
        pos = torch.from_numpy(np.array(self.pos_mm[:, start:end], copy=False)).to(torch.long).unsqueeze(1)

        # embeds: stored (total_tokens,2048) -> (1,L,2048)
        emb = torch.from_numpy(np.array(self.emb_mm[start:end, :], copy=False)).to(torch.float16).unsqueeze(0)

        # deepstack: blocks (K,64,2048)
        d0 = int(self.deep_offsets[idx])
        K = int(self.deep_k[idx])
        if K > 0:
            deep = torch.from_numpy(np.array(self.deep_mm[d0:d0 + K, :, :], copy=False)).to(torch.float16)
        else:
            deep = torch.zeros((0, 64, 2048), dtype=torch.float16)

        return {
            "id": ex_id,
            "prompt": prompt,
            "label": torch.tensor(lab, dtype=torch.long),

            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": pos,
            "input_embeds": emb,
            "visual_pos_masks": visual_pos_masks,
            "deepstack_visual_embeds": deep,
        }


# ============================================================
# Part 3: Your collate (unchanged logic, now works for memmap too)
# ============================================================

def make_collate_from_cache(*, padding_side: str = "right", pad_token_id: int = 0):
    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        ids_ = [b["id"] for b in batch]
        prompts_ = [b.get("prompt", "") for b in batch]
        labels = torch.stack([b["label"] for b in batch], dim=0)

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
        visual_pos_masks = torch.stack(
            [_pad_1d(v.to(torch.bool), Lmax, pad_value=0, padding_side=padding_side).to(torch.bool) for v in vmask_list],
            dim=0,
        )

        # deepstack stacked-64: (B, Kmax*64, 2048)
        deepstack_visual_embeds = torch.stack(
            [_pad_deepstack_stack64(d, Kmax, padding_side=padding_side) for d in deep_list],
            dim=0,
        )

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


# ============================================================
# Part 4: Dataloader wrapper (your interface, memmap toggle)
# ============================================================

class ESNLI_MemmapDataloader:
    def __init__(
        self,
        config,
        *,
        split: str = "validation",
        max_items: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        padding_side: str = "right",
        pad_token_id: int = 0,
        print_first_batch_stats: bool = True,
        use_memmap: bool = True,
        memmap_dir: Optional[str] = None,
    ):
        cache_root = config.dataset.cache_root
        batch_size = int(config.training_params.batch_size)

        ds = ESNLI_MemmapDataset(
            cache_root=cache_root,
            split=split,
            use_memmap=use_memmap,
            memmap_dir=memmap_dir,
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
            persistent_workers=(int(num_workers) > 0),
        )

        self.train_loader = self.loader
        self.valid_loader = self.loader
        self.test_loader = self.loader

        if print_first_batch_stats:
            batch = next(iter(self.loader))
            ex = batch["data"]
            print("\n[cache-reader] first batch loaded (memmap padded from cache):")
            for k in ["input_ids", "attention_mask", "position_ids", "input_embeds", "visual_pos_masks", "deepstack_visual_embeds"]:
                print(_stats(k, ex[k]))


# ============================================================
# Optional CLI to build memmaps once
# ============================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="build memmap from existing shards")
    ap.add_argument("--cache_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    if args.build:
        out = build_memmap_from_shards(cache_root=args.cache_root, split=args.split, out_dir=args.out_dir)
        print(f"[memmap] built at: {out}")
