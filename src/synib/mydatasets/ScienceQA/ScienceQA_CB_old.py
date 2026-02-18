"""
ScienceQA memmap loader (low-RAM, worker-friendly)

This file replaces the "torch.load(list-of-dicts shard)" approach with a
memory-mapped cache that supports true random access without deserializing
entire shards per worker process.

What it does:
- On first run (or if memmap cache missing), it converts your existing cached
  shards (manifest.jsonl + .pt shards containing list[dict]) into a compact
  memmap cache per split.
- Then it loads examples on-demand via np.memmap (OS page cache shared across
  workers), avoiding CPU OOM from per-worker shard caching.

Outputs returned (same as your original):
  - input_ids
  - attention_mask
  - image_mask
  - hint_mask
  - label
  - id

Notes:
- This uses uint8 for masks/attention_mask in storage; it converts to torch.long/torch.bool in __getitem__/collate.
- The conversion is one-time; subsequent runs skip it.
- Safe-ish for multi-process launches: uses a simple lock file per split.

Run:
  python scienceqa_memmap_loader.py

Then integrate:
  from scienceqa_memmap_loader import ScienceQA_MemmapDataloader
"""

import os
import json
import time
import random
import multiprocessing
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# =========================
# Utilities
# =========================

def _atomic_write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _try_acquire_lock(lock_path: str, wait_seconds: int = 600) -> bool:
    """
    Create lock file exclusively.
    Returns True if lock acquired, False if timed out.
    """
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"pid={os.getpid()} time={time.time()}\n")
            return True
        except FileExistsError:
            if time.time() - start > wait_seconds:
                return False
            time.sleep(0.25)


def _release_lock(lock_path: str) -> None:
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# =========================
# Memmap cache format
# =========================
# Per split directory:  <cache_root>/<split>/
# Source (existing):
#   manifest.jsonl
#   shard_*.pt  (torch.save(list[dict]))
#
# New memmap cache written under:
#   <cache_root>/<split>/_memmap/
# with files:
#   meta.json
#   offsets.npy      int64 [N]
#   lengths.npy      int32 [N]
#   labels.npy       int64 [N]
#   ids.jsonl        lines of ids (string)
#   input_ids.bin    int32 or int64 (configurable; default int32)
#   attention.bin    uint8
#   image_mask.bin   uint8
#   hint_mask.bin    uint8
#
# Tensors are stored concatenated 1D, and offset/length index each example.
# This enables random access reads without loading all examples into RAM.


def build_memmap_from_token_shards(
    *,
    split_dir: str,
    out_dir: str,
    input_ids_dtype: str = "int32",
) -> None:
    """
    Convert existing torch shards in split_dir into memmap cache in out_dir.
    This is a one-time conversion.

    input_ids_dtype: "int32" (recommended) or "int64"
      - Use int32 if your vocab size fits (it does for standard tokenizers).
    """
    manifest_path = os.path.join(split_dir, "manifest.jsonl")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    _ensure_dir(out_dir)

    meta_path = os.path.join(out_dir, "meta.json")
    done_flag = os.path.join(out_dir, ".DONE")
    lock_path = os.path.join(out_dir, ".LOCK")

    # If already built, do nothing.
    if os.path.isfile(done_flag) and os.path.isfile(meta_path):
        return

    if not _try_acquire_lock(lock_path, wait_seconds=600):
        raise RuntimeError(f"Timed out waiting for lock: {lock_path}")

    try:
        # Re-check after acquiring lock (another process may have built it).
        if os.path.isfile(done_flag) and os.path.isfile(meta_path):
            return

        # -------- Pass 1: compute N and total token count --------
        shard_recs: List[Dict[str, Any]] = []
        N = 0
        total_tokens = 0

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                shard_path = os.path.join(split_dir, rec["shard"])
                num_items = int(rec["num_items"])
                shard_recs.append({"path": shard_path, "num_items": num_items})
                N += num_items

        if N == 0:
            raise RuntimeError(f"No items found in manifest: {manifest_path}")

        # We need total_tokens, which requires peeking at lengths.
        # We do this by loading each shard once (still okay during conversion).
        for r in shard_recs:
            items = torch.load(r["path"], map_location="cpu")
            if len(items) != r["num_items"]:
                # Trust manifest but warn via raise to be safe
                raise RuntimeError(
                    f"Shard length mismatch: {r['path']} manifest={r['num_items']} actual={len(items)}"
                )
            for ex in items:
                input_ids = ex["input_ids"]
                total_tokens += int(input_ids.numel())

        # -------- Allocate index arrays --------
        offsets = np.zeros((N,), dtype=np.int64)
        lengths = np.zeros((N,), dtype=np.int32)
        labels = np.zeros((N,), dtype=np.int64)

        ids_path = os.path.join(out_dir, "ids.jsonl")
        # We'll write ids as plain JSON per line to preserve string ids safely.
        ids_f = open(ids_path + ".tmp", "w", encoding="utf-8")

        # -------- Allocate memmaps (contiguous storage) --------
        if input_ids_dtype not in ("int32", "int64"):
            raise ValueError("input_ids_dtype must be 'int32' or 'int64'")

        input_ids_np_dtype = np.int32 if input_ids_dtype == "int32" else np.int64

        input_ids_bin = os.path.join(out_dir, "input_ids.bin")
        attn_bin = os.path.join(out_dir, "attention.bin")
        image_bin = os.path.join(out_dir, "image_mask.bin")
        hint_bin = os.path.join(out_dir, "hint_mask.bin")

        input_ids_mm = np.memmap(input_ids_bin, mode="w+", dtype=input_ids_np_dtype, shape=(total_tokens,))
        attn_mm = np.memmap(attn_bin, mode="w+", dtype=np.uint8, shape=(total_tokens,))
        image_mm = np.memmap(image_bin, mode="w+", dtype=np.uint8, shape=(total_tokens,))
        hint_mm = np.memmap(hint_bin, mode="w+", dtype=np.uint8, shape=(total_tokens,))

        # -------- Pass 2: fill everything --------
        cursor = 0
        i = 0

        for r in shard_recs:
            items = torch.load(r["path"], map_location="cpu")
            for ex in items:
                masks = ex.get("masks", None)
                if masks is None or ("image" not in masks) or ("hint" not in masks):
                    raise KeyError(
                        f"Example missing masks['image'] or masks['hint'] in shard {r['path']}. "
                        f"Rebuild your original token cache with masks enabled."
                    )

                inp: torch.Tensor = ex["input_ids"]
                att: torch.Tensor = ex["attention_mask"]
                img: torch.Tensor = masks["image"]
                hnt: torch.Tensor = masks["hint"]

                L = int(inp.numel())
                offsets[i] = cursor
                lengths[i] = L

                # label should be scalar-ish
                lab = ex["label"]
                if isinstance(lab, torch.Tensor):
                    lab = int(lab.item())
                labels[i] = int(lab)

                # id can be anything JSON-serializable; preserve exact value
                ids_f.write(json.dumps(ex["id"]) + "\n")

                # Copy into memmaps
                # input_ids
                input_ids_mm[cursor:cursor + L] = inp.detach().cpu().numpy().astype(input_ids_np_dtype, copy=False)

                # attention_mask -> uint8 0/1
                attn_mm[cursor:cursor + L] = att.detach().cpu().numpy().astype(np.uint8, copy=False)

                # masks -> uint8 0/1
                image_mm[cursor:cursor + L] = img.detach().cpu().numpy().astype(np.uint8, copy=False)
                hint_mm[cursor:cursor + L] = hnt.detach().cpu().numpy().astype(np.uint8, copy=False)

                cursor += L
                i += 1

        if i != N or cursor != total_tokens:
            raise RuntimeError(f"Conversion mismatch: wrote i={i}/{N}, cursor={cursor}/{total_tokens}")

        # Flush memmaps
        input_ids_mm.flush()
        attn_mm.flush()
        image_mm.flush()
        hint_mm.flush()
        ids_f.close()

        # Save index arrays
        np.save(os.path.join(out_dir, "offsets.npy"), offsets)
        np.save(os.path.join(out_dir, "lengths.npy"), lengths)
        np.save(os.path.join(out_dir, "labels.npy"), labels)

        # Finalize ids
        os.replace(ids_path + ".tmp", ids_path)

        # Write meta
        meta = {
            "version": 1,
            "N": int(N),
            "total_tokens": int(total_tokens),
            "input_ids_dtype": input_ids_dtype,
            "paths": {
                "offsets": "offsets.npy",
                "lengths": "lengths.npy",
                "labels": "labels.npy",
                "ids": "ids.jsonl",
                "input_ids": "input_ids.bin",
                "attention": "attention.bin",
                "image_mask": "image_mask.bin",
                "hint_mask": "hint_mask.bin",
            },
        }
        _atomic_write_json(meta_path, meta)

        # Done flag
        with open(done_flag, "w", encoding="utf-8") as f:
            f.write("ok\n")

        print(f"[ScienceQA Memmap] Built: {out_dir}  N={N}  total_tokens={total_tokens}  dtype={input_ids_dtype}")

    finally:
        _release_lock(lock_path)


# =========================
# Dataset (memmap)
# =========================

class ScienceQA_MemmapDataset(Dataset):
    """
    Uses memmap cache built by build_memmap_from_token_shards().
    Returns:
      - id (python object loaded from ids.jsonl line)
      - label (torch.long scalar tensor)
      - input_ids (torch tensor 1D)
      - attention_mask (torch tensor 1D)
      - image_mask (torch bool tensor 1D)
      - hint_mask (torch bool tensor 1D)
    """

    def __init__(self, cache_root: str, split: str):
        super().__init__()
        self.split_dir = os.path.join(cache_root, split)
        assert os.path.isdir(self.split_dir), f"Missing split dir: {self.split_dir}"

        self.mem_dir = os.path.join(self.split_dir, "_memmap")
        meta_path = os.path.join(self.mem_dir, "meta.json")
        done_flag = os.path.join(self.mem_dir, ".DONE")

        if not (os.path.isfile(meta_path) and os.path.isfile(done_flag)):
            raise RuntimeError(
                f"Memmap cache missing for split={split}. Expected: {self.mem_dir}\n"
                f"Run conversion first (see __main__) or call build_memmap_from_token_shards()."
            )

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.N = int(meta["N"])
        self.total_tokens = int(meta["total_tokens"])
        self.input_ids_dtype = meta["input_ids_dtype"]

        # Load index arrays in RAM (small)
        self.offsets = np.load(os.path.join(self.mem_dir, "offsets.npy"), mmap_mode=None)  # int64 [N]
        self.lengths = np.load(os.path.join(self.mem_dir, "lengths.npy"), mmap_mode=None)  # int32 [N]
        self.labels = np.load(os.path.join(self.mem_dir, "labels.npy"), mmap_mode=None)    # int64 [N]

        # ids: keep file handle open; random access by seeking is annoying
        # simplest stable approach: read all ids into RAM (usually small compared to tokens).
        # If ids are huge, we can optimize later.
        ids_path = os.path.join(self.mem_dir, "ids.jsonl")
        self.ids: List[Any] = []
        with open(ids_path, "r", encoding="utf-8") as f:
            for line in f:
                self.ids.append(json.loads(line))
        if len(self.ids) != self.N:
            raise RuntimeError(f"ids.jsonl length mismatch: {len(self.ids)} vs N={self.N}")

        # Memmaps for token fields (shared OS cache, low RSS)
        input_dtype = np.int32 if self.input_ids_dtype == "int32" else np.int64

        self.input_ids_mm = np.memmap(
            os.path.join(self.mem_dir, "input_ids.bin"),
            mode="r",
            dtype=input_dtype,
            shape=(self.total_tokens,)
        )
        self.attn_mm = np.memmap(
            os.path.join(self.mem_dir, "attention.bin"),
            mode="r",
            dtype=np.uint8,
            shape=(self.total_tokens,)
        )
        self.image_mm = np.memmap(
            os.path.join(self.mem_dir, "image_mask.bin"),
            mode="r",
            dtype=np.uint8,
            shape=(self.total_tokens,)
        )
        self.hint_mm = np.memmap(
            os.path.join(self.mem_dir, "hint_mask.bin"),
            mode="r",
            dtype=np.uint8,
            shape=(self.total_tokens,)
        )

        print(f"[ScienceQA MemmapDataset] split={split} N={self.N} mem_dir={self.mem_dir}")

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        off = int(self.offsets[idx])
        L = int(self.lengths[idx])

        # Create torch tensors from numpy slices.
        # .copy() avoids keeping a view into the memmap (sometimes helps with worker safety / lifetime).
        # If you want max speed, you can remove .copy(), but keep it for fewer surprises.
        input_ids = torch.from_numpy(np.array(self.input_ids_mm[off:off + L], copy=True))
        attention_mask = torch.from_numpy(np.array(self.attn_mm[off:off + L], copy=True)).to(torch.long)

        image_mask = torch.from_numpy(np.array(self.image_mm[off:off + L], copy=True)).to(torch.bool)
        hint_mask = torch.from_numpy(np.array(self.hint_mm[off:off + L], copy=True)).to(torch.bool)

        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        return {
            "id": self.ids[idx],
            "label": label,
            "input_ids": input_ids.to(torch.long),
            "attention_mask": attention_mask,
            "image_mask": image_mask,
            "hint_mask": hint_mask,
        }


# =========================
# Collate (LEFT pad)
# =========================

def _left_pad_1d(seqs: List[torch.Tensor], pad_val: int, dtype: torch.dtype) -> torch.Tensor:
    max_len = max(int(s.numel()) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_val, dtype=dtype)
    for i, s in enumerate(seqs):
        L = int(s.numel())
        out[i, -L:] = s.to(dtype)
    return out


def _left_pad_bool(seqs: List[torch.Tensor]) -> torch.Tensor:
    max_len = max(int(s.numel()) for s in seqs)
    out = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        L = int(s.numel())
        out[i, -L:] = s.bool()
    return out


def scienceqa_memmap_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    ids = [b["id"] for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)

    input_ids = _left_pad_1d([b["input_ids"] for b in batch], pad_val=0, dtype=torch.long)
    attention_mask = _left_pad_1d([b["attention_mask"] for b in batch], pad_val=0, dtype=torch.long)

    image_mask = _left_pad_bool([b["image_mask"] for b in batch])
    hint_mask = _left_pad_bool([b["hint_mask"] for b in batch])

    return {
        "ids": ids,
        "label": labels,
        "data": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_mask": image_mask,
            "hint_mask": hint_mask,
        },
    }


# =========================
# Dataloader wrapper
# =========================

class ScienceQA_MemmapDataloader:
    """
    Builds memmap caches if needed (one-time), then exposes train/valid/test loaders.
    """

    def __init__(
        self,
        config,
        *,
        input_ids_dtype: str = "int32",
        num_workers: Optional[int] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        pin_memory: bool = True,
    ):
        cache_root = config.dataset.cache_root
        batch_size = int(config.training_params.batch_size)

        g = torch.Generator()
        g.manual_seed(0)

        def seed_worker(worker_id: int):
            worker_seed = torch.initial_seed() % (2**32)
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Reasonable default: do not explode workers.
        # If user doesn't specify, pick something conservative.
        if num_workers is None:
            total_cpus = multiprocessing.cpu_count()
            # Conservative: 4 or (cpus//8), whichever larger, capped
            num_workers = max(2, min(8, total_cpus // 8 if total_cpus >= 16 else 4))

        print(
            f"[ScienceQA MemmapLoader] CPUs={multiprocessing.cpu_count()} | num_workers={num_workers} | "
            f"prefetch_factor={prefetch_factor} | persistent_workers={persistent_workers} | pin_memory={pin_memory}"
        )

        # Ensure memmap exists BEFORE creating DataLoader workers
        for split in ("train", "validation", "test"):
            split_dir = os.path.join(cache_root, split)
            mem_dir = os.path.join(split_dir, "_memmap")
            meta_path = os.path.join(mem_dir, "meta.json")
            done_flag = os.path.join(mem_dir, ".DONE")
            if not (os.path.isfile(meta_path) and os.path.isfile(done_flag)):
                print(f"[ScienceQA MemmapLoader] Building memmap cache for split={split} ...")
                build_memmap_from_token_shards(
                    split_dir=split_dir,
                    out_dir=mem_dir,
                    input_ids_dtype=input_ids_dtype,
                )

        self.collate_fn = scienceqa_memmap_collate

        def make_loader(split: str, shuffle: bool):
            ds = ScienceQA_MemmapDataset(cache_root=cache_root, split=split)
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                generator=g if shuffle else None,
                worker_init_fn=seed_worker,
                collate_fn=self.collate_fn,
                num_workers=int(num_workers),
                pin_memory=bool(pin_memory),
                prefetch_factor=int(prefetch_factor) if int(num_workers) > 0 else None,
                persistent_workers=bool(persistent_workers) if int(num_workers) > 0 else False,
            )

        self.train_loader = make_loader("train", shuffle=True)
        self.valid_loader = make_loader("validation", shuffle=False)
        self.test_loader = make_loader("test", shuffle=False)


# =========================
# Main: pre-process + smoke test
# =========================

if __name__ == "__main__":
    import types

    # ---- Config stub (match your structure) ----
    config = types.SimpleNamespace()
    config.training_params = types.SimpleNamespace()
    config.dataset = types.SimpleNamespace()

    config.training_params.batch_size = 8

    # Point this to your existing token cache root that contains:
    #   <cache_root>/train/manifest.jsonl + shards
    #   <cache_root>/validation/manifest.jsonl + shards
    #   <cache_root>/test/manifest.jsonl + shards
    config.dataset.cache_root = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ScienceQA/cache_tokens2B"  # <-- set this

    # ---- Build + load ----
    loader = ScienceQA_MemmapDataloader(
        config,
        input_ids_dtype="int32",       # usually safe and smaller than int64
        num_workers=4,                 # conservative default; increase if you want
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )

    # ---- Smoke test ----
    batch = next(iter(loader.train_loader))
    print("ids[0:3]:", batch["ids"][:3])
    print("label:", batch["label"].shape, batch["label"].dtype)
    print("input_ids:", batch["data"]["input_ids"].shape, batch["data"]["input_ids"].dtype)
    print("attention_mask:", batch["data"]["attention_mask"].shape, batch["data"]["attention_mask"].dtype)
    print("image_mask:", batch["data"]["image_mask"].shape, batch["data"]["image_mask"].dtype)
    print("hint_mask:", batch["data"]["hint_mask"].shape, batch["data"]["hint_mask"].dtype)
