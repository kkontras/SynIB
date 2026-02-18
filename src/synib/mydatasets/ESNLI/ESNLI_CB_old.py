"""
ESNLI-VE Qwen3-VL codebook memmap loader (low-RAM, worker-friendly) + token masks

Converts existing cached shards:
  <cache_root>/<split>/manifest.jsonl
  <cache_root>/<split>/shard_*.pt    (torch.save(list[dict]))

Shards must contain per-example keys:
  - input_ids (1D tensor)
  - attention_mask (1D tensor)
  - label
  - id
  - masks: {"image": (1D uint8/bool), "text": (1D uint8/bool)}   <-- REQUIRED for this version

Optional:
  - image_grid_thw
  - vision_embeds
  - pixel_values (if you enabled it in your cache builder)

Memmap cache:
  <cache_root>/<split>/_memmap/
    meta.json
    offsets.npy         int64 [N]
    lengths.npy         int32 [N]
    labels.npy          int64 [N]
    ids.jsonl           N lines of JSON id
    input_ids.bin       int32/int64 concatenated tokens
    attention.bin       uint8 concatenated 0/1
    image_mask.bin      uint8 concatenated 0/1    <-- NEW
    text_mask.bin       uint8 concatenated 0/1    <-- NEW
    image_grid_thw.npy  int32 [N, 3]   (if present)
    vision_offsets.npy  int64 [N]      (if vision_embeds present)
    vision_lengths.npy  int32 [N]      (#image tokens per sample, if 2D)
    vision_dim.npy      int32 scalar   (D)
    vision.bin          float16/float32 concatenated (sum_i N_i * D)
    pixel_values.bin    float16 memmap [N, C, H, W] (optional, if enabled & present)

Returned per item:
  - id
  - label
  - input_ids
  - attention_mask
  - image_mask   (bool)
  - text_mask    (bool)
  - image_grid_thw (optional)
  - vision_embeds + vision_mask (optional)
  - pixel_values (optional)

Usage:
  from esnli_memmap_loader_masks import ESNLI_MemmapDataloader
"""

import os
import json
import time
import random
import multiprocessing
from typing import Dict, Any, List, Optional, Tuple

import einops
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
# Memmap builder
# =========================

def build_memmap_from_token_shards(
    *,
    split_dir: str,
    out_dir: str,
    input_ids_dtype: str = "int32",
    vision_dtype: str = "float16",
    store_pixel_values: bool = False,
) -> None:
    manifest_path = os.path.join(split_dir, "manifest.jsonl")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    _ensure_dir(out_dir)

    meta_path = os.path.join(out_dir, "meta.json")
    done_flag = os.path.join(out_dir, ".DONE")
    lock_path = os.path.join(out_dir, ".LOCK")

    # Already built
    if os.path.isfile(done_flag) and os.path.isfile(meta_path):
        return

    if not _try_acquire_lock(lock_path, wait_seconds=600):
        raise RuntimeError(f"Timed out waiting for lock: {lock_path}")

    try:
        if os.path.isfile(done_flag) and os.path.isfile(meta_path):
            return

        # -------- Read manifest --------
        shard_recs: List[Dict[str, Any]] = []
        N = 0
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                shard_path = os.path.join(split_dir, rec["shard"])
                n_items = int(rec.get("n", rec.get("num_items", 0)))
                shard_recs.append({"path": shard_path, "n": n_items})
                N += n_items

        if N <= 0:
            raise RuntimeError(f"No items found in manifest: {manifest_path}")

        # -------- Pass 1: compute totals + discover shapes --------
        total_tokens = 0
        has_grid = False
        has_vision = False
        has_pixel = False

        vision_dim: Optional[int] = None
        total_vision_elems = 0

        pixel_shape: Optional[Tuple[int, int, int]] = None  # (C,H,W)

        for r in shard_recs:
            items = torch.load(r["path"], map_location="cpu")
            if len(items) != r["n"]:
                raise RuntimeError(f"Shard length mismatch: {r['path']} manifest={r['n']} actual={len(items)}")

            for ex in items:
                inp: torch.Tensor = ex["input_ids"]
                total_tokens += int(inp.numel())

                # masks required
                masks = ex.get("masks", None)
                if masks is None or ("image" not in masks) or ("text" not in masks):
                    raise KeyError(
                        f"Example missing masks['image'] or masks['text'] in shard {r['path']}.\n"
                        f"Rebuild your cache shards to include token masks."
                    )

                has_grid = True

                ve = ex["vision_embeds"]
                if not torch.is_tensor(ve):
                    raise TypeError(f"vision_embeds must be torch.Tensor, got {type(ve)}")
                has_vision = True
                if ve.dim() == 1:
                    D = int(ve.numel()); nimg = 1
                elif ve.dim() == 2:
                    nimg, D = int(ve.shape[0]), int(ve.shape[1])
                else:
                    raise ValueError(f"vision_embeds must be 1D or 2D, got shape={tuple(ve.shape)}")

                if vision_dim is None:
                    vision_dim = D
                elif vision_dim != D:
                    raise ValueError(f"vision_dim mismatch: saw {vision_dim} then {D}")

                total_vision_elems += int(nimg * D)

                # pv = ex["pixel_values"]
                # pv = einops.rearrange(ex["pixel_values"], "h (w c) -> c h w",c=3)
                # if not torch.is_tensor(pv):
                #     raise TypeError(f"pixel_values must be torch.Tensor, got {type(pv)}")
                # if pv.dim() != 2:
                #     raise ValueError(f"pixel_values must be [C,H,W], got shape={tuple(pv.shape)}")
                # has_pixel = True
                # print(pv.shape)
                # shp = (int(pv.shape[0]), int(pv.shape[1]), int(pv.shape[2]))
                # if pixel_shape is None:
                #     pixel_shape = shp
                # elif pixel_shape != shp:
                #     raise ValueError(f"pixel_values shape mismatch: saw {pixel_shape} then {shp}")

        if input_ids_dtype not in ("int32", "int64"):
            raise ValueError("input_ids_dtype must be 'int32' or 'int64'")
        if vision_dtype not in ("float16", "float32"):
            raise ValueError("vision_dtype must be 'float16' or 'float32'")

        input_np_dtype = np.int32 if input_ids_dtype == "int32" else np.int64
        vision_np_dtype = np.float16 if vision_dtype == "float16" else np.float32

        # -------- Allocate index arrays --------
        offsets = np.zeros((N,), dtype=np.int64)
        lengths = np.zeros((N,), dtype=np.int32)
        labels = np.zeros((N,), dtype=np.int64)

        grid_thw = np.zeros((N, 3), dtype=np.int32) if has_grid else None

        if vision_dim is None:
            raise RuntimeError("has_vision True but vision_dim None")
        vision_offsets = np.zeros((N,), dtype=np.int64)
        vision_lengths = np.zeros((N,), dtype=np.int32)


        # ids file
        ids_path = os.path.join(out_dir, "ids.jsonl")
        ids_f = open(ids_path + ".tmp", "w", encoding="utf-8")

        # -------- Allocate memmaps --------
        input_ids_bin = os.path.join(out_dir, "input_ids.bin")
        attn_bin = os.path.join(out_dir, "attention.bin")
        image_mask_bin = os.path.join(out_dir, "image_mask.bin")
        text_mask_bin = os.path.join(out_dir, "text_mask.bin")

        input_ids_mm = np.memmap(input_ids_bin, mode="w+", dtype=input_np_dtype, shape=(total_tokens,))
        attn_mm = np.memmap(attn_bin, mode="w+", dtype=np.uint8, shape=(total_tokens,))
        image_mask_mm = np.memmap(image_mask_bin, mode="w+", dtype=np.uint8, shape=(total_tokens,))
        text_mask_mm = np.memmap(text_mask_bin, mode="w+", dtype=np.uint8, shape=(total_tokens,))

        vision_bin = os.path.join(out_dir, "vision.bin")
        vision_mm = np.memmap(vision_bin, mode="w+", dtype=vision_np_dtype, shape=(total_vision_elems,))


        # C, H, W = pixel_shape
        # pixel_bin = os.path.join(out_dir, "pixel_values.bin")
        # pixel_mm = np.memmap(pixel_bin, mode="w+", dtype=np.float16, shape=(N, C, H, W))


        # -------- Pass 2: fill --------
        token_cursor = 0
        vision_cursor = 0
        i = 0

        for r in shard_recs:
            items = torch.load(r["path"], map_location="cpu")
            for ex in items:
                inp: torch.Tensor = ex["input_ids"]
                att: torch.Tensor = ex["attention_mask"]
                masks = ex["masks"]
                imgm: torch.Tensor = masks["image"]
                txtm: torch.Tensor = masks["text"]

                L = int(inp.numel())
                if int(att.numel()) != L or int(imgm.numel()) != L or int(txtm.numel()) != L:
                    raise RuntimeError(
                        f"Length mismatch at global idx={i} in {r['path']}: "
                        f"input_ids={L}, att={att.numel()}, image_mask={imgm.numel()}, text_mask={txtm.numel()}"
                    )

                offsets[i] = token_cursor
                lengths[i] = L

                lab = ex["label"]
                if isinstance(lab, torch.Tensor):
                    lab = int(lab.item())
                labels[i] = int(lab)

                ids_f.write(json.dumps(ex["id"]) + "\n")

                input_ids_mm[token_cursor:token_cursor + L] = inp.detach().cpu().numpy().astype(input_np_dtype, copy=False)
                attn_mm[token_cursor:token_cursor + L] = att.detach().cpu().numpy().astype(np.uint8, copy=False)

                image_mask_mm[token_cursor:token_cursor + L] = imgm.detach().cpu().numpy().astype(np.uint8, copy=False)
                text_mask_mm[token_cursor:token_cursor + L] = txtm.detach().cpu().numpy().astype(np.uint8, copy=False)

                if has_grid and grid_thw is not None:
                    g = ex.get("image_grid_thw", None)
                    if g is None:
                        grid_thw[i, :] = 0
                    else:
                        if isinstance(g, torch.Tensor):
                            g = g.detach().cpu().view(-1).tolist()
                        g = list(g)
                        if len(g) != 3:
                            raise ValueError(f"image_grid_thw must have len=3, got {g}")
                        grid_thw[i, :] = np.array(g, dtype=np.int32)

                ve = ex.get("vision_embeds", None)

                if ve is None:
                    vision_offsets[i] = vision_cursor
                    vision_lengths[i] = 0
                else:
                    if not torch.is_tensor(ve):
                        raise TypeError(f"vision_embeds must be torch.Tensor, got {type(ve)}")
                    if ve.dim() == 1:
                        ve = ve.view(1, -1)
                    if ve.dim() != 2:
                        raise ValueError(f"vision_embeds must be 2D after reshape, got {tuple(ve.shape)}")

                    nimg, D = int(ve.shape[0]), int(ve.shape[1])
                    if vision_dim != D:
                        raise ValueError(f"vision_dim mismatch at item {i}: expected {vision_dim}, got {D}")

                    vision_offsets[i] = vision_cursor
                    vision_lengths[i] = nimg

                    flat = ve.detach().cpu().numpy().astype(vision_np_dtype, copy=False).reshape(-1)
                    n_elems = int(flat.size)
                    vision_mm[vision_cursor:vision_cursor + n_elems] = flat
                    vision_cursor += n_elems

                # if pixel_mm is not None:
                #     pv = ex.get("pixel_values", None)
                #     if pv is None:
                #         pixel_mm[i, ...] = 0
                #     else:
                #         arr = pv.detach().cpu().numpy().astype(np.float16, copy=False)
                #         pixel_mm[i, ...] = arr

                token_cursor += L
                i += 1

        if i != N or token_cursor != total_tokens:
            raise RuntimeError(f"Conversion mismatch: wrote i={i}/{N}, token_cursor={token_cursor}/{total_tokens}")
        if has_vision and vision_cursor != total_vision_elems:
            raise RuntimeError(f"Vision mismatch: vision_cursor={vision_cursor} total={total_vision_elems}")

        # Flush
        input_ids_mm.flush()
        attn_mm.flush()
        image_mask_mm.flush()
        text_mask_mm.flush()
        if vision_mm is not None:
            vision_mm.flush()
        # if pixel_mm is not None:
        #     pixel_mm.flush()

        ids_f.close()
        os.replace(ids_path + ".tmp", ids_path)

        # Save index arrays
        np.save(os.path.join(out_dir, "offsets.npy"), offsets)
        np.save(os.path.join(out_dir, "lengths.npy"), lengths)
        np.save(os.path.join(out_dir, "labels.npy"), labels)

        if has_grid and grid_thw is not None:
            np.save(os.path.join(out_dir, "image_grid_thw.npy"), grid_thw)

        np.save(os.path.join(out_dir, "vision_offsets.npy"), vision_offsets)
        np.save(os.path.join(out_dir, "vision_lengths.npy"), vision_lengths)
        np.save(os.path.join(out_dir, "vision_dim.npy"), np.array([int(vision_dim)], dtype=np.int32))

        meta = {
            "version": 2,
            "N": int(N),
            "total_tokens": int(total_tokens),
            "input_ids_dtype": input_ids_dtype,
            "has_image_grid_thw": bool(has_grid),
            "has_vision_embeds": bool(has_vision),
            "vision_dtype": vision_dtype if has_vision else None,
            "vision_dim": int(vision_dim) if vision_dim is not None else None,
            "store_pixel_values": bool(store_pixel_values and has_pixel),
            "pixel_shape": list(pixel_shape) if (
                        store_pixel_values and has_pixel and pixel_shape is not None) else None,
            "has_token_masks": True,
            "paths": {
                "offsets": "offsets.npy",
                "lengths": "lengths.npy",
                "labels": "labels.npy",
                "ids": "ids.jsonl",
                "input_ids": "input_ids.bin",
                "attention": "attention.bin",
                "image_mask": "image_mask.bin",
                "text_mask": "text_mask.bin",
                "image_grid_thw": "image_grid_thw.npy" if has_grid else None,
                "vision_offsets": "vision_offsets.npy" if has_vision else None,
                "vision_lengths": "vision_lengths.npy" if has_vision else None,
                "vision_dim": "vision_dim.npy" if has_vision else None,
                "vision": "vision.bin" if has_vision else None,
                "pixel_values": "pixel_values.bin" if (store_pixel_values and has_pixel) else None,
            },
        }
        _atomic_write_json(os.path.join(out_dir, "meta.json"), meta)

        with open(done_flag, "w", encoding="utf-8") as f:
            f.write("ok\n")

        print(
            f"[ESNLI Memmap] Built: {out_dir} | N={N} | total_tokens={total_tokens} | "
            f"masks=True | vision={has_vision} | pixel_values={bool(store_pixel_values and has_pixel)}"
        )

    finally:
        _release_lock(lock_path)


# =========================
# Dataset (memmap)
# =========================

class ESNLI_MemmapDataset(Dataset):
    """
    Loads ESNLI codebook examples from memmap.
    Returns per item:
      - id
      - label
      - input_ids (1D long)
      - attention_mask (1D long)
      - image_mask (bool 1D)
      - text_mask (bool 1D)
      - image_grid_thw (long[3]) if present
      - vision_embeds (float tensor [Nimg, D]) + vision_len (long) if present
      - pixel_values (float16 [C,H,W]) if stored
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
                f"Build it first (call build_memmap_from_token_shards())."
            )

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.N = int(meta["N"])
        self.total_tokens = int(meta["total_tokens"])
        self.input_ids_dtype = meta["input_ids_dtype"]
        self.has_grid = bool(meta.get("has_image_grid_thw", True))
        self.has_vision = bool(meta.get("has_vision_embeds", True))
        self.vision_dtype = meta.get("vision_dtype", None)
        self.vision_dim = meta.get("vision_dim", None)
        self.store_pixel_values = bool(meta.get("store_pixel_values", False))
        self.pixel_shape = meta.get("pixel_shape", None)
        self.has_token_masks = bool(meta.get("has_token_masks", False))

        if not self.has_token_masks:
            raise RuntimeError(
                f"This loader expects token masks (image/text) in memmap, but meta says has_token_masks=False.\n"
                f"Delete {self.mem_dir} and rebuild with the masks-enabled converter."
            )

        # small index arrays in RAM
        self.offsets = np.load(os.path.join(self.mem_dir, "offsets.npy"))
        self.lengths = np.load(os.path.join(self.mem_dir, "lengths.npy"))
        self.labels = np.load(os.path.join(self.mem_dir, "labels.npy"))

        # ids in RAM
        self.ids: List[Any] = []
        with open(os.path.join(self.mem_dir, "ids.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                self.ids.append(json.loads(line))
        if len(self.ids) != self.N:
            raise RuntimeError(f"ids.jsonl mismatch: {len(self.ids)} vs N={self.N}")

        # memmaps
        input_dtype = np.int32 if self.input_ids_dtype == "int32" else np.int64

        self.input_ids_mm = np.memmap(
            os.path.join(self.mem_dir, "input_ids.bin"),
            mode="r",
            dtype=input_dtype,
            shape=(self.total_tokens,),
        )
        self.attn_mm = np.memmap(
            os.path.join(self.mem_dir, "attention.bin"),
            mode="r",
            dtype=np.uint8,
            shape=(self.total_tokens,),
        )
        self.image_mask_mm = np.memmap(
            os.path.join(self.mem_dir, "image_mask.bin"),
            mode="r",
            dtype=np.uint8,
            shape=(self.total_tokens,),
        )
        self.text_mask_mm = np.memmap(
            os.path.join(self.mem_dir, "text_mask.bin"),
            mode="r",
            dtype=np.uint8,
            shape=(self.total_tokens,),
        )

        self.vision_offsets = np.load(os.path.join(self.mem_dir, "vision_offsets.npy"))
        self.vision_lengths = np.load(os.path.join(self.mem_dir, "vision_lengths.npy"))
        vd = np.load(os.path.join(self.mem_dir, "vision_dim.npy"))
        self.D = int(vd[0])

        v_dtype = np.float16 if self.vision_dtype == "float16" else np.float32
        total_vision_elems = int((self.vision_lengths.astype(np.int64) * self.D).sum())

        self.vision_mm = np.memmap(
            os.path.join(self.mem_dir, "vision.bin"),
            mode="r",
            dtype=v_dtype,
            shape=(total_vision_elems,),
        )
        # C, H, W = self.pixel_shape
        # self.pixel_mm = np.memmap(
        #     os.path.join(self.mem_dir, "pixel_values.bin"),
        #     mode="r",
        #     dtype=np.float16,
        #     shape=(self.N, int(C), int(H), int(W)),
        # )

        print(f"[ESNLI MemmapDataset] split={split} N={self.N} mem_dir={self.mem_dir}")

    def __len__(self) -> int:
        # return self.N
        return 10

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        off = int(self.offsets[idx])
        L = int(self.lengths[idx])

        input_ids = torch.from_numpy(np.array(self.input_ids_mm[off:off + L], copy=True)).to(torch.long)
        attention_mask = torch.from_numpy(np.array(self.attn_mm[off:off + L], copy=True)).to(torch.long)

        image_mask = torch.from_numpy(np.array(self.image_mask_mm[off:off + L], copy=True)).to(torch.bool)
        text_mask = torch.from_numpy(np.array(self.text_mask_mm[off:off + L], copy=True)).to(torch.bool)

        out: Dict[str, Any] = {
            "id": self.ids[idx],
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_mask": image_mask,
            "text_mask": text_mask,
        }

        voff = int(self.vision_offsets[idx])
        nimg = int(self.vision_lengths[idx])

        if nimg == 0:
            vision = torch.empty((0, self.D), dtype=torch.float16 if self.vision_dtype == "float16" else torch.float32)
        else:
            start = voff
            end = voff + nimg * self.D
            flat = np.array(self.vision_mm[start:end], copy=True)
            vision = torch.from_numpy(flat).view(nimg, self.D)

        out["vision_embeds"] = vision
        out["vision_len"] = torch.tensor(nimg, dtype=torch.long)
        # out["pixel_values"] = torch.from_numpy(np.array(self.pixel_mm[idx], copy=True))  # float16

        return out


# =========================
# Collate (LEFT pad text; pad masks)
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


def _pad_2d_by_rows(seqs: List[torch.Tensor], pad_val: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads a list of [Ni, D] tensors to [B, Nmax, D].
    Returns (padded, mask) where mask is [B, Nmax] bool (True for real rows).
    """
    B = len(seqs)
    D = int(seqs[0].shape[1]) if B > 0 and seqs[0].dim() == 2 else 0
    Nmax = max(int(x.shape[0]) for x in seqs) if B > 0 else 0

    padded = torch.full((B, Nmax, D), float(pad_val), dtype=seqs[0].dtype if B > 0 else torch.float16)
    mask = torch.zeros((B, Nmax), dtype=torch.bool)

    for i, x in enumerate(seqs):
        n = int(x.shape[0])
        if n > 0:
            padded[i, :n, :] = x
            mask[i, :n] = True

    return padded, mask


def esnli_memmap_collate(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, Any]:
    ids = [b["id"] for b in batch]
    labels = torch.stack([b["label"] for b in batch], dim=0)

    input_ids = _left_pad_1d([b["input_ids"] for b in batch], pad_val=int(pad_token_id), dtype=torch.long)
    attention_mask = _left_pad_1d([b["attention_mask"] for b in batch], pad_val=0, dtype=torch.long)

    image_mask = _left_pad_bool([b["image_mask"] for b in batch])
    text_mask = _left_pad_bool([b["text_mask"] for b in batch])

    data: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image_mask": image_mask,
        "text_mask": text_mask,
    }

    # data["image_grid_thw"] = torch.stack([b["image_grid_thw"] for b in batch], dim=0)

    # data["pixel_values"] = torch.stack([b["pixel_values"] for b in batch], dim=0)

    if "vision_embeds" in batch[0]:
        vis_list = [b["vision_embeds"] for b in batch]
        non_empty = [v for v in vis_list if v.numel() > 0]
        if len(non_empty) == 0:
            data["vision_embeds"] = torch.empty((len(batch), 0, 0), dtype=torch.float16)
            data["vision_mask"] = torch.empty((len(batch), 0), dtype=torch.bool)
        else:
            fixed = []
            D = int(non_empty[0].shape[1])
            for v in vis_list:
                if v.dim() == 1:
                    v = v.view(1, -1)
                if v.numel() == 0:
                    v = torch.empty((0, D), dtype=non_empty[0].dtype)
                fixed.append(v)
            vis_pad, vis_mask = _pad_2d_by_rows(fixed, pad_val=0.0)
            data["vision_embeds"] = vis_pad
            data["vision_mask"] = vis_mask

    return {"ids": ids, "label": labels, "data": data}


# =========================
# Dataloader wrapper
# =========================

class ESNLI_MemmapDataloader:
    """
    Builds memmap caches if needed (one-time), then exposes train/validation/test loaders.

    Expects your config to have:
      config.dataset.cache_root
      config.training_params.batch_size
      optionally config.model.pad_token_id (else default 0)
    """

    def __init__(
        self,
        config,
        *,
        input_ids_dtype: str = "int32",
        vision_dtype: str = "float16",
        store_pixel_values: bool = False,
        num_workers: Optional[int] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        force_rebuild_memmap: bool = False,   # <-- NEW: force rebuild without manual deletion
    ):
        cache_root = config.dataset.cache_root
        batch_size = int(config.training_params.batch_size)
        pad_token_id = int(getattr(getattr(config, "model", None), "pad_token_id", 0))

        g = torch.Generator()
        g.manual_seed(0)

        def seed_worker(worker_id: int):
            worker_seed = torch.initial_seed() % (2**32)
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        if num_workers is None:
            total_cpus = multiprocessing.cpu_count()
            num_workers = max(2, min(8, total_cpus // 8 if total_cpus >= 16 else 4))

        print(
            f"[ESNLI MemmapLoader] CPUs={multiprocessing.cpu_count()} | num_workers={num_workers} | "
            f"prefetch_factor={prefetch_factor} | persistent_workers={persistent_workers} | pin_memory={pin_memory} | "
            f"store_pixel_values={store_pixel_values} | force_rebuild_memmap={force_rebuild_memmap}"
        )

        # Ensure memmap exists BEFORE creating DataLoader workers
        for split in ( "validation", "test"):
            split_dir = os.path.join(cache_root, split)
            mem_dir = os.path.join(split_dir, "_memmap")
            meta_path = os.path.join(mem_dir, "meta.json")
            done_flag = os.path.join(mem_dir, ".DONE")

            if force_rebuild_memmap and os.path.isdir(mem_dir):
                # nuke just the completion markers + meta to force rebuild safely
                # (builder will overwrite the .bin files anyway)
                for p in [done_flag, meta_path]:
                    try:
                        os.remove(p)
                    except FileNotFoundError:
                        pass

            if not (os.path.isfile(meta_path) and os.path.isfile(done_flag)):
                print(f"[ESNLI MemmapLoader] Building memmap cache for split={split} ...")
                build_memmap_from_token_shards(
                    split_dir=split_dir,
                    out_dir=mem_dir,
                    input_ids_dtype=input_ids_dtype,
                    vision_dtype=vision_dtype,
                    store_pixel_values=store_pixel_values,
                )

        def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            return esnli_memmap_collate(batch, pad_token_id=pad_token_id)

        self.collate_fn = collate_fn

        def make_loader(split: str, shuffle: bool):
            ds = ESNLI_MemmapDataset(cache_root=cache_root, split=split)
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

        self.train_loader = make_loader("validation", shuffle=True)
        self.valid_loader = make_loader("validation", shuffle=False)
        self.test_loader = make_loader("test", shuffle=False)
