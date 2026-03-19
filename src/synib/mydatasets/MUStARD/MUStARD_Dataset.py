"""
MUStARD raw-video dataset and dataloader.

Returns items with the project-standard schema:
    {
        "data": {
            "text":               str,
            "speaker":            str,
            "speaker_id":         int,
            "video_frames":       Tensor (num_frames, 3, H, W) float32 in [0, 1],
            "context":            list[str],
        },
        "label": LongTensor,   # 0 = not sarcastic, 1 = sarcastic
        "id":    str,          # clip ID
    }
"""

import json
import logging
import multiprocessing
import os
import random
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

__all__ = ["MUStARD_RawDataset", "collate_mustard_raw", "MUStARD_Raw_Dataloader"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MUStARD_RawDataset(Dataset):
    """
    Loads MUStARD raw video clips from disk and returns (data, label, id) triples.

    Config fields consumed:
        config.dataset.data_roots        – path to metadata.json *or* the mustard_raw/ dir
        config.dataset.fps               – frames per second to sample (default 1)
        config.dataset.image_size        – spatial resize target (default 224)
        config.dataset.val_split_rate    – fraction for validation (default 0.15)
        config.dataset.test_split_rate   – fraction for test (default 0.15)
        config.training_params.seed      – random seed for splits (default 42)
    """

    def __init__(self, config, split: str = "train"):
        super().__init__()
        self.split = split

        # ── locate metadata.json and videos/ ─────────────────────────────────
        data_roots = config.dataset.data_roots
        if os.path.isfile(data_roots):
            metadata_path = data_roots
            self.videos_dir = os.path.join(os.path.dirname(metadata_path), "videos")
        else:
            metadata_path = os.path.join(data_roots, "metadata.json")
            self.videos_dir = os.path.join(data_roots, "videos")

        with open(metadata_path) as f:
            all_records = json.load(f)

        # ── build speaker2id from the full corpus (consistent across splits) ─
        speakers = sorted({r["speaker"] for r in all_records})
        self.speaker2id: Dict[str, int] = {s: i for i, s in enumerate(speakers)}

        # ── stratified train / val / test split ───────────────────────────────
        seed = getattr(config.training_params, "seed", 42)
        val_rate = getattr(config.dataset, "val_split_rate", 0.15)
        test_rate = getattr(config.dataset, "test_split_rate", 0.15)

        labels = [int(r["sarcasm"]) for r in all_records]

        train_val, test_records = train_test_split(
            all_records,
            test_size=test_rate,
            random_state=seed,
            stratify=labels,
        )
        tv_labels = [int(r["sarcasm"]) for r in train_val]
        val_frac = val_rate / (1.0 - test_rate)
        train_records, val_records = train_test_split(
            train_val,
            test_size=val_frac,
            random_state=seed,
            stratify=tv_labels,
        )

        if split == "train":
            self.records = train_records
        elif split in ("val", "validation", "dev"):
            self.records = val_records
        elif split == "test":
            self.records = test_records
        else:
            raise ValueError(f"Unknown split: {split!r}. Use 'train', 'val', or 'test'.")

        # ── video extraction settings ─────────────────────────────────────────
        self.fps = getattr(config.dataset, "fps", 1)
        self.image_size = getattr(config.dataset, "image_size", 224)

        log.info(
            "MUStARD_RawDataset split=%s: %d samples, fps=%s, %d speakers",
            split, len(self.records), self.fps, len(self.speaker2id),
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def _extract_frames(self, video_path: str) -> torch.Tensor:
        """
        Extract frames from *video_path* sampled at self.fps.

        Returns a float32 tensor of shape (n_frames, 3, H, W) with values in [0, 1].
        Falls back to a single zero frame if the video cannot be opened.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.warning("Could not open video: %s — returning zero frame.", video_path)
            return torch.zeros(1, 3, self.image_size, self.image_size)

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        duration_secs = total_frames / native_fps
        n_target = max(1, int(duration_secs * self.fps))

        # Native frame indices we want to capture
        target_indices = [
            min(int(i * native_fps / self.fps), total_frames - 1)
            for i in range(n_target)
        ]
        target_set = set(target_indices)
        stop_at = max(target_set)

        # Sequential read — capture only the needed frames
        frame_map: Dict[int, torch.Tensor] = {}
        frame_idx = 0
        last_tensor: Optional[torch.Tensor] = None

        while frame_idx <= stop_at:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in target_set:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(frame_rgb, (self.image_size, self.image_size))
                t = torch.from_numpy(resized.copy()).permute(2, 0, 1).float() / 255.0
                frame_map[frame_idx] = t
                last_tensor = t
            frame_idx += 1

        cap.release()

        zero = torch.zeros(3, self.image_size, self.image_size)
        fallback = last_tensor if last_tensor is not None else zero

        frames = [frame_map.get(fi, fallback) for fi in target_indices]
        return torch.stack(frames, dim=0)  # (n_target, 3, H, W)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        video_path = os.path.join(self.videos_dir, f"{rec['id']}.mp4")
        video_frames = self._extract_frames(video_path)

        return {
            "data": {
                "text": rec["utterance"],
                "speaker": rec["speaker"],
                "speaker_id": self.speaker2id[rec["speaker"]],
                "video_frames": video_frames,         # (F, 3, H, W)
                "context": list(rec.get("context", [])),
            },
            "label": torch.tensor(int(rec["sarcasm"]), dtype=torch.long),
            "id": rec["id"],
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_mustard_raw(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate MUStARD_RawDataset items into a padded batch.

    video_frames is padded to max_frames in the batch by repeating the last frame.
    attention_mask_video has 1 for real frames and 0 for padding.

    Returns:
        {
            "data": {
                "text":                list[str],
                "speaker":             list[str],
                "speaker_id":          LongTensor (B,),
                "video_frames":        FloatTensor (B, max_F, 3, H, W),
                "attention_mask_video": LongTensor (B, max_F),
                "context":             list[list[str]],
            },
            "label": LongTensor (B,),
            "id":    list[str],
        }
    """
    max_frames = max(b["data"]["video_frames"].shape[0] for b in batch)

    padded_frames = []
    attention_masks = []
    for b in batch:
        frames = b["data"]["video_frames"]  # (F, 3, H, W)
        n = frames.shape[0]
        pad_n = max_frames - n
        if pad_n > 0:
            pad = frames[-1:].expand(pad_n, -1, -1, -1)
            frames = torch.cat([frames, pad], dim=0)
        padded_frames.append(frames)

        mask = torch.zeros(max_frames, dtype=torch.long)
        mask[:n] = 1
        attention_masks.append(mask)

    return {
        "data": {
            "text": [b["data"]["text"] for b in batch],
            "speaker": [b["data"]["speaker"] for b in batch],
            "speaker_id": torch.tensor(
                [b["data"]["speaker_id"] for b in batch], dtype=torch.long
            ),
            "video_frames": torch.stack(padded_frames, dim=0),       # (B, max_F, 3, H, W)
            "attention_mask_video": torch.stack(attention_masks, dim=0),  # (B, max_F)
            "context": [b["data"]["context"] for b in batch],
        },
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "id": [b["id"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

class MUStARD_Raw_Dataloader:
    """
    Wraps MUStARD_RawDataset into train / valid / test DataLoaders.

    Follows the ESNLI_VE_Dataloader pattern:
      - seeded generator + worker_init_fn for reproducibility
      - shuffle=True for train, False for val/test
    """

    def __init__(self, config):
        batch_size = config.training_params.batch_size
        seed = getattr(config.training_params, "seed", 42)

        g = torch.Generator()
        g.manual_seed(seed)

        def seed_worker(worker_id: int) -> None:
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        total_cpus = multiprocessing.cpu_count()
        num_gpus = max(1, len(getattr(config.training_params, "gpu_device", [0])))
        workers = max(1, min(12, total_cpus // num_gpus))
        # workers = 1

        log.info("MUStARD_Raw_Dataloader | CPUs: %d | workers: %d", total_cpus, workers)

        self.train_loader = DataLoader(
            MUStARD_RawDataset(config=config, split="train"),
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            worker_init_fn=seed_worker,
            collate_fn=collate_mustard_raw,
            num_workers=workers,
            pin_memory=True,
            drop_last=True,
        )

        self.valid_loader = DataLoader(
            MUStARD_RawDataset(config=config, split="val"),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_mustard_raw,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )

        self.test_loader = DataLoader(
            MUStARD_RawDataset(config=config, split="test"),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_mustard_raw,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )


# ---------------------------------------------------------------------------
# Quick exploration entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    from easydict import EasyDict as edict

    cfg = edict({
        "dataset": {
            "data_roots": "src/synib/mydatasets/MUStARD/prepared/mustard_raw",
            "fps": 1,
            "image_size": 224,
            "val_split_rate": 0.15,
            "test_split_rate": 0.15,
        },
        "training_params": {"batch_size": 2, "seed": 42},
    })

    dl = MUStARD_Raw_Dataloader(config=cfg)
    batch = next(iter(dl.train_loader))
    print("Keys:", batch.keys())
    print("Data keys:", batch["data"].keys())
    print("video_frames shape:", tuple(batch["data"]["video_frames"].shape))
    print("attention_mask_video shape:", tuple(batch["data"]["attention_mask_video"].shape))
    print("label:", batch["label"])
    print("id:", batch["id"])
    print("text:", batch["data"]["text"])
    print("speaker:", batch["data"]["speaker"])
    print("context:", batch["data"]["context"])


if __name__ == "__main__":
    main()
